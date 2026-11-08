"""자동 실행기 (plan/01 5 · 6장). 서버에서 한 번 걸면 끝까지 돌고 멈춘다. 죽으면 다시 띄워 저장 지점부터 잇는다.

단계
  0. 복제 수 K 정하기 — 넓게 훑기 앞 48 점을 복제 8 번으로 굴려, 복제 흩어짐이 점 사이 차이보다 충분히 작은 가장 작은 K
  1. 넓게 훑기 — 서보 가속 흉내 끔 · 켬 각각 Sobol 점. 상위 점을 복제 2K 로 다시 굴려 손실이 낮은 쪽을 고른다
  2. BIPOP-CMA-ES — 넓게 훑기 최저점에서 시작, 재시작마다 큰 · 작은 개체 수를 번갈아 쓴다
  3. 프로파일 — 값 하나를 범위 9 칸에 고정하고 나머지를 다시 맞춰, 그 값이 기록으로 정해지는지 본다
  4. 최종 — 최저점을 복제 4K 로 다시 굴려 기록별 표와 함께 결과를 쓴다

출력은 real_fit/out/ 의 progress.log(진행), status.json(진행 화면 `python -m real_fit.watch` 가 읽는다),
state.pkl(저장 지점), result.json(결과), DONE(끝 표시) 이다.
"""
import json
import math
import os
import pickle
import time
from pathlib import Path

import cma  # noqa: F401  — 2 단계에서 처음 쓰지만 없으면 시작할 때 멈춘다
import jax
import jax.numpy as jp
import numpy as np

from real_fit import env as E, fingerprint as FP, model as M, policy as P, score as S
from real_fit.pack import OUT as PACK

ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get("FIT_SMOKE") == "1"  # 맥에서 끝까지 도는지만 보는 축소 설정. 출력 폴더도 따로 쓴다
OUTDIR = Path(__file__).resolve().parent / ("out_smoke" if SMOKE else "out")
STATE = OUTDIR / "state.pkl"
STATUS = OUTDIR / "status.json"
STAGE_NAMES = ["복제 수 정하기", "넓게 훑기", "진화 탐색", "프로파일 확인", "최종 재평가"]
LOG = OUTDIR / "progress.log"
MAX_ENVS = int(os.environ.get("FIT_MAX_ENVS", "8192"))
IMPL = os.environ.get("FIT_IMPL", "warp")
DEG = math.pi / 180

# (이름, 하한, 상한, 시뮬 단위 배율). 범위는 plan/01 1장.
SPACE = [
    ("dx_mm", -10.0, 40.0, 1e-3),
    ("dy_mm", -15.0, 15.0, 1e-3),
    ("hx_mm", -20.0, 20.0, 1e-3),
    ("hy_mm", -20.0, 20.0, 1e-3),
    ("leg_play_deg", 0.0, 2.0, DEG),        # 반각
    ("head_play_deg", 0.0, 6.0, DEG),       # 반각
    ("backlash_deg", 0.0, 2.5, DEG / 2),    # 전체 → 반각
    ("mu_maru", 0.3, 2.0, 1.0),
    ("mu_tile", 0.3, 2.0, 1.0),
]
DIM = len(SPACE)
MIN_HALF = 1e-4  # 유격 0 은 범위 0 관절이 되므로 아주 작은 값으로 둔다
MODES = (0.0, E.SERVO_ACCEL_253)
# 기록 길이(주기)별로 나눠 굴린다. 기록 54 개의 길이에서 꼬리 낭비가 가장 작은 6 묶음이다(합 43,770 스텝, 실제 41,280).
# 처음 (600, 1200, 2000) 은 60,600 스텝으로 32 % 를 버렸다.
BUCKETS = (292, 600, 750, 1050, 1300, 2000)

N_PILOT, K_PILOT = 48, 8
N_SOBOL = 4096          # 모드마다
SOBOL_BATCH = 256
TOP_RECHECK = 16
LAMBDA0 = 32
MAX_RESTARTS = 9
CMA_BUDGET = 30000      # 후보 수
CMA_MAXITER = 250
# 잡음 바닥에서는 CMA 자체 멈춤 조건(손실 차이 · σ)이 걸리지 않아 최대 세대까지 예산을 쓴다. 실행이 STALL_MIN 세대를 넘긴
# 뒤 STALL_EVERY 세대마다, 최근 STALL_WIN 세대와 STALL_GENS 세대 앞 STALL_WIN 세대의 세대 중앙 평균을 비교한다. 개선에
# 두 평균 차 오차(세대 중앙의 흩어짐으로 잰다)의 2 배를 더해도 STALL_DMIN 보다 작은 판정이 두 번 이어지면 끝낸다(개선의
# 위쪽 한계가 값어치 아래). 세대 최저는 λ 개 중 최저라 아래로 치우쳐 쓰지 않는다.
STALL_MIN, STALL_EVERY, STALL_WIN, STALL_GENS = 40, 5, 10, 20
STALL_DMIN = 5.0  # 이어 갈 값어치가 있는 20 세대 개선(손실 점). 최저점 근처 후보 손실 표준오차(4.66)다
FINAL_TOL = 5.0   # 4 단계에서 고른 점이 다른 후보보다 나빠도 되는 손실 차
FINAL_VER = 3     # 4 단계 재평가 방식 판. 저장된 값의 판이 다르면 버리고 다시 잰다
PROFILE_LEVELS = 9
PROFILE_LAMBDA, PROFILE_ITERS = 16, 30
JITTER_RAD = 1e-3
K_MAX = 256  # 복제 흔들림 수. 앞 64 개는 64 로 만들 때와 같다(난수를 앞에서부터 채운다)
SMOKE_RECORDS = ("log_v227b_vx005", "run_225400")  # 두 형식 · 정책 둘
PROFILE_DIMS = DIM  # 프로파일 · 이웃 검사를 하는 값의 수
if SMOKE:
    BUCKETS = (20,)
    N_PILOT, K_PILOT, N_SOBOL, SOBOL_BATCH, TOP_RECHECK = 2, 2, 2, 2, 1
    LAMBDA0, MAX_RESTARTS, CMA_BUDGET, CMA_MAXITER = 2, 0, 2, 1
    PROFILE_LEVELS, PROFILE_LAMBDA, PROFILE_ITERS, PROFILE_DIMS = 2, 2, 1, 1


# 목적함수가 같은 이전 판의 지문. 이 판들의 저장 지점은 그대로 이어받는다.
# ecfedf563803cb80: 2026-10-12 01:42 서버 실행. 이 판과 다른 것은 길이 묶음 분할(꼬리 낭비만 줄임)과 쓰지 않는
# 보상 계산 생략뿐이다. 둘 다 기록 구간의 시뮬 · 지표 · 손실 식을 바꾸지 않는다.
# rest: 그 실행 때 서버에서 `python -m real_fit.fingerprint model.py,policy.py,score.py 16384` 로 잰 값.
# per: 그 판의 후보 한 점 · 복제 하나 계산량(길이 묶음 (600, 1200, 2000)). 이어받은 진행량을 새 단위로 환산한다.
# 68c88b1f72adb2c9: 2026-10-12 05:47 서버 실행(1e156b3). 이 판과 다른 것은 후보가 적을 때 기록을 가장 긴 묶음 한
# 번에 태우는 것뿐이다. 점수는 기록 끝 안쪽만 보고 난수는 기록 번호로 정해진다.
# 6f54c22e941d6535: 2026-10-12 11:24 서버 실행(5f90c0e). 이 판과 다른 것은 GPU 호출 크기를 남은 환경 수에 맞추는 것과
# 2 단계 후보 예산(6 만 → 3 만)뿐이다. 같은 후보도 호출 크기 · 자리 배치에 따라 손실이 조금 달라진다. 차이는 복제 흔들림 크기이고 치우침이 없다
# (research/04 36장).
# aea59082ed376144: 2026-10-12 11:54 서버 실행(7dd3de9). 이 판과 다른 것은 2 단계 정체 멈춤(STALL_*)뿐이다.
# ae22adb0468242fd: 2026-10-12 14:22 서버 실행(928c33b). 이 판과 다른 것은 정체 멈춤 판정식뿐이다.
# 66fe234bab027ffd: 2026-10-12 14:42 서버 실행(1a8ea56). 이 판과 다른 것은 4 단계 재평가(새 복제 · 고정 호출 조건 ·
# 짝지은 차이)와 evaluate 의 rep0 · canon 인자뿐이다. 2 단계 탐색 평가는 그대로다.
COMPATIBLE_FINGERPRINTS = {"ecfedf563803cb80": dict(rest="25be1fc770e0877a", per=60600),
                           "68c88b1f72adb2c9": dict(rest="25be1fc770e0877a", per=43770),
                           "6f54c22e941d6535": dict(rest="25be1fc770e0877a", per=43770),
                           "aea59082ed376144": dict(rest="25be1fc770e0877a", per=43770),
                           "ae22adb0468242fd": dict(rest="25be1fc770e0877a", per=43770),
                           "66fe234bab027ffd": dict(rest="25be1fc770e0877a", per=43770)}


def fingerprint() -> str:
    """데이터 · 코드 · 정책 · 탐색 설정의 지문. 저장 지점의 지문과 다르면 그 저장 지점은 이어받지 않는다."""
    return FP.digest(FP.CODE, SMOKE, IMPL, MAX_ENVS)


def legacy_ok(old: str) -> bool:
    """옛 판 저장 지점을 이어받아도 되는가. 바뀐 두 파일(env.py · search.py)을 뺀 나머지 전부의 지문이 옛 판 실행
    때 서버에서 잰 값과 같아야 한다(데이터 · 다른 코드 · 정책 · 학습 환경 · 라이브러리 판 · 실행 설정)."""
    want = COMPATIBLE_FINGERPRINTS.get(old)
    return want is not None and FP.digest(("model.py", "policy.py", "score.py"), SMOKE, IMPL, MAX_ENVS) == want["rest"]


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(state))
    tmp.replace(STATE)


def to_phys(u: np.ndarray) -> dict:
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return {n: lo + (hi - lo) * u[..., i] for i, (n, lo, hi, _) in enumerate(SPACE)}


class Progress:
    """진행 화면용 상태. 단위는 환경-스텝이다. 단계별 계획량은 복제 수 K 가 정해지면 다시 센다."""

    def __init__(self, ev, st):
        self.ev, self.st = ev, st
        self.t_last, self.rate = time.time(), None

    def plan(self) -> list[float]:
        per = float(sum(BUCKETS[b] for b in self.ev.bucket))  # 후보 한 점 · 복제 하나
        K = self.st.get("K", 4)
        K2 = self.st.get("K2", 8)  # 1 단계 끝에 정한다. 그 전에는 8 로 어림한다
        return [N_PILOT * K_PILOT * per,
                (2 * N_SOBOL + 2 * TOP_RECHECK * 2) * K * per,
                CMA_BUDGET * K2 * per,
                DIM * PROFILE_LEVELS * PROFILE_LAMBDA * PROFILE_ITERS * K2 * per,
                (11 * 4 + 2 * DIM * 2 + 2 + 2 + DIM * PROFILE_LEVELS * 4) * K2 * per]  # 후보 11 × 4K, 이웃 18 × 2K, 전압 · 지표표 2K, 프로파일 81 × 4K

    def add(self, steps: float, note: str = "") -> None:
        st = self.st
        done = st.setdefault("done_steps", [0.0] * len(STAGE_NAMES))
        done[st["stage"]] += steps
        now = time.time()
        r = steps / max(now - self.t_last, 1e-6)
        self.rate = r if self.rate is None else 0.8 * self.rate + 0.2 * r
        self.t_last = now
        if note:
            st["note"] = note
        self.write()

    def write(self) -> None:
        st = self.st
        plan = self.plan()
        done = st.get("done_steps", [0.0] * len(STAGE_NAMES))
        i = min(st["stage"], len(STAGE_NAMES) - 1)
        frac = [1.0 if j < st["stage"] else min(done[j] / plan[j], 0.99) if j == st["stage"] else 0.0
                for j in range(len(STAGE_NAMES))]
        overall = sum(f * p for f, p in zip(frac, plan)) / sum(plan)
        remain = sum((1 - f) * p for f, p in zip(frac, plan))
        best = st.get("final_u", st.get("x_best"))
        s = dict(time=time.time(), stage=st["stage"], stage_name=STAGE_NAMES[i] if st["stage"] < len(STAGE_NAMES) else "끝",
                 stage_frac=frac[i] if st["stage"] < len(STAGE_NAMES) else 1.0, overall=overall,
                 eta_h=remain / self.rate / 3600 if self.rate else None, rate=self.rate, K=st.get("K"),
                 mode=st.get("mode"), f_best=st.get("final_loss", st.get("f_best")), note=st.get("note", ""),
                 checks=st.get("checks"),
                 best={n: float(v) for n, v in to_phys(best).items()} if best is not None else None)
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATUS)


class Evaluator:
    def __init__(self):
        d = dict(np.load(PACK))
        if SMOKE:
            keep = np.array([list(d["name"]).index(n) for n in SMOKE_RECORDS])
            n0 = len(d["name"])
            d = {k: (v[keep] if np.ndim(v) and len(v) == n0 else v) for k, v in d.items()}
        self.d = d
        self.names, self.floor = d["name"], d["floor"]
        self.R = len(self.names)
        self.end, self.move = d["end"], d["move"]
        if SMOKE:
            self.end, self.move = np.minimum(self.end, BUCKETS[-1]), np.minimum(self.move, 10)
        self.act = P.load_many([ROOT / "policies" / "v2" / f"{p}_300M.onnx" for p in d["policies"]])
        self.cmd = jp.asarray(d["cmd"])
        real = S.metrics(d["real_acc"].transpose(1, 0, 2), d["real_knee"].transpose(1, 0, 2),
                         d["real_tg"].transpose(1, 0, 2), self.move, self.end, d["fall_at_end"] & ~SMOKE)
        self.real = [S.pick(real, r) for r in range(self.R)]
        self.bucket = np.array([next(i for i, b in enumerate(BUCKETS) if e <= b) for e in self.end])
        self.xml = M.build()
        self.envs, self.fns = {}, {}
        self.progress = None
        self.metrics = None
        mj = M.load(self.xml)
        self.act_qadr = np.array([mj.jnt_qposadr[mj.actuator_trnid[i, 0]] for i in range(mj.nu)])
        # 복제 흔들림은 공통 난수다. 같은 복제 · 같은 기록이면 어느 후보든 같은 흔들림을 받아 후보끼리의 비교에서
        # 흔들림 차이가 빠진다. 복제 0 은 흔들지 않는다.
        self.jitter = np.random.default_rng(12345).uniform(-JITTER_RAD, JITTER_RAD, (K_MAX, self.R, mj.nu))
        self.jitter[0] = 0.0

    def _fn(self, mode: int, b: int, volt: bool = False, size: int = MAX_ENVS):
        key = (mode, b, volt, size)
        if key not in self.fns:
            if (mode, volt, size) not in self.envs:
                self.envs[(mode, volt, size)] = E.FitEnv(self.xml, E.fit_config(IMPL, size, MODES[mode], volt))
            env = self.envs[(mode, volt, size)]
            T = BUCKETS[b]
            roll = E.make_rollout(env, self.act, self.cmd[:, :T + 1], T)
            holder = {}

            def run(p, rec, pid, home, init_q, phase0, cfix, raw):
                mv, ia = E.batched_model(env, p)
                if "f" not in holder:
                    holder["f"] = jax.jit(lambda mv, *a: roll(mv, ia, *a))
                return holder["f"](mv, rec, pid, home, init_q, phase0, cfix, raw, jax.random.PRNGKey(0))

            self.fns[key] = run
        return self.fns[key]

    @staticmethod
    def _chunks(n: int) -> list:
        """환경 n 개를 GPU 호출로 나눈다 → [(시작, 호출 크기)]. 호출 크기는 MAX_ENVS 또는 그 1/2 · 1/4 · 1/8 중 남은
        수를 담는 가장 작은 것이다. 빈자리도 채워 돌기 때문에 크기를 남은 수에 맞추면 그만큼 계산이 준다.
        같은 후보라도 호출 크기가 다르면 손실이 조금 달라진다(같은 크기면 자리 배치와 상관없이 같다)."""
        sizes = [MAX_ENVS >> i for i in range(4) if MAX_ENVS >> i]
        out, s = [], 0
        while s < n:
            out.append((s, min(z for z in sizes if z >= min(n - s, MAX_ENVS))))
            s += out[-1][1]
        return out

    def evaluate(self, U: np.ndarray, mode: int, K: int, volt: bool = False, keep_metrics: bool = False,
                 rep0: int = 0, canon: bool = False) -> np.ndarray:
        """U (C, DIM) → 기록별 · 복제별 손실 (C, K, R). keep_metrics 면 self.metrics[(c, r)] 에 복제별 지표를 모은다.
        rep0: 복제 흔들림 번호 시작(rep0 ~ rep0 + K − 1). 탐색은 0 부터 쓰고 재평가는 탐색에 쓰지 않은 번호를 쓴다.
        canon: 호출 조건을 고정한다(묶음별, 호출 크기 MAX_ENVS). 같은 후보도 호출 조건에 따라 손실이 조금 달라지므로
        서로 비교할 재평가는 이 조건으로 맞춘다(research/04 36장)."""
        assert rep0 + K <= K_MAX, (rep0, K)
        self.metrics = {} if keep_metrics else None
        U = np.atleast_2d(U)
        C = len(U)
        ph = to_phys(U)
        loss = np.zeros((C, K, self.R))
        t0, steps = time.time(), 0
        groups = [(b, np.where(self.bucket == b)[0]) for b in range(len(BUCKETS)) if (self.bucket == b).any()]
        # GPU 는 부를 때마다 호출 크기를 채워 돈다. 후보가 적으면(2 · 3 단계) 묶음마다 부르는 계산이 대부분 빈자리
        # 채움이라, 기록 전부를 가장 긴 묶음 한 번에 태우는 쪽이 싸다. 계산량은 기록 원래 묶음 길이로 센다.
        cost = lambda n, T: sum(z for _, z in self._chunks(n)) * T  # noqa: E731
        split = sum(cost(C * K * len(r), BUCKETS[b]) for b, r in groups)
        merged = cost(C * K * self.R, BUCKETS[int(self.bucket.max())])
        if merged < split and not canon:
            groups = [(int(self.bucket.max()), np.arange(self.R))]
        for b, recs in groups:
            cc, kk, rr = np.meshgrid(np.arange(C), np.arange(K), recs, indexing="ij")
            cc, kk, rr = cc.ravel(), kk.ravel(), rr.ravel()
            n = len(cc)
            pending = []
            for s, size in ([(s, MAX_ENVS) for s in range(0, n, MAX_ENVS)] if canon else self._chunks(n)):
                fn = self._fn(mode, b, volt, size)
                sl = np.arange(s, min(s + size, n))
                pad = np.concatenate([sl, np.full(size - len(sl), sl[-1])])
                c, k, r = cc[pad], kk[pad], rr[pad]
                mu = np.where(self.floor[r] == "타일", ph["mu_tile"][c], ph["mu_maru"][c])
                p = {nm: jp.asarray(np.maximum(ph[nm][c] * f, MIN_HALF if "play" in nm or "backlash" in nm else -np.inf),
                                    jp.float32)
                     for nm, _, _, f in SPACE if not nm.startswith("mu_")}
                p = dict(dx=p["dx_mm"], dy=p["dy_mm"], hx=p["hx_mm"], hy=p["hy_mm"], leg=p["leg_play_deg"],
                         head=p["head_play_deg"], backlash=jp.repeat(p["backlash_deg"][:, None], 10, 1),
                         floor=jp.asarray(mu, jp.float32))
                iq = self.d["init_q"][r].copy()
                iq[:, self.act_qadr] += self.jitter[rep0 + k, r]  # 구동 관절 시작각을 ±0.001 rad 안에서 흔든다
                out = fn(p, jp.asarray(r), jp.asarray(self.d["pid"][r]), jp.asarray(self.d["home"][r], jp.float32),
                         jp.asarray(iq, jp.float32), jp.asarray(self.d["phase0"][r]),
                         jp.asarray(self.d["contact_fixed"][r]), jp.asarray(self.d["raw_imu"][r]))
                nom = int(np.asarray(BUCKETS)[self.bucket[r[:len(sl)]]].sum())
                pending.append((sl, len(sl), c, k, r, out, nom))
                steps += nom
                if len(pending) > 1:
                    self._score(pending.pop(0), loss)
            for item in pending:
                self._score(item, loss)
        dt = time.time() - t0
        self.rate = steps / max(dt, 1e-9)
        return loss

    def _score(self, item, loss):
        sl, m, c, k, r, out, nom = item
        acc, knee, tg = (np.asarray(o)[:, :m] for o in out)
        c, k, r = c[:m], k[:m], r[:m]
        sim = S.metrics(acc, knee, tg, self.move[r], self.end[r])
        for rec in np.unique(r):
            i = np.where(r == rec)[0]
            sub = {kk: v[i] for kk, v in sim.items()}
            loss[c[i], k[i], rec] = S.record_loss(sub, self.real[rec])
            if self.metrics is not None:
                for j in i:
                    self.metrics.setdefault((int(c[j]), int(rec)), []).append(S.pick(sim, j))
        if self.progress is not None:  # GPU 결과를 받아 채점까지 끝난 양으로 센다
            self.progress.add(nom)


def total(L: np.ndarray) -> np.ndarray:
    """(C, K, R) → 후보별 손실(복제마다 기록 손실을 매기고 복제 평균, 기록 합). 유한하지 않으면 큰 값으로 바꾼다."""
    t = L.mean(1).sum(-1)
    return np.where(np.isfinite(t), t, 1e6)


def stage_pilot(ev, st):
    from scipy.stats import qmc
    U = qmc.Sobol(DIM, scramble=True, seed=0).random(N_PILOT)
    L = ev.evaluate(U, 1, K_PILOT)
    tot = L.sum(-1)  # (C, K)
    sd = tot.std(1, ddof=1)
    iqr = np.subtract(*np.percentile(tot.mean(1), [75, 25]))
    K = 16
    for k in (2, 4, 8, 16):  # K_MAX 안
        if np.median(sd) / math.sqrt(k) < 0.05 * iqr:
            K = k
            break
    if SMOKE:
        K = 1
    log(f"[0] 복제 흩어짐 중앙 {np.median(sd):.2f}, 점 사이 IQR {iqr:.2f} → K = {K}  (처리량 {ev.rate:,.0f} 환경-스텝/초)")
    st.update(stage=1, K=K, pilot=dict(U=U, L=L))


def stage_sobol(ev, st):
    from scipy.stats import qmc
    K = st["K"]
    sob = st.setdefault("sobol", {m: dict(U=qmc.Sobol(DIM, scramble=True, seed=1 + m).random(N_SOBOL), loss=[]) for m in (0, 1)})
    for m in (0, 1):
        s = sob[m]
        while len(s["loss"]) < N_SOBOL:
            i = len(s["loss"])
            U = s["U"][i:i + SOBOL_BATCH]
            t0 = time.time()
            s["loss"].extend(total(ev.evaluate(U, m, K)).tolist())
            save(st)
            done = sum(len(sob[x]["loss"]) for x in (0, 1))
            eta = (time.time() - t0) / len(U) * (2 * N_SOBOL - done) / 3600
            log(f"[1] 모드 {m} {len(s['loss'])}/{N_SOBOL}  최저 {min(s['loss']):.2f}  (처리량 {ev.rate:,.0f}/초, 1단계 남은 {eta:.1f} h)")
            st["note"] = f"모드 {m} {len(s['loss'])}/{N_SOBOL} 점, 이 모드 최저 {min(s['loss']):.2f}"
    best, sds = {}, {}
    for m in (0, 1):
        s = sob[m]
        top = np.argsort(s["loss"])[:TOP_RECHECK]
        L = ev.evaluate(s["U"][top], m, 2 * K)
        re = total(L)
        j = int(np.argmin(re))
        best[m] = (float(re[j]), s["U"][top[j]])
        sds[m] = float(np.median(L.sum(-1).std(1, ddof=1)))
        log(f"[1] 모드 {m}(가속 흉내 {MODES[m]}) 상위 {TOP_RECHECK} 재평가 최저 {re[j]:.2f}, 복제 흩어짐 {sds[m]:.2f}  {fmt(s['U'][top[j]])}")
    mode = min(best, key=lambda m: best[m][0])
    # 2 ~ 4 단계 복제 수. 0 단계 K 는 범위 전체에 흩은 점 사이의 차이로 정해 넓게 훑기에만 맞다. 최저점 근처는
    # 후보끼리 차이가 작아서, 상위 점의 복제 흩어짐으로 후보 손실 오차가 최저 손실의 1 % 이하가 되는 K 를 쓴다.
    K2 = 16
    for k in (4, 8, 16):
        if sds[mode] / math.sqrt(k) <= 0.01 * best[mode][0]:
            K2 = k
            break
    K2 = max(K2, K) if not SMOKE else K
    log(f"[1] 고른 모드 {mode} (가속 흉내 {MODES[mode]}), 2 ~ 4 단계 복제 수 K2 = {K2}")
    st.update(stage=2, mode=mode, x_best=best[mode][1], f_best=best[mode][0], K2=K2)


def fmt(u) -> str:
    ph = to_phys(u)
    return ", ".join(f"{n} {float(ph[n]):.3g}" for n, *_ in SPACE)


def stall_check(hist, run: int, it: int):
    """→ (정체, 개선, 오차). hist 의 (실행, 세대, 세대 최저, 세대 중앙) 만 써서 이어받아도 같은 판정이 나온다.
    판정 세대 t: 개선 = 앞 창(t−G−W+1 ~ t−G) 중앙 평균 − 뒤 창(t−W+1 ~ t) 중앙 평균, 오차 = √(s₁²/W + s₂²/W).
    t 와 t − STALL_EVERY 둘 다 개선 + 2 × 오차 < STALL_DMIN 이면 정체."""
    med = {i: m for r, i, _, m in hist if r == run}
    W, G = STALL_WIN, STALL_GENS

    def judge(t):
        a = [med.get(i) for i in range(t - G - W + 1, t - G + 1)]
        b = [med.get(i) for i in range(t - W + 1, t + 1)]
        if t < STALL_MIN or None in a or None in b:
            return None
        gain = float(np.mean(a) - np.mean(b))
        err = float(math.sqrt(np.var(a, ddof=1) / W + np.var(b, ddof=1) / W))
        return gain + 2 * err < STALL_DMIN, gain, err

    if it % STALL_EVERY:
        return False, math.nan, math.nan
    now, before = judge(it), judge(it - STALL_EVERY)
    if now is None:
        return False, math.nan, math.nan
    return bool(now[0] and before and before[0]), now[1], now[2]


def stage_cma(ev, st):
    import cma
    K, mode = st.get("K2", st["K"]), st["mode"]
    c = st.setdefault("cma", dict(runs=0, evals=0, ev_large=0, ev_small=0, i_large=0, lam_large=LAMBDA0, es=None,
                                  regime="first", hist=[]))
    while True:
        if c["es"] is None:
            if c["runs"] > MAX_RESTARTS or c["evals"] >= CMA_BUDGET:
                break
            rng = np.random.default_rng(100 + c["runs"])  # 재시작 선택은 실행 번호로 정해져 재개해도 같다
            if c["regime"] == "first":
                lam, sig, x0 = LAMBDA0, 0.25, st["x_best"]
            elif c["ev_large"] <= c["ev_small"]:
                c["i_large"] += 1
                lam = LAMBDA0 * 2 ** c["i_large"]
                c["lam_large"], sig, x0, c["regime"] = lam, 0.25, rng.uniform(0, 1, DIM), "large"
            else:
                u = rng.uniform()
                lam = max(8, int(LAMBDA0 * (c["lam_large"] / LAMBDA0) ** (u ** 2)))
                sig, x0, c["regime"] = 0.25 * 10 ** (-2 * rng.uniform()), rng.uniform(0, 1, DIM), "small"
            c["es"] = cma.CMAEvolutionStrategy(list(x0), sig, {"popsize": lam, "bounds": [0, 1], "maxiter": CMA_MAXITER,
                                                               "verbose": -9, "seed": 1000 + c["runs"]})
            log(f"[2] 실행 {c['runs']} 시작: {c['regime']} λ {lam} σ {sig:.3g}")
        es = c["es"]
        X = np.array(es.ask())
        f = total(ev.evaluate(X, mode, K))
        es.tell(list(X), list(f))
        c["evals"] += len(X)
        # 첫 실행은 기본 개체 수라 BIPOP 의 큰 개체 수 계열로 센다.
        c["ev_small" if c["regime"] == "small" else "ev_large"] += len(X)
        j = int(np.argmin(f))
        if f[j] < st["f_best"]:
            st["f_best"], st["x_best"] = float(f[j]), X[j]
        c["hist"].append((c["runs"], es.countiter, float(f.min()), float(np.median(f))))
        stalled, gain, err = stall_check(c["hist"], c["runs"], es.countiter)
        st["note"] = (f"실행 {c['runs']}/{MAX_RESTARTS} ({c['regime']}, λ {len(X)}) 세대 {es.countiter}  σ {es.sigma:.3g}  "
                      f"이 세대 최저 {f.min():.2f}  누적 후보 {c['evals']}/{CMA_BUDGET}")
        if es.countiter % 5 == 0:
            log(f"[2] 실행 {c['runs']} 세대 {es.countiter} 최저 {f.min():.2f} 중앙 {np.median(f):.2f} σ {es.sigma:.3g}  "
                f"전체 최저 {st['f_best']:.2f}  누적 {c['evals']}/{CMA_BUDGET}  (처리량 {ev.rate:,.0f}/초)")
        over = c["evals"] >= CMA_BUDGET
        if es.stop() or over or stalled:
            why = (dict(es.stop()) if es.stop() else "예산 소진" if over else
                   f"정체 — {STALL_GENS} 세대 세대 중앙 개선 {gain:+.2f} + 오차 {err:.2f} × 2 < {STALL_DMIN:g}")
            log(f"[2] 실행 {c['runs']} 끝 {why}  이 실행 최저 {es.result.fbest:.2f}")
            c.setdefault("run_best", []).append((float(es.result.fbest), np.asarray(es.result.xbest)))
            c["runs"] += 1
            c["es"] = None
            if c["regime"] == "first":
                c["regime"] = "large"
        save(st)
    # 잡음 속 최저값은 낮게 치우친다(승자의 저주). 4 단계에서 복제를 늘려 다시 재고 고른다.
    log(f"[2] 끝. 전체 최저 {st['f_best']:.2f}  {fmt(st['x_best'])}")
    st.update(stage=3)


def stage_profile(ev, st):
    import cma
    K, mode = st.get("K2", st["K"]), st["mode"]
    x_best = np.asarray(st["x_best"])
    pr = st.setdefault("profile", {})
    for i in range(PROFILE_DIMS):
        for lv in np.linspace(0, 1, PROFILE_LEVELS):
            key = (i, round(float(lv), 4))
            if key in pr and pr[key].get("done"):
                continue
            item = pr.setdefault(key, dict(done=False))
            free = [j for j in range(DIM) if j != i]
            if "es" not in item or item["es"] is None:
                item["es"] = cma.CMAEvolutionStrategy(list(x_best[free]), 0.1, {
                    "popsize": PROFILE_LAMBDA, "bounds": [0, 1], "maxiter": PROFILE_ITERS, "verbose": -9, "seed": 7})
                item["best"] = math.inf
            es = item["es"]
            while not es.stop():
                Y = np.array(es.ask())
                X = np.empty((len(Y), DIM))
                X[:, free], X[:, i] = Y, lv
                f = total(ev.evaluate(X, mode, K))
                es.tell(list(Y), list(f))
                if f.min() < item["best"]:
                    item["best"], item["x"] = float(f.min()), X[int(np.argmin(f))]
                save(st)
            item.update(done=True, stop=str(dict(es.stop())), es=None)
            save(st)
            n, lo, hi, _ = SPACE[i]
            log(f"[3] {n} = {lo + (hi - lo) * lv:.3g}  최저 {item['best']:.2f}")
            st["note"] = f"{n} = {lo + (hi - lo) * lv:.3g} 끝 (값 {i + 1}/{DIM})"
    st.update(stage=4)


def stage_final(ev, st):
    """최종 점을 고르고 검증한다. 통과 여부는 result.json 의 checks 에 적는다(실패해도 결과는 남긴다).
    재평가는 탐색에 쓴 복제(시범 · 1 단계 재평가 · 2 · 3 단계) 뒤 번호의 새 복제와 고정 호출 조건(canon)으로 한다.
    한 번 고르는 데 쓴 복제로 다시 비교하면 고른 점이 낮게 치우쳐 비교가 기울기 때문에 고르는 묶음과 재는 묶음을 나눈다.
      프로파일 81 점: PS 로 기준점을 고르고 PC 로 기준점과 비교해 폭을 낸다
      최종 후보: FA 로 고르고 FB 로 손실을 보고하며 이웃 · 전압 · 지표표도 FB 로 잰다"""
    from scipy.stats import t as student
    K, mode = st.get("K2", st["K"]), st["mode"]
    n2 = 2 * K
    base = max(K_PILOT, 2 * st["K"], K)
    PS, PC, FA, FB = base, base + n2, base + 2 * n2, base + 3 * n2
    f = st.get("final", {})
    if f.get("ver") != FINAL_VER:  # 다른 판의 재평가 값은 복제 · 호출 조건이 달라 섞지 않는다
        f = st["final"] = dict(ver=FINAL_VER)
    T = lambda L: L.sum(-1)  # noqa: E731 — (C, 복제, 기록) → 복제별 기록 합

    def paired(Ta, tb):
        """Ta (C, n) 와 기준 tb (n,) 를 같은 복제끼리 뺀 차 → (평균, 표준오차)."""
        d = Ta - tb
        return d.mean(-1), d.std(-1, ddof=1) / math.sqrt(d.shape[-1])

    # 1. 프로파일 최저점 다시 재기. 탐색 값(480 개 중 최저)은 아래로 치우쳐 폭을 좁게 만든다.
    pkeys = sorted(k for k, v in st.get("profile", {}).items() if "x" in v)
    if pkeys and "Lp" not in f:
        f["Lp"] = ev.evaluate(np.stack([st["profile"][k]["x"] for k in pkeys]), mode, 2 * n2, rep0=PS, canon=True)
        save(st)
    prof, width = {}, {}
    if pkeys:
        Tps, Tpc = T(f["Lp"][:, :n2]), T(f["Lp"][:, n2:])
        b = int(np.argmin(Tps.mean(-1)))
        dm, ds = paired(Tpc, Tpc[b])
        thr = float(student.ppf(1 - 0.05 / max(len(pkeys) - 1, 1), n2 - 1))  # 한쪽 · Bonferroni
        for q, (i, lv) in enumerate(pkeys):
            n, lo, hi, _ = SPACE[i]
            v = st["profile"][(i, lv)]
            prof.setdefault(n, []).append(dict(value=lo + (hi - lo) * lv, loss=float(Tpc[q].mean()), diff=float(dm[q]),
                                               diff_se=float(ds[q]), inside=bool(dm[q] <= thr * ds[q]),
                                               search_loss=v.get("best"), stop=v.get("stop")))
        for n, rows in prof.items():
            rows.sort(key=lambda r: r["value"])
            inside = [r["value"] for r in rows if r["inside"]]
            width[n] = [min(inside), max(inside)] if inside else None

    # 2. 최종 후보: 전체 최저 · 실행별 최저 5 · 다시 잰 프로파일 최저 5 (PS 기준)
    cands = [np.asarray(st["x_best"])]
    cands += [np.asarray(x) for _, x in sorted(st.get("cma", {}).get("run_best", []), key=lambda t: t[0])[:5]]
    if pkeys:
        cands += [np.asarray(st["profile"][pkeys[q]]["x"]) for q in np.argsort(Tps.mean(-1))[:5]]
    U = np.unique(np.round(np.stack(cands), 6), axis=0)
    if "LF" not in f:
        f["LF"] = ev.evaluate(U, mode, 2 * n2, rep0=FA, canon=True)  # FA · FB 가 이어진 번호다
        save(st)
    TA, TB = T(f["LF"][:, :n2]), T(f["LF"][:, n2:])
    tA, tB = TA.mean(-1), TB.mean(-1)
    j = int(np.argmin(tA))
    u, Tu = U[j], TB[j]
    loss = float(tB[j])
    se = float(Tu.std(ddof=1) / math.sqrt(n2))
    dc, sc = paired(Tu, TB)  # 고른 점 − 다른 후보 (B)
    thr_c = float(student.ppf(1 - 0.05 / max(len(U) - 1, 1), n2 - 1))  # 한쪽 · Bonferroni
    log(f"[4] 후보 {len(U)} 개, A 로 고른 점의 B 손실 {loss:.2f} ± {se:.2f}  (A 1 위 {j}, B 1 위 {int(np.argmin(tB))})")
    # 이웃: 축마다 정규화 폭 ±0.03 (범위의 3 %) 두 점
    if "Ln" not in f:
        nb = []
        for i in range(PROFILE_DIMS):
            for dlt in (-0.03, 0.03):
                x = u.copy()
                x[i] = np.clip(x[i] + dlt, 0, 1)
                nb.append(x)
        f["nb"] = np.array(nb)
        # 최종점을 같은 호출에 함께 태운다. 같은 호출 크기라도 자리 · 함께 탄 환경에 따라 손실이 조금 달라진다
        f["Ln"] = ev.evaluate(np.vstack([u[None], f["nb"]]), mode, n2, rep0=FB, canon=True)
        save(st)
    tn = total(f["Ln"][1:])
    dn, sdn = paired(T(f["Ln"][1:]), T(f["Ln"][:1])[0])
    thr_n = float(student.ppf(1 - 0.05 / len(dn), n2 - 1))
    # 전압 강하를 켜고 같은 점
    if "Lv" not in f:
        f["Lv"] = ev.evaluate(u[None], mode, n2, volt=True, rep0=FB, canon=True)
        save(st)
    tv = float(total(f["Lv"])[0])
    dvolt, sdvolt = (float(x[0]) for x in paired(T(f["Lv"]), Tu))
    # 기록별 지표표
    ev.evaluate(u[None], mode, n2, keep_metrics=True, rep0=FB, canon=True)
    table = {}
    for r in range(ev.R):
        ms = ev.metrics.get((0, r), [])
        row = dict(floor=str(ev.floor[r]), loss=float(f["LF"][j, n2:, r].mean()))
        for key in ("stand_pitch", "stand_roll", "walk_pitch", "fall_t", "peak"):
            sv = np.array([m[key] for m in ms], float)
            row[key] = dict(real=_num(ev.real[r][key]), sim_mean=_num(np.nanmean(sv)) if np.isfinite(sv).any() else None,
                            sim_sd=_num(np.nanstd(sv)) if np.isfinite(sv).sum() > 1 else None)
        # 방향은 원형 값이다(179° 와 −179° 의 평균은 180°).
        dv = np.radians(np.array([m["fall_dir"] for m in ms], float))
        dv = dv[np.isfinite(dv)]
        if len(dv):
            cs, sn = np.cos(dv).mean(), np.sin(dv).mean()
            row["fall_dir"] = dict(real=_num(ev.real[r]["fall_dir"]), sim_mean=float(np.degrees(np.arctan2(sn, cs))),
                                   sim_spread=float(np.degrees(np.sqrt(-2 * np.log(max(np.hypot(cs, sn), 1e-12))))))
        else:
            row["fall_dir"] = dict(real=_num(ev.real[r]["fall_dir"]), sim_mean=None, sim_spread=None)
        row["finite"] = bool(all(bool(m["finite"]) for m in ms))
        row["fell"] = dict(real=bool(ev.real[r]["fell"]), sim_rate=float(np.mean([bool(m["fell"]) for m in ms])))
        kn = np.array([m["knee"] for m in ms], float)
        row["knee"] = dict(real=[_num(x) for x in ev.real[r]["knee"]], sim_mean=[_num(x) for x in np.nanmean(kn, 0)])
        table[str(ev.names[r])] = row
    checks = dict(
        numeric_ok=bool(np.isfinite(loss) and loss < S.FAILED and all(v["finite"] for v in table.values())),
        # 고른 점이 B 에서 어느 후보보다도 FINAL_TOL 넘게 나쁘다는 근거가 검출되지 않는다(한쪽 · Bonferroni)
        near_best=bool(np.all(dc - thr_c * sc <= FINAL_TOL)),
        # 명백히 더 나은 이웃이 없다(한쪽 · Bonferroni). 통과가 '이웃보다 낫다' 를 증명하지는 않는다
        beats_neighbours=bool(np.all(dn >= -thr_n * sdn)),
        # 전압 강하를 켠 차가 2 SE 를 더해도 손실의 2 % 안이다(같음 판정)
        voltage_no_change=bool(abs(dvolt) + 2 * sdvolt <= 0.02 * loss),
    )
    res = dict(mode=MODES[mode], K=K,
               replicas=dict(search=[0, base], profile_select=[PS, PS + n2], profile_compare=[PC, PC + n2],
                             final_select=[FA, FA + n2], final_report=[FB, FB + n2]),
               loss=loss, loss_se=se, loss_A=float(tA[j]), rank_A=j, rank_B=int(np.argmin(tB)), loss_voltage=tv,
               voltage_diff=dvolt, voltage_diff_se=sdvolt,
               values={n: float(v) for n, v in to_phys(u).items()}, u=u.tolist(), checks=checks,
               neighbours=dict(loss_min=float(tn.min()), loss_max=float(tn.max()), diff=[float(x) for x in dn],
                               diff_se=[float(x) for x in sdn], threshold_se=thr_n),
               candidates=[dict(loss_A=float(a), loss_B=float(bb), diff_B=float(d), diff_B_se=float(e),
                                values={n: float(v) for n, v in to_phys(x).items()})
                           for a, bb, d, e, x in zip(tA, tB, dc, sc, U)],
               profile=prof, profile_width=width, per_record=table)
    (OUTDIR / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[4] 최종 손실 {loss:.2f} ± {se:.2f}  검증 {checks}  {fmt(u)}")
    st.update(stage=5, checks=checks, final_u=u, final_loss=loss)


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def main():
    OUTDIR.mkdir(exist_ok=True)
    fp = fingerprint()
    st = pickle.loads(STATE.read_bytes()) if STATE.exists() else dict(stage=0, fingerprint=fp)
    if st.get("fingerprint") in COMPATIBLE_FINGERPRINTS and legacy_ok(st["fingerprint"]):
        old = COMPATIBLE_FINGERPRINTS[st["fingerprint"]]
        per = float(sum(next((b for b in BUCKETS if e <= b), BUCKETS[-1]) for e in np.load(PACK)["end"]))
        st["done_steps"] = [x * per / old["per"] for x in st.get("done_steps", [])]
        log(f"목적함수가 같은 이전 판의 저장 지점({st['fingerprint']})을 이어받는다 — 단계 {st['stage']}, "
            f"진행량 {per:.0f}/{old['per']} 로 환산")
        st["fingerprint"] = fp
        save(st)
    if st.get("fingerprint") != fp:
        # 데이터 · 코드가 바뀌었으면 옛 손실과 새 손실을 섞지 않는다. 옛 출력은 옮겨 두고 새로 시작한다.
        old = OUTDIR.parent / f"{OUTDIR.name}_{st.get('fingerprint', 'old')}_{time.strftime('%m%d%H%M')}"
        OUTDIR.rename(old)
        OUTDIR.mkdir()
        log(f"지문이 바뀌었다({st.get('fingerprint')} → {fp}). 옛 출력은 {old.name} 로 옮기고 새로 시작한다")
        st = dict(stage=0, fingerprint=fp)
    log(f"시작/재개 — 단계 {st['stage']}, 장치 {jax.devices()}, 한 번에 환경 {MAX_ENVS}")
    ev = Evaluator()
    ev.progress = Progress(ev, st)
    ev.progress.write()
    stages = [stage_pilot, stage_sobol, stage_cma, stage_profile, stage_final]
    while st["stage"] < len(stages):
        stages[st["stage"]](ev, st)
        save(st)
        ev.progress.write()
    ok = all(st.get("checks", {}).values())
    (OUTDIR / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S ") + ("검증 통과\n" if ok else f"검증 실패 {st.get('checks')}\n"))
    log("끝 — " + ("검증 통과" if ok else f"검증 실패 {st.get('checks')}"))


if __name__ == "__main__":
    main()
