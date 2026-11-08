"""실물 바닥 시험 기록을 읽어 시뮬과 같은 식의 지표를 낸다 (plan/01 2 · 3장).

기록은 두 형식이다. `log_*` 는 걷기 시험 도구(`real_walk.py`)가 관측 101 을 통째로 남긴 것이고,
`run_*` 는 실물 진입점(`real.run`)이 명령 · 관절각 · 목표각을 따로 남긴 것이다.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RUNTIME = Path(__file__).resolve().parents[2] / "AIRookie_Runtime"
LOGS = RUNTIME / "logs" / "jetson"
DT = 0.02
# 걷기 시험 도구는 장착각을 되돌리지 않은 IMU 값을 남겼다. 몸통 앞숙임 = IMU 앞숙임 + 0.74°
# (Runtime research/10 17-1, real/config.json 의 mount_pitch_deg −0.74). 실물 진입점 기록은 이미 되돌린 값이다.
MOUNT_PITCH_DEG = -0.74
FALL_DEG = 45.0
# 2026-10-09 v2_8 시험은 원값 기울기 50° 에서 토크를 끄고 기록을 닫았다. 그 표본은 기록에 남지 않아 저역 45° 에
# 닿기 전에 기록이 끝난다. 그런 기록은 표의 `FALL_AT_END` 로 끝을 넘어짐으로 본다(Runtime research/10 6장 표).
FALL_AT_END = {"log_ground_v28_253_vx005", "log_ground_v28_253_vx005_15s", "log_ground_v28_253_vx005_15s_d",
               "log_ground_v28_253_vx005_15s_e", "log_ground_v28_253_vx005g", "log_ground_v28_253_vx005h"}
LPF_TAU = 0.3
KNEES = (3, 12)
LEGS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)

MARU, TILE = "마루", "타일"

# (날짜, 이름, 정책, 바닥). 정책은 이름 · Runtime research/10 표 · 관측 재생 대조(`identify_policy`)로 정했다.
# 뺀 기록: 매단 시험, 홈 유지만 한 시험(`log_imu_floor_hold253`, 정책 없음), 사람이 민 시험(`log_v215_push30`),
# 3 초짜리 가속 미설정 기록(`log_024022`), 시작 2.5 초 동안 발이 뜬 시험(`log_ground_v28_253_vx005_15s_b`, R8).
TABLE = [
    ("2026-10-08", "log_ground_v28_253", "v2_8", MARU),
    ("2026-10-08", "log_ground_v28_253b", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005b", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005c", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005d", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005e", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005f", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005_15s", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005_15s_c", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005_15s_d", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005_15s_e", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005g", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005h", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005_nosafe", "v2_8", MARU),
    ("2026-10-09", "log_ground_v28_253_vx005_nosafe_15s", "v2_8", MARU),
    ("2026-10-10", "log_v215_vx005_15s", "v2_15", MARU),
    ("2026-10-10", "log_v215_temp_walk60_stand60", "v2_15", MARU),
    ("2026-10-10", "log_v215_vx01_20s", "v2_15", MARU),
    ("2026-10-10", "log_v215_vx005_20s", "v2_15", MARU),
    ("2026-10-10", "log_v215_vx0075_20s_chg", "v2_15", MARU),
    ("2026-10-10", "log_v215_vx005_20s_chg", "v2_15", MARU),
    ("2026-10-10", "log_v220_vx005_20s", "v2_20", MARU),
    ("2026-10-10", "log_v220_vx0111_20s", "v2_20", MARU),
    ("2026-10-10", "log_imu_floor_vx005_1", "v2_15", MARU),
    ("2026-10-10", "log_imu_floor_vx005_2", "v2_15", MARU),
    ("2026-10-10", "log_imu_floor_vx005_3", "v2_15", MARU),
    ("2026-10-10", "log_imu_floor_vx005_4_nochg", "v2_15", MARU),
    ("2026-10-10", "log_imu_floor_vx005_5_nochg", "v2_15", MARU),
    ("2026-10-10", "log_imu_floor_vx005_60s_nochg", "v2_15", MARU),
    ("2026-10-11", "log_v227_vx005_nokick", "v2_27", MARU),
    ("2026-10-11", "log_v227_vx005_kick", "v2_27", MARU),
    ("2026-10-11", "log_v227_vx008_kick", "v2_27", MARU),
    ("2026-10-11", "log_v227_vx0111", "v2_27", MARU),
    ("2026-10-11", "log_v227b_vx005", "v2_27", MARU),
    ("2026-10-11", "log_v227b_vx005_kick", "v2_27", MARU),
    ("2026-10-11", "log_v227b_vx008_kick", "v2_27", MARU),
    ("2026-10-11", "log_v229_vx005", "v2_29", MARU),
    ("2026-10-11", "log_v228_vx005_kick", "v2_28", MARU),
    ("2026-10-11", "log_v228_vx008", "v2_28", MARU),
    ("2026-10-11", "log_v215_vx005_1011", "v2_15", MARU),
    ("2026-10-11", "log_v223_vx005", "v2_23", MARU),
    ("2026-10-11", "log_v230_vx005", "v2_30", TILE),
    ("2026-10-11", "log_v215_vx005_1011b", "v2_15", TILE),
    ("2026-10-11", "log_v230_vx005_2", "v2_30", TILE),
    ("2026-10-11", "log_v215_vx005_1011c", "v2_15", TILE),
    ("2026-10-11", "log_v215_vx0111_1011", "v2_15", TILE),
    ("2026-10-11", "log_v230_vx0111", "v2_30", TILE),
    ("2026-10-11", "run_224527", "v2_15", TILE),
    ("2026-10-11", "run_225400", "v2_15", TILE),
    ("2026-10-11", "run_225632", "v2_15", TILE),
    ("2026-10-11", "run_225809", "v2_15", TILE),
    ("2026-10-11", "run_225930", "v2_15", TILE),
    ("2026-10-11", "run_230154", "v2_30", TILE),
]


@dataclass
class Record:
    name: str
    floor: str
    policy: str | None
    cmd: np.ndarray        # (T, 7) 주기마다 정책 관측에 들어간 명령
    angle: np.ndarray      # (T, 14) 관절각 rad (엔코더)
    target: np.ndarray     # (T, 14) 목표각 rad
    accel: np.ndarray      # (T, 3) 몸통 좌표 가속도 (장착각 되돌림)
    contact: np.ndarray    # (T, 2)
    obs: np.ndarray | None = None   # (T, 101) 걷기 시험 도구 기록만
    home: np.ndarray | None = None  # (14,) 짝 홈 자세
    contact_fixed: bool = False
    start: int = 0         # 시뮬을 시작하는 주기(출발 3 초 전, 서 있는 자세)
    move: int = 0          # 첫 이동 명령 주기
    end: int = 0           # 비교 끝 주기(포함하지 않음)
    metrics: dict = field(default_factory=dict)


def _mount(accel: np.ndarray) -> np.ndarray:
    """IMU 값을 몸통 좌표로 돌린다(앞숙임만). `real/imu.py` 의 mount_rotation 과 같은 방향."""
    c, s = np.cos(np.radians(MOUNT_PITCH_DEG)), np.sin(np.radians(MOUNT_PITCH_DEG))
    # 몸통이 수평일 때 IMU 가 앞숙임 p 를 보고하면 가속도는 (−sin p, 0, cos p) g 이다. 이것을 z 로 돌린다.
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return accel @ r.T


def lpf(x: np.ndarray, tau: float = LPF_TAU) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    a = DT / tau
    for i in range(1, len(x)):
        y[i] = y[i - 1] + (x[i] - y[i - 1]) * a
    return y


def pitch_roll(accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """가속도로 본 앞숙임(+ 앞) · 왼기울임(+ 왼) 도."""
    return (np.degrees(np.arctan2(-accel[:, 0], accel[:, 2])),
            np.degrees(np.arctan2(-accel[:, 1], accel[:, 2])))


def tilt(accel: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(accel, axis=1)
    return np.degrees(np.arccos(np.clip(accel[:, 2] / np.maximum(n, 1e-6), -1, 1)))


def metrics(accel, angle, target, move, end, fall_at_end: bool = False) -> dict:
    """기록과 시뮬에 같이 쓰는 지표 (plan/01 3장). 배열은 시뮬 시작 주기부터다."""
    a = lpf(accel)
    p, r = pitch_roll(a)
    tl = tilt(a)
    pre = slice(max(0, move - 100), move)
    out = {"stand_pitch": float(np.mean(pitch_roll(accel[pre])[0])) if move > 0 else np.nan,
           "stand_roll": float(np.mean(pitch_roll(accel[pre])[1])) if move > 0 else np.nan}
    over = np.where(tl[move:end] > FALL_DEG)[0]
    fall = move + int(over[0]) if len(over) else (end - 1 if fall_at_end else None)
    if fall is not None:
        k = fall
        d = ("앞" if p[k] > 0 else "뒤") if abs(p[k]) >= abs(r[k]) else ("왼" if r[k] > 0 else "오른")
        out.update(fall_t=(fall - move) * DT, fall_dir=d)
    else:
        out.update(fall_t=None, fall_dir=None)
    w0, w1 = move + int(1.0 / DT), (fall if fall is not None else end)
    if w1 - w0 >= 25:
        out["walk_pitch"] = float(np.mean(pitch_roll(accel[w0:w1])[0]))
        sa, st = angle[w0:w1][:, KNEES].std(0), target[w0:w1][:, KNEES].std(0)
        out["knee_track"] = [float(x) for x in sa / np.maximum(st, 1e-6)]
    else:
        out["walk_pitch"], out["knee_track"] = np.nan, [np.nan, np.nan]
    return out


def _read(date: str, name: str) -> dict:
    d = np.load(LOGS / date / f"{name}.npz", allow_pickle=True)
    if name.startswith("run_"):
        return dict(cmd=d["command"], angle=d["servo_angle"], target=d["target"], accel=d["accel"],
                    contact=d["contact"].astype(bool), obs=None)
    obs = d["obs"]
    home = None  # 관측의 관절각 칸은 관절각 − 짝 홈 자세다. 짝 홈 자세는 아래에서 푼다
    return dict(cmd=obs[:, 6:13], angle=None, target=d["target"], accel=_mount(obs[:, 3:6]),
                contact=obs[:, 97:99] > 0.5, obs=obs, home=home)


# 목표각을 자르는 두 규칙(Runtime gait.py, 학습 joystick.py 와 같다): 직전 목표각 기준 속도 상한, 관절 범위.
MAX_MOTOR_VELOCITY = 4.83


def solve_home(obs: np.ndarray, target: np.ndarray, scale: np.ndarray, joint_range: np.ndarray) -> np.ndarray:
    """짝 홈 자세. 주기 k 의 목표각은 주기 k+1 관측의 직전 행동에서 나온다: 목표 = 홈 + 배율 × 행동.

    속도 상한 · 관절 범위에 잘린 표본은 이 식이 맞지 않아 축마다 뺀다. 잘린 표본을 넣고 중앙값을 내면 왼쪽
    hip yaw 홈이 −0.052 rad 로 틀린다(잘리지 않은 표본은 모두 +0.0015). 머리는 명령 직결이라 이 식이 맞지
    않는다. 머리 홈은 학습 장면 값으로 둔다(호출하는 쪽)."""
    act = obs[1:, 41:55]
    tg, prev = target[1:], target[:-1]
    h = target[:-1] - scale * act
    lo, hi = joint_range[:, 0], joint_range[:, 1]
    tg0 = target[:-1]
    prev0 = np.vstack([tg0[:1], tg0[:-1]])
    eps = 1e-6
    ok = (tg0 > lo + eps) & (tg0 < hi - eps) & (np.abs(tg0 - prev0) < MAX_MOTOR_VELOCITY * DT - 1e-5)
    home = np.empty(target.shape[1])
    for j in range(target.shape[1]):
        v = h[ok[:, j], j]
        home[j] = np.median(v) if len(v) >= 20 else np.nan
    # 짝 홈 자세는 좌우 거울이다(yaw · roll · pitch 부호 반대, 무릎 · 발목 같음). 한쪽이 늘 잘려 표본이 없으면
    # 반대쪽 값으로 채운다(log_v227b_vx005 오른쪽 hip yaw).
    mirror = np.array([-1.0, -1.0, -1.0, 1.0, 1.0])
    left, right = home[0:5], home[9:14]
    home[0:5] = np.where(np.isnan(left), right * mirror, left)
    home[9:14] = np.where(np.isnan(right), left * mirror, right)
    return home


def load(date: str, name: str, policy: str | None, floor: str, scale: np.ndarray, head_home: np.ndarray,
         homes: dict | None = None, joint_range: np.ndarray | None = None) -> Record:
    r = _read(date, name)
    cmd = r["cmd"]
    moving = np.abs(cmd[:, :3]).sum(1) > 0
    move = int(np.argmax(moving)) if moving.any() else len(cmd)
    if r["obs"] is not None:
        home = solve_home(r["obs"], r["target"], scale, joint_range)
        home[5:9] = head_home
        assert np.isfinite(home).all(), (name, home)
        angle = r["obs"][:, 13:27] + home
    else:
        home = homes[policy].copy()
        angle = r["angle"]
    start = max(0, move - int(3.0 / DT))
    rec = Record(name=name, floor=floor, policy=policy, cmd=cmd[start:], angle=angle[start:],
                 target=r["target"][start:], accel=r["accel"][start:], contact=r["contact"][start:],
                 obs=None if r["obs"] is None else r["obs"][start:], home=home, start=start)
    rec.move = move - start
    t_end = len(rec.cmd)
    # 비교는 첫 넘어짐 뒤 5 초까지만 한다. 그 뒤는 사람이 잡은 구간이다. 길어도 40 초에서 자른다.
    m0 = metrics(rec.accel, rec.angle, rec.target, rec.move, t_end)
    if m0["fall_t"] is not None:
        t_end = min(t_end, rec.move + int(round(m0["fall_t"] / DT)) + int(5.0 / DT))
    rec.end = min(t_end, int(40.0 / DT))
    walk = slice(rec.move, rec.end)
    rec.contact_fixed = bool(len(rec.contact[walk]) > 50 and rec.contact[walk].all(1).mean() > 0.98)
    rec.metrics = metrics(rec.accel, rec.angle, rec.target, rec.move, rec.end, name in FALL_AT_END)
    return rec


def identify_policy(rec: Record, candidates: list[str]) -> dict[str, float]:
    """관측을 후보 정책에 넣어 기록의 다음 행동과의 평균 차(rad)를 낸다. 걷기 시험 도구 기록만 된다."""
    from real_fit.policy import load as load_policy
    import jax.numpy as jp
    out = {}
    obs = rec.obs[: rec.end]
    for c in candidates:
        act = np.asarray(load_policy(Path(__file__).resolve().parents[1] / "policies" / "v2" / f"{c}_300M.onnx")(
            jp.asarray(obs[:-1], dtype=jp.float32)))
        out[c] = float(np.abs(act - obs[1:, 41:55]).mean())
    return out


def phase0(date: str, name: str, start: int) -> int:
    """시작 주기의 걸음 위상 칸(0 ~ 26). 걷기 시험 도구는 관측의 위상, 실물 진입점은 `gait.phase` 의 시각 식."""
    d = np.load(LOGS / date / f"{name}.npz")
    if name.startswith("run_"):
        return int((round(float(d["t"][start]) / DT) + 1) % 27)
    o = d["obs"][start]
    return int(round(np.arctan2(o[100], o[99]) / (2 * np.pi) * 27)) % 27


def load_all(scale: np.ndarray, head_home: np.ndarray, joint_range: np.ndarray) -> list[Record]:
    """표의 기록 전부. 실물 진입점 기록의 짝 홈 자세는 같은 정책 걷기 시험 기록에서 푼 값의 중앙값이다."""
    recs, homes = [], {}
    for date, name, pol, floor in TABLE:
        if not name.startswith("run_"):
            r = load(date, name, pol, floor, scale, head_home, joint_range=joint_range)
            r.date = date
            recs.append(r)
            homes.setdefault(pol, []).append(r.home)
    homes = {k: np.median(v, 0) for k, v in homes.items()}
    for date, name, pol, floor in TABLE:
        if name.startswith("run_"):
            r = load(date, name, pol, floor, scale, head_home, homes, joint_range)
            r.date = date
            recs.append(r)
    for r in recs:
        r.phase0 = phase0(r.date, r.name, r.start)
    return recs
