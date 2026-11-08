"""지표와 손실 (plan/01 3장). 실물과 시뮬에 같은 함수를 쓴다.

지표는 환경 축으로 한꺼번에 계산한다. 배열은 (T, N, ...) 이고 시간 0 은 시뮬 시작(출발 3 초 전)이다.
넘어짐은 '넘어졌나' 대신 넘어진 시각(안 넘어지면 구간 끝으로 자른 값), 넘어지기 전 최대 기울기, 넘어진 방향의
각도로 비교한다. 45° 를 넘는 순간 시각과 걷기 구간이 바뀌므로 손실은 완전히 연속은 아니다(plan/01 3장).
"""
import numpy as np

DT = 0.02
LPF_TAU = 0.3
FALL_DEG = 45.0

# 손실 정규화 폭. 이만큼 어긋나면 그 항이 1 이다. 실물 반복 시험의 흩어짐 크기에 맞춘 판단값이다
# (v2_15 0.05 마루 기록 사이 서기 앞숙임 2.7 ~ 5.3°, 무릎 추종 0.48 ~ 0.87).
SCALE = dict(stand_pitch=1.0, stand_roll=1.0, walk_pitch=2.0, fall_t=1.0, peak=10.0, fall_dir=45.0, knee=0.15)
HUBER = 3.0  # 기록 하나의 오차(정규화 제곱합의 제곱근)가 이것을 넘으면 선형으로 센다
# 실물에 있는 지표가 시뮬에 없을 때(시뮬이 일찍 넘어져 걷기 구간이 없음, 계산 실패) 그 항의 벌점. 오차 3 배와 같다.
# 결측이 오차 0 이 되면 일찍 넘어지거나 깨진 후보가 이득을 본다.
MISSING = 9.0
# 비교 구간 안에서 시뮬 값이 유한하지 않으면(물리 계산 실패) 그 기록 손실. 후보를 사실상 뺀다.
FAILED = 1e4


def _lpf(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    a = DT / LPF_TAU
    for i in range(1, len(x)):
        y[i] = y[i - 1] + (x[i] - y[i - 1]) * a
    return y


def _pitch_roll(a):
    return np.degrees(np.arctan2(-a[..., 0], a[..., 2])), np.degrees(np.arctan2(-a[..., 1], a[..., 2]))


def _tilt(a):
    n = np.linalg.norm(a, axis=-1)
    return np.degrees(np.arccos(np.clip(a[..., 2] / np.maximum(n, 1e-6), -1, 1)))


def _masked_mean(x, m):
    # 곱셈으로 가리면 구간 밖의 NaN 이 NaN × 0 = NaN 으로 새어 든다. where 로 가린다. 모두 float32 로 둔다.
    c = m.sum(0, dtype=np.float32)
    s = np.where(m, x, np.float32(0)).sum(0, dtype=np.float32)
    return np.where(c > 0, s / np.maximum(c, np.float32(1)), np.float32(np.nan))


def _masked_std(x, m):
    mu = _masked_mean(x, m)
    c = m.sum(0, dtype=np.float32)
    v = np.where(m, (x - mu) ** 2, np.float32(0)).sum(0, dtype=np.float32) / np.maximum(c, np.float32(1))
    return np.where(c > 1, np.sqrt(v), np.float32(np.nan))


def metrics(acc, knee, tg, move, end, fall_at_end=None) -> dict:
    """acc (T, N, 3), knee · tg (T, N, 2) — 무릎은 실제각(상수 차이는 무관) · 목표각. move · end (N,) 주기.
    fall_at_end: 안전 정지로 끝난 실물 기록이면 참(끝을 넘어짐으로 본다)."""
    acc, knee, tg = (np.asarray(x, np.float32) for x in (acc, knee, tg))
    T, N = acc.shape[:2]
    t = np.arange(T)[:, None]
    move = np.asarray(move)[None, :]
    end = np.asarray(end)[None, :]
    fae = np.zeros(N, bool) if fall_at_end is None else np.asarray(fall_at_end)
    a = _lpf(acc)
    p, r = _pitch_roll(a)
    tl = _tilt(a)
    rp, rr = _pitch_roll(acc)

    stand = (t >= move - 100) & (t < move) & (move > 0)
    out = {"stand_pitch": _masked_mean(rp, stand), "stand_roll": _masked_mean(rr, stand)}
    # 비교 구간(0 ~ end) 안에서 시뮬 값이 하나라도 유한하지 않으면 물리 계산 실패다.
    span = t < end
    fin = np.isfinite(acc).all(-1) & np.isfinite(knee).all(-1) & np.isfinite(tg).all(-1)
    out["finite"] = ~(span & ~fin).any(0)

    walk = (t >= move) & (t < end)
    over = walk & (tl > FALL_DEG)
    has = over.any(0)
    first = np.where(has, over.argmax(0), np.where(fae, end[0] - 1, -1))
    fell = first >= 0
    w_end = np.where(fell, first, end[0])
    win = end[0] - move[0]
    out["fell"] = fell
    out["fall_t"] = np.where(fell, (first - move[0]) * DT, win * DT)  # 안 넘어지면 구간 끝으로 자른 값
    idx = np.clip(first, 0, T - 1)
    out["fall_dir"] = np.where(fell, np.degrees(np.arctan2(r[idx, np.arange(N)], p[idx, np.arange(N)])), np.nan)
    pre = (t >= move) & (t < w_end[None, :])
    out["peak"] = np.where(pre.any(0), np.max(np.where(pre, tl, 0.0), 0), np.nan)

    ww = (t >= move + int(1.0 / DT)) & (t < w_end[None, :])
    enough = ww.sum(0) >= 25
    out["walk_pitch"] = np.where(enough, _masked_mean(rp, ww), np.nan)
    ks = np.stack([_masked_std(knee[..., j], ww) / np.maximum(_masked_std(tg[..., j], ww), 1e-6) for j in range(2)], -1)
    out["knee"] = np.where(enough[:, None], ks, np.nan)
    return out


def _term(sim_v, real_v, s):
    """실물 값이 없으면 비교하지 않는다(0). 실물 값이 있는데 시뮬 값이 없거나 유한하지 않으면 MISSING."""
    if real_v is None or not np.isfinite(real_v):
        return np.zeros(np.shape(sim_v))
    d = (np.asarray(sim_v, float) - real_v) / s
    return np.where(np.isfinite(d), d ** 2, MISSING)


def record_loss(sim: dict, real: dict) -> np.ndarray:
    """기록 하나에 대한 손실 (N,). real 의 값은 스칼라(기록 하나)이고 sim 은 (N,)."""
    e2 = _term(sim["stand_pitch"], real["stand_pitch"], SCALE["stand_pitch"])
    e2 = e2 + _term(sim["stand_roll"], real["stand_roll"], SCALE["stand_roll"])
    e2 = e2 + _term(sim["walk_pitch"], real["walk_pitch"], SCALE["walk_pitch"])
    e2 = e2 + _term(sim["fall_t"], real["fall_t"], SCALE["fall_t"])
    e2 = e2 + _term(sim["peak"], real["peak"], SCALE["peak"])
    if real["fell"]:
        # 실물만 넘어지고 시뮬은 안 넘어지면 시뮬 방향이 NaN 이라 MISSING 이 붙는다.
        dd = (np.asarray(sim["fall_dir"], float) - real["fall_dir"] + 180.0) % 360.0 - 180.0
        e2 = e2 + _term(dd, 0.0, SCALE["fall_dir"])
    for j in range(2):
        e2 = e2 + _term(sim["knee"][:, j], real["knee"][j], SCALE["knee"])
    e = np.sqrt(e2)
    loss = np.where(e < HUBER, 0.5 * e ** 2, HUBER * (e - 0.5 * HUBER))
    return np.where(np.asarray(sim["finite"]), loss, FAILED)


def pick(m: dict, i) -> dict:
    """지표 묶음에서 환경 하나(실물 기록)를 꺼낸다."""
    return {k: (v[i] if np.ndim(v) else v) for k, v in m.items()}
