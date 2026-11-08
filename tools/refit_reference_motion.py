"""시범 동작(polynomial_coefficients.pkl)을 서보 속도 및 관절 한계 안으로 투영한다.

핵심 두 가지:

1. 제약을 27점 값이 아니라 **다항식 계수**에 건다.
   15차 다항식은 27개 점을 정확히 지날 수 없으므로,
   점을 먼저 투영한 뒤 재적합하면 제약이 다시 깨진다.

2. 계수는 반드시 **t 기저에서 직접** 푼다.
   u = 2t-1 기저에서 풀고 t 기저로 환산하면 계수가 1e3 -> 1e8 로 폭발하고,
   JAX 가 float32 로 평가하므로 값이 완전히 깨진다.
   t 기저는 조건수가 나쁘므로 릿지 항으로 최소 노름 해에 고정한다.

    minimize ||V c - r||^2 + lam * ||c||^2
    s.t.  |D V c| <= dmax      (프레임 간 변화, 주기 순환)
          lo <= V c <= hi      (관절 한계)

검증은 실제 학습 경로와 같은 float32 로 수행한다.

변경 범위는 다리 10축의 위치(0~4, 11~15)와 속도(16~20, 27~31) 채널뿐이다.
머리, 안테나, 발접촉, 몸통 속도 채널은 손대지 않는다.
"""
import pickle, sys, time
import numpy as np
from scipy.optimize import minimize
import xml.etree.ElementTree as ET

PKL_IN, PKL_OUT, XML = sys.argv[1], sys.argv[2], sys.argv[3]
MARGIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.90
NBINS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
# 데이터시트 값이다. M-05c 는 실물 측정이 아니다.
#
# 이 제약은 **속도**만 건다. 실물의 진짜 상한은 가속이며 약 435 deg/s^2 다
# (AIRookie_Runtime/docs/archive/research/06_servo_motion_limit.md). 걸음의 관절
# 이동각이 22~25도뿐이라 서보가 등속에 도달하기 전에 감속을 시작하므로,
# 속도 제약만으로는 보행 구간이 구속되지 아니한다. 가속은 시범 동작의
# **주기**로 맞춘다 — 가속은 주기 제곱에 반비례한다.
LIMIT_DEG_S = 276.86
LAM = 1e-12                               # 릿지. 계수 폭발 방지용
EPS = 3e-3                                # 제약 여유 (rad). float32 오차 흡수

LEG = {0: "left_hip_yaw", 1: "left_hip_roll", 2: "left_hip_pitch",
       3: "left_knee", 4: "left_ankle",
       11: "right_hip_yaw", 12: "right_hip_roll", 13: "right_hip_pitch",
       14: "right_knee", 15: "right_ankle"}
POS2VEL = {0: 16, 1: 17, 2: 18, 3: 19, 4: 20,
           11: 27, 12: 28, 13: 29, 14: 30, 15: 31}

root = ET.parse(XML).getroot()
lim = {j.get("name"): tuple(float(v) for v in j.get("range").split())
       for j in root.iter("joint")
       if j.get("name") in LEG.values() and j.get("range")}
assert len(lim) == 10, sorted(lim)

data = pickle.load(open(PKL_IN, "rb"))
keys = list(data.keys())
if NBINS:
    keys = keys[:NBINS]
N = data[keys[0]]["nb_steps_in_period"]
PERIOD = data[keys[0]]["period"]
NC = len(data[keys[0]]["coefficients"]["dim_0"])
DT = PERIOD / N
DMAX = np.radians(LIMIT_DEG_S * MARGIN) * DT
T = np.arange(N) / N
V = np.vander(T, NC, increasing=True)
D = np.zeros((N, N))
for k in range(N):
    D[k, k] = -1.0
    D[k, (k + 1) % N] = 1.0
DV = D @ V

def ev32(c):
    """학습 경로와 동일한 float32 평가."""
    return np.polyval(np.flip(np.asarray(c, np.float32)).astype(np.float32),
                      T.astype(np.float32))

print(f"bin {len(keys)}개 / 프레임 {N} / 주기 {PERIOD}s / 계수 {NC}개")
print(f"프레임당 허용 변화 {np.degrees(DMAX):.3f} deg "
      f"(= {LIMIT_DEG_S} × {MARGIN:.2f} deg/s), 여유 {np.degrees(EPS):.3f} deg")

def solve(r, c0, lo, hi):
    q0 = V @ c0
    if (np.abs(D @ q0).max() <= DMAX and q0.min() >= lo and q0.max() <= hi):
        return c0, False
    res = minimize(
        lambda c: float(np.sum((V @ c - r) ** 2) + LAM * np.sum(c * c)),
        x0=c0,
        jac=lambda c: 2.0 * (V.T @ (V @ c - r) + LAM * c),
        constraints=[
            {"type": "ineq", "fun": lambda c: DMAX - EPS - DV @ c, "jac": lambda c: -DV},
            {"type": "ineq", "fun": lambda c: DMAX - EPS + DV @ c, "jac": lambda c: DV},
            {"type": "ineq", "fun": lambda c: V @ c - (lo + EPS), "jac": lambda c: V},
            {"type": "ineq", "fun": lambda c: (hi - EPS) - V @ c, "jac": lambda c: -V},
        ],
        method="SLSQP",
        options={"maxiter": 800, "ftol": 1e-16},
    )
    return res.x, True

t0 = time.time()
n_touch = n_tot = n_fail = 0
errs_rms, errs_max, cmax = [], [], 0.0
cvmax = vel_err = vel_absmax = 0.0
for bi, key in enumerate(keys):
    C = data[key]["coefficients"]
    for dim, name in LEG.items():
        c0 = np.asarray(C[f"dim_{dim}"], dtype=float)
        r = np.polyval(np.flip(c0), T)
        lo, hi = lim[name]
        c, touched = solve(r, c0, lo, hi)
        n_tot += 1
        cmax = max(cmax, float(np.abs(c).max()))
        if touched:
            n_touch += 1
            e = np.degrees(np.abs(V @ c - r))
            errs_rms.append(float(np.sqrt(np.mean(e ** 2))))
            errs_max.append(float(e.max()))
        # float32 로 최종 검증
        q = ev32(c)
        if (np.degrees(np.abs(np.roll(q, -1) - q)).max() / DT > LIMIT_DEG_S
                or q.min() < lo - 1e-3 or q.max() > hi + 1e-3):
            n_fail += 1
        C[f"dim_{dim}"] = list(c)

        # 속도 채널은 위치 다항식을 해석적으로 미분해서 만든다.
        #
        # 원본은 프레임 중앙차분(np.roll)으로 27개 속도값을 만든 뒤 릿지 없는
        # lstsq 로 다시 적합했다. 그런데 그 27개 값은 위치 다항식이 표본점 사이에서
        # 출렁이는 것을 뭉갠 신호라, 같은 15차 다항식으로 되살리려면 계수가
        # 1e10 까지 커진다. float64 로 읽으면 값이 맞지만 학습은 JAX float32 라
        # 상쇄가 무너져 참조 속도가 517 rad/s(서보 한계의 107 배)로 깨졌다.
        # 이 파일 머리말이 경고한 바로 그 함정에 속도 채널만 빠져 있었다.
        #
        # 해석적 미분은 적합을 하지 않으므로 조건수 문제가 없다. 계수는 1.9e6
        # 규모로 float32 오차가 0.1 rad/s 이내이며(원본 pkl 0.03 rad/s 와 같은
        # 수준), 무엇보다 정책이 따라가야 할 바로 그 궤적의 속도다.
        #
        # 계수는 낮은 차수부터 저장하므로 미분은 한 칸 당겨 차수를 곱하고,
        # 시간 단위를 정규화 t 에서 실제 초로 되돌리기 위해 주기로 나눈다.
        # 길이는 NC 로 유지한다(최고차 항은 0).
        cv = np.zeros(NC)
        cv[: NC - 1] = (np.asarray(c) * np.arange(NC))[1:] / PERIOD
        C[f"dim_{POS2VEL[dim]}"] = list(cv)

        # 속도 채널도 float32 로 검증한다. 위치만 보던 것이 위 사고의 원인이다.
        cvmax = max(cvmax, float(np.abs(cv).max()))
        v32 = ev32(cv)
        v64 = np.polyval(np.flip(cv), T)
        vel_err = max(vel_err, float(np.abs(v32 - v64).max()))
        vel_absmax = max(vel_absmax, float(np.abs(v64).max()))
    if (bi + 1) % 60 == 0:
        print(f"  {bi+1}/{len(keys)} ({time.time()-t0:.0f}s)")

pickle.dump(data, open(PKL_OUT, "wb"))
print()
print(f"수정한 궤적          {n_touch} / {n_tot} ({100*n_touch/n_tot:.1f}%)")
print(f"float32 검증 실패    {n_fail}")
print(f"위치 계수 절대값 최대 {cmax:.3e}")
print(f"속도 계수 절대값 최대 {cvmax:.3e}")
print(f"속도 float32 오차    {vel_err:.4f} rad/s  (참조 속도 |max| {vel_absmax:.2f})")
if vel_err > 1.0:
    print("  경고: 속도 채널이 float32 에서 깨졌다. 학습은 float32 로 평가한다.")
if errs_rms:
    print(f"원본 대비 오차       RMS 평균 {np.mean(errs_rms):.2f} deg / "
          f"최대 {np.max(errs_max):.2f} deg")
print(f"저장 {PKL_OUT} ({time.time()-t0:.0f}s)")
