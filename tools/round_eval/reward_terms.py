"""학습 환경(joystick.Joystick)으로 정책을 굴려 명령 일정별 보상 항을 잰다. 무작위화(domain_randomize)는 걸지 않는다.
일정 × 시드를 한 묶음(jax.vmap)으로 같이 굴린다. 정책 onnx 는 입력이 1 개 고정이라 환경마다 따로 부른다.
보상에 없는 후보 항도 가중치 없이 잰다. ang_vel_xy 는 몸통 좌우 · 앞뒤 각속도 제곱합, ang_vel_xy_ref 는 시범 동작의
몸통 각속도(imitation 이 견주는 값)와의 차이 제곱합이다. 단위는 (rad/s)². tracking_lin_vel_low 는 전진 허용폭을
명령 크기에 비례해 좁힌 추종 값(가중치 곱하기 전)이고, '후보 합' 은 지금 합에서 추종을 이것으로 바꾸고 ang_vel_xy × −1 을 더한 값이다.
추종만 바꾼 합 셋을 함께 낸다. 어느 학습 코드의 환경에서 재든 같은 기준으로 견줄 수 있다.
  '넓은 폭 합' 은 전진 폭 고정(tracking_low_speed_ref 없음), '저속 폭 합' 은 0.111 아래 모든 명령에서 전진 폭을 좁힌 값,
  '전진만 저속 폭 합' 은 옆 · 회전 명령이 둘 다 0 일 때만 좁힌 값이다.
일정의 명령은 vx 하나 또는 (vx, vy, wz) 다.
정책 × 바닥 마찰 조합마다 프로세스 하나를 **동시에** 띄운다. 병목은 환경마다 onnx 를 부르는 CPU 이고 GPU 는 놀아서
프로세스 수만큼 빨라진다. jax 가 프로세스마다 GPU 메모리를 미리 잡지 않게 XLA_PYTHON_CLIENT_PREALLOCATE=false 로 띄운다.
사용: python reward_terms.py <onnx>... [--impl jax] [--seeds 4] [--mu 바닥 마찰...] [--acc 서보 가속 흉내 rad/s²]
  --mu 를 주지 않으면 장면 XML 값이다. --acc 를 주면 학습 끝값을 이것으로 바꾼다."""
import argparse
import itertools
import os
import subprocess
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("onnx", nargs="+")
ap.add_argument("--impl", default="jax")
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--mu", type=float, nargs="+", default=[None])
ap.add_argument("--acc", type=float, default=None)
ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # 조합 하나를 재는 작업 프로세스
A = ap.parse_args()

if not A.one:
    jobs = list(itertools.product(A.onnx, A.mu))
    env_ = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    run = []
    for pol, mu in jobs:
        cmd = [sys.executable, "-u", __file__, pol, "--one", "--impl", A.impl, "--seeds", str(A.seeds)]
        if mu is not None:
            cmd += ["--mu", str(mu)]
        if A.acc is not None:
            cmd += ["--acc", str(A.acc)]
        # 파이프로 받으면 출력이 큰 프로세스가 버퍼에 막힌다. 파일로 받는다.
        out = tempfile.TemporaryFile("w+")
        run.append((pol, mu, subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True, env=env_), out))
    print(f"{len(run)} 개를 동시에 띄웠다", flush=True)
    rc = 0
    for pol, mu, p, out in run:
        p.wait()
        out.seek(0)
        lines = out.read().splitlines()
        print(f"#### {pol}  바닥 마찰 {'XML' if mu is None else mu}  (종료 코드 {p.returncode})", flush=True)
        keep = lines if p.returncode else [l for l in lines if l.startswith(("==", "   "))]
        print("\n".join(keep), flush=True)
        rc |= p.returncode
    sys.exit(rc)

import numpy as np
import jax
import jax.numpy as jp
sys.path.insert(0, ".")
from playground.open_duck_mini_v2 import joystick
from playground.common.onnx_infer import OnnxInfer

POL, IMPL, NS, ACC, MU = A.onnx[0], A.impl, A.seeds, A.acc, A.mu[0]
over = {"impl": IMPL} if ACC is None else {"impl": IMPL, "servo_max_accel": ACC}
env = joystick.Joystick(task="flat_terrain_backlash", config_overrides=over)
# 학습 끝 조건(servo_accel_ramp · knee_kp_ramp 의 끝값)으로 잰다.
env.set_train_progress(1.0)
if MU is not None:
    # 바닥 마찰을 고정한다. 학습에서는 randomize.py 가 판마다 덮어쓰는 값이다.
    _floor = env._mj_model.geom("floor").id
    env._mjx_model = env._mjx_model.tree_replace(
        {"geom_friction": env._mjx_model.geom_friction.at[_floor, 0].set(MU)}
    )
pol = OnnxInfer(POL, awd=True)
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
local_vx = jax.jit(jax.vmap(lambda d: env.get_local_linvel(d)[0]))
ang_xy = jax.jit(jax.vmap(lambda d: jp.sum(jp.square(env.get_gyro(d)[:2]))))
# imitation 의 ang_vel_xy 와 같은 값(몸통 자유 관절 각속도 3:5 대 시범 37:39)을 포화 없이 본다.
ang_xy_ref = jax.jit(jax.vmap(lambda d, r: jp.sum(jp.square(env.get_floating_base_qvel(d.qvel)[3:5] - r[37:39]))))
SIG = env._config.reward_config.tracking_sigma
W_TRACK = env._config.reward_config.scales.tracking_lin_vel


def _track(d, info, narrow_all, narrow_fwd):
    # 전진 허용폭 = SIG × (|명령| / 0.111)², 0.2 배 아래로는 줄이지 않는다. 정지 명령과 옆 성분은 SIG 그대로다.
    # narrow_all 은 0.111 아래 모든 명령에서, narrow_fwd 는 옆 · 회전 명령이 둘 다 0 일 때만 좁힌다.
    c = info["command"]; v = env.get_local_linvel(d)
    r = jp.clip(jp.abs(c[0]) / 0.111, 0.2, 1.0)
    narrow = (jp.abs(c[0]) > 0) & (narrow_all | (narrow_fwd & (c[1] == 0) & (c[2] == 0)))
    sx = jp.where(narrow, SIG * r ** 2, SIG)
    return jp.exp(-(jp.square(c[0] - v[0]) / sx + jp.square(c[1] - env._mean_vy(info)) / SIG))


track_low = jax.jit(jax.vmap(lambda d, i: _track(d, i, True, False)))
track_wide = jax.jit(jax.vmap(lambda d, i: _track(d, i, False, False)))
track_fwd = jax.jit(jax.vmap(lambda d, i: _track(d, i, False, True)))
local_vy = jax.jit(jax.vmap(lambda d: env.get_local_linvel(d)[1]))
yaw_rate = jax.jit(jax.vmap(lambda d: env.get_gyro(d)[2]))
dt = env.dt
TQ = env._config.reward_config.scales.torques
# (이름, 일정[(끝 시각, vx 또는 (vx, vy, wz))], 재는 창)
SC = [
    ("정지 명령 서기", [(13.0, 0.0)], (5.0, 13.0)),
    ("서기 3s → 0.05", [(3.0, 0.0), (13.0, 0.05)], (5.0, 13.0)),
    ("서기 3s → 0.111 0.5s → 0.05", [(3.0, 0.0), (3.5, 0.111), (13.5, 0.05)], (5.5, 13.5)),
    ("서기 3s → 0.08", [(3.0, 0.0), (13.0, 0.08)], (5.0, 13.0)),
    ("서기 3s → 0.111 0.5s → 0.08", [(3.0, 0.0), (3.5, 0.111), (13.5, 0.08)], (5.5, 13.5)),
    ("서기 3s → 0.09", [(3.0, 0.0), (13.0, 0.09)], (5.0, 13.0)),
    ("서기 3s → 0.111", [(3.0, 0.0), (13.0, 0.111)], (5.0, 13.0)),
    ("서기 3s → 0.222", [(3.0, 0.0), (13.0, 0.222)], (5.0, 13.0)),
    ("서기 3s → 후진 0.111", [(3.0, 0.0), (13.0, -0.111)], (5.0, 13.0)),
    ("서기 3s → 0.111 0.5s → 후진 0.111", [(3.0, 0.0), (3.5, 0.111), (13.5, -0.111)], (5.5, 13.5)),
    ("서기 3s → 후진 0.05", [(3.0, 0.0), (13.0, -0.05)], (5.0, 13.0)),
    ("서기 3s → 0.111 0.5s → 후진 0.05", [(3.0, 0.0), (3.5, 0.111), (13.5, -0.05)], (5.5, 13.5)),
    ("서기 3s → 회전 1.111", [(3.0, 0.0), (13.0, (0.0, 0.0, 1.111))], (5.0, 13.0)),
    ("서기 3s → 회전 0.5", [(3.0, 0.0), (13.0, (0.0, 0.0, 0.5))], (5.0, 13.0)),
    ("서기 3s → 옆 0.148", [(3.0, 0.0), (13.0, (0.0, 0.148, 0.0))], (5.0, 13.0)),
    ("서기 3s → 0.05 + 회전 0.5", [(3.0, 0.0), (13.0, (0.05, 0.0, 0.5))], (5.0, 13.0)),
    ("서기 3s → 0.05 + 옆 0.1", [(3.0, 0.0), (13.0, (0.05, 0.1, 0.0))], (5.0, 13.0)),
    ("서기 3s → 0.05 + 옆 0.1 + 회전 0.5", [(3.0, 0.0), (13.0, (0.05, 0.1, 0.5))], (5.0, 13.0)),
    ("서기 3s → 0.111 + 회전 0.5", [(3.0, 0.0), (13.0, (0.111, 0.0, 0.5))], (5.0, 13.0)),
]


def cmd_at(sched, t):
    for end, v in sched:
        if t < end: break
    return v if isinstance(v, tuple) else (v, 0.0, 0.0)


# 환경 순서는 일정마다 시드 0 .. NS-1 이다.
N = len(SC) * NS
sc_of = np.repeat(np.arange(len(SC)), NS)
st = reset(jax.vmap(jax.random.PRNGKey)(jp.tile(jp.arange(NS), len(SC))))
st = st.replace(info={**st.info, "torque_scale": jp.full((N,), TQ, dtype=jp.float32)})
zero_step = jp.zeros_like(st.info["step"])
n_steps = max(int(round(s[-1][0] / dt)) for _, s, _ in SC)
alive = np.ones(N, bool)
fell = np.zeros(N, bool)
acc, cnt, vx_sum, vy_sum, wz_sum = {}, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
for i in range(n_steps):
    t = i * dt
    c = np.zeros((N, 7), np.float32)
    c[:, 0:3] = [cmd_at(SC[s][1], t) for s in sc_of]
    c = jp.asarray(c)
    st = st.replace(info={**st.info, "command": c, "command_target": c, "step": zero_step})
    obs = np.asarray(st.obs["state"])
    a = np.stack([pol.infer(obs[j]) for j in range(N)])
    st = step(st, jp.asarray(a))
    done = np.asarray(st.done) > 0
    fell |= alive & done
    alive &= ~done
    # 일정마다 길이가 달라 끝난 일정은 더 세지 않는다.
    win = np.array([SC[s][2][0] <= t < SC[s][2][1] for s in sc_of]) & alive
    if win.any():
        for k, v in st.metrics.items():
            if k.startswith(("reward/", "cost/")):
                acc.setdefault(k, np.zeros(N))
                acc[k] += np.where(win, np.asarray(v), 0.0)
        vx_sum += np.where(win, np.asarray(local_vx(st.data)), 0.0)
        vy_sum += np.where(win, np.asarray(local_vy(st.data)), 0.0)
        wz_sum += np.where(win, np.asarray(yaw_rate(st.data)), 0.0)
        acc.setdefault("후보/ang_vel_xy", np.zeros(N))
        acc["후보/ang_vel_xy"] += np.where(win, np.asarray(ang_xy(st.data)), 0.0)
        acc.setdefault("후보/tracking_lin_vel_low", np.zeros(N))
        acc["후보/tracking_lin_vel_low"] += np.where(win, np.asarray(track_low(st.data, st.info)), 0.0)
        acc.setdefault("후보/tracking_lin_vel_wide", np.zeros(N))
        acc["후보/tracking_lin_vel_wide"] += np.where(win, np.asarray(track_wide(st.data, st.info)), 0.0)
        acc.setdefault("후보/tracking_lin_vel_fwd", np.zeros(N))
        acc["후보/tracking_lin_vel_fwd"] += np.where(win, np.asarray(track_fwd(st.data, st.info)), 0.0)
        acc.setdefault("후보/ang_vel_xy_ref", np.zeros(N))
        acc["후보/ang_vel_xy_ref"] += np.where(win, np.asarray(ang_xy_ref(st.data, st.info["current_reference_motion"])), 0.0)
        cnt += win

for s, (name, _, _) in enumerate(SC):
    m = sc_of == s; n = max(cnt[m].sum(), 1)
    print(f"== {name}  넘어짐 {int(fell[m].sum())}/{NS}  vx {vx_sum[m].sum() / n:+.3f}  vy {vy_sum[m].sum() / n:+.3f}  wz {wz_sum[m].sum() / n:+.3f}", flush=True)
    tot = 0.0
    for k in sorted(k for k in acc if not k.startswith("후보/")):
        v = acc[k][m].sum() / n * (1 if k.startswith("reward/") else -1)
        tot += v
        print(f"   {k:28s} {v:+8.3f}", flush=True)
    print(f"   {'합(스텝당, ×dt 전)':28s} {tot:+8.3f}", flush=True)
    base = tot - acc["reward/tracking_lin_vel"][m].sum() / n
    for label, k in (("넓은 폭 합", "wide"), ("저속 폭 합", "low"), ("전진만 저속 폭 합", "fwd")):
        print(f"   {label:28s} {base + W_TRACK * acc[f'후보/tracking_lin_vel_{k}'][m].sum() / n:+8.3f}", flush=True)
    for k in sorted(k for k in acc if k.startswith("후보/")):
        print(f"   {k:28s} {acc[k][m].sum() / n:8.3f}  (가중치 없음)", flush=True)
    if "cost/ang_vel_xy" not in acc:  # 후보를 이미 학습 보상으로 쓰는 코드(v2_45 부터)에서는 지금 합이 곧 후보 합이다
        new = tot - acc["reward/tracking_lin_vel"][m].sum() / n + W_TRACK * acc["후보/tracking_lin_vel_low"][m].sum() / n - acc["후보/ang_vel_xy"][m].sum() / n
        print(f"   {'후보 합':28s} {new:+8.3f}", flush=True)
