"""걸으며 머리 명령을 아홉 방식으로 바꾸는 시험(이동 아홉 × 머리 아홉, 무작위는 시드 셋 = 117 번). 사용: python tools/round_eval/head_stress.py <onnx> <머리 직결 1|0>"""
import sys
import mujoco
import numpy as np
sys.path.insert(0, ".")
from playground.open_duck_mini_v2 import actuation
from playground.open_duck_mini_v2.joystick import default_config
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer
from tools.eval_policy_behavior import DEFAULT_MODEL, DEFAULT_REFERENCE

POLICY = sys.argv[1]
HFC = sys.argv[2] == "1"
SECONDS, SETTLE = 15.0, 2.0
MOVES = [("정지", [0, 0, 0]), ("전진 0.05", [0.05, 0, 0]), ("전진 0.111", [0.111, 0, 0]), ("전진 0.222", [0.222, 0, 0]),
         ("후진 0.111", [-0.111, 0, 0]), ("좌 0.148", [0, 0.148, 0]), ("우 0.148", [0, -0.148, 0]),
         ("회전 +1.111", [0, 0, 1.111]), ("회전 -1.111", [0, 0, -1.111])]

infer = MjInfer(DEFAULT_MODEL, DEFAULT_REFERENCE, POLICY, standing=False, head_from_command=HFC)
infer.servo = lambda goal, dt: np.asarray(goal).copy()
infer.servo.reset = lambda: None
LO, HI = actuation.config_head_command_range(infer.model, default_config())
MID, AMP = (LO + HI) / 2, (HI - LO) / 2
model, data = infer.model, infer.data
dt = infer.sim_dt * infer.decimation


def sweep(axis, f):
    def h(t, rng):
        v = np.zeros(4); v[axis] = MID[axis] + AMP[axis] * np.sin(2 * np.pi * f * t); return v
    return h


def all4(t, rng):
    return MID + AMP * np.sin(2 * np.pi * np.array([0.6, 0.9, 0.5, 0.8]) * t + np.array([0, 1, 2, 3]))


def snaps(period):
    state = {}
    def h(t, rng):
        k = int(t / period)
        if k not in state:
            state[k] = rng.uniform(LO, HI)
        return state[k]
    return h


def shake(t, rng):
    v = np.zeros(4); v[2] = 0.3 * np.sign(np.sin(2 * np.pi * 2.0 * t)); return v


# 시험마다 새로 만든다. 무작위 홱은 뽑은 목표를 들고 있어 재사용하면 시드가 안 먹는다.
HEADS = [("없음", lambda: (lambda t, r: np.zeros(4))), ("좌우 돌리기 0.5Hz", lambda: sweep(2, 0.5)), ("끄덕 1Hz", lambda: sweep(1, 1.0)),
         ("갸웃 0.7Hz", lambda: sweep(3, 0.7)), ("목 앞뒤 0.5Hz", lambda: sweep(0, 0.5)), ("네 축 동시", lambda: all4),
         ("무작위 홱 1초", lambda: snaps(1.0)), ("무작위 홱 0.4초", lambda: snaps(0.4)), ("좌우 털기 2Hz", lambda: shake)]


def run(cmd, head, seed=0):
    rng = np.random.default_rng(seed); np.random.seed(seed)
    mujoco.mj_resetData(model, data); mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    for a in ("last_action", "last_last_action", "last_last_last_action"):
        setattr(infer, a, np.zeros(infer.num_dofs))
    infer.motor_targets = infer.default_actuator.copy(); infer.prev_motor_targets = infer.default_actuator.copy()
    infer.imitation_i = 0
    delayed = ActionDelay(infer.default_actuator, "train"); goal = infer.default_actuator.copy()
    n = int(round(SECONDS / dt)); k = 0
    tilt, rp, vel, wz, tq = [], [], [], [], []
    fell = False
    for _ in range(n * infer.decimation):
        data.ctrl = goal; mujoco.mj_step(model, data); k += 1
        if k % infer.decimation: continue
        t = k * infer.sim_dt
        infer.commands = list(cmd) + list(head(t, rng))
        infer.imitation_i = (infer.imitation_i + infer.phase_frequency_factor) % infer.PRM.nb_steps_in_period
        ph = infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi
        infer.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
        act = infer.policy.infer(infer.get_obs(data, infer.commands))
        infer.last_last_last_action = infer.last_last_action.copy(); infer.last_last_action = infer.last_action.copy()
        infer.last_action = act.copy()
        tg = infer.targets_from_action(act)
        tg = np.clip(tg, infer.prev_motor_targets - infer.max_motor_velocity * dt, infer.prev_motor_targets + infer.max_motor_velocity * dt)
        tg = np.clip(tg, infer.ctrl_lower, infer.ctrl_upper)
        infer.motor_targets = tg.copy(); infer.prev_motor_targets = tg.copy(); goal = delayed(tg)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, data.qpos[3:7]); R = R.reshape(3, 3)
        if data.qpos[2] < 0.10 or R[2, 2] < 0.5:
            fell = True; break
        if t < SETTLE: continue
        tilt.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        rp.append(data.qvel[3:5].copy())  # 자유 관절 각속도는 몸통 좌표다
        vel.append(R.T @ data.qvel[0:3]); wz.append(data.qvel[5]); tq.append(np.abs(data.actuator_force).sum())
    if fell or not tilt:
        return dict(fell=True, t=k * infer.sim_dt)
    rp = np.array(rp); vel = np.array(vel)
    return dict(fell=False, tilt_mean=np.mean(tilt), tilt_max=np.max(tilt), rp_rms=np.sqrt((rp ** 2).mean(0)).sum(),
                vx=vel[:, 0].mean(), vy=vel[:, 1].mean(), wz=np.mean(wz), tq=np.mean(tq))


print("머리 명령 범위 rad 하한/상한:", np.round(LO, 2), np.round(HI, 2))
print("칸: 몸통 기울기 평균/최대 °, 롤·피치 각속도 RMS 합 rad/s, 속도(전진·옆 m/s, 회전 rad/s), Σ|τ| N·m. 넘어지면 '넘어짐 t초'")
for mname, cmd in MOVES:
    print(f"\n== {mname}")
    for hname, h in HEADS:
        seeds = (0, 1, 2) if "무작위" in hname else (0,)
        for sd in seeds:
            r = run(cmd, h(), sd)
            lab = hname + (f" s{sd}" if len(seeds) > 1 else "")
            if r["fell"]:
                print(f"  {lab:16s} 넘어짐 {r['t']:.1f}초"); continue
            spd = f"wz {r['wz']:+.3f}" if cmd[2] else (f"vy {r['vy']:+.3f}" if cmd[1] else f"vx {r['vx']:+.3f}")
            print(f"  {lab:16s} 기울기 {r['tilt_mean']:4.1f}/{r['tilt_max']:4.1f}  흔들림 {r['rp_rms']:.2f}  {spd}  Σ|τ| {r['tq']:5.2f}")
