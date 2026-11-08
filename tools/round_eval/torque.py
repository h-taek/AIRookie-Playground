"""정책을 명령별 12 초 걸려 14 축 토크 절댓값 합과 포트별 부하 합(%), 월드 속도를 잰다. 머리 목표각은 홈(머리 명령 0 직결, v2_12 ~ 학습 조건)이다. 사용: python tools/round_eval/torque.py <onnx> [몸통 휨 강성]
강성을 주면 flex_site.sway 와 같은 몸(좌우 강성, 앞뒤 × 4 / 5.1, 감쇠 0.03 × √(강성 / 5.1))에서 잰다. 보고 비율 기준 v2_48 은 강성 50 이다 (research/04 47-3)"""
import sys
import mujoco
import numpy as np
sys.path.insert(0, ".")
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer
from tools.eval_policy_behavior import DEFAULT_MODEL, DEFAULT_REFERENCE

POLICY = sys.argv[1]
CASES = [("정지", [0, 0, 0]), ("전진 0.05", [0.05, 0, 0]), ("전진 0.111", [0.111, 0, 0]),
         ("전진 0.222", [0.222, 0, 0]), ("회전 1.111", [0, 0, 1.111]), ("옆 0.148", [0, 0.148, 0])]
SECONDS, SETTLE = 12.0, 2.0
HEADS = ("neck_pitch","head_pitch","head_yaw","head_roll")
HI = [infer_names.index(h) for h in HEADS] if False else None

MODEL = DEFAULT_MODEL
if len(sys.argv) > 2:
    from tools.round_eval import flex_site
    k = float(sys.argv[2]); d = 0.03 * np.sqrt(k / 5.1)
    MODEL = flex_site.make(0, k, d, k * 4.0 / 5.1, d)
infer = MjInfer(MODEL, DEFAULT_REFERENCE, POLICY, standing=False)
infer.servo = lambda goal, dt: np.asarray(goal).copy()   # v2_8 학습 조건: 가속 흉내 없음
infer.servo.reset = lambda: None
model, data = infer.model, infer.data
names = infer.actuator_names
HI = [names.index(h) for h in HEADS]
fr = model.actuator_forcerange[:, 1]
dt = infer.sim_dt * infer.decimation
rows = {}
VEL = {}
for label, c in CASES:
    np.random.seed(0)
    mujoco.mj_resetData(model, data); mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    for a in ("last_action", "last_last_action", "last_last_last_action"):
        setattr(infer, a, np.zeros(infer.num_dofs))
    infer.motor_targets = infer.default_actuator.copy(); infer.prev_motor_targets = infer.default_actuator.copy()
    infer.imitation_i = 0; infer.commands = list(c) + [0, 0, 0, 0]
    delayed = ActionDelay(infer.default_actuator, "train"); goal = infer.default_actuator.copy()
    T, AR, H, Q = [], [], [], []
    n = int(round(SECONDS / dt)); k = 0
    for _ in range(n * infer.decimation):
        data.ctrl = goal; mujoco.mj_step(model, data); k += 1
        if k % infer.decimation: continue
        infer.imitation_i = (infer.imitation_i + infer.phase_frequency_factor) % infer.PRM.nb_steps_in_period
        ph = infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi
        infer.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
        act = infer.policy.infer(infer.get_obs(data, infer.commands))
        AR.append(float(np.sum((act - infer.last_action) ** 2)))
        infer.last_last_last_action = infer.last_last_action.copy(); infer.last_last_action = infer.last_action.copy()
        infer.last_action = act.copy()
        t = infer.default_actuator + act * infer.action_scale
        t = np.clip(t, infer.prev_motor_targets - infer.max_motor_velocity * dt, infer.prev_motor_targets + infer.max_motor_velocity * dt)
        t = np.clip(t, infer.ctrl_lower, infer.ctrl_upper)
        t[HI] = infer.default_actuator[HI]
        infer.motor_targets = t.copy(); infer.prev_motor_targets = t.copy(); goal = delayed(t)
        if len(T) == int(SETTLE / dt): X0 = data.qpos[:2].copy()
        T.append(data.actuator_force.copy()); H.append(float(data.qpos[2])); Q.append([data.qpos[model.jnt_qposadr[model.joint(n).id]] for n in HEADS])
    s = int(SETTLE / dt)
    VEL[label] = (data.qpos[0] - X0[0]) / (SECONDS - SETTLE), (data.qpos[1] - X0[1]) / (SECONDS - SETTLE)
    T = np.array(T)[s:]; AR = np.array(AR)[s:]; H = np.array(H)
    load = np.abs(T) / fr * 100  # 실물 부하 % 와 같은 단위
    rows[label] = dict(q=np.degrees(np.array(Q)[s:]), tq=np.sum(T ** 2, 1), ar=AR, load=load, fell=bool(np.any(H < 0.10)))

print("항 크기 (스텝 평균, 가중치 곱한 값) / alive +2.0")
print(f"{'명령':12s} {'Σtau²':>7s} {'torques':>9s} {'action_rate':>11s} {'부하합%평균':>10s} {'부하합%p95':>10s} {'부하합%최대':>10s} 넘어짐")
for l, r in rows.items():
    ls = r["load"].sum(1)
    print(f"{l:12s} {r['tq'].mean():7.3f} {-1e-3*r['tq'].mean():9.5f} {-0.5*r['ar'].mean():11.4f} {ls.mean():10.0f} {np.percentile(ls,95):10.0f} {ls.max():10.0f} {r['fell']}")
print("\n축별 평균 부하 % (|tau|/forcerange)")
print(f"{'축':16s}" + "".join(f"{l:>11s}" for l in rows))
for i, nm in enumerate(names):
    print(f"{nm:16s}" + "".join(f"{r['load'][:, i].mean():11.0f}" for r in rows.values()))

print("\nforcerange:", np.unique(fr))
L = [names.index(n) for n in names if n.startswith("left_")] + [names.index(n) for n in ("neck_pitch","head_pitch","head_yaw","head_roll")]
R = [names.index(n) for n in names if n.startswith("right_")]
Hd = [names.index(n) for n in ("neck_pitch","head_pitch","head_yaw","head_roll")]
print(f"{'명령':12s} {'Σ|tau| N·m':>10s} {'왼다리+머리%':>11s} {'그중 머리%':>9s} {'오른다리%':>9s}")
for l, r in rows.items():
    ld = r["load"]
    print(f"{l:12s} {(ld*fr/100).sum(1).mean():10.2f} {ld[:,L].sum(1).mean():11.0f} {ld[:,Hd].sum(1).mean():9.0f} {ld[:,R].sum(1).mean():9.0f}")

print("\n머리 관절 각도 흔들림: 표준편차 deg / 최대-최소 deg")
for l, r in rows.items():
    q = r["q"]
    print(f"{l:12s} " + "  ".join(f"{h}:{q[:,i].std():4.1f}/{np.ptp(q[:,i]):4.1f}" for i, h in enumerate(HEADS)))

print("\n월드 기준 평균 속도 x, y m/s")
for l,v in VEL.items(): print(f"{l:12s} {v[0]:+.3f} {v[1]:+.3f}")
