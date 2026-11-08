"""서기 3 s → 전진 VX 10 s → 서기 3 s 에서 몸통 자이로 RMS(°/s, x 좌우 · y 앞뒤)를 실물 real_walk 기록과 같은 구간으로 잰다.
사용: PYTHONPATH=. python sim_sway.py <onnx> <가속> <vx> <시드 수>"""
import sys, math
import mujoco, numpy as np
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer, ServoProfile
from tools.eval_policy_behavior import DEFAULT_MODEL, DEFAULT_REFERENCE
POL, ACC, VX, NS = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
infer = MjInfer(DEFAULT_MODEL, DEFAULT_REFERENCE, POL, standing=False, head_from_command=True)
m, d = infer.model, infer.data
dt = infer.sim_dt * infer.decimation; names = infer.actuator_names; home = infer.default_actuator.copy()
R = []; fell = 0
for seed in range(NS):
    np.random.seed(seed)
    mujoco.mj_resetData(m, d); mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
    for a in ("last_action", "last_last_action", "last_last_last_action"): setattr(infer, a, np.zeros(infer.num_dofs))
    infer.motor_targets = home.copy(); infer.prev_motor_targets = home.copy(); infer.imitation_i = 0
    delayed = ActionDelay(home, "train", seed=seed); infer.servo = ServoProfile(home, accel=ACC); goal = home.copy()
    T, W = [], []; k = 0
    for _ in range(int(16.0 / dt) * infer.decimation):
        d.ctrl = infer.servo(goal, infer.sim_dt); mujoco.mj_step(m, d); k += 1
        if k % infer.decimation: continue
        t = k * infer.sim_dt
        infer.commands = [VX if 3.0 <= t < 13.0 else 0.0, 0, 0, 0, 0, 0, 0]
        infer.imitation_i = (infer.imitation_i + infer.phase_frequency_factor) % infer.PRM.nb_steps_in_period
        ph = infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi
        infer.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
        act = infer.policy.infer(infer.get_obs(d, infer.commands))
        infer.last_last_last_action = infer.last_last_action.copy(); infer.last_last_action = infer.last_action.copy(); infer.last_action = act.copy()
        tg = infer.targets_from_action(act)
        tg = np.clip(tg, infer.prev_motor_targets - infer.max_motor_velocity * dt, infer.prev_motor_targets + infer.max_motor_velocity * dt)
        tg = np.clip(tg, infer.ctrl_lower, infer.ctrl_upper)
        infer.motor_targets = tg.copy(); infer.prev_motor_targets = tg.copy(); goal = delayed(tg)
        T.append(t); W.append(np.degrees(np.asarray(infer.get_gyro(d))))
    up = np.array([0, 0, 1]) @ d.xmat[1].reshape(3, 3) @ np.array([0, 0, 1])
    fell += up < 0.5
    T, W = np.array(T), np.array(W)
    w1 = (T >= 3.5) & (T < 6.5); w2 = (T >= 3.5) & (T < 13.0)
    R.append([np.sqrt(np.mean(W[w1, 0] ** 2)), np.sqrt(np.mean(W[w1, 1] ** 2)), np.sqrt(np.mean(W[w2, 0] ** 2)), np.sqrt(np.mean(W[w2, 1] ** 2))])
R = np.array(R)
print(f"# {POL} 가속 {ACC} vx {VX} 시드 {NS} 넘어짐 {fell}")
print(f"걷기 0.5~3.5 s RMS x {R[:,0].mean():.1f} ({R[:,0].min():.1f}~{R[:,0].max():.1f})  y {R[:,1].mean():.1f} ({R[:,1].min():.1f}~{R[:,1].max():.1f})")
print(f"걷기 0.5~10 s  RMS x {R[:,2].mean():.1f} ({R[:,2].min():.1f}~{R[:,2].max():.1f})  y {R[:,3].mean():.1f} ({R[:,3].min():.1f}~{R[:,3].max():.1f})")
