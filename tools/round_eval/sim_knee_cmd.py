"""sim_sway 와 같은 걷기에서 무릎 명령 거칠기(research/10 26-7 지표)를 잰다.
사용: PYTHONPATH=.:tools/round_eval python tools/round_eval/sim_knee_cmd.py <onnx> <가속> <vx> <시드 수>"""
import sys, math
import mujoco, numpy as np
from cmd_rough import rough
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
    T, W, K = [], [], []; k = 0
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
        T.append(t); W.append(np.degrees(np.asarray(infer.get_gyro(d)))); K.append(np.degrees(tg[[names.index('left_knee'), names.index('right_knee')]]))
    up = np.array([0, 0, 1]) @ d.xmat[1].reshape(3, 3) @ np.array([0, 0, 1])
    fell += up < 0.5
    T, K = np.array(T), np.array(K)
    for (lo, hi) in ((3.5, 13.0), (3.5, 6.5)):
        w = (T >= lo) & (T < hi)
        print(f"시드 {seed} {lo}~{hi}  " + "   ".join(f"{s_} {r[0]:.2f}° {100*r[1]:.0f}% {100*r[2]:.0f}%" for s_, r in zip("lr", (rough(K[w, 0]), rough(K[w, 1])))))
print(f"# {POL} 가속 {ACC} vx {VX} 시드 {NS}")
