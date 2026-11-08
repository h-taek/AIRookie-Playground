"""sim_sway 와 같은 걷기에서 센서 관측(자이로 · 가속도 · 관절각 · 관절속도)의 5 Hz 위 성분 RMS 를 잰다.
사용: PYTHONPATH=.:<scratch> python sim_obs_hf.py <onnx> <가속> <vx> <시드 수>"""
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
    T, W, K, OB = [], [], [], []; k = 0
    for _ in range(int(16.0 / dt) * infer.decimation):
        d.ctrl = infer.servo(goal, infer.sim_dt); mujoco.mj_step(m, d); k += 1
        if k % infer.decimation: continue
        t = k * infer.sim_dt
        infer.commands = [VX if 3.0 <= t < 13.0 else 0.0, 0, 0, 0, 0, 0, 0]
        infer.imitation_i = (infer.imitation_i + infer.phase_frequency_factor) % infer.PRM.nb_steps_in_period
        ph = infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi
        infer.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
        ob = infer.get_obs(d, infer.commands); act = infer.policy.infer(ob)
        infer.last_last_last_action = infer.last_last_action.copy(); infer.last_last_action = infer.last_action.copy(); infer.last_action = act.copy()
        tg = infer.targets_from_action(act)
        tg = np.clip(tg, infer.prev_motor_targets - infer.max_motor_velocity * dt, infer.prev_motor_targets + infer.max_motor_velocity * dt)
        tg = np.clip(tg, infer.ctrl_lower, infer.ctrl_upper)
        infer.motor_targets = tg.copy(); infer.prev_motor_targets = tg.copy(); goal = delayed(tg)
        T.append(t); W.append(np.degrees(np.asarray(infer.get_gyro(d)))); K.append(np.degrees(tg[[names.index('left_knee'), names.index('right_knee')]])); OB.append(ob)
    up = np.array([0, 0, 1]) @ d.xmat[1].reshape(3, 3) @ np.array([0, 0, 1])
    fell += up < 0.5
    T, OB = np.array(T), np.array(OB); w = (T >= 3.5) & (T < 6.5)
    def lp(x, fc=5.0, dt=0.02):
        X = np.fft.rfft(x, axis=0); f = np.fft.rfftfreq(len(x), dt); X[f > fc] = 0
        return np.fft.irfft(X, n=len(x), axis=0)
    for g, sl in (("gyro", slice(0, 3)), ("accel", slice(3, 6)), ("joint_ang", slice(13, 27)), ("joint_vel", slice(27, 41))):
        x = OB[w][:, sl]; hf = x - lp(x)
        print(f"시드 {seed} {g:10s} 5 Hz 위 RMS {np.sqrt(np.mean(hf ** 2)):.4f}  전체 표준편차 {x.std(0).mean():.4f}")
print(f"# {POL} 가속 {ACC} vx {VX} 시드 {NS}")
