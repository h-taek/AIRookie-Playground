"""몸통 무게중심을 앞으로 옮기거나 IMU 를 앞으로 기울여 달고, 서기 3 초 뒤 전진 0.05 · 20 초.

사용: python tools/round_eval/com_shift.py <onnx> <dx_mm,...> <저항배율,...(0=강하 없음)> <시드 수> [IMU 앞기울임 도,...]
IMU 앞기울임 θ 는 몸통이 수평일 때 IMU 가 θ 만큼 앞숙임을 보고하도록 자이로 · 가속도 관측을 돌린다.
"""
import sys
import mujoco
import numpy as np
sys.path.insert(0, ".")
from playground.open_duck_mini_v2.joystick import default_config
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer
from tools.eval_policy_behavior import DEFAULT_MODEL, DEFAULT_REFERENCE

POLICY = sys.argv[1]
DXS = [float(x) for x in sys.argv[2].split(",")]
RS = [float(x) for x in sys.argv[3].split(",")]
SEEDS = int(sys.argv[4])
IMU_DEGS = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.0]
STAND, WALK = 3.0, 20.0

infer = MjInfer(DEFAULT_MODEL, DEFAULT_REFERENCE, POLICY, standing=False, head_from_command=True)
model, data = infer.model, infer.data
dt = infer.sim_dt * infer.decimation
vd = default_config().voltage_drop
trunk = model.body("trunk_assembly").id
ipos0 = model.body_ipos[trunk].copy()
port_b = ("right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
port = np.array([int(model.actuator(i).name in port_b) for i in range(model.nu)])
act_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
act_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
damp, arm = model.dof_damping[act_dof], model.dof_armature[act_dof]
reach_line = model.actuator_forcerange[:, 1] / model.actuator_gainprm[:, 0]
r0 = np.stack([np.array(vd.servo_r_a), np.array(vd.servo_r_b)])


gyro0, acc0 = infer.get_gyro, infer.get_accelerometer
imu_rot = np.eye(3)
infer.get_gyro = lambda d: imu_rot.T @ gyro0(d)
infer.get_accelerometer = lambda d: imu_rot.T @ acc0(d)


def set_imu(deg):
    global imu_rot
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    imu_rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def imu_lean(d):
    a = infer.get_accelerometer(d)
    return np.degrees(np.arctan2(-a[0], a[2]))  # 실물 기록과 같은 가속도 앞숙임


def lean(R):
    return np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))  # + 앞숙임


def run(dx_mm, rscale, seed):
    np.random.seed(seed)
    model.body_ipos[trunk] = ipos0 + np.array([dx_mm / 1000.0, 0, 0])
    mujoco.mj_resetData(model, data); mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    for a in ("last_action", "last_last_action", "last_last_last_action"):
        setattr(infer, a, np.zeros(infer.num_dofs))
    infer.motor_targets = infer.default_actuator.copy(); infer.prev_motor_targets = infer.default_actuator.copy()
    infer.imitation_i = 0
    delayed = ActionDelay(infer.default_actuator, "train"); goal = infer.default_actuator.copy()
    port_r = r0 * rscale
    n = int(round((STAND + WALK) / dt)); k = 0
    stand_lean, walk_lean, vmin, x0, stand_imu, walk_imu, stand_q = [], [], [], None, [], [], []
    for _ in range(n * infer.decimation):
        ctrl = goal
        if rscale > 0:
            ext = data.actuator_force - damp * data.qvel[act_dof] - arm * data.qacc[act_dof]
            cur = vd.cur_gain * np.maximum(np.abs(ext) - vd.cur_hold, 0.0)
            pc = np.zeros(2); np.add.at(pc, port, cur)
            volt = vd.v_nominal - pc @ port_r
            vmin.append(volt[:5].min())
            scale = np.clip(volt / vd.v_nominal, vd.min_scale, 1.0)
            q = data.qpos[act_qpos]
            reach = reach_line * scale
            ctrl = q + np.clip((goal - q) * scale, -reach, reach)
        data.ctrl = ctrl; mujoco.mj_step(model, data); k += 1
        if k % infer.decimation: continue
        t = k * infer.sim_dt
        infer.commands = ([0.05, 0, 0] if t >= STAND else [0, 0, 0]) + [0, 0, 0, 0]
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
            return dict(fell=t - STAND, dir="앞" if lean(R) > 0 else "뒤", roll=np.degrees(np.arcsin(np.clip(R[2, 1], -1, 1))),
                        stand=np.mean(stand_lean) if stand_lean else np.nan, stand_imu=np.mean(stand_imu) if stand_imu else np.nan,
                        vmin=min(vmin) if vmin else np.nan)
        if 1.0 <= t < STAND:
            stand_lean.append(lean(R)); stand_imu.append(imu_lean(data))
            stand_q.append(infer.get_measured_joint_angles(data) - infer.default_actuator)
        if t >= STAND:
            walk_lean.append(lean(R)); walk_imu.append(imu_lean(data))
            if x0 is None: x0 = data.qpos[0]
    return dict(fell=None, stand=np.mean(stand_lean), stand_imu=np.mean(stand_imu), walk_imu=np.mean(walk_imu),
                stand_q=np.degrees(np.mean(stand_q, 0)), walk=np.mean(walk_lean), walk_max=np.max(walk_lean),
                vx=(data.qpos[0] - x0) / WALK, vmin=min(vmin) if vmin else np.nan)


for deg in IMU_DEGS:
    set_imu(deg)
    for dx in DXS:
        for rs in RS:
            rows = [run(dx, rs, s) for s in range(SEEDS)]
            falls = [r for r in rows if r["fell"] is not None]
            ok = [r for r in rows if r["fell"] is None]
            st = np.mean([r["stand"] for r in rows]); si = np.mean([r["stand_imu"] for r in rows])
            txt = (f"IMU {deg:+4.1f}°  몸통 dx {dx:+5.0f} mm  저항 ×{rs:.1f}  서기 앞숙임 실제 {st:+5.1f}° · IMU {si:+5.1f}°"
                   f"  넘어짐 {len(falls)}/{len(rows)}")
            if falls: txt += "  [" + ", ".join(f"{r['fell']:.1f}s {r['dir']}" for r in falls) + "]"
            if ok: txt += f"  걷기 앞숙임 평균 {np.mean([r['walk'] for r in ok]):+.1f}° 최대 {np.max([r['walk_max'] for r in ok]):+.1f}°  vx {np.mean([r['vx'] for r in ok]):+.3f}"
            if rs > 0: txt += f"  왼다리 최저 {min(r['vmin'] for r in rows):.1f} V"
            print(txt, flush=True)
