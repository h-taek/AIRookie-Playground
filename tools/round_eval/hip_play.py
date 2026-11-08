# 외장 유격 관절(`*_play`)이 있던 모델을 전제로 한다. 지금 모델에는 외장 유격이 없으므로 다시 쓸 때 고친다 (2026-10-21).
"""외장 유격을 움직이는 틈으로 넣고(관측에 안 보임), 서기 3 초 뒤 전진 0.05 · 20 초. 진단용이다.

사용: python tools/round_eval/hip_play.py <onnx> <반폭 도,...> <저항배율,...(0=강하 없음)> <시드 수> <hip|leg> [몸통 dx mm,...] [--view]
--view 는 뷰어 창에 실제 속도로 띄우고 조건을 차례로 되풀이한다(mjpython 으로 돌린다).
hip 은 좌우 엉덩이 피치, leg 는 좌우 엉덩이 피치 · 무릎 · 발목의 유격 관절 범위를 ±반폭으로 둔다.
그 관절들의 유격각은 정책 관측에서 뺀다. 나머지 유격 관절은 XML 기본(±0.5°)이고 관측에 들어간다.
"""
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
sys.path.insert(0, ".")
from playground.open_duck_mini_v2.joystick import default_config
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer
from tools.eval_policy_behavior import DEFAULT_MODEL, DEFAULT_REFERENCE

VIEW = "--view" in sys.argv
if VIEW:
    sys.argv.remove("--view")
VIEWER = None
POLICY = sys.argv[1]
OFFS = [float(x) for x in sys.argv[2].split(",")]
RS = [float(x) for x in sys.argv[3].split(",")]
SEEDS = int(sys.argv[4])
WHICH = sys.argv[5] if len(sys.argv) > 5 else "hip"
DXS = [float(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else [0.0]
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


names = [model.actuator(i).name for i in range(model.nu)]
PLAY = [f"{s}_{j}" for s in ("left", "right") for j in (("hip_pitch",) if WHICH == "hip" else ("hip_pitch", "knee", "ankle"))]
# v2_27 부터 모델에 외장 유격 관절(`*_play`)이 있고 뷰어가 이미 관측에서 뺀다. 옛 모델이면 유격 관절을 넓히고 여기서 뺀다.
HAS_PLAY = all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n + "_play") >= 0 for n in PLAY)
bl_jnt = np.array([model.joint(n + ("_play" if HAS_PLAY else "_backlash")).id for n in PLAY])
bl_adr = model.jnt_qposadr[bl_jnt]
act_i = np.array([names.index(n) for n in PLAY])
range0 = model.jnt_range[bl_jnt].copy()
meas0 = infer.get_measured_joint_angles
def measured(d):
    q = meas0(d).copy()
    if not HAS_PLAY:
        q[act_i] -= d.qpos[bl_adr]
    return q
infer.get_measured_joint_angles = measured
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


def run(half_deg, rscale, seed, dx_mm=0.0):
    np.random.seed(seed)
    model.body_ipos[trunk] = ipos0 + np.array([dx_mm / 1000.0, 0, 0])
    model.jnt_range[bl_jnt] = np.array([-1.0, 1.0]) * np.radians(half_deg)
    mujoco.mj_resetData(model, data); mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    for a in ("last_action", "last_last_action", "last_last_last_action"):
        setattr(infer, a, np.zeros(infer.num_dofs))
    infer.motor_targets = infer.default_actuator.copy(); infer.prev_motor_targets = infer.default_actuator.copy()
    infer.imitation_i = 0
    delayed = ActionDelay(infer.default_actuator, "train"); goal = infer.default_actuator.copy()
    port_r = r0 * rscale
    n = int(round((STAND + WALK) / dt)); k = 0
    stand_lean, walk_lean, vmin, x0, stand_imu, walk_imu, stand_q = [], [], [], None, [], [], []
    wall0 = time.time()
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
        if VIEWER is not None and k % infer.decimation == 0:
            if not VIEWER.is_running():
                sys.exit(0)
            VIEWER.sync()
            lag = k * infer.sim_dt - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
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


def table():
    set_imu(0.0)
    for dx in DXS:
      for off in OFFS:
        for rs in RS:
            rows = [run(off, rs, s, dx) for s in range(SEEDS)]
            falls = [r for r in rows if r["fell"] is not None]
            ok = [r for r in rows if r["fell"] is None]
            st = np.mean([r["stand"] for r in rows]); si = np.mean([r["stand_imu"] for r in rows])
            txt = (f"몸통 dx {dx:+3.0f} mm  유격 {WHICH} ±{off:3.1f}°  저항 ×{rs:.1f}  서기 앞숙임 {st:+5.1f}°"
                   f"  넘어짐 {len(falls)}/{len(rows)}")
            if falls: txt += "  [" + ", ".join(f"{r['fell']:.1f}s {r['dir']}" for r in falls) + "]"
            if ok:
                txt += f"  걷기 앞숙임 평균 {np.mean([r['walk'] for r in ok]):+.1f}° 최대 {np.max([r['walk_max'] for r in ok]):+.1f}°  vx {np.mean([r['vx'] for r in ok]):+.3f}"
                q = np.mean([r["stand_q"] for r in ok], 0)
                txt += f"  서기 관측 엉덩이 피치(홈 기준) 왼 {q[names.index('left_hip_pitch')]:+.1f}° 오른 {q[names.index('right_hip_pitch')]:+.1f}°"
            if rs > 0: txt += f"  왼다리 최저 {min(r['vmin'] for r in rows):.1f} V"
            print(txt, flush=True)


if VIEW:
    set_imu(0.0)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        VIEWER = v
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = trunk
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9, 90.0, -10.0
        while v.is_running():
            for dx in DXS:
                for off in OFFS:
                    for rs in RS:
                        print(f"▶ 몸통 dx {dx:+.0f} mm · 유격 {WHICH} ±{off:.2f}° · 저항 ×{rs:.1f}  (서기 {STAND:.0f} 초 → 전진 0.05)", flush=True)
                        r = run(off, rs, 0, dx)
                        print("   앞으로 넘어짐" if r["fell"] is not None and r["dir"] == "앞" else
                              (f"   {r['dir']}로 넘어짐" if r["fell"] is not None else f"   끝까지 걸음 vx {r['vx']:+.3f}"),
                              (f"{r['fell']:.1f} 초" if r["fell"] is not None else ""), flush=True)
                        t_end = time.time() + 2.0
                        while time.time() < t_end and v.is_running():
                            v.sync(); time.sleep(0.02)
else:
    table()
