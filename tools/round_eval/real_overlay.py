# 외장 유격 관절(`*_play`)이 있던 모델을 전제로 한다. 지금 모델에는 외장 유격이 없으므로 다시 쓸 때 고친다 (2026-10-21).
"""실물 기록과 맞추는 격자 시뮬. 정책 하나를 짝 홈 자세로 서기 3 초 뒤 전진 CMD · 20 초 굴린다.

사용: CMD=0.05 DXS=0,10,20 PLS=0.85,1.0 NS=8 [RSC=1.0] [KP=1.0] [FR=1.0] \
      python tools/round_eval/real_overlay.py <onnx> <짝 home_pose.json> 1 1.0
DXS 는 몸통 무게중심 x(학습 XML 기준 mm), PLS 는 외장 유격 반각(°), RSC 는 전압 강하 배율(0 = 없음),
KP 는 서보 kp 배율, FR 은 바닥 마찰 배율이다. CONTACT=fixed 면 관측의 발 접촉을 양발 닿음으로 고정한다(핀 설정이 풀린
실물, Runtime research/10 19장). KICK=0.111,0.5 면 걷기 첫 0.5 초만 전진 명령을 0.111 로 준다(`real_walk.py --kick-vx --kick`). 서보 유격은 표본마다 학습 범위에서 뽑는다.
줄마다 서기 · 걷기 몸통 앞숙임(1 ~ 3 s, 걷기 1 s 뒤부터), 45° 넘음 비율 · 시각 · 방향, 걸은 표본 vx 를 JSON 으로 낸다.
"""
import sys
import mujoco
import numpy as np
sys.path.insert(0, ".")
from playground.open_duck_mini_v2.joystick import default_config
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer
from tools.eval_policy_behavior import DEFAULT_MODEL, DEFAULT_REFERENCE

import json
from playground.common import randomize as RZ
POLICY = sys.argv[1]
PAIR = json.load(open(sys.argv[2]))
N = int(sys.argv[3])
RS = [float(x) for x in sys.argv[4].split(",")]
import os
STAND, WALK = 3.0, 20.0
CMD=float(os.environ['CMD'])

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


for n, v in PAIR["joints"].items():
    model.key_qpos[0, model.jnt_qposadr[model.joint(n).id]] = v
    model.key_ctrl[0, model.actuator(n).id] = v
    infer.default_actuator[model.actuator(n).id] = v
bl = [j for j in range(model.njnt) if model.joint(j).name.endswith("_backlash")]
pl = [j for j in range(model.njnt) if model.joint(j).name.endswith("_play")]
gyro0, acc0 = infer.get_gyro, infer.get_accelerometer
imu_rot = np.eye(3)
infer.get_gyro = lambda d: imu_rot.T @ gyro0(d)
infer.get_accelerometer = lambda d: imu_rot.T @ acc0(d)
if os.environ.get('CONTACT') == 'fixed':
    infer.get_feet_contacts = lambda d: np.ones(2)
KICK = [float(x) for x in os.environ['KICK'].split(',')] if os.environ.get('KICK') else None


def set_imu(deg):
    global imu_rot
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    imu_rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def imu_lean(d):
    a = infer.get_accelerometer(d)
    return np.degrees(np.arctan2(-a[0], a[2]))  # 실물 기록과 같은 가속도 앞숙임


def lean(R):
    return np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))  # + 앞숙임


_KP=float(os.environ.get('KP','1.0')); _FR=float(os.environ.get('FR','1.0'))
model.actuator_gainprm[:,0]*=_KP; model.actuator_biasprm[:,1]*=_KP; model.geom_friction[:,0]*=_FR
reach_line = model.actuator_forcerange[:, 1] / model.actuator_gainprm[:, 0]

def run(sample, rscale, seed):
    np.random.seed(seed)
    model.body_ipos[trunk] = ipos0 + sample["dpos"]
    for j, h in zip(bl, sample["bl"]): model.jnt_range[j] = [-h, h]
    for j, h in zip(pl, sample["pl"]): model.jnt_range[j] = [-h, h]
    mujoco.mj_resetData(model, data); mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    for a in ("last_action", "last_last_action", "last_last_last_action"):
        setattr(infer, a, np.zeros(infer.num_dofs))
    infer.motor_targets = infer.default_actuator.copy(); infer.prev_motor_targets = infer.default_actuator.copy()
    infer.imitation_i = 0
    delayed = ActionDelay(infer.default_actuator, "train"); goal = infer.default_actuator.copy()
    port_r = r0 * rscale
    n = int(round((STAND + WALK) / dt)); k = 0
    stand_lean, walk_lean, vmin, x0, stand_imu, walk_imu, stand_q = [], [], [], None, [], [], []
    t45 = None
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
        vx = KICK[0] if KICK and STAND <= t < STAND + KICK[1] else CMD
        infer.commands = ([vx, 0, 0] if t >= STAND else [0, 0, 0]) + [0, 0, 0, 0]
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
        if t45 is None and t >= STAND and R[2, 2] < np.cos(np.radians(45)):
            t45 = (t - STAND, '앞' if lean(R) > 0 else '뒤')
        if data.qpos[2] < 0.10 or R[2, 2] < 0.5:
            return dict(t45=t45, walk=np.mean(walk_lean) if walk_lean else np.nan, fell=t - STAND, dir="앞" if lean(R) > 0 else "뒤", roll=np.degrees(np.arcsin(np.clip(R[2, 1], -1, 1))),
                        stand=np.mean(stand_lean) if stand_lean else np.nan, stand_imu=np.mean(stand_imu) if stand_imu else np.nan,
                        vmin=min(vmin) if vmin else np.nan)
        if 1.0 <= t < STAND:
            stand_lean.append(lean(R)); stand_imu.append(imu_lean(data))
            stand_q.append(infer.get_measured_joint_angles(data) - infer.default_actuator)
        if t >= STAND:
            walk_lean.append(lean(R)); walk_imu.append(imu_lean(data))
            if x0 is None: x0 = data.qpos[0]
    return dict(t45=t45, fell=None, stand=np.mean(stand_lean), stand_imu=np.mean(stand_imu), walk_imu=np.mean(walk_imu),
                stand_q=np.degrees(np.mean(stand_q, 0)), walk=np.mean(walk_lean), walk_max=np.max(walk_lean),
                vx=(data.qpos[0] - x0) / WALK, vmin=min(vmin) if vmin else np.nan)



import itertools, json as _j
DXS=[float(x) for x in os.environ['DXS'].split(',')]; PLS=[float(x) for x in os.environ['PLS'].split(',')]
NS=int(os.environ.get('NS','3')); RSC=float(os.environ.get('RSC','1.0'))
for dx, ph in itertools.product(DXS, PLS):
    rows=[]
    for seed in range(NS):
        rg=np.random.default_rng(1000+seed)
        smp=dict(dpos=np.array([dx/1000,0,0]), bl=np.radians(rg.uniform(RZ.BACKLASH_TOTAL_MIN_DEG, RZ.BACKLASH_TOTAL_MAX_DEG, len(bl))/2), pl=np.radians(np.full(len(pl),ph)))
        rows.append(run(smp, RSC, seed))
    st=np.nanmean([r['stand'] for r in rows]); wk=np.nanmean([r.get('walk',np.nan) for r in rows])
    t45=[r['t45'] for r in rows]; nf=sum(1 for x in t45 if x is not None); fa=[x[0] for x in t45 if x is not None]; fd=''.join(sorted(set(x[1] for x in t45 if x is not None)))
    vx=[r['vx'] for r in rows if r.get('fell') is None]
    print(_j.dumps(dict(dx=dx,ph=ph,stand=round(float(st),2),walk=round(float(wk),2),fall=f'{nf}/{len(rows)}', tfall=(round(min(fa),1),round(max(fa),1)) if fa else None, fdir=fd,vx=round(float(np.mean(vx)),3) if vx else None),ensure_ascii=False),flush=True)
