# 외장 유격 관절(`*_play`)이 있던 모델을 전제로 한다. 지금 모델에는 외장 유격이 없으므로 다시 쓸 때 고친다 (2026-10-21).
"""실물 기록의 지령을 정책 없이 시뮬에 그대로 넣는다.

사용: DXS=0,10,20 PLS=1.0 NS=8 [RSC=1.0] \
      python tools/round_eval/replay_real.py <log.npz> <짝 home_pose.json> [끝 초]
DXS 는 몸통 무게중심 x 를 CAD 기준 mm 로 준다(학습 XML 의 값과 무관). PLS 는 외장 유격 반각(°),
RSC 는 전압 강하 배율(0 = 없음)이다. 서보 유격은 표본마다 학습 범위에서 뽑는다.
시작 관절각은 기록 첫 주기의 관측(관절각 − 짝 홈 자세)에 짝 홈 자세를 더한 값이다. 몸통은 서 있는 자세 ·
속도 0 에서 시작한다. 지령은 기록 `target` 을 20 ms 마다 학습과 같은 지령 지연(`ActionDelay`)으로 넣는다.
실물 기울기는 가속도 기울기(`real_walk.py` `tilt_deg`)의 0.2 초 평균이고, 45° 를 처음 넘은 시각을 넘어짐으로 본다.
"""
import itertools
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from playground.common import randomize as RZ
from playground.open_duck_mini_v2.joystick import default_config
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay
from tools.eval_policy_behavior import DEFAULT_MODEL

CAD_X = -0.0471231  # 몸통 무게중심 x, CAD 값
LOG = np.load(sys.argv[1], allow_pickle=True)
PAIR = json.load(open(sys.argv[2]))
names = [str(n) for n in LOG["names"]]
home = np.array([PAIR["joints"][n] for n in names])
angle = LOG["obs"][:, 13:27] + home
target = LOG["target"]
acc = LOG["accel"]
tilt = np.degrees(np.arccos(np.clip(acc[:, 2] / np.linalg.norm(acc, axis=1), -1, 1)))
lean_real = np.degrees(np.arctan2(-acc[:, 0], acc[:, 2])) + 0.74  # 몸통 앞숙임 (장착 각도 보정)
k5 = np.convolve(tilt, np.ones(10) / 10, "same")
over = np.flatnonzero(k5 > 45)
k_fall = int(over[0]) if over.size else None
END = float(sys.argv[3]) if len(sys.argv) > 3 else (LOG["t"][k_fall] + 0.5 if k_fall is not None else LOG["t"][-1])
n_ctrl = min(len(target), int(round(END / 0.02)))

model = mujoco.MjModel.from_xml_path(DEFAULT_MODEL)
data = mujoco.MjData(model)
sim_dt = model.opt.timestep
dec = int(round(0.02 / sim_dt))
vd = default_config().voltage_drop
trunk = model.body("trunk_assembly").id
ipos0 = model.body_ipos[trunk].copy()
act = [model.actuator(n).id for n in names]
act_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
act_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
damp, arm = model.dof_damping[act_dof], model.dof_armature[act_dof]
reach_line = model.actuator_forcerange[:, 1] / model.actuator_gainprm[:, 0]
port_b = ("right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
port = np.array([int(model.actuator(i).name in port_b) for i in range(model.nu)])
r0 = np.stack([np.array(vd.servo_r_a), np.array(vd.servo_r_b)])
bl = [j for j in range(model.njnt) if model.joint(j).name.endswith("_backlash")]
pl = [j for j in range(model.njnt) if model.joint(j).name.endswith("_play")]
key = model.keyframe("home").id
SETTLE = 1.0


def to_ctrl(v):
    out = model.key_ctrl[key].copy()
    out[act] = v
    return out


def lean(R):
    return np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))


def run(dx, ph, seed):
    rg = np.random.default_rng(1000 + seed)
    model.body_ipos[trunk] = ipos0
    model.body_ipos[trunk][0] = CAD_X + dx / 1000
    for j, h in zip(bl, np.radians(rg.uniform(RZ.BACKLASH_TOTAL_MIN_DEG, RZ.BACKLASH_TOTAL_MAX_DEG, len(bl)) / 2)):
        model.jnt_range[j] = [-h, h]
    for j in pl:
        model.jnt_range[j] = np.radians([-ph, ph])
    mujoco.mj_resetDataKeyframe(model, data, key)
    for i, n in enumerate(names):
        data.qpos[model.jnt_qposadr[model.joint(n).id]] = angle[0, i]
    data.qpos[2] += 0.005  # 홈 자세가 키프레임과 달라 살짝 띄워 내려놓는다
    c0 = to_ctrl(angle[0])
    data.ctrl = c0
    for _ in range(int(SETTLE / sim_dt)):  # 첫 관절각을 잡고 선다
        mujoco.mj_step(model, data)
    delayed = ActionDelay(c0, "train")
    goal = c0.copy()
    data.ctrl = goal
    rows, t45, err = [], None, []
    for k in range(n_ctrl):
        goal = delayed(to_ctrl(target[k]))
        for _ in range(dec):
            ctrl = goal
            if RSC > 0:
                ext = data.actuator_force - damp * data.qvel[act_dof] - arm * data.qacc[act_dof]
                cur = vd.cur_gain * np.maximum(np.abs(ext) - vd.cur_hold, 0.0)
                pc = np.zeros(2)
                np.add.at(pc, port, cur)
                volt = vd.v_nominal - pc @ (r0 * RSC)
                scale = np.clip(volt / vd.v_nominal, vd.min_scale, 1.0)
                q = data.qpos[act_qpos]
                ctrl = q + np.clip((goal - q) * scale, -reach_line * scale, reach_line * scale)
            data.ctrl = ctrl
            mujoco.mj_step(model, data)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[3:7])
        R = R.reshape(3, 3)
        lt = lean(R)
        rows.append(lt)
        if k + 1 < len(angle):
            err.append(data.qpos[act_qpos][act] - angle[k + 1])
        if t45 is None and R[2, 2] < np.cos(np.radians(45)):
            t45 = ((k + 1) * 0.02, "앞" if lt > 0 else "뒤")
            break
    return np.array(rows), t45, np.degrees(np.array(err))


DXS = [float(x) for x in os.environ.get("DXS", "0,10,20").split(",")]
PLS = [float(x) for x in os.environ.get("PLS", "1.0").split(",")]
NS = int(os.environ.get("NS", "8"))
RSC = float(os.environ.get("RSC", "1.0"))
real_fall = (round(float(LOG["t"][k_fall]), 2), "앞" if lean_real[k_fall] > 0 else "뒤") if k_fall is not None else None
print(json.dumps(dict(log=os.path.basename(sys.argv[1]), real_fall=real_fall, end=round(END, 2),
                     real_lean_1s=[round(float(lean_real[i:i + 50].mean()), 1) for i in range(0, n_ctrl, 50)]),
                 ensure_ascii=False), flush=True)
for dx, ph in itertools.product(DXS, PLS):
    res = [run(dx, ph, s) for s in range(NS)]
    falls = [r[1] for r in res if r[1] is not None]
    L = min(len(r[0]) for r in res)
    mean_lean = np.mean([r[0][:L] for r in res], 0)
    e = np.concatenate([np.abs(r[2]) for r in res])
    print(json.dumps(dict(dx=dx, ph=ph, fall=f"{len(falls)}/{NS}",
                          tfall=(round(min(f[0] for f in falls), 2), round(max(f[0] for f in falls), 2)) if falls else None,
                          fdir="".join(sorted(set(f[1] for f in falls))),
                          sim_lean_1s=[round(float(mean_lean[i:i + 50].mean()), 1) for i in range(0, L, 50)],
                          joint_err_deg={n: round(float(v), 1) for n, v in zip(names, e.mean(0)) if "head" not in n and "neck" not in n}),
                     ensure_ascii=False), flush=True)
