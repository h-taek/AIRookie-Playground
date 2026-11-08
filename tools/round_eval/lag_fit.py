# 외장 유격 관절(`*_play`)이 있던 모델을 전제로 한다. 지금 모델에는 외장 유격이 없으므로 다시 쓸 때 고친다 (2026-10-21).
"""v2_15 에 서보 가속 흉내 · 힘 지연을 넣고 관절 지연과 넘어짐을 잰다. 실물 지연과 같은 계산법.

사용 (Playground 루트): ACCS=0,38.8 TAUS=0,20,40 RSCS=0,1 NS=8 [DX=10 PL=1.0] python tools/round_eval/lag_fit.py <onnx> <짝 home_pose.json>
요약: python tools/round_eval/lag_sum.py lag_fit.jsonl (실물 지연은 Runtime research/10 24-5)
지연 d: mean|목표[k-1-d] - 관절각[k]| 를 가장 작게 하는 d (실물 기록과 같은 정의). 걷기 1 s 뒤부터, 넘어지기 1 s 전까지.
힘 지연 T: 서보 힘이 kp × (내부 기준 − 관절각) 을 시간 상수 T 의 1 차 지연으로 따라간다.
"""
import sys, os, json, itertools
sys.path.insert(0, ".")
import numpy as np
import mujoco
from multiprocessing import Pool

POLICY, PAIR = sys.argv[1], json.load(open(sys.argv[2]))
STAND, WALK = 3.0, 20.0
CMD = float(os.environ.get('CMD', '0.05'))
DX, PL = float(os.environ.get('DX', '10')), float(os.environ.get('PL', '1.0'))
LEG = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
       'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']


def run(args):
    acc, tau, rsc, seed = args
    from playground.open_duck_mini_v2.joystick import default_config
    from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer, ServoProfile
    from tools.eval_policy_behavior import DEFAULT_MODEL, DEFAULT_REFERENCE
    from playground.common import randomize as RZ
    infer = MjInfer(DEFAULT_MODEL, DEFAULT_REFERENCE, POLICY, standing=False, head_from_command=True)
    model, data = infer.model, infer.data
    dt = infer.sim_dt * infer.decimation; sdt = infer.sim_dt
    vd = default_config().voltage_drop
    trunk = model.body("trunk_assembly").id
    for n, v in PAIR["joints"].items():
        model.key_qpos[0, model.jnt_qposadr[model.joint(n).id]] = v
        model.key_ctrl[0, model.actuator(n).id] = v
        infer.default_actuator[model.actuator(n).id] = v
    bl = [j for j in range(model.njnt) if model.joint(j).name.endswith("_backlash")]
    pl = [j for j in range(model.njnt) if model.joint(j).name.endswith("_play")]
    port_b = ("right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
    port = np.array([int(model.actuator(i).name in port_b) for i in range(model.nu)])
    act_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    act_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    damp, arm = model.dof_damping[act_dof], model.dof_armature[act_dof]
    kp = model.actuator_gainprm[:, 0].copy(); fr = model.actuator_forcerange[:, 1].copy()
    r0 = np.stack([np.array(vd.servo_r_a), np.array(vd.servo_r_b)]) * rsc
    leg_i = [model.actuator(n).id for n in LEG]

    rg = np.random.default_rng(1000 + seed); np.random.seed(seed)
    model.body_ipos[trunk] = model.body_ipos[trunk] + np.array([DX / 1000, 0, 0])
    for j, h in zip(bl, np.radians(rg.uniform(RZ.BACKLASH_TOTAL_MIN_DEG, RZ.BACKLASH_TOTAL_MAX_DEG, len(bl)) / 2)):
        model.jnt_range[j] = [-h, h]
    for j in pl: model.jnt_range[j] = [-np.radians(PL), np.radians(PL)]
    mujoco.mj_resetData(model, data); mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    for a in ("last_action", "last_last_action", "last_last_last_action"):
        setattr(infer, a, np.zeros(infer.num_dofs))
    infer.motor_targets = infer.default_actuator.copy(); infer.prev_motor_targets = infer.default_actuator.copy()
    infer.imitation_i = 0
    delayed = ActionDelay(infer.default_actuator, "train", seed=seed); goal = infer.default_actuator.copy()
    servo = ServoProfile(infer.default_actuator, accel=acc)
    force = np.zeros(model.nu)
    n = int(round((STAND + WALK) / dt)); k = 0
    TG, Q, T = [], [], []
    fell = None
    for _ in range(n * infer.decimation):
        ref = servo(goal, sdt)
        q = data.qpos[act_qpos]
        want = np.clip(kp * (ref - q), -fr, fr)
        if rsc > 0:
            ext = data.actuator_force - damp * data.qvel[act_dof] - arm * data.qacc[act_dof]
            cur = vd.cur_gain * np.maximum(np.abs(ext) - vd.cur_hold, 0.0)
            pc = np.zeros(2); np.add.at(pc, port, cur)
            scale = np.clip((vd.v_nominal - pc @ r0) / vd.v_nominal, vd.min_scale, 1.0)
            want = np.clip(want * scale, -fr * scale, fr * scale)
        force = want if tau <= 0 else force + (want - force) * min(1.0, sdt / (tau / 1000))
        data.ctrl = q + force / kp
        mujoco.mj_step(model, data); k += 1
        if k % infer.decimation: continue
        t = k * sdt
        infer.commands = ([CMD, 0, 0] if t >= STAND else [0, 0, 0]) + [0, 0, 0, 0]
        infer.imitation_i = (infer.imitation_i + infer.phase_frequency_factor) % infer.PRM.nb_steps_in_period
        ph = infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi
        infer.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
        obs = infer.get_obs(data, infer.commands)
        Q.append(infer.get_measured_joint_angles(data)[leg_i].copy()); T.append(t)
        act = infer.policy.infer(obs)
        infer.last_last_last_action = infer.last_last_action.copy(); infer.last_last_action = infer.last_action.copy()
        infer.last_action = act.copy()
        tg = infer.targets_from_action(act)
        tg = np.clip(tg, infer.prev_motor_targets - infer.max_motor_velocity * dt, infer.prev_motor_targets + infer.max_motor_velocity * dt)
        tg = np.clip(tg, infer.ctrl_lower, infer.ctrl_upper)
        infer.motor_targets = tg.copy(); infer.prev_motor_targets = tg.copy(); goal = delayed(tg)
        TG.append(tg[leg_i].copy())
        R = np.zeros(9); mujoco.mju_quat2Mat(R, data.qpos[3:7]); R = R.reshape(3, 3)
        if t >= STAND and R[2, 2] < np.cos(np.radians(45)):
            pitch = np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))
            fell = (round(t - STAND, 2), '앞' if pitch > 0 else '뒤'); break
    TG, Q, T = np.array(TG), np.array(Q), np.array(T)
    end = (fell[0] + STAND - 1.0) if fell else T[-1] + 1
    use = (T >= STAND + 1.0) & (T < end)
    sums = np.zeros((21, len(LEG))); cnt = 0
    nT = len(T)
    for d in range(21):
        e = np.full((nT, len(LEG)), np.nan)
        e[1 + d:] = TG[:nT - 1 - d] - Q[1 + d:]
        m = use & ~np.isnan(e[:, 0])
        sums[d] = np.abs(e[m]).sum(0)
        if d == 0: cnt = m.sum()
    return dict(acc=acc, tau=tau, rsc=rsc, seed=seed, fell=fell, sums=sums.tolist(), cnt=int(cnt))


if __name__ == '__main__':
    ACCS = [float(x) for x in os.environ['ACCS'].split(',')]
    TAUS = [float(x) for x in os.environ['TAUS'].split(',')]
    RSCS = [float(x) for x in os.environ['RSCS'].split(',')]
    NS = int(os.environ.get('NS', '8'))
    jobs = list(itertools.product(ACCS, TAUS, RSCS, range(NS)))
    out = os.environ.get('OUT', 'lag_fit.jsonl')
    with Pool(int(os.environ.get('NP', '32'))) as p, open(out, 'w') as f:
        for r in p.imap_unordered(run, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
    print('ALL_DONE_FILE', out)
