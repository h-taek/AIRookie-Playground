"""학습된 정책을 헤드리스로 굴려 "실제로 걷는가"를 숫자로 판정한다.

보상값만으로는 걷는 정책과 제자리에 버티는 정책을 구분할 수 없다 — alive 가 크면
"안 넘어지고 가만히 있기"가 국소 최적이 되는데 보상 곡선은 똑같이 올라간다.
mujoco_infer.MjInfer 의 제어 루프를 그대로 쓰되 viewer 와 실시간 sleep 을 빼고, 학습의 행동 지연
(ActionDelay)을 넣었다. 관측에는 지연 전 목표각, 서보에는 지연된 목표각이 간다. 뷰어에는 지연이 없다.

사용:
    cd <저장소 루트>
    PYTHONPATH="$PWD" uv run python tools/eval_policy_behavior.py -o <정책>.onnx
"""

import argparse

import mujoco
import numpy as np

from playground.open_duck_mini_v2 import constants
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer

DEFAULT_MODEL = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"
DEFAULT_REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

# 명령은 [lin_vel_x, lin_vel_y, ang_vel, neck_pitch, head_pitch, head_yaw, head_roll]
# 명령 축이 셋이므로 여섯 방향을 모두 본다 — 3 차에 좌이동을 빠뜨려 오판한 적이 있다.
# 후진은 5 차부터 명령 범위 밖이라 판정에 넣지 않되 거동은 계속 기록한다.
# 머리 명령값은 joystick.py 가 실제로 뽑는 범위(관절 소프트 한계 0.95 교집합) 안이다 —
# 5 ~ 7 차에는 도달 불가능한 각도를 넣고 낙제를 매기고 있었다.
CASES = [
    ("정지", [0.0, 0.0, 0.0, 0, 0, 0, 0]),
    ("전진 최대", [0.222, 0.0, 0.0, 0, 0, 0, 0]),
    ("전진 절반", [0.111, 0.0, 0.0, 0, 0, 0, 0]),
    ("후진 최대 (5 차 범위 밖)", [-0.111, 0.0, 0.0, 0, 0, 0, 0]),
    ("좌이동 최대", [0.0, 0.148, 0.0, 0, 0, 0, 0]),
    ("우이동 최대", [0.0, -0.148, 0.0, 0, 0, 0, 0]),
    ("제자리 회전", [0.0, 0.0, 1.111, 0, 0, 0, 0]),
    ("전진 + 머리 명령", [0.222, 0.0, 0.0, 0.20, 0.40, 0.50, 0.30]),
    ("회전 + 머리 명령", [0.0, 0.0, 0.8, -0.20, -0.40, -0.50, -0.30]),
]

FALL_HEIGHT = 0.10  # m. 이 아래로 내려가면 무너진 것으로 본다

# 직진 쏠림 표. 정책은 행동 지연 0 ~ 2 스텝 아래에서 배웠고 쏠림이 지연에 따라 달라지므로
# 지연마다 따로 재고, 지연 사이 폭(민감도)을 함께 낸다. 앞 5 초는 기동 과도라 뺀다 (research/04 8장).
DRIFT_CASES = [0.05, 0.111, 0.15, 0.222, -0.05, -0.111]
DRIFT_DELAYS = [0, 1, 2]
DRIFT_SECONDS = 20.0
DRIFT_SETTLE = 5.0


def _geom_ids(model, names):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
    if -1 in ids:
        missing = [n for n, i in zip(names, ids) if i == -1]
        raise RuntimeError(f"geom 이름 조회 실패: {missing}")
    return set(ids)


def rollout(infer: MjInfer, command, seconds: float, delay="train", settle: float = 0.0):
    """명령 하나를 주고 굴린 뒤 거동 지표를 낸다. delay 는 ActionDelay 의 mode 다."""
    model, data = infer.model, infer.data
    mujoco.mj_resetData(model, data)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)

    infer.last_action = np.zeros(infer.num_dofs)
    infer.last_last_action = np.zeros(infer.num_dofs)
    infer.last_last_last_action = np.zeros(infer.num_dofs)
    infer.motor_targets = infer.default_actuator.copy()
    infer.prev_motor_targets = infer.default_actuator.copy()
    infer.imitation_i = 0
    infer.commands = list(command)
    delayed = ActionDelay(infer.default_actuator, delay)
    infer.servo.reset()
    servo_goal = infer.default_actuator.copy()

    left = _geom_ids(model, constants.LEFT_FEET_GEOMS)
    right = _geom_ids(model, constants.RIGHT_FEET_GEOMS)
    floor = _geom_ids(model, ["floor"])
    trunk_id = model.body("trunk_assembly").id
    forcerange = model.actuator_forcerange.copy()
    head_idx = [infer.actuator_names.index(n)
                for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]

    control_dt = infer.sim_dt * infer.decimation
    n_control = int(round(seconds / control_dt))

    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    actuator_qpos_adr = [model.jnt_qposadr[model.joint(n).id] for n in infer.actuator_names]
    actuator_qvel_adr = [model.jnt_dofadr[model.joint(n).id] for n in infer.actuator_names]
    height, up_axis, yaw_rate = [], [], []
    contact, torque, ctrl, tracking_error, joint_qpos = [], [], [], [], []
    joint_qvel = []
    # 출발점에서 도착점까지의 직선 변위만 재면, 원을 그리며 걷는 정책이
    # "걷지 못하는" 것으로 잘못 보인다. 경로 길이와 몸통 기준 전진 속도를
    # 함께 재서 "못 걷는 것"과 "걷지만 휘는 것"을 구분한다.
    path_len = 0.0
    prev_xy = (x0, y0)
    body_vx, body_vy = [], []

    counter = 0
    for _ in range(n_control * infer.decimation):
        # 서보 안의 가속 상한을 물리 스텝마다 흉내 낸다 (학습 joystick.py _servo_step 과 같은 식).
        data.ctrl = infer.servo(servo_goal, infer.sim_dt)
        mujoco.mj_step(model, data)
        counter += 1
        if counter % infer.decimation:
            continue

        infer.imitation_i = (infer.imitation_i + infer.phase_frequency_factor) % infer.PRM.nb_steps_in_period
        phase = infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi
        infer.imitation_phase = np.array([np.cos(phase), np.sin(phase)])

        action = infer.policy.infer(infer.get_obs(data, infer.commands))
        infer.last_last_last_action = infer.last_last_action.copy()
        infer.last_last_action = infer.last_action.copy()
        infer.last_action = action.copy()

        targets = infer.targets_from_action(action)
        # 세 곳(학습 joystick.py · 뷰어 mujoco_infer.py · 여기)이 같은 식을
        # 써야 한다. 다르면 실물에 없는 걸음을 통과시킨다. 18 차부터 속도
        # 클립만 남긴다 — 가속 클립은 근거가 무효라 삭제했다.
        targets = np.clip(
            targets,
            infer.prev_motor_targets - infer.max_motor_velocity * control_dt,
            infer.prev_motor_targets + infer.max_motor_velocity * control_dt,
        )
        # infer.motor_targets 를 갱신하지 않아 관측 83~96 칸(서보 목표각)이 홈에 고정돼
        # 있었다. 학습과 뷰어는 매 스텝 갱신하고 평가 도구만 빠져 있었다.
        targets = np.clip(targets, infer.ctrl_lower, infer.ctrl_upper)
        infer.motor_targets = targets.copy()
        infer.prev_motor_targets = targets.copy()
        # 서보에는 지연된 목표각이 간다. 관측(infer.motor_targets)은 지연 전 값이다.
        servo_goal = delayed(targets)

        height.append(float(data.qpos[2]))
        rot = data.xmat[trunk_id].reshape(3, 3)
        # 몸통 상방축이 세계 +Z 와 이루는 정렬도. +1 정상, 음수면 뒤집힘
        up_axis.append(float(rot[2, 2]))
        yaw_rate.append(float(data.qvel[5]))

        xy = (float(data.qpos[0]), float(data.qpos[1]))
        path_len += float(np.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1]))
        prev_xy = xy
        v_body = rot.T @ data.qvel[:3]
        body_vx.append(float(v_body[0]))
        body_vy.append(float(v_body[1]))

        touching_left = touching_right = False
        for i in range(data.ncon):
            pair = {data.contact[i].geom1, data.contact[i].geom2}
            if pair & floor:
                touching_left |= bool(pair & left)
                touching_right |= bool(pair & right)
        contact.append((touching_left, touching_right))
        torque.append(np.abs(data.actuator_force).copy())
        ctrl.append(data.ctrl.copy())
        joint_qpos.append(data.qpos[actuator_qpos_adr].copy())
        joint_qvel.append(data.qvel[actuator_qvel_adr].copy())
        tracking_error.append(data.ctrl - data.qpos[actuator_qpos_adr])

    height = np.array(height)
    contact = np.array(contact)
    torque = np.array(torque)
    ctrl = np.array(ctrl)
    joint_qpos = np.array(joint_qpos)
    joint_qvel = np.array(joint_qvel)
    tracking_error = np.array(tracking_error)

    travelled = float(np.hypot(data.qpos[0] - x0, data.qpos[1] - y0))
    expected = float(np.hypot(command[0], command[1]) * seconds)
    body_vx = np.array(body_vx)
    body_vy = np.array(body_vy)
    below = np.where(height < FALL_HEIGHT)[0]

    saturation = {infer.actuator_names[i]: float(np.mean(torque[:, i] > 0.95 * forcerange[i, 1]))
                  for i in range(model.nu)}
    # 포화 비율만 보면 부하가 크다고 오해하기 쉽다 — 위치 제어라 도달 불가능한 각도를
    # 요구하기만 해도 부하와 무관하게 토크가 상한에 붙는다. 추종 오차도 함께 낸다.
    kp = float(model.actuator_gainprm[0, 0])
    # 요구/상한. kp x |목표각 - 실제각| 이 forcerange 의 몇 배인가이고, 1.0 을 넘으면
    # 서보가 못 내는 힘을 요구한다는 뜻이다. cost_torques 는 잘린 뒤의 값이라 못 본다.
    kp_per = model.actuator_gainprm[:, 0]
    reach_line = forcerange[:, 1] / kp_per
    err_median = np.median(np.abs(tracking_error), axis=0)
    overdemand = {
        infer.actuator_names[i]: float(err_median[i] / reach_line[i])
        for i in range(model.nu)
    }
    # 머리 4 축 각속도 제곱합. 9 차의 head_vel 항이 직접 줄이려는 값이다.
    # 이동량/방향전환은 목표각의 떨림이고 이쪽은 실제 관절의 움직임이다.
    head_qvel_sq = float(np.mean(np.sum(joint_qvel[:, 5:9] ** 2, axis=1)))
    leg_qvel_sq = float(
        np.mean(np.sum(joint_qvel[:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]] ** 2, axis=1))
    )
    worst = max(range(model.nu), key=lambda i: saturation[infer.actuator_names[i]])
    torque_detail = dict(
        joint=infer.actuator_names[worst],
        median=float(np.median(torque[:, worst])),
        rating=float(forcerange[worst, 1]),
        median_ratio=float(np.median(torque[:, worst]) / forcerange[worst, 1]),
        error_median=float(np.median(np.abs(tracking_error[:, worst]))),
        error_for_rating=float(forcerange[worst, 1] / kp),
    )
    # 머리는 둘을 따로 본다 — 이동량·방향 전환은 떨림이고, 자세 오차는 명령 추종이다.
    head_cmd = dict(zip(("neck_pitch", "head_pitch", "head_yaw", "head_roll"), command[3:]))
    head = {}
    for i in head_idx:
        name = infer.actuator_names[i]
        head[name] = dict(
            travel=float(np.sum(np.abs(np.diff(ctrl[:, i])))),
            reversals=int(np.sum(np.diff(np.sign(np.diff(ctrl[:, i]))) != 0)),
            cmd=float(head_cmd[name]),
            err_mean=float(np.mean(np.abs(joint_qpos[:, i] - head_cmd[name]))),
        )

    return dict(
        seconds=seconds,
        travelled=travelled,
        expected=expected,
        follow_ratio=(travelled / expected) if expected > 1e-9 else None,
        path_len=path_len,
        path_ratio=(path_len / expected) if expected > 1e-9 else None,
        # 몸통 기준 속도. 휘어져도 "앞으로 얼마나 가는가"는 이 값이 답한다
        body_vx_mean=float(body_vx.mean()),
        body_vy_mean=float(body_vy.mean()),
        vx_ratio=(float(body_vx.mean()) / command[0]) if abs(command[0]) > 1e-9 else None,
        vy_ratio=(float(body_vy.mean()) / command[1]) if abs(command[1]) > 1e-9 else None,
        yaw_rate_mean=float(np.mean(yaw_rate)),
        yaw_rate_settled=float(np.mean(yaw_rate[int(round(settle / control_dt)):])),
        yaw_rate_command=command[2],
        height_mean=float(height.mean()),
        height_end=float(height[-1]),
        up_axis_end=float(up_axis[-1]),
        flipped=bool(up_axis[-1] < 0.0),
        fall_time=(float(below[0] * control_dt) if len(below) else None),
        contact_both=float(np.mean(contact.all(axis=1))),
        contact_single=float(np.mean(contact[:, 0] != contact[:, 1])),
        contact_none=float(np.mean(~contact.any(axis=1))),
        step_count=int(np.sum(np.diff(contact[:, 0].astype(int)) != 0)),
        saturation=saturation,
        overdemand=overdemand,
        head_qvel_sq=head_qvel_sq,
        leg_qvel_sq=leg_qvel_sq,
        torque_detail=torque_detail,
        head=head,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="학습된 정책의 실제 거동 측정")
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--reference_data", type=str, default=DEFAULT_REFERENCE)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--delay", default="train",
                        help="행동 지연. train(학습과 같이 매 스텝 0 ~ 2 무작위) 또는 0 / 1 / 2 스텝 고정")
    parser.add_argument("--skip-drift", action="store_true", help="지연별 직진 쏠림 표를 건너뛴다")
    parser.add_argument("--head-from-command", action="store_true",
                        help="머리 목표각을 머리 명령으로 준다(v2_12 부터의 학습 조건)")
    args = parser.parse_args()
    delay = args.delay if args.delay == "train" else int(args.delay)

    infer = MjInfer(
        model_path=args.model_path,
        reference_data=args.reference_data,
        onnx_model_path=args.onnx_model_path,
        standing=False,
        head_from_command=args.head_from_command,
    )
    home_height = float(infer.model.keyframe("home").qpos[2])
    print(f"\n정책: {args.onnx_model_path}")
    print(f"홈 자세 몸통 높이: {home_height:.4f} m")
    print(f"행동 지연: {delay}" + (" (기본 지연은 시드 0 으로 뽑는다)" if delay == "train" else ""))

    for name, command in CASES:
        r = rollout(infer, command, args.seconds, delay)
        ratio = "-" if r["follow_ratio"] is None else f"{r['follow_ratio'] * 100:.1f} %"
        worst = max(r["saturation"].items(), key=lambda kv: kv[1])
        print(f"\n===== {name}  cmd=({command[0]:+.3f}, {command[1]:+.3f}, {command[2]:+.2f}) "
              f"— {r['seconds']:.0f} 초 =====")
        p_ratio = "-" if r["path_ratio"] is None else f"{r['path_ratio'] * 100:.1f} %"
        vx_r = "-" if r["vx_ratio"] is None else f"{r['vx_ratio'] * 100:.1f} %"
        vy_r = "-" if r["vy_ratio"] is None else f"{r['vy_ratio'] * 100:.1f} %"
        print(f"  직선 변위    {r['travelled']:.4f} m / 기대 {r['expected']:.4f} m  (추종률 {ratio})")
        print(f"  경로 길이    {r['path_len']:.4f} m  (경로 기준 {p_ratio})  ← 직선보다 크면 휘어서 걷는 것")
        print(f"  몸통 기준 속도  전진 {r['body_vx_mean']:+.4f} m/s (명령 {command[0]:+.3f}, 추종 {vx_r})"
              f" / 좌우 {r['body_vy_mean']:+.4f} m/s (명령 {command[1]:+.3f}, 추종 {vy_r})")
        print(f"  yaw 각속도   {r['yaw_rate_mean']:+.4f} rad/s / 명령 {r['yaw_rate_command']:+.2f}")
        print(f"  몸통 높이    평균 {r['height_mean']:.4f} m, 끝 {r['height_end']:.4f} m (홈 {home_height:.4f})")
        print(f"  상방축 끝값  {r['up_axis_end']:+.3f}  뒤집힘 {r['flipped']}"
              + (f", {r['fall_time']:.2f} 초에 무너짐" if r["fall_time"] is not None else ""))
        print(f"  접촉         양발 {r['contact_both'] * 100:.1f} % / 한발 {r['contact_single'] * 100:.1f} %"
              f" / 공중 {r['contact_none'] * 100:.1f} %")
        print(f"  좌발 접촉 전환 {r['step_count']} 회  (보행이면 10 초에 수십 회)")
        t = r["torque_detail"]
        print(f"  토크 포화 최대 {worst[0]} {worst[1] * 100:.1f} %")
        print(f"    └ {t['joint']} 토크 중앙값 {t['median']:.3f} N.m (정격 {t['rating']:.3f} 의 {t['median_ratio'] * 100:.0f} %)")
        print(f"      추종 오차 중앙값 {t['error_median']:.4f} rad / 정격 도달선 {t['error_for_rating']:.4f} rad")
        print(f"      도달선 = forcerange / kp. 오차가 이 선을 넘은 만큼 요구 토크가 잘린다")
        od = sorted(r["overdemand"].items(), key=lambda kv: -kv[1])[:4]
        print("  요구/상한 상위  " + "  ".join(f"{k} {v:.2f}" for k, v in od)
              + "   ← 1.0 초과 = 낼 수 없는 힘을 요구")
        print(f"  관절 각속도²   머리 4 축 {r['head_qvel_sq']:.2f} / 다리 10 축 {r['leg_qvel_sq']:.2f}")
        for joint, h in r["head"].items():
            print(f"  머리 {joint:12s} 이동량 {h['travel']:6.2f} rad, 방향전환 {h['reversals']:4d} 회"
                  f"  | 명령 {h['cmd']:+.2f} rad, 자세 오차 평균 {h['err_mean']:.3f} rad")

    if args.skip_drift:
        return
    print(f"\n===== 직진 쏠림 — 요 각속도 (rad/s), {DRIFT_SETTLE:.0f} ~ {DRIFT_SECONDS:.0f} 초 평균 =====")
    print("  전진·후진 명령, 회전 명령 0. 열은 고정 행동 지연(스텝), 폭은 지연 사이 최대 - 최소")
    print("  " + f"{'명령 (m/s)':>12s}" + "".join(f"{('지연 ' + str(d)):>10s}" for d in DRIFT_DELAYS)
          + f"{'폭':>10s}{'폭 (°/10 s)':>14s}")
    for vx in DRIFT_CASES:
        ys = [rollout(infer, [vx, 0.0, 0.0, 0, 0, 0, 0], DRIFT_SECONDS, d, DRIFT_SETTLE)["yaw_rate_settled"]
              for d in DRIFT_DELAYS]
        spread = max(ys) - min(ys)
        print("  " + f"{vx:>+12.3f}" + "".join(f"{y:>+10.4f}" for y in ys)
              + f"{spread:>10.4f}{np.degrees(spread) * 10:>14.1f}")


if __name__ == "__main__":
    main()
