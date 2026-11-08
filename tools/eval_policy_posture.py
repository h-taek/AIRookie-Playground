"""정책의 자세와 보상 항 값을 잰다. eval_policy_behavior.py 가 안 보는 것들이다.

`eval_policy_behavior.py` 는 "걷는가" 를 판정한다. 이 도구는 "어떤 자세로 걷는가" 와
"새 보상 항을 얼마로 둘 것인가" 를 판정한다. 22 ~ 24 차 판정 기준이 전부 여기서 나온다.

내는 값:
  1. 몸통 좌표계 피치·롤   — 양수 피치가 앞으로 숙임. 부호는 arcsin(-fwd_z) 규약.
                             무게중심 쏠림은 두 발 중앙 기준이라 걸어가도 유효하다
  2. 좌우 관절 대칭성      — 미러 규약(hip 3 축 부호 반대, knee/ankle 동부호) 기준
  3. 발 체공과 미끄러짐    — 최고 체공 높이, 평균 체공 시간, 착지 횟수
  4. 보상 항 실측값        — orientation / feet_slip / feet_air_time / head_pos / pose

가중치를 정할 때는 이 값에 후보 가중치를 곱해 같은 조건의 action_rate 실측 벌점
(전진에서 -0.17 점/스텝)과 같은 급인지 본다. 짐작으로 넣으면 회차를 통째로 날린다.

사용:
    cd <저장소 루트>
    PYTHONPATH="$PWD" uv run python tools/eval_policy_posture.py -o <정책>.onnx
"""

import argparse
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from playground.open_duck_mini_v2 import constants  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer  # noqa: E402

DEFAULT_MODEL = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"
DEFAULT_REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

# 미러 규약. MJCF 의 jnt_range 로 확인했다 — hip 3 축은 좌우 부호가 반대이고
# knee/ankle 은 같은 부호다. 대칭이면 right == MIRROR[j] * left 다.
MIRROR = {"hip_yaw": -1, "hip_roll": -1, "hip_pitch": -1, "knee": +1, "ankle": +1}

# reward_feet_air_time 의 상수. rewards.py 와 같아야 한다.
AIR_MIN, AIR_MAX = 0.1, 0.5

CASES = [
    ("정지", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
    ("전진 최대", [0.222, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
    ("전진+머리명령", [0.222, 0.0, 0.0], [0.20, 0.40, 0.50, 0.30]),
    ("제자리 회전", [0.0, 0.0, 1.111], [0.0, 0.0, 0.0, 0.0]),
    ("좌이동", [0.0, 0.148, 0.0], [0.0, 0.0, 0.0, 0.0]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="정책의 자세와 보상 항 값 측정")
    ap.add_argument("-o", "--onnx_model_path", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--reference_data", type=str, default=DEFAULT_REFERENCE)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--delay", default="train",
                    help="행동 지연. train(학습과 같이 매 스텝 0 ~ 2 무작위) 또는 0 / 1 / 2 스텝 고정")
    args = ap.parse_args()
    delay = args.delay if args.delay == "train" else int(args.delay)

    infer = MjInfer(args.model_path, args.reference_data, args.onnx_model_path, False, False)
    m = infer.model
    floor = m.geom("floor").id
    left_geoms = {m.geom(n).id for n in constants.LEFT_FEET_GEOMS}
    right_geoms = {m.geom(n).id for n in constants.RIGHT_FEET_GEOMS}
    feet_site = [m.site(n).id for n in constants.FEET_SITES]
    imu = m.site("imu").id
    # 구동 관절의 qpos 주소. backlash 씬에는 여분 관절이 있어 qpos[7:7+nu] 로는 안 된다.
    act_adr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    # cost_feet_slip 이 실제로 받는 값이다. 유한차분으로 재면 30 % 어긋난다.
    linvel_adr = np.array(
        [
            list(range(m.sensor_adr[m.sensor(f"{s}_global_linvel").id],
                       m.sensor_adr[m.sensor(f"{s}_global_linvel").id]
                       + m.sensor_dim[m.sensor(f"{s}_global_linvel").id]))
            for s in constants.FEET_SITES
        ]
    )
    # cost_pose 의 가중치는 joystick._weights 와 같아야 한다(다리 10 축).
    pose_w = np.array([1.0, 1.0, 0.01, 0.01, 1.0, 1.0, 1.0, 0.01, 0.01, 1.0])
    home = np.array(m.keyframe("home").ctrl)
    pose_ref = np.concatenate([home[:5], home[9:]])

    ctrl_dt = 0.02
    n_sub = int(round(ctrl_dt / float(m.opt.timestep)))

    def touching(d):
        left = right = False
        for i in range(d.ncon):
            pair = {d.contact[i].geom1, d.contact[i].geom2}
            if floor in pair:
                left |= bool(pair & left_geoms)
                right |= bool(pair & right_geoms)
        return np.array([left, right])

    def rollout(move_cmd, head_cmd):
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("home").qpos
        infer.data = d
        infer.motor_targets = infer.default_actuator.copy()
        infer.prev_motor_targets = infer.default_actuator.copy()
        infer.last_action = np.zeros(infer.num_dofs)
        infer.last_last_action = np.zeros(infer.num_dofs)
        infer.last_last_last_action = np.zeros(infer.num_dofs)
        infer.commands = np.array(move_cmd + head_cmd, dtype=float)
        infer.imitation_i = 0
        delayed = ActionDelay(infer.default_actuator, delay)
        infer.servo.reset()
        cmd_norm = float(np.linalg.norm(np.array(move_cmd)))
        prev = infer.default_actuator.copy()
        air = np.zeros(2)
        peak = np.zeros(2)
        con_prev = np.zeros(2, dtype=bool)
        acc = {k: [] for k in ("ori", "slip", "air", "head", "pose")}
        peaks, pitches, rolls, com_y = [], [], [], []
        joints, airs = [], []
        nb = infer.PRM.nb_steps_in_period
        for k in range(int(args.seconds / ctrl_dt)):
            obs = infer.get_obs(d, infer.commands)
            action = infer.policy.infer(obs)
            infer.last_last_last_action = infer.last_last_action.copy()
            infer.last_last_action = infer.last_action.copy()
            infer.last_action = action.copy()
            target = action * infer.action_scale + infer.default_actuator
            target = prev + np.clip(
                (target - prev) / ctrl_dt,
                -infer.max_motor_velocity,
                infer.max_motor_velocity,
            ) * ctrl_dt
            target = np.clip(target, infer.ctrl_lower, infer.ctrl_upper)
            # 관측 83 ~ 96 칸(서보 목표각)이다. 갱신하지 않으면 홈에 고정된다(eval_policy_behavior 와 같은 결함).
            infer.motor_targets = target.copy()
            # 서보에는 지연된 목표각이 간다(학습과 같다). 관측은 지연 전 값이다.
            # 서보 안의 가속 상한을 물리 스텝마다 흉내 낸다 (학습 joystick.py _servo_step 과 같은 식).
            goal = delayed(target)
            for _ in range(n_sub):
                d.ctrl = infer.servo(goal, m.opt.timestep)
                mujoco.mj_step(m, d)
            infer.imitation_i = (infer.imitation_i + 1) % nb
            infer.imitation_phase = np.array(
                [np.cos(infer.imitation_i / nb * 2 * np.pi),
                 np.sin(infer.imitation_i / nb * 2 * np.pi)]
            )
            con = touching(d)
            first = con & (~con_prev)
            q = d.qpos[act_adr]
            # 초반 1 초는 기동 과도라 버린다.
            if k > 50:
                rot_imu = d.site_xmat[imu].reshape(3, 3)
                g = rot_imu.T @ np.array([0.0, 0.0, -1.0])
                rot = d.xmat[1].reshape(3, 3)
                acc["ori"].append(float(np.sum(g[:2] ** 2)))
                fv = d.sensordata[linvel_adr]
                # rewards.py 와 같게 미끄러짐은 수평 속도만, 짧은 체공은 0 점이다.
                acc["slip"].append(float(np.sum(np.linalg.norm(fv[..., :2], axis=-1) * con)))
                t = np.clip(air - AIR_MIN, 0.0, AIR_MAX - AIR_MIN)
                acc["air"].append(float(np.sum(t * first)) * (cmd_norm > 0.01))
                acc["head"].append(float(np.sum(np.square(q[5:9] - np.array(head_cmd)))))
                leg = np.concatenate([q[:5], q[9:]])
                acc["pose"].append(float(np.sum(np.square(leg - pose_ref) * pose_w)))
                peaks.append(float(np.max(peak)))
                # 몸통 자세는 표본 평균을 쓴다. 마지막 한 프레임은 보행 위상에 따라 흔들린다.
                fwd_k = rot @ np.array([1.0, 0.0, 0.0])
                lft_k = rot @ np.array([0.0, 1.0, 0.0])
                pitches.append(np.degrees(np.arcsin(-fwd_k[2])))
                rolls.append(np.degrees(np.arcsin(lft_k[2])))
                # 무게중심 좌우 쏠림은 두 발 중앙 기준이다. 월드 y 는 걸어가면 무의미해진다.
                mujoco.mj_comPos(m, d)
                mid = float(np.mean(d.site_xpos[feet_site][:, 1]))
                com_y.append((float(d.subtree_com[0][1]) - mid) * 1000.0)
                joints.append(q.copy())
                for i in range(2):
                    if first[i] and air[i] > 0:
                        airs.append(float(air[i]))
            air = np.where(con, 0.0, air + ctrl_dt)
            peak = np.where(con, 0.0, np.maximum(peak, d.site_xpos[feet_site][:, 2]))
            con_prev = con
            prev = target.copy()

        return {
            "pitch": float(np.mean(pitches)),
            "roll": float(np.mean(rolls)),
            "com_y": float(np.mean(com_y)),
            "peak": float(np.max(peaks)) if peaks else 0.0,
            "q": np.degrees(np.mean(joints, axis=0)),
            "air_mean": float(np.mean(airs)) if airs else 0.0,
            "landings": len(airs),
            **{k: float(np.mean(v)) for k, v in acc.items()},
        }

    print(f"\n정책 {args.onnx_model_path}")
    print(f"모델 {args.model_path}   시나리오당 {args.seconds:.0f} 초\n")

    results = {}
    for label, move, head in CASES:
        results[label] = rollout(move, head)

    print("=== 1. 몸통 자세 (양수 피치 = 앞으로 숙임) ===")
    print(f"{'명령':16s} {'피치':>8s} {'롤':>8s} {'쏠림 y':>11s}")
    for label in results:
        r = results[label]
        print(f"{label:16s} {r['pitch']:+8.2f} {r['roll']:+8.2f} {r['com_y']:+10.1f} mm")

    print("\n=== 2. 좌우 대칭 (정지) ===")
    q = results["정지"]["q"]
    print(f"{'관절':12s} {'좌':>8s} {'우':>8s} {'대칭이면 우':>11s} {'어긋남':>8s}")
    total = []
    for joint, sign in MIRROR.items():
        left = q[names.index(f"left_{joint}")]
        right = q[names.index(f"right_{joint}")]
        print(f"{joint:12s} {left:+8.2f} {right:+8.2f} {sign*left:+11.2f} {right - sign*left:+8.2f}")
        total.append(abs(right - sign * left))
    print(f"어긋남 합 {sum(total):.2f} 도, 최대 {max(total):.2f} 도")

    print("\n=== 3. 발 체공과 미끄러짐 ===")
    print(f"{'명령':16s} {'최고 체공':>10s} {'평균 체공':>10s} {'착지':>6s}")
    for label in results:
        r = results[label]
        print(f"{label:16s} {r['peak']*1000:9.1f} mm {r['air_mean']:9.3f}s {r['landings']:6d}")

    print("\n=== 4. 보상 항 실측값 (가중치를 곱해 action_rate -0.17 급인지 본다) ===")
    print(f"{'명령':16s} {'orientation':>12s} {'feet_slip':>10s} {'air_time':>10s}"
          f" {'head_pos':>9s} {'pose':>9s}")
    for label in results:
        r = results[label]
        print(f"{label:16s} {r['ori']:12.5f} {r['slip']:10.4f} {r['air']:10.5f}"
              f" {r['head']:9.4f} {r['pose']:9.5f}")


if __name__ == "__main__":
    main()
