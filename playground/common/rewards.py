"""
Set of commonly used rewards
For examples on how to use some rewards, look at https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py
"""

import jax
import jax.numpy as jp


# Tracking rewards.
def reward_tracking_lin_vel(
    commands: jax.Array,
    local_vel: jax.Array,
    tracking_sigma: float,
    low_speed_ref: float | None = None,
) -> jax.Array:
    # low_speed_ref 를 주면 전진 성분의 폭을 명령 크기에 비례해 좁힌다: 폭 = sigma × clip(|vx 명령| / ref, 0.2, 1)².
    # ref 아래의 어느 명령이든 서 있으면 ref 에서와 같은 점수를 받는다. 정지 명령과 좌우 성분은 sigma 그대로다.
    sigma_x = tracking_sigma
    if low_speed_ref is not None:
        r = jp.clip(jp.abs(commands[0]) / low_speed_ref, 0.2, 1.0)
        sigma_x = jp.where(jp.abs(commands[0]) > 0, tracking_sigma * r**2, tracking_sigma)
    # 좌우 이동 허용 오차. 0 이다 — 부르는 쪽(joystick.py)이 좌우 속도를 한 걸음 평균으로
    # 넘기므로 걸음마다의 옆 흔들림을 따로 봐줄 필요가 없다. 0.03 은 작은 옆 명령을 서 있어도
    # 0.96 점으로 만들었다 (research/04 4장).
    y_tol = 0.0
    error_x = jp.square(commands[0] - local_vel[0])
    error_y = jp.clip(jp.abs(local_vel[1] - commands[1]) - y_tol, 0.0, None)
    lin_vel_error = error_x / sigma_x + jp.square(error_y) / tracking_sigma
    return jp.nan_to_num(jp.exp(-lin_vel_error))


def reward_tracking_ang_vel(
    commands: jax.Array,
    ang_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.nan_to_num(jp.exp(-ang_vel_error / tracking_sigma))


# Base-related rewards.


def cost_lin_vel_z(global_linvel) -> jax.Array:
    return jp.nan_to_num(jp.square(global_linvel[2]))


def cost_ang_vel_xy(global_angvel) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(global_angvel[:2])))


def cost_ang_vel_x(gyro) -> jax.Array:
    # 몸통 좌우 기울기 축(자이로 x) 각속도의 제곱. 시범 동작은 이 값이 0 이다(research/04 52-5).
    return jp.nan_to_num(jp.square(gyro[0]))


def cost_orientation(torso_zaxis: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torso_zaxis[:2])))


def cost_base_height(base_height: jax.Array, base_height_target: float) -> jax.Array:
    return jp.nan_to_num(jp.square(base_height - base_height_target))


def reward_base_y_swing(
    base_y_speed: jax.Array,
    freq: float,
    amplitude: float,
    t: float,
    tracking_sigma: float,
) -> jax.Array:
    target_y_speed = amplitude * jp.sin(2 * jp.pi * freq * t)
    y_speed_error = jp.square(target_y_speed - base_y_speed)
    return jp.nan_to_num(jp.exp(-y_speed_error / tracking_sigma))


# Energy related rewards.


def cost_torques(torques: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torques)))
    # return jp.nan_to_num(jp.sum(jp.abs(torques)))


def cost_torques_abs(torques: jax.Array) -> jax.Array:
    """토크 절댓값 합. 서보 전류는 토크에 비례하고, 전압 강하는 전선 저항 × 전류 합이라 선형으로 본다."""
    return jp.nan_to_num(jp.sum(jp.abs(torques)))


def cost_energy(qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)))


def cost_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    c1 = jp.nan_to_num(jp.sum(jp.square(act - last_act)))
    return c1


def cost_action_acc(act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(act - 2.0 * last_act + last_last_act)))


# Other rewards.


def cost_joint_pos_limits(
    qpos: jax.Array, soft_lowers: float, soft_uppers: float
) -> jax.Array:
    out_of_limits = -jp.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - soft_uppers, 0.0, None)
    return jp.nan_to_num(jp.sum(out_of_limits))


def cost_torque_overdemand(
    motor_targets: jax.Array, qpos: jax.Array, reach_line: jax.Array
) -> jax.Array:
    """서보가 낼 수 없는 힘을 요구하는 정도. 도달선(forcerange / kp) 초과분만 벌한다.

    cost_torques 는 잘린 뒤의 힘을 보므로 과요구를 못 본다(포화하면 기울기가 0).
    오차 전체를 벌하면 정책이 목표를 현재 위치에 붙여 아무것도 안 하는 쪽으로 도망친다.
    """
    excess = jp.clip(jp.abs(motor_targets - qpos) - reach_line, 0.0, None)
    return jp.nan_to_num(jp.sum(excess))


def cost_head_vel(joints_qvel: jax.Array) -> jax.Array:
    """머리 4 축이 빨리 움직이는 것만 벌한다. 어디 있는지는 묻지 않는다.

    머리·목이 질량의 21.6 % 라 정책이 무게중심 이동에 쓴다 — 위치를 묶으면 전진이
    0.1649 -> 0.0492 m/s 로 떨어지지만, 속도만 벌하면 저주파 사용은 남고 떨림만 사라진다.
    """
    return jp.nan_to_num(jp.sum(jp.square(joints_qvel[5:9])))


def cost_stand_still(
    commands: jax.Array,
    qpos: jax.Array,
    qvel: jax.Array,
    default_pose: jax.Array,
    ignore_head: bool = False,
) -> jax.Array:
    # TODO no hard coded slices
    cmd_norm = jp.linalg.norm(commands[:3])
    if not ignore_head:
        pose_cost = jp.sum(jp.abs(qpos - default_pose))
        vel_cost = jp.sum(jp.abs(qvel))
    else:
        left_leg_pos = qpos[:5]
        right_leg_pos = qpos[9:]
        left_leg_vel = qvel[:5]
        right_leg_vel = qvel[9:]
        left_leg_default = default_pose[:5]
        right_leg_default = default_pose[9:]
        pose_cost = jp.sum(jp.abs(left_leg_pos - left_leg_default)) + jp.sum(
            jp.abs(right_leg_pos - right_leg_default)
        )
        vel_cost = jp.sum(jp.abs(left_leg_vel)) + jp.sum(jp.abs(right_leg_vel))

    return jp.nan_to_num(pose_cost + vel_cost) * (cmd_norm < 0.01)


def cost_cmd_idle(
    command: jax.Array,
    target: jax.Array,
    mean_vel: jax.Array,
    step: jax.Array,
    delay_steps: int,
    ratio: float,
) -> jax.Array:
    # 이동 명령 목표가 바뀌고 delay_steps 가 지났는데 한 걸음 평균 속도가 명령 방향으로 명령 크기의
    # ratio 배에 못 미치면 모자란 만큼 0 ~ 1 로 벌한다. 서기 · 회전만 목표와 출발 구간은 벌하지 않는다.
    c = command[:2]
    r = jp.dot(mean_vel, c) / jp.maximum(jp.dot(c, c), 1e-6)
    on = (step >= delay_steps) & (jp.sum(jp.abs(target[:2])) > 0)
    return jp.clip((ratio - r) / ratio, 0.0, 1.0) * on


def cost_termination(done: jax.Array) -> jax.Array:
    return done


def reward_alive() -> jax.Array:
    return jp.array(1.0)


# Pose-related rewards.


def cost_yaw_sway(gyro_z: jax.Array, mean_gyro_z: jax.Array) -> jax.Array:
    """걸음 안 회전 흔들림. 한 걸음 평균에서 벗어난 순간 회전 속도의 제곱이다.

    회전 추적은 평균만 보므로 평균 둘레의 흔들림은 이 항이 맡는다.
    """
    return jp.nan_to_num(jp.square(gyro_z - mean_gyro_z))


def cost_head_pos(
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    cmd: jax.Array,
) -> jax.Array:
    head_cmd = cmd[3:]
    head_pos = joints_qpos[5:9]
    # head_vel = joints_qvel[5:9]

    # target_head_qvel = jp.zeros_like(head_cmd)

    head_pos_error = jp.sum(jp.square(head_pos - head_cmd))

    # head_vel_error = jp.sum(jp.square(head_vel - target_head_qvel))

    # 게이트가 없다(14 차부터). 5~8 차는 걷는 중만, 12~13 차는 정지 중만 켜져
    # 나머지 구간에서 머리가 방목 상태가 됐다 — 걷는 중에는 stand_still 도 꺼진다.
    # 대가는 안다: 걷는 중 머리를 명령에 묶으면 무게중심 이동에 못 쓴다.
    return jp.nan_to_num(head_pos_error)
    # return jp.nan_to_num(head_pos_error + head_vel_error)


# FIXME
def cost_joint_deviation_hip(
    qpos: jax.Array, cmd: jax.Array, hip_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    cost = jp.sum(jp.abs(qpos[hip_indices] - default_pose[hip_indices]))
    cost *= jp.abs(cmd[1]) > 0.1
    return jp.nan_to_num(cost)


# FIXME
def cost_joint_deviation_knee(
    qpos: jax.Array, knee_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    return jp.nan_to_num(
        jp.sum(jp.abs(qpos[knee_indices] - default_pose[knee_indices]))
    )


# FIXME
def cost_pose(
    qpos: jax.Array, default_pose: jax.Array, weights: jax.Array
) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(qpos - default_pose) * weights))


# Feet related rewards.


# FIXME
def cost_feet_slip(contact: jax.Array, global_linvel: jax.Array) -> jax.Array:
    # global_linvel 은 두 발의 (2, 3) 속도다. 미끄러짐은 수평 속도만 본다. 3 축을 다 쓰면
    # 디딘 발을 들고 내리는 수직 움직임까지 벌해 feet_air_time 과 부딪혔다.
    foot_vel_xy = global_linvel[..., :2]
    reward = jp.sum(jp.linalg.norm(foot_vel_xy, axis=-1) * contact)
    return jp.nan_to_num(reward)


# FIXME
def cost_feet_clearance(
    feet_vel: jax.Array,
    foot_pos: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    # feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - max_foot_height)
    return jp.nan_to_num(jp.sum(delta * vel_norm))


# FIXME
def cost_feet_height(
    swing_peak: jax.Array,
    first_contact: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    error = swing_peak / max_foot_height - 1.0
    return jp.nan_to_num(jp.sum(jp.square(error) * first_contact))


# FIXME
def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    commands: jax.Array,
    threshold_min: float = 0.1,  # 0.2
    threshold_max: float = 0.5,
) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands[:3])
    # threshold_min 보다 짧은 체공은 0 점이다. 음수로 두면 짧게 드는 것이 안 드는 것보다
    # 손해라 발을 끌게 된다(v2_1 은 1억 4천만 스텝까지 이 항이 음수였다, research/04 1장).
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, 0.0, threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)


# FIXME
def reward_feet_phase(
    foot_pos: jax.Array,
    rz: jax.Array,
) -> jax.Array:
    # Reward for tracking the desired foot height.
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    # rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return jp.nan_to_num(reward)
