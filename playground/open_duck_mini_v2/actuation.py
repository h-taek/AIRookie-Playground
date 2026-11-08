"""행동을 서보 목표각으로 바꾸는 배율과 머리 명령 범위.

학습 환경(joystick.py, standing.py)과 뷰어·평가 도구(mujoco_infer.py)가 이 둘을 같이 쓰고,
tools/set_home_pose.py 가 같은 값을 Runtime home_pose.json 에 적는다. 따로 적으면 어긋난다 —
1 ~ 7 차에는 머리 명령 범위가 상수 사본으로 어긋나 있었다.
"""

import numpy as np

HEAD_ACTUATORS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def action_scale(mj_model, home, leg_scale: float) -> np.ndarray:
    """구동기별 행동 배율(rad). 목표각 = 홈 + 배율 x 행동(-1 ~ 1).

    다리는 leg_scale 이다. 머리 4 축은 홈에서 관절 한계까지 먼 쪽 거리라 정책이 관절 범위
    전체에 닿는다. 머리도 0.25 였던 v1 은 머리를 홈 +-0.25 rad 밖으로 못 보내 머리 명령의
    57 ~ 63 % 에 닿지 못했다(research/03 11장).
    """
    scale = np.full(mj_model.nu, float(leg_scale))
    for i in range(mj_model.nu):
        if mj_model.actuator(i).name in HEAD_ACTUATORS:
            lo, hi = mj_model.jnt_range[mj_model.actuator_trnid[i, 0]]
            scale[i] = max(abs(lo - home[i]), abs(hi - home[i]))
    return scale


def head_command_range(mj_model, fence_lo, fence_hi, soft_factor: float):
    """머리 4 축 명령 범위. 관절 소프트 한계와 설정 울타리의 교집합이다."""
    lo, hi = [], []
    for name, f_lo, f_hi in zip(HEAD_ACTUATORS, fence_lo, fence_hi):
        a, b = mj_model.jnt_range[mj_model.joint(name).id]
        c, r = (a + b) / 2, b - a
        lo.append(max(f_lo, c - 0.5 * r * soft_factor))
        hi.append(min(f_hi, c + 0.5 * r * soft_factor))
    return np.array(lo), np.array(hi)


def config_head_command_range(mj_model, config):
    """학습 설정(joystick.default_config)이 정하는 머리 4 축 명령 범위."""
    f = config.head_range_factor
    fences = np.array([config.neck_pitch_range, config.head_pitch_range,
                       config.head_yaw_range, config.head_roll_range]) * f
    return head_command_range(mj_model, fences[:, 0], fences[:, 1],
                              config.soft_joint_pos_limit_factor)
