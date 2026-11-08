# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

from . import actuation
from . import constants
from . import base as open_duck_mini_v2_base

# from playground.common.utils import LowPassActionFilter
from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques_abs,
    cost_action_rate,
    cost_action_acc,
    cost_stand_still,
    cost_cmd_idle,
    cost_orientation,
    cost_ang_vel_xy,
    cost_ang_vel_x,
    cost_torque_overdemand,
    cost_feet_slip,
    reward_feet_air_time,
    cost_pose,
    cost_head_vel,
    cost_head_pos,
    cost_yaw_sway,
    reward_alive,
)
from playground.open_duck_mini_v2.custom_rewards import reward_imitation

# if set to false, won't require the reference data to be present and won't compute the reference motions polynoms for nothing
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # 물리 백엔드. "warp" 가 MJWarp 이고 "jax" 가 기존 MJX 경로다.
        # runner 의 --impl 로 덮어쓴다. CUDA 가 없는 곳에서는 jax 를 쓴다.
        impl="warp",
        # Warp 백엔드가 미리 잡아 두는 접촉·제약 상한이다. jax 백엔드에서는 무시된다.
        # naconmax 는 전 세계 합이다. 충돌 기하가 발바닥 메시 둘뿐이고 메시-평면 한 쌍이
        # 최대 4 점이라 세계당 8 점이 상한이다. 넘친 접촉은 경고 없이 버려진다.
        # njmax 는 세계당이고 실측 최대 59 행이다 (docs/archive/research/10_mjwarp_backend.md 7장).
        naconmax=8 * 8192,
        njmax=128,
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        # 다리 10 축의 행동 배율(rad). 머리 4 축은 관절 범위에서 유도한다 (actuation.py).
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        # 4.83 과 상류 5.24 는 둘 다 데이터시트 값이고 실물로 확인된 적이 없다.
        max_motor_velocity=4.83,  # 21차: 걷기가 확인된 우리 값으로 복귀
        # max_motor_accel 은 근거였던 435 deg/s^2 가 우리 제약 아래에서 잰 값이라 18 차에 삭제했다.
        # 서보 내부 가속 상한(rad/s^2). STS3215 는 목표각을 받으면 내부 기준을 가속 레지스터(41)
        # 값 × 8.79 deg/s^2 상한 궤적으로 옮긴다. 0·254 는 50 과 같은 느린 상한이다. 실물은 253
        # (2224 deg/s^2)으로 걷는다. 매단 무릎 실측(가속 50 · 100 · 150 · 253)을 이 한 값으로
        # 재현한다 (Runtime research/10). 에피소드마다 servo_accel_range 배율을 곱한다.
        # 0 이면 흉내를 끄고 목표각을 PD 에 바로 준다(v2_8 까지와 같다). 38.8 을 켠 v2_9 는 서 있기만
        # 배웠다. v2_10 은 전류 벌점 하나만 v2_8 과 다르게 하려고 끈다.
        servo_max_accel=0.0,  # v2_39: v2_15 와 같이 끈다
        servo_accel_range=[0.85, 1.15],
        # 무릎 두 축의 kp 배율(base.py). 실물 무릎은 목표각 진폭의 33 ~ 66 % 만 따라가고, 시뮬에서 0.3 ~ 0.5 가
        # 그 응답을 낸다 (research/04 42장).
        knee_kp_scale=1.0,
        # 학습 중 서보 가속 상한을 servo_accel_start 에서 servo_max_accel 로 옮긴다. None 이면 처음부터
        # servo_max_accel 이다. 학습 진행 비율 [0] 까지 시작값, [1] 까지 로그 선형으로 옮기고 그 뒤는 끝값이다.
        # torques_start 처럼 평가 주기마다 reset 할 때 정한다. 400 rad/s² 는 목표각을 5 ms 안에 따라가는
        # 사실상 흉내 없음이다 (research/04 41장).
        servo_accel_ramp=[1 / 3, 2 / 3],
        servo_accel_start=400.0,
        # 학습 중 무릎 두 축의 kp 를 1.0 에서 knee_kp_range 안의 끝값으로 옮긴다. None 이면 끈다. 진행 비율 [0] 까지
        # 1.0, [1] 까지 선형으로 옮기고 그 뒤는 끝값이다. [0, 0] 이면 처음부터 끝값이다. 끝값은 reset 마다 환경별 ·
        # 무릎별로 뽑고, 노이즈 레벨 0 이면 범위 가운데다. knee_kp_scale 위에 곱한다.
        # 범위는 [왼무릎, 오른무릎] 이다. 실물 오른무릎 서보는 각도 차이 1° 당 힘이 왼쪽의 절반이라, 공중 시험
        # 맞춤(가속 38.8 고정)에서 왼 ×1.325, 오른 ×0.325 ~ 0.35 다 (research/04 42-9).
        knee_kp_ramp=None,  # v2_39: v2_15 와 같이 끈다
        knee_kp_range=[[1.0, 1.3], [0.3, 0.35]],
        # 몸통 휨. 다리 뿌리마다 좌우 · 앞뒤 스프링 관절(*_flex_roll · *_flex_pitch, 엔코더에 안 보임)의 강성
        # (Nm/rad)과 감쇠(Nm·s/rad)를 reset 마다 [좌우, 앞뒤] 범위에서 뽑는다. 두 다리는 같은 값이다. 범위는
        # 실물 툭 치기의 흩어짐(좌우 3.9 ~ 4.5 Hz · 앞뒤 2.3 ~ 2.8 Hz, 감쇠비 0.12 ~ 0.16)을 시뮬 툭 치기로 옮긴
        # 값이다 (Runtime research/10 26-12, research/04 43-4). 노이즈 레벨 0 이면 범위 가운데다.
        # ramp 가 None 이 아니면 진행 비율 [0] 까지 k_start, [1] 까지 로그 선형으로 끝값에 옮긴다. 감쇠는
        # 감쇠비가 같도록 √(강성 / 끝 강성) 배다.
        flex=config_dict.create(
            # v2_48: 끝값을 위 실측 범위(가운데 5.1 / 4.0)의 9.8 배(가운데 50 / 39)로 올린다. 감쇠는 감쇠비가
            # 같도록 √9.8 배다. v2_47 은 19.6 배다. 실측 끝값까지 내린 v2_42r 은 실물 0.111 에서 넘어지고 200 고정 v2_43 은 걸었다
            # (Runtime research/10 27장). 실측 범위는 k_range [[4.2, 6.0], [3.2, 4.8]] · d_range [[0.02, 0.04]] × 2 다.
            k_range=[[41.2, 58.8], [31.4, 47.1]],
            d_range=[[0.063, 0.125], [0.063, 0.125]],
            ramp=[1 / 3, 2 / 3],  # v2_47 · v2_48: v2_42r 과 같다(1 억까지 200, 2 억까지 끝값)
            k_start=200.0,
        ),
        # 전원 포트의 전압 강하. 서보마다 전압 = v_nominal − 포트 A 전류 합 × servo_r_a − 포트 B 전류 합
        # × servo_r_b 로 처지고, 처진 비율만큼 그 서보의 힘 상한이 준다. 포트 A 는 왼다리 · 머리 ·
        # right_hip_yaw, 포트 B 는 오른다리 넷이다. 저항은 실물 v2_15 걷기 로그에서 서보 전압을 두 포트
        # 전류 합으로 회귀한 계수다(Runtime research/10 13장). 서보 전류는 cur_gain × max(|바깥 하중| −
        # cur_hold, 0) 이고 바깥 하중은 서보 힘 − 관절 감쇠 × 속도 − armature × 가속이다. 실물 무릎 전류는
        # 발을 든 동안 거의 0 이라 감쇠 · 회전자 관성 몫은 전류로 보지 않는다(research/04 20-12).
        # 에피소드마다 포트별로 servo_r_a · servo_r_b 에 servo_r_scale_range 배율을 곱한다.
        # 처진 전압 비율은 서보 힘의 기울기(kp)와 상한을 함께 줄인다. 서보 힘은 출력 비율 × 전압이다.
        # 상한만 줄인 v2_22 는 처져도 걷기가 그대로라 처짐을 배울 신호가 없었다(research/04 21장).
        # v2_21 은 포트별 저항 하나와 '포화 동안 2.87 A' 전류로 학습해 후진을 잃었다(research/04 19장).
        # v2_30 부터 끈다. 켠 v2_23 · v2_27 ~ v2_29 는 실물에서 앞으로 고꾸라지고 끈 v2_15 만 걸었다
        # (Runtime research/10 20장).
        voltage_drop=config_dict.create(
            enable=False,
            v_nominal=7.4,
            cur_gain=1.29,  # A / N·m
            cur_hold=0.417,  # N·m
            # 서보 순서는 모델 액추에이터 순서(왼다리 5, 머리 4, 오른다리 5)다.
            servo_r_a=[0.32, 0.38, 0.47, 0.54, 0.52, 0.28, 0.19, 0.08, 0.13, 0.23, 0.03, 0.03, 0.02, 0.03],
            servo_r_b=[0.03, 0.03, 0.03, 0.04, 0.03, 0.04, 0.04, 0.02, 0.03, 0.02, 0.08, 0.13, 0.21, 0.20],
            servo_r_scale_range=[0.7, 1.3],
            min_scale=0.3,
        ),
        # 머리 4 축(행동 5:9) 목표각을 정책 행동 대신 머리 명령으로 바로 준다. 행동 크기 14 는 그대로다.
        # v2_11 은 걸을 때 머리를 1 ~ 2 도만 움직이면서 왼쪽 포트 부하의 41 % 를 머리에 썼다. 머리 목표만
        # 기본 자세로 고정해도 전진 0.05 의 토크 합이 8.50 -> 6.50 N·m 이고 속도는 그대로였다 (research/04 15장).
        head_from_command=True,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            # 행동 지연. 에피소드마다 기본 지연을 [min, max) 에서 뽑고, 스텝마다 확률
            # action_delay_jitter 로 ±1 스텝 흔든다. 직전보다 2 스텝 이상 늘지 않게 잘라 먼저 낸
            # 행동이 나중 행동 뒤에 서보에 닿지 않는다. v2_5 까지는 스텝마다 0 ~ 2 를 새로 뽑아
            # 11 % 스텝에서 목표가 과거로 돌아갔고, 지속 지연은 겪지 못했다 (research/04 8장).
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps (미포함)
            action_delay_jitter=0.1,
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            # 실물 발 접촉은 발마다 스위치 하나(바닥판이 누르는 구조)다. 오판정·지연·고장을 학습에 넣는다.
            contact_min_delay=0,  # env steps
            contact_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad, for each hip joint
                knee_pos=0.05,  # rad, for each knee joint
                ankle_pos=0.08,  # rad, for each ankle joint
                # 노이즈가 실물과 거꾸로였다 — 관절속도는 신호의 153 %, IMU 는 1~3 % 였다.
                joint_vel=0.8,  # rad/s. 신호 std 0.94 대비 약 46 %
                gravity=0.1,
                linvel=0.1,
                gyro=0.25,  # 신호 std 0.8~2.4 rad/s 대비 약 6~18 %
                accelerometer=0.4,  # 신호 std 2.4~4.8 m/s² 대비 약 5~10 %
                # 스텝마다 발 하나의 접촉 판정이 뒤집힐 확률
                contact_flip=0.05,
                # 에피소드 내내 한쪽 접촉 추정이 죽어 있을 확률 (발마다 독립)
                contact_dropout=0.02,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # 2 차에 정책이 회전만 배웠다. 7.0 이면 전진 동기가 6.27 로 회전 5.89 와 대등해진다.
                tracking_lin_vel=7.0,  # 21차: 걷기가 확인된 우리 값으로 복귀
                tracking_ang_vel=6.0,
                # v2_10 에 제곱합 -1e-3 에서 절댓값 합 -0.01 로 바꾼다. 제곱합 -1e-3 은 v2_8 에서
                # alive 의 0.1 ~ 0.8 % 라 전류를 줄일 동기가 없었다. -0.01 은 v2_8 기준 정지 -0.045,
                # 전진 0.05 -0.090, 전진 0.222 -0.127 점/스텝으로 action_rate 와 같은 급이다 (research/04 11장).
                # v2_10 은 전류를 거의 못 줄였다(13장). v2_11 에 5 배로 올린다. 걸을 때 더 쓰는 토크 약 5.6 N·m
                # 기준으로 서는 쪽이 추종 손해보다 이득인 명령은 0.023 m/s 아래다.
                # v2_12 에 -0.1 로 올린다. v2_11 의 걸을 때 더 쓰는 토크 3.65 N·m 기준으로 그 경계는 0.026 m/s 다.
                # v2_13 은 머리 목표를 자주 바꾸는 것만 다르고 -0.1 그대로다. v2_14 부터 걷는 최대값을 찾는다
                # (research/04 18장). v2_15(학습 중 -0.1 에서 -0.2 로)로 정했다. 끝값 -0.3(v2_18) · -0.25(v2_19)는
                # 전진 전류가 v2_15 와 같고 머리 시험 넘어짐 · 쏠림이 늘었다.
                torques=-0.2,  # v2_39: v2_15
                action_rate=-0.5,  # was -1.5
                # 행동 2차 변화. action_rate 는 한 스텝 변화만 봐 매 스텝 방향이 뒤집히는 떨림을 거의 벌하지 않는다.
                # 서보 가속 흉내를 넣은 v2_36r 은 목표각 방향이 스텝의 42 % 에서 뒤집혔다(v2_15 25 %).
                action_acc=-0.5,  # v2_43 · v2_42. v2_40 은 0
                # 3 차에 tracking_lin_vel 을 2.8 배 올리자 그 동기가 정지 명령에서도 새어 나왔다.
                stand_still=-1.0,  # 21차: 걷기가 확인된 우리 값으로 복귀
                # v2_52 신규. 이동 명령 3 s 뒤에도 서 있으면 벌한다. 서 있어도 받는 몫(alive · tracking_ang_vel ·
                # tracking_lin_vel 일부)이 커서 버티기가 걷기의 86 % 를 받았다(research/04 51-6 · 51-8).
                cmd_idle=-4.0,
                # 1 차의 20.0 은 넘어질 손해(200 점)가 걸어 버는 이득(45 점)을 눌러 걷기를 포기시켰다.
                alive=2.0,  # 21차: 걷기가 확인된 우리 값으로 복귀
                imitation=1.0,  # v2_39: v2_15
                # 21 차까지 몸통 자세를 보는 항이 없어 v21·v16 이 똑같이 +11 도 숙이고 섰다.
                # -5.0(22 차)이 +6.99 도, -12.0(23 차)이 +2.09 도까지 내렸다.
                orientation=-12.0,
                # v2_45 신규. 몸통 휨 몸에서 넘어지기 전 몸통 흔들림을 직접 벌한다. 넘어짐 없는 걷기 0.34 ~ 1.05,
                # 서기 0.07 ~ 0.25, 넘어짐 잦은 걷기 0.66 ~ 2.44 (rad/s)² 다(research/04 45-4).
                ang_vel_xy=0.0,  # v2_47 · v2_48: 끈다(v2_43 과 같다). v2_45 · v2_46 은 −1.0
                # v2_53 신규. 좌우 기울기 각속도만 벌한다. 시범 동작은 0 인데 v2_48 · v2_52 는 걸을 때
                # 23 ~ 46 °/s 로 흔들고 그 크기가 시드마다 갈렸다. −2.0 에서 걷기 − 서기 차는 v2_52 의 85 % 이상 남는다(research/04 52-5).
                ang_vel_x=-2.0,
                # 21 차가 머리를 무게추로 휘둘러 head_yaw 요구/상한이 1.11(서보가 못 내는 힘)이 됐다.
                # 11 차에 뺀 근거는 다리 얘기였고, 관성력만 내는 head_yaw 에는 적용되지 않는다.
                torque_overdemand=-1.5,
                # 23 차에 -1.5 로 넣었더니 회전 체공이 15.4 -> 10.1 mm 로 **더 나빠졌다** —
                # 비비는 것을 벌하니 정책이 발을 아예 안 움직이는 쪽으로 도망갔다.
                # 이 항은 정상 보행에서도 큰 값이 나와(v23 전진 0.0933) 걷기 자체를 누른다.
                # 24 차에 -0.5 로 내리고 feet_air_time 을 키워 드는 쪽이 이기게 한다.
                feet_slip=-0.5,  # 24차: 미검증
                # feet_slip 의 짝이다. 23 차의 +20.0 은 약했다 — 회전에서 feet_slip 이
                # -0.104 인데 이 항은 -0.039 로 2.7 배 밀렸고, 그래서 발을 안 들었다.
                # 24 차에 +40.0 으로 올린다(회전 -0.078 대 feet_slip -0.035 로 역전).
                # 체공 하한이 0.1 초라 지금(0.054 초) 값이 음수이고, 들수록 벌점이 줄어든다.
                feet_air_time=40.0,  # v2_39: v2_15
                # 머리를 어디 두는지가 아니라 얼마나 빨리 움직이는지만 벌하는 항이다.
                head_vel=0.0,  # 18차: 상류값 복귀 (항 제거)
                # 21~22 차의 0 은 머리 명령을 완전히 무시하게 만들었다. -20.0 으로 되살리자
                # v23 이 자세 오차 0.072~0.259 rad 로 v16 과 같은 수준을 재현했다.
                # 이 로봇은 머리를 무게추로 쓰므로 구속 자체가 비용이고, 걷기 속도는 내준다.
                head_pos=-20.0,  # 23차: 목 명령 추종 복구
                # v2_4 신규. 전진에서 v2_3 −1.2 ~ −1.5, v2_1 −0.03 ~ −0.06, 정지 0 점/스텝이다.
                yaw_sway=-1.0,  # v2_39: v2_15
                # v2_5 신규, v2_6 에서 뺐다가 v2_7 에 되살린다. tracking_ang_vel 과 같은 걸음 평균
                # 회전 오차를 폭 yaw_drift_sigma 로 좁게 본다. v2_5 는 지연이 보여 이 항을 지연별 보정으로
                # 피했다(학습 조건에서만 직진, research/04 8장). 지연을 숨기면 모든 지연에서 직진해야 피한다.
                # v2_8 에서 다시 뺀다. v2_7 의 쏠림 감소가 지연 숨기기와 이 항 중 어느 쪽 몫인지 가린다.
                yaw_drift=0.0,
                # 24 차 신규. **좌우 대칭을 요구하는 항이 하나도 없다.** 홈은 22 차에
                # 대칭으로 맞췄는데 v23 이 정지에서 hip_roll 을 좌 +9.22 / 우 +2.44 도로
                # (미러라면 부호가 반대여야 한다) 잡고 무게중심이 y +16.5 mm 쏠려 있다.
                # 발 위치는 1 mm 안에서 맞으므로 여러 관절이 상쇄하는 자세이고, 유격과
                # 마찰이 있는 실물에서는 그 상쇄가 재현되지 않는다.
                #
                # cost_pose 는 대칭이 아니라 "홈에 붙어 있기" 를 요구한다. 홈이 대칭이라
                # 결과적으로 대칭을 끌어당기지만 겨냥이 정확하지는 않다. _weights 가
                # hip_pitch·knee 를 0.01 로 낮춰 두어 걷기에 필요한 굽힘은 거의 안 벌한다.
                # v23 실측 0.0838~0.1369 중 발목이 0.070, hip_roll 이 0.031 이다 —
                # 즉 이 항의 대부분은 발목을 홈으로 되돌리는 압력이고, 그것이 원하는
                # 효과인지는 확인되지 않았다.
                #
                # -1.5 면 정지 -0.154, 전진 -0.144 점/스텝으로 action_rate 급이다.
                pose=-1.5,  # 24차: 미검증
            ),
            # 시그마는 손잡이가 아니다 — 키우면 정지 보상이 올라 걷는 동기가 줄고, 좁히면
            # 초기 기울기가 죽어 걷기가 무너진다. 2·5·6 차 세 회차를 여기에 썼다.
            tracking_sigma=0.01,  # 4 차 0.01 -> 5 차 0.002 -> 6 차 0.005 (둘 다 실패) -> 7 차 0.01 복귀
            # v2_45: 전진 폭만 이 속도 아래에서 명령 크기에 비례해 좁힌다(rewards.reward_tracking_lin_vel). 몸통 휨
            # 몸에서 0.05 명령에 서 있으면 지금 폭으로 0.78 배를 받아 걷기 이득이 작았다(research/04 44-3 · 45-4).
            # 이 속도 이상의 폭은 그대로라 초기 기울기는 그대로다. None 이면 끈다.
            tracking_low_speed_ref=None,  # v2_47 · v2_48: 끈다(v2_43 과 같다). v2_45 · v2_46 은 0.111
            # 명령 범위가 ±1.0 rad/s 라 0.01 이면 오차 1.0 에서 기울기가 수치적으로 0 이다.
            # cmd_idle 이 켜지는 명령 목표 전환 뒤 시간과 속도비 경계. 3 s 는 명령 시상수 상한 1 s 의 3 배다.
            cmd_idle_delay=3.0,  # s
            cmd_idle_ratio=0.5,
            tracking_ang_vel_sigma=0.25,  # 21차: 걷기가 확인된 우리 값으로 복귀
            # yaw_drift 의 폭. 걸음 평균 회전 오차의 자연 흔들림(표준편차 0.012 ~ 0.045)보다 넓다.
            # 큰 오차에서는 기울기가 0 이라 회전을 배우는 일은 넓은 항이 맡는다.
            yaw_drift_sigma=0.01,
            # 학습 중 torques 가중치를 이 값에서 scales.torques 로 옮긴다. None 이면 처음부터 scales.torques 다.
            # 학습 진행 비율 torques_ramp[0] 까지 시작값, [1] 까지 선형으로 옮기고 그 뒤는 끝값이다. 값은 brax 가
            # 평가 주기마다 학습 환경을 reset 할 때 정해 그 주기 동안 쓴다(3 억 스텝에 15 단계). v2_14 는 -0.2 를
            # 처음부터 걸어 후진을 못 찾았다 (research/04 18장).
            torques_start=-0.1,  # v2_15 부터. 끝값은 scales.torques
            torques_ramp=[1 / 3, 2 / 3],
            # imitation 이 견주는 시범 동작을 이 스텝 수만큼 늦춘다. 서보 가속 흉내가 있으면 관절이 목표를
            # 늦게 따라가 늦추지 않은 시범 동작에서는 걷기와 서기의 점수가 같아진다. 3 이 걷기 점수가 가장 높다
            # (research/04 41장). 관측의 위상은 늦추지 않는다.
            imitation_delay=0,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            # 1.0 m/s 를 받으려면 무게중심 앞 146 mm 에 발을 놓아야 하는데 보폭은 60 mm 다.
            # v2_1 평가의 조기 종료는 전부 밀기였다(research/04 1장).
            magnitude_range=[0.1, 0.5],
        ),
        # 명령 범위는 시범 동작 격자가 지원하는 범위와 같아야 tracking 과 imitation 이
        # 같은 목표를 준다. 후진은 지금 격자(2026-10-07)의 라벨과 몸통 속도가 맞아 넣는다
        # (−0.111 → −0.115 m/s, research/04 4장).
        lin_vel_x=[-0.111, 0.222],
        lin_vel_y=[-0.148, 0.148],
        ang_vel_yaw=[-1.111, 1.111],
        # 아래 네 값은 바깥 울타리로만 쓴다 — 실제 범위는 _post_init 에서 관절 소프트
        # 한계(0.95)와의 교집합으로 정한다. 여기 적힌 상한은 전부 관절이 못 가는 각도다.
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],
        # 네 축에 같은 배율을 곱하므로 이것으로는 한계 초과를 못 고친다(neck_pitch 는 비대칭).
        head_range_factor=1.0,  # to make it easier
        # 명령을 연속으로 바꾼다 (spec/01 2장). 500 스텝마다 뽑는 값은 목표이고, 실제 명령은
        # 매 스텝 1 차 지연으로 목표를 따라간다. 시상수는 에피소드마다 뽑아 게임패드처럼 빠른
        # 변화(0.02 s)부터 추종 제어기처럼 느린 변화(1 s)까지 덮는다. 흔들림은 채널 범위
        # 절반폭에 대한 비율의 정규분포이고, 이동 채널은 목표가 정지일 때 끈다(7장 정지 유지).
        command_config=config_dict.create(
            tau_range=[0.02, 1.0],  # s
            jitter=0.03,
            # 이동 명령 종류: 정지, 제자리 회전, 전진만, 후진만, 옆만, 세 축 섞음. 세 축을 따로 뽑기만 하면
            # 제자리 회전(spec/01 3장)이 이동 명령의 1.2 % 뿐이었고, 옆만 가는 명령은 없었다.
            # v2_15 까지는 앞뒤를 한 종류(15 %)로 뽑아 후진만이 5 % 였고 후진을 학습 후반에야 배웠다
            # (research/04 18장). v2_16 은 정지 15 · 전진만 15 · 후진만 10 · 옆 15 · 섞음 25 % 로 빠른 전진만 배웠다.
            # v2_17 은 v2_15 에서 섞음 5 % 를 후진만으로 옮겼으나 후진 0.05 는 그대로 서 있었다.
            # v2_18 부터 v2_15 비율(앞뒤만 15 % 를 전진만 10 · 후진만 5 % 로 나눈 것과 같다)로 돌아간다.
            mode_probs=[0.2, 0.2, 0.1, 0.05, 0.1, 0.35],
            # 머리 목표는 이동 명령과 따로 이 간격(s)마다 새로 뽑는다. v2_12 까지는 이동 명령과 같이
            # 10 초마다였고, 걸으며 머리를 자주 움직이면 117 번 중 9 번 넘어졌다 (research/04 17장).
            head_interval_range=[0.3, 3.0],
            # 머리 목표를 따라가는 시상수(s). 목표마다 뽑는다.
            head_tau_range=[0.02, 0.5],
            # 새 머리 목표가 정면(0)일 확률. 나머지는 머리 명령 범위에서 고르게 뽑는다.
            head_home_prob=0.2,
        ),
    )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()
        self._train_frac = 0.0

    def set_train_progress(self, frac: float) -> None:
        """러너가 평가 주기마다 학습 진행 비율(0 ~ 1)을 알린다. torques_start 가 쓴다."""
        self._train_frac = float(frac)

    def _torque_scale_host(self) -> float:
        rc = self._config.reward_config
        a, b = rc.torques_ramp
        w = min(max((self._train_frac - a) / max(b - a, 1e-9), 0.0), 1.0)
        return (1 - w) * rc.torques_start + w * rc.scales.torques

    def _servo_accel_host(self) -> float:
        c = self._config
        a, b = c.servo_accel_ramp
        w = min(max((self._train_frac - a) / max(b - a, 1e-9), 0.0), 1.0)
        return float(np.exp((1 - w) * np.log(c.servo_accel_start) + w * np.log(c.servo_max_accel)))

    def _knee_kp_w_host(self) -> float:
        a, b = self._config.knee_kp_ramp
        if self._train_frac >= b:
            return 1.0
        return min(max((self._train_frac - a) / max(b - a, 1e-9), 0.0), 1.0)

    def _knee_kp(self, rng: jax.Array) -> jax.Array:
        c = self._config
        lo, hi = jp.array(c.knee_kp_range, jp.float32).T
        end = jp.where(
            c.noise_config.level > 0, jax.random.uniform(rng, (2,), minval=lo, maxval=hi), 0.5 * (lo + hi)
        )
        w = jax.pure_callback(
            lambda v: np.full(np.shape(v), self._knee_kp_w_host(), np.float32),
            jax.ShapeDtypeStruct((2,), jp.float32), end, vmap_method="broadcast_all",
        )
        return 1.0 - w * (1.0 - end)

    def _flex_w_host(self) -> float:
        a, b = self._config.flex.ramp
        return min(max((self._train_frac - a) / max(b - a, 1e-9), 0.0), 1.0)

    def _flex(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """이 에피소드의 몸통 휨 [좌우, 앞뒤] 강성 · 감쇠."""
        c = self._config.flex
        k_lo, k_hi = jp.array(c.k_range, jp.float32).T
        d_lo, d_hi = jp.array(c.d_range, jp.float32).T
        k_rng, d_rng = jax.random.split(rng)
        noisy = self._config.noise_config.level > 0
        k = jp.where(noisy, jax.random.uniform(k_rng, (2,), minval=k_lo, maxval=k_hi), 0.5 * (k_lo + k_hi))
        d = jp.where(noisy, jax.random.uniform(d_rng, (2,), minval=d_lo, maxval=d_hi), 0.5 * (d_lo + d_hi))
        if c.ramp is None:
            return k, d
        w = jax.pure_callback(
            lambda v: np.full(np.shape(v), self._flex_w_host(), np.float32),
            jax.ShapeDtypeStruct((2,), jp.float32), k, vmap_method="broadcast_all",
        )
        k_now = jp.exp((1 - w) * jp.log(c.k_start) + w * jp.log(k))
        return k_now, d * jp.sqrt(k_now / k)

    def _servo_accel_nominal(self, rng: jax.Array) -> jax.Array:
        if self._config.servo_accel_ramp is None or self._config.servo_max_accel <= 0:
            return jp.float32(self._config.servo_max_accel)
        x = jax.random.uniform(rng)
        return jax.pure_callback(
            lambda v: np.full(np.shape(v), self._servo_accel_host(), np.float32),
            jax.ShapeDtypeStruct((), jp.float32), x, vmap_method="broadcast_all",
        )

    def _torque_scale(self, rng: jax.Array) -> jax.Array:
        if self._config.reward_config.torques_start is None:
            return jp.float32(self._config.reward_config.scales.torques)
        # reset 안에서 한 번 호스트 값을 읽는다. rng 는 vmap 묶음 모양을 맞추는 데만 쓴다.
        x = jax.random.uniform(rng)
        return jax.pure_callback(
            lambda v: np.full(np.shape(v), self._torque_scale_host(), np.float32),
            jax.ShapeDtypeStruct((), jp.float32), x, vmap_method="broadcast_all",
        )

    def _post_init(self) -> None:

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(
                "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
            )

        # tracking_ang_vel 은 자이로 z 를 한 걸음 주기 동안 평균해 명령과 견준다. 순간값으로
        # 매기면 v2_2 가 제자리 회전 명령에 몸통을 좌우로 비틀어(평균 0.1, 표준편차 1.4 rad/s)
        # 명령 속도를 지날 때마다 점수를 받았다. 주기 평균은 걸음마다의 흔들림도 상쇄한다
        # (research/04 2장).
        self._yaw_window = self.PRM.nb_steps_in_period if USE_IMITATION_REWARD else 27

        # Note: First joint is freejoint.
        # get the range of the joints
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # weights for computing the cost of each joints compared to a reference pose
        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # left leg.
                # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        self._njoints = self._mj_model.njnt  # number of joints
        self._actuators = self._mj_model.nu  # number of actuators

        # 구동 관절 14 개의 관절 한계. reset 에서 초기 자세를 이 안으로 자른다.
        # mj_model 은 numpy 이므로 여기서 상수로 뽑아 둔다(추적 안에서 인덱싱 불가).
        _act_range = self._mj_model.jnt_range[np.array(self.actuator_joint_ids)]
        self._actuator_jnt_lower = jp.array(_act_range[:, 0])
        self._actuator_jnt_upper = jp.array(_act_range[:, 1])

        # 머리 명령 범위는 관절 한계에서 유도한다. 설정 상수로 두면 XML 과 어긋나고,
        # 1 ~ 7 차 내내 어긋나 뽑은 명령의 84.4 % 가 도달 불가능한 각도를 요구했다.
        # 소프트 한계(0.95)와 설정값의 교집합이고, 뷰어(mujoco_infer.py)도 같은 함수를 쓴다.
        _head_lo, _head_hi = actuation.config_head_command_range(self._mj_model, self._config)
        self._head_cmd_lower = jp.array(_head_lo)
        self._head_cmd_upper = jp.array(_head_hi)

        # 목표각을 자르는 범위. 구동기 ctrlrange 는 관절 범위를 물려받는다(inheritrange).
        # 자르지 않으면 관측·속도 제한·과요구 벌점이 실제 구동기에 간 값과 다른 값을 쓴다.
        self._ctrl_lower = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_upper = jp.array(self._mj_model.actuator_ctrlrange[:, 1])

        # 구동기별 행동 배율. 머리는 관절 범위 전체에 닿는다 (actuation.py).
        self._action_scale = jp.array(
            actuation.action_scale(self._mj_model, self._default_actuator, self._config.action_scale)
        )

        # 명령 7 채널(이동 3 + 머리 4)의 범위와 흔들림 표준편차.
        self._cmd_lower = jp.concatenate([
            jp.array([self._config.lin_vel_x[0], self._config.lin_vel_y[0], self._config.ang_vel_yaw[0]]),
            self._head_cmd_lower,
        ])
        self._cmd_upper = jp.concatenate([
            jp.array([self._config.lin_vel_x[1], self._config.lin_vel_y[1], self._config.ang_vel_yaw[1]]),
            self._head_cmd_upper,
        ])
        self._cmd_jitter_std = (
            0.5 * (self._cmd_upper - self._cmd_lower) * self._config.command_config.jitter
        )

        # 서보가 포화하기 시작하는 추종 오차(forcerange / kp). 모델에서 유도한다 —
        # backlash 판과 비-backlash 판의 kp 가 달라 상수로 두면 한쪽과 반드시 어긋난다.
        self._reach_line = jp.array(
            self._mj_model.actuator_forcerange[:, 1]
            / self._mj_model.actuator_gainprm[:, 0]
        )

        # 서보가 물린 전원 포트 (0 = A, 1 = B). 오른다리 hip_roll · hip_pitch · knee · ankle 만 B 다.
        port_b = ("right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
        self._actuator_port = jp.array(
            [int(self._mj_model.actuator(i).name in port_b) for i in range(self._mj_model.nu)]
        )
        self._forcerange = jp.array(self._mj_model.actuator_forcerange[:, 1])
        self._knee_ids = jp.array([self._mj_model.actuator(n).id for n in ("left_knee", "right_knee")])
        flex = [self._mj_model.joint(f"{s}_flex_{a}").id for s in ("left", "right") for a in ("roll", "pitch")]
        self._flex_jnt = jp.array(flex)
        self._flex_dof = jp.array(self._mj_model.jnt_dofadr[flex])
        self._flex_axis = jp.array([0, 1, 0, 1])
        act_dof = self._mj_model.jnt_dofadr[self._mj_model.actuator_trnid[:, 0]]
        self._act_dof = jp.array(act_dof)
        self._act_damping = jp.array(self._mj_model.dof_damping[act_dof])
        self._act_armature = jp.array(self._mj_model.dof_armature[act_dof])

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        # 발바닥-바닥 접촉은 MJCF 접촉 센서로 읽는다. 씬 XML 이 발 기하마다
        # "<geom>_floor_found" 센서를 선언해 두었고 순서는 constants.FEET_GEOMS 와 같다.
        #
        # 0.0.3 에서는 mujoco_playground._src.collision.geoms_colliding 으로 기하 쌍을
        # 직접 질의했으나 그 모듈이 0.2.0 에서 삭제됐다. Warp 백엔드는 data.contact 를
        # 같은 방식으로 주소 지정하지 못한다.
        #
        # mj_model.sensor(name) 은 이름이 없으면 예외를 던진다. mj_name2id 처럼 -1 을
        # 돌려주지 않으므로 센서 선언이 빠진 모델은 조용히 통과하지 못한다.
        self._feet_floor_found_adr = np.array(
            [
                self._mj_model.sensor_adr[
                    self._mj_model.sensor(f"{name}_floor_found").id
                ]
                for name in constants.FEET_GEOMS
            ]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # # noise in the simu?
        qpos_noise_scale = np.zeros(self._actuators)

        # 인덱스는 반드시 실제 액추에이터 이름 순서에서 뽑는다. 원본은 머리가 빠진
        # 10 축 순번을 14 축 배열에 박아 6 번째부터 전부 밀렸고, 오른다리 4 축의 관절각
        # 노이즈가 0 이었다. 크기가 14 로 맞아 오류가 안 나고 학습도 정상으로 보인다.
        scales = self._config.noise_config.scales
        for idx, name in enumerate(self.actuator_names):
            if "_hip" in name:
                qpos_noise_scale[idx] = scales.hip_pos
            elif "_knee" in name:
                qpos_noise_scale[idx] = scales.knee_pos
            elif "_ankle" in name:
                qpos_noise_scale[idx] = scales.ankle_pos
            # 머리 4 축은 0 으로 둔다.
        # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # self.action_filter = LowPassActionFilter(
        #     1 / self._config.ctrl_dt, cutoff_frequency=37.5
        # )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # the complete qpos
        # print(f'DEBUG0 init qpos: {qpos}')
        qvel = jp.zeros(self.mjx_model.nv)

        # init position/orientation in environment
        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # floating base
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )  # x y noise

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )  # yaw noise

        base_qpos = base_qpos.at[3:7].set(new_quat)

        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        # print(f'DEBUG1 base qpos: {qpos}')
        # init joint position
        # qpos[7:]=*U(0.0, 0.1)
        rng, key = jax.random.split(rng)

        # 초기 자세 흔들기는 덧셈으로 한다. 원본의 곱셈(U(0.5,1.5))은 홈값이 0 인 6 축을
        # 한 번도 안 흔들고 무릎은 ±37 도를 흔들어 reset 의 10.9 % 가 관절 한계 밖이었다.
        qpos_j = self.get_actuator_joints_qpos(qpos) + jax.random.uniform(
            key, (self._actuators,), minval=-0.1, maxval=0.1
        )
        # 관절 한계 안으로 자른다. 한계 밖에서 시뮬을 시작하지 아니한다.
        qpos_j = jp.clip(
            qpos_j, self._actuator_jnt_lower, self._actuator_jnt_upper
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)
        # print(f'DEBUG2 joint qpos: {qpos}')
        # init joint vel
        # d(xyzrpy)=U(-0.05, 0.05)
        rng, key = jax.random.split(rng)
        # qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )
        # print(f'DEBUG3 base qvel: {qvel}')
        ctrl = self.get_actuator_joints_qpos(qpos)
        # print(f'DEBUG4 ctrl: {ctrl}')
        # make_data 에는 mjx 모델이 아니라 원본 MjModel 을 넘긴다. Warp 구현은
        # MjModel 만 받는다.
        #
        # 0.0.3 의 mjx_env.init 은 마지막에 mjx.forward 를 불렀지만 0.2.0 의
        # make_data 는 부르지 아니한다. 여기서 부르지 않으면 reset 이 돌려주는
        # 관측이 전방 운동학 전 값이 되어 site_xpos 와 sensordata 가 비어 있다.
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)
        # 명령 1 차 지연 계수. 시상수를 에피소드마다 뽑는다 (spec/01 2장).
        rng, tau_rng = jax.random.split(rng)
        cmd_tau = jax.random.uniform(
            tau_rng,
            minval=self._config.command_config.tau_range[0],
            maxval=self._config.command_config.tau_range[1],
        )
        cmd_alpha = 1.0 - jp.exp(-self.dt / cmd_tau)
        rng, head_rng, tq_rng = jax.random.split(rng, 3)
        _, head_steps, head_alpha = self._sample_head(head_rng)
        torque_scale = self._torque_scale(tq_rng)

        # 이 에피소드의 서보 가속 상한. 노이즈 레벨 0 이면 공칭값이다.
        rng, servo_rng = jax.random.split(rng)
        servo_scale = jax.random.uniform(
            servo_rng,
            minval=self._config.servo_accel_range[0],
            maxval=self._config.servo_accel_range[1],
        )
        servo_accel = self._servo_accel_nominal(servo_rng) * jp.where(
            self._config.noise_config.level > 0, servo_scale, 1.0
        )
        # 이 에피소드의 [왼, 오른] 무릎 kp 배율 (_servo_step). 끈 때는 난수를 쓰지 않는다.
        knee_kp = jp.ones(2, jp.float32)
        if self._config.knee_kp_ramp is not None:
            rng, knee_rng = jax.random.split(rng)
            knee_kp = self._knee_kp(knee_rng)
        # 이 에피소드의 몸통 휨 강성 · 감쇠 [왼 좌우, 왼 앞뒤, 오른 좌우, 오른 앞뒤] (_servo_step).
        rng, flex_rng = jax.random.split(rng)
        flex_k, flex_d = self._flex(flex_rng)
        flex_k, flex_d = flex_k[self._flex_axis], flex_d[self._flex_axis]
        # 이 에피소드의 서보별 경로 저항 [포트 A 전류 계수, 포트 B 전류 계수] (_servo_step).
        # 끈 때는 난수를 쓰지 않아 v2_15 와 같은 난수열이다.
        vd = self._config.voltage_drop
        port_r = jp.stack([jp.array(vd.servo_r_a), jp.array(vd.servo_r_b)])
        if vd.enable:
            rng, port_rng = jax.random.split(rng)
            port_r = port_r * jp.where(
                self._config.noise_config.level > 0,
                jax.random.uniform(
                    port_rng, (2, 1), minval=vd.servo_r_scale_range[0], maxval=vd.servo_r_scale_range[1]
                ),
                1.0,
            )

        # Sample push interval.
        # 이 에피소드의 기본 행동 지연 (step 의 지연 처리).
        rng, delay_rng = jax.random.split(rng)
        delay_base = jax.random.randint(
            delay_rng,
            (),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )

        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        if USE_IMITATION_REWARD:
            current_reference_motion = self.PRM.get_reference_motion(
                cmd[0], cmd[1], cmd[2], 0
            )
        else:
            current_reference_motion = jp.zeros(0)

        # 이 에피소드에서 접촉 추정이 죽어 있을 발을 뽑는다 (P0-1).
        # 노이즈 레벨이 0 이면 랜덤화 전체를 끄는 기존 규약을 따른다.
        rng, drop_rng = jax.random.split(rng)
        contact_dropped = jax.random.bernoulli(
            drop_rng,
            p=self._config.noise_config.scales.contact_dropout
            * self._config.noise_config.level,
            shape=(2,),
        )

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "command_target": cmd,
            # 넘어진 뒤 첫 상태로 돌릴 때 쓴다 (step 끝의 리셋 처리).
            "command_init": cmd,
            "episode_step": 0,
            "command_alpha": cmd_alpha,
            # 다음 머리 목표까지 남은 스텝과 지금 머리 목표의 1 차 지연 계수.
            "head_steps": head_steps,
            "head_alpha": head_alpha,
            # 이 reset 에서 정한 torques 가중치 (torques_start)
            "torque_scale": torque_scale,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # History related.
            # 정책이 낸 목표각의 이력 [이번, 1 스텝 전, ...]. 서보에는 지연만큼 지난 것이 간다.
            "target_history": jp.tile(
                self._default_actuator, self._config.noise_config.action_max_delay
            ),
            # 서보 PD 가 실제로 쫓는 내부 기준각(지연·가속 상한을 거친 값). 관측에는 넣지 않는다.
            "applied_targets": self._default_actuator,
            # 서보 내부 기준의 위치·속도와 이 에피소드의 가속 상한 (_servo_step).
            "servo_pos": self._default_actuator,
            "servo_vel": jp.zeros(self.mjx_model.nu),
            "servo_accel": servo_accel,
            "knee_kp": knee_kp,
            "flex_k": flex_k,
            "flex_d": flex_d,
            "port_r": port_r,
            # 서보마다 지금 힘 상한의 배율(처진 전압 / 공칭). 포화 판정과 전류에 쓴다.
            "force_scale": jp.ones(self.mjx_model.nu),
            "action_delay_base": delay_base,
            # 직전 스텝에 쓴 지연. 다음 지연은 이보다 1 스텝까지만 늘 수 있다.
            "action_delay": delay_base,
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            # 접촉 관측 랜덤화 (P0-1)
            "contact_history": jp.zeros(
                self._config.noise_config.contact_max_delay * 2
            ),
            # 이 에피소드에서 죽어 있는 발. 에피소드 시작 때 한 번만 뽑는다.
            "contact_dropped": contact_dropped,
            # imitation related
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
            # 자이로 z 이력과 채운 칸 수. 에피소드 초반에는 채운 칸만 평균한다.
            "yaw_history": jp.zeros(self._yaw_window),
            # 몸통 좌우 속도 이력. tracking_lin_vel 의 좌우 성분이 같은 창으로 평균한다.
            "vy_history": jp.zeros(self._yaw_window),
            # 몸통 앞뒤 속도 이력. cmd_idle 이 같은 창으로 평균한다.
            "vx_history": jp.zeros(self._yaw_window),
            "window_count": 0,
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = data.sensordata[self._feet_floor_found_adr] > 0
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        if USE_IMITATION_REWARD:
            state.info["imitation_i"] += 1
            state.info["imitation_i"] = (
                state.info["imitation_i"] % self.PRM.nb_steps_in_period
            )  # not critical, is already moduloed in get_reference_motion
            state.info["imitation_phase"] = jp.array(
                [
                    jp.cos(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                    jp.sin(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                ]
            )
        else:
            state.info["imitation_i"] = 0

        if USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                state.info["command"][0],
                state.info["command"][1],
                state.info["command"][2],
                state.info["imitation_i"],
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)

        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"], 4
        )

        # 정책이 낸 목표각. Runtime(gait.py)과 같이 직전 목표각 기준 속도 상한과 관절 범위로
        # 자른다. 관측의 목표각 칸은 이 값이다 — 실물의 지연은 Runtime 뒤(서보 버스·서보)에서
        # 생겨 Runtime 이 알 수 없다. v2_6 까지는 지연된 목표각을 관측에 넣어 정책이 지연을
        # 볼 수 있었고, 숨기면 전진 0.15 의 지연 민감도가 6.6° 에서 28.6° 였다 (research/04 10장).
        motor_targets = self._default_actuator + action * self._action_scale
        if self._config.head_from_command:
            motor_targets = motor_targets.at[5:9].set(state.info["command"][3:7])
        if USE_MOTOR_SPEED_LIMITS:
            # 18 차에 상류와 동일한 속도 클립만 남긴다. 가속 클립은 삭제했다
            # (근거는 default_config 의 max_motor_velocity 주석).
            prev_motor_targets = state.info["motor_targets"]
            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets - self._config.max_motor_velocity * self.dt,
                prev_motor_targets + self._config.max_motor_velocity * self.dt,
            )
        motor_targets = jp.clip(motor_targets, self._ctrl_lower, self._ctrl_upper)

        # 서보 지연. 낸 목표각의 이력에서 지연만큼 지난 것을 서보에 적용한다.
        target_history = (
            jp.roll(state.info["target_history"], self._actuators)
            .at[: self._actuators]
            .set(motor_targets)
        )
        state.info["target_history"] = target_history
        noise = self._config.noise_config
        jitter_rng, sign_rng = jax.random.split(action_delay_rng)
        jitter = jax.random.bernoulli(jitter_rng, noise.action_delay_jitter) * jp.where(
            jax.random.bernoulli(sign_rng), 1, -1
        )
        action_delay = jp.clip(
            state.info["action_delay_base"] + jitter,
            noise.action_min_delay,
            noise.action_max_delay - 1,
        )
        action_delay = jp.minimum(action_delay, state.info["action_delay"] + 1)
        state.info["action_delay"] = action_delay
        applied_targets = target_history.reshape((-1, self._actuators))[action_delay]

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
        ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )  # floating base x,y
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        ####

        data, servo_pos, servo_vel = self._servo_step(state.data, applied_targets, state.info)

        state.info["motor_targets"] = motor_targets
        state.info["servo_pos"] = servo_pos
        state.info["servo_vel"] = servo_vel
        state.info["applied_targets"] = servo_pos

        contact = data.sensordata[self._feet_floor_found_adr] > 0
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        state.info["yaw_history"] = (
            jp.roll(state.info["yaw_history"], 1).at[0].set(self.get_gyro(data)[2])
        )
        state.info["vy_history"] = (
            jp.roll(state.info["vy_history"], 1).at[0].set(self.get_local_linvel(data)[1])
        )
        state.info["vx_history"] = (
            jp.roll(state.info["vx_history"], 1).at[0].set(self.get_local_linvel(data)[0])
        )
        state.info["window_count"] = jp.minimum(state.info["window_count"] + 1, self._yaw_window)

        # 보상을 이력 갱신보다 먼저 계산한다 — cost_action_rate 가 info["last_act"] 를
        # 직전 행동으로 읽으므로, 갱신 뒤에 계산하면 자기 자신과 비교해 항상 0 이 된다.
        done = self._get_termination(data)
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )

        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        obs = self._get_obs(data, state.info, contact)
        # FIXME
        rewards = {
            k: v * (state.info["torque_scale"] if k == "torques" else self._config.reward_config.scales[k])
            for k, v in rewards.items()
        }
        # 하한을 0 에서 풀었다 — imitation 의 joint_pos 가 생 2 차식이라 음수가 나오는데
        # 원본이 그것을 0 으로 눌러 실측 3.3 % 의 스텝에서 학습 신호가 사라졌다.
        reward = jp.clip(sum(rewards.values()) * self.dt, -100.0, 10000.0)  # 21차: 걷기가 확인된 우리 값으로 복귀
        # jax.debug.print('STEP REWARD: {}',reward)
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        # 행동 이력 갱신은 위(_get_obs 앞)로 옮겼다. 여기서 다시 밀면 한 스텝에
        # 두 번 밀려 이력이 통째로 어긋난다.
        state.info["rng"], cmd_rng, jitter_rng, head_rng = jax.random.split(state.info["rng"], 4)
        # 이동 명령은 500 스텝마다, 머리 목표는 제 간격마다 바꾼다.
        target = state.info["command_target"]
        target = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng).at[3:].set(target[3:]),
            target,
        )
        new_head, head_steps, head_alpha = self._sample_head(head_rng)
        renew = state.info["head_steps"] <= 0
        state.info["command_target"] = target.at[3:].set(jp.where(renew, new_head, target[3:]))
        state.info["head_steps"] = jp.where(renew, head_steps, state.info["head_steps"] - 1)
        state.info["head_alpha"] = jp.where(renew, head_alpha, state.info["head_alpha"])
        alpha = jp.concatenate([
            jp.full(3, state.info["command_alpha"]), jp.full(4, state.info["head_alpha"])
        ])
        state.info["command"] = self.advance_command(
            state.info["command"],
            state.info["command_target"],
            alpha,
            jitter_rng,
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        # 래퍼(full_reset=False)는 끝난 환경의 물리와 관측만 첫 상태로 돌리고 info 는 남긴다.
        # 그대로 두면 다시 선 직후 행동 이력·목표각·위상이 넘어지기 직전 값이다. 첫 관측과
        # 맞도록 info 도 첫 상태로 돌린다. 에피소드 길이로 잘릴 때(래퍼 종료)도 같다.
        state.info["episode_step"] += 1
        restart = done | (state.info["episode_step"] >= self._config.episode_length)
        zero = lambda x: jp.where(restart, jp.zeros_like(x), x)
        for k in ("last_act", "last_last_act", "last_last_last_act",
                  "imu_history", "contact_history", "feet_air_time", "swing_peak",
                  "imitation_phase", "yaw_history", "vy_history", "vx_history"):
            state.info[k] = zero(state.info[k])
        state.info["last_contact"] = state.info["last_contact"] & ~restart
        for k in ("motor_targets", "applied_targets", "servo_pos"):
            state.info[k] = jp.where(restart, self._default_actuator, state.info[k])
        state.info["servo_vel"] = zero(state.info["servo_vel"])
        state.info["force_scale"] = jp.where(restart, 1.0, state.info["force_scale"])
        state.info["target_history"] = jp.where(
            restart,
            jp.tile(self._default_actuator, self._config.noise_config.action_max_delay),
            state.info["target_history"],
        )
        for k in ("imitation_i", "push_step", "episode_step", "step", "window_count"):
            state.info[k] = jp.where(restart, 0, state.info[k])
        state.info["action_delay"] = jp.where(
            restart, state.info["action_delay_base"], state.info["action_delay"]
        )
        for k in ("command", "command_target"):
            state.info[k] = jp.where(restart, state.info["command_init"], state.info[k])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        # 전복만으로는 무릎 접고 주저앉아 버티기를 못 거른다. 임계 0.11 m 는 홈 높이
        # 0.165 m 의 67 % 이고, reset 이 높이를 안 건드리므로 직후 오작동할 여지가 없다.
        crouch_termination = self.get_floating_base_qpos(data.qpos)[2] < 0.11
        return (
            fall_termination
            | crouch_termination
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        # accelerometer[0] += 1.3 # TODO testing
        accelerometer.at[0].set(accelerometer[0] + 1.3)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        # Handle IMU delay
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        # joint_angles = data.qpos[7:]

        # Handling backlash
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)

        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)

        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        # joint_vel = data.qvel[6:]
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        # 접촉 관측 랜덤화 (P0-1).
        #
        # 여기서 만든 오염된 값은 관측에만 들어간다. 보상 쪽(first_contact,
        # feet_air_time, reward_imitation)은 이 함수 밖에서 깨끗한 contact 를
        # 그대로 받는다. 보상까지 오염시키면 정책이 배울 신호 자체가 망가진다.
        noise_level = self._config.noise_config.level
        noisy_contact = contact.astype(jp.float32)

        # 1) 판정이 틀린다
        info["rng"], flip_rng = jax.random.split(info["rng"])
        flip = jax.random.bernoulli(
            flip_rng,
            p=self._config.noise_config.scales.contact_flip * noise_level,
            shape=noisy_contact.shape,
        )
        noisy_contact = jp.where(flip, 1.0 - noisy_contact, noisy_contact)

        # 2) 늦게 알려준다. 행동/IMU 지연과 같은 이력 버퍼 방식이다.
        contact_history = (
            jp.roll(info["contact_history"], 2).at[:2].set(noisy_contact)
        )
        info["contact_history"] = contact_history
        info["rng"], delay_rng = jax.random.split(info["rng"])
        contact_idx = jax.random.randint(
            delay_rng,
            (1,),
            minval=self._config.noise_config.contact_min_delay,
            maxval=self._config.noise_config.contact_max_delay,
        )
        noisy_contact = contact_history.reshape((-1, 2))[contact_idx[0]]

        # 3) 한쪽이 에피소드 내내 죽어 있다
        noisy_contact = jp.where(info["contact_dropped"], 0.0, noisy_contact)

        linvel = self.get_local_linvel(data)
        # info["rng"], noise_rng = jax.random.split(info["rng"])
        # noisy_linvel = (
        #     linvel
        #     + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        #     * self._config.noise_config.level
        #     * self._config.noise_config.scales.linvel
        # )

        state = jp.hstack(
            [
                # noisy_linvel,  # 3
                # noisy_gyro,  # 3
                # noisy_gravity,  # 3
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                info["command"],  # 7 (이동 3 + 머리 4)
                noisy_joint_angles - self._default_actuator,  # 14
                noisy_joint_vel * self._config.dof_vel_scale,  # 14
                info["last_act"],  # 14
                info["last_last_act"],  # 14
                info["last_last_last_act"],  # 14
                info["motor_targets"],  # 14
                noisy_contact,  # 2 (P0-1: 랜덤화된 접촉 추정)
                # info["current_reference_motion"],
                # info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,  # 1
                data.actuator_force,  # 10
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
                info["current_reference_motion"],
                info["imitation_i"],
                info["imitation_phase"],
                self._mean_yaw(info),  # 1 (tracking_ang_vel 이 견주는 값)
                self._mean_vy(info),  # 1 (tracking_lin_vel 좌우 성분이 견주는 값)
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _servo_step(
        self, data: mjx.Data, goal: jax.Array, info: dict[str, Any]
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        """서보 안의 가속 상한을 흉내 내며 물리 스텝을 n_substeps 번 돈다.

        서보는 받은 목표각으로 내부 기준을 바로 옮기지 않고, 가속 상한 안에서 목표에 멈춰 설 수
        있는 속도로 옮긴다. PD 는 그 내부 기준을 쫓는다. 뷰어·평가 도구의 ServoProfile 과 같은 식이다.
        """
        vd = self._config.voltage_drop
        knee = self._config.knee_kp_ramp is not None
        # 이 에피소드의 몸통 휨 강성 · 감쇠를 넣은 모델.
        model = self.mjx_model.replace(
            jnt_stiffness=self.mjx_model.jnt_stiffness.at[self._flex_jnt].set(info["flex_k"]),
            dof_damping=self.mjx_model.dof_damping.at[self._flex_dof].set(info["flex_d"]),
        )
        if self._config.servo_max_accel <= 0 and not vd.enable and not knee:
            data = mjx_env.step(model, data, goal, self.n_substeps)
            return data, goal, jp.zeros_like(goal)
        accel = info["servo_accel"]
        dt = self._config.sim_dt
        # 위치 서보의 힘은 kp × (ctrl − q) 라 무릎만 ctrl − q 에 배율을 곱하면 kp 를 바꾼 것과 같다.
        knee_gain = jp.ones(self._mj_model.nu).at[self._knee_ids].set(info["knee_kp"])

        def substep(carry, _):
            d, pos, vel, scale = carry
            if self._config.servo_max_accel > 0:
                err = goal - pos
                want = jp.sign(err) * jp.sqrt(2.0 * accel * jp.abs(err))
                vel = vel + jp.clip(want - vel, -accel * dt, accel * dt)
                pos = pos + vel * dt
            else:
                pos = goal
            ctrl = pos
            if vd.enable:
                # 직전 서브스텝의 바깥 하중으로 서보 전류와 서보별 전압을 구한 뒤, 처진 전압 비율만큼 힘의
                # 기울기와 상한을 줄인다. 위치 서보의 힘은 kp × (ctrl − q) 라 ctrl − q 에 비율을 곱하고
                # q ± 줄어든 상한 / kp 로 자르면 된다.
                ext = (
                    d.actuator_force
                    - self._act_damping * d.qvel[self._act_dof]
                    - self._act_armature * d.qacc[self._act_dof]
                )
                current = vd.cur_gain * jp.maximum(jp.abs(ext) - vd.cur_hold, 0.0)
                port_current = jp.zeros(2).at[self._actuator_port].add(current)
                volt = vd.v_nominal - port_current @ info["port_r"]
                scale = jp.clip(volt / vd.v_nominal, vd.min_scale, 1.0)
                q = self.get_actuator_joints_qpos(d.qpos)
                reach = self._reach_line * scale
                ctrl = q + jp.clip((ctrl - q) * scale, -reach, reach)
            if knee:
                q = self.get_actuator_joints_qpos(d.qpos)
                ctrl = q + (ctrl - q) * knee_gain
            d = mjx.step(model, d.replace(ctrl=ctrl))
            return (d, pos, vel, scale), None

        (data, pos, vel, scale), _ = jax.lax.scan(
            substep,
            (data, info["servo_pos"], info["servo_vel"], info["force_scale"]),
            (),
            self.n_substeps,
        )
        info["force_scale"] = scale
        return data, pos, vel

    def _mean_yaw(self, info: dict[str, Any]) -> jax.Array:
        """직전 한 걸음 주기 동안의 자이로 z 평균. 채운 칸만 평균한다."""
        return jp.sum(info["yaw_history"]) / jp.maximum(info["window_count"], 1)

    def _mean_vy(self, info: dict[str, Any]) -> jax.Array:
        """직전 한 걸음 주기 동안의 몸통 좌우 속도 평균. 채운 칸만 평균한다."""
        return jp.sum(info["vy_history"]) / jp.maximum(info["window_count"], 1)

    def _mean_vx(self, info: dict[str, Any]) -> jax.Array:
        """직전 한 걸음 주기 동안의 몸통 앞뒤 속도 평균. 채운 칸만 평균한다."""
        return jp.sum(info["vx_history"]) / jp.maximum(info["window_count"], 1)

    def _imitation_reference(self, info: dict[str, Any]) -> jax.Array:
        d = self._config.reward_config.imitation_delay
        if not USE_IMITATION_REWARD or d == 0:
            return info["current_reference_motion"]
        c = info["command"]
        return self.PRM.get_reference_motion(c[0], c[1], c[2], info["imitation_i"] - d)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.

        ret = {
            # 좌우 성분은 한 걸음 평균 속도로, 불감대 없이 견준다. 순간값이면 걸음마다의 옆
            # 흔들림까지 벌해 불감대 0.03 이 필요했고, 그 불감대가 작은 옆 명령을 서 있어도
            # 0.96 점으로 만들었다(research/04 4장). 전진 성분은 순간값 그대로다.
            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"],
                self.get_local_linvel(data).at[1].set(self._mean_vy(info)),
                self._config.reward_config.tracking_sigma,
                self._config.reward_config.get("tracking_low_speed_ref", None),
            ),
            # v2_45 신규. 몸통 좌우 · 앞뒤 각속도 제곱합. imitation 안의 같은 성분은 0.5 × exp(−2 × 오차²) 라
            # 오차 1 (rad/s)² 위에서 신호가 거의 없는데, 끝값 몸에서 넘어지는 걸음은 1.2 ~ 2.4 다(research/04 45-4).
            "ang_vel_xy": cost_ang_vel_xy(self.get_gyro(data)),
            "ang_vel_x": cost_ang_vel_x(self.get_gyro(data)),
            # 걸음 안 회전 흔들림. 회전 추적을 한 걸음 평균으로 매긴 뒤 v2_3 이 걸음마다 몸통을
            # ±7 ~ 8° 비틀었다. 평균에서 벗어난 순간 회전 속도의 제곱이다 (research/04 4장).
            "yaw_sway": cost_yaw_sway(self.get_gyro(data)[2], self._mean_yaw(info)),
            "tracking_ang_vel": reward_tracking_ang_vel(
                info["command"],
                self.get_gyro(data).at[2].set(self._mean_yaw(info)),
                self._config.reward_config.tracking_ang_vel_sigma,
            ),
            # 같은 오차를 좁게 본다. 작은 쏠림을 넓은 항은 거의 벌하지 않는다 (research/04 5장).
            "yaw_drift": reward_tracking_ang_vel(
                info["command"],
                self.get_gyro(data).at[2].set(self._mean_yaw(info)),
                self._config.reward_config.yaw_drift_sigma,
            ),
            # 22 차에 되살린다. get_gravity 는 몸통 좌표계에서 본 중력 방향이므로
            # 그 xy 성분이 0 이면 몸통이 똑바로 서 있다는 뜻이다. 레퍼런스와 비교할
            # 것이 없어 배열 인덱스 문제도 없다.
            "orientation": cost_orientation(self.get_gravity(data)),
            "torques": cost_torques_abs(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "action_acc": cost_action_acc(action, info["last_act"], info["last_last_act"]),
            "alive": reward_alive(),
            "imitation": reward_imitation(  # FIXME, this reward is so adhoc...
                self.get_floating_base_qpos(data.qpos),  # floating base qpos
                self.get_floating_base_qvel(data.qvel),  # floating base qvel
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                contact,
                self._imitation_reference(info),
                info["command"],
                USE_IMITATION_REWARD,
            ),
            "cmd_idle": cost_cmd_idle(
                info["command"],
                info["command_target"],
                jp.stack([self._mean_vx(info), self._mean_vy(info)]),
                info["step"],
                int(round(self._config.reward_config.cmd_idle_delay / self.dt)),
                self._config.reward_config.cmd_idle_ratio,
            ),
            "stand_still": cost_stand_still(
                # info["command"], data.qpos[7:], data.qvel[6:], self._default_pose
                info["command"],
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                self._default_actuator,
                # 정지 중에도 머리는 명령을 따른다. 머리를 넣으면 head_pos 와 충돌한다.
                ignore_head=True,
            ),
            # 24 차 신규. 다리 10 축만 본다 — _weights 가 10 개이고 머리는 head_pos 가 맡는다.
            "pose": cost_pose(
                jp.concatenate(
                    [
                        self.get_actuator_joints_qpos(data.qpos)[:5],
                        self.get_actuator_joints_qpos(data.qpos)[9:],
                    ]
                ),
                jp.concatenate([self._default_actuator[:5], self._default_actuator[9:]]),
                self._weights,
            ),
            # 23 차 신규. contact 는 이 함수에 인자로 들어온 깨끗한 값이다
            # (관측용으로 뒤집고 떨어뜨린 값이 아니다).
            "feet_slip": cost_feet_slip(
                contact,
                data.sensordata[self._foot_linvel_sensor_adr],
            ),
            # 23 차 신규. feet_air_time 은 이번 스텝에서 이미 갱신됐고 접지로
            # 0 이 되는 것은 _get_reward 뒤이므로, 착지 순간의 체공 시간이 그대로
            # 들어온다.
            "feet_air_time": reward_feet_air_time(
                info["feet_air_time"],
                first_contact,
                info["command"],
            ),
            # info["applied_targets"] 는 이 data 를 만든 마지막 물리 스텝의 서보 내부 기준각이다.
            # step() 에서 _servo_step 직후에 대입되고 _get_reward 는 그 뒤에 불리므로 짝이 맞는다.
            "torque_overdemand": cost_torque_overdemand(
                info["applied_targets"],
                self.get_actuator_joints_qpos(data.qpos),
                self._reach_line,
            ),
            # 게이트가 없다. 정지와 이동 양쪽에서 켜진다.
            "head_vel": cost_head_vel(self.get_actuator_joints_qvel(data.qvel)),
            # 14 차부터 게이트가 없다 - 걷는 중에도 켜진다(rewards.py 주석).
            "head_pos": cost_head_pos(
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                info["command"],
            ),
        }

        return ret

    def advance_command(
        self, cmd: jax.Array, target: jax.Array, alpha: jax.Array, rng: jax.Array
    ) -> jax.Array:
        """명령을 목표 쪽으로 한 스텝 옮기고 흔들림을 얹는다 (spec/01 2장).

        흔들림은 목표가 0 이 아닌 이동 채널에만 얹는다. 정지 명령에서 명령이 떨면 정지
        유지(7장)를 배울 수 없고, 제자리 회전의 전진·옆 채널도 0 이어야 한다. 머리 4 채널은
        얹지 아니한다.
        """
        moving = jp.any(target[:3] != 0.0)
        # 이동 채널은 목표가 0 이 아닌 채널에만 흔들림을 넣는다(제자리 회전의 vx·vy 는 0 이어야 한다).
        # 머리 채널은 넣지 아니한다. 범위 3 % 가 매 스텝 0.02 rad, 1 rad/s 의 떨림이 되어
        # head_pos 가 그 떨림을 따라가게 만들었다(research/04 1장).
        mask = jp.concatenate([target[:3] != 0.0, jp.zeros(4, dtype=bool)])
        jitter = jax.random.normal(rng, shape=cmd.shape) * self._cmd_jitter_std
        nxt = cmd + alpha * (target - cmd) + jp.where(mask, jitter, 0.0)
        # 정지 목표로 줄어들다가 정지 판정 문턱(이동 명령 크기 0.01, rewards.py)
        # 아래로 들어오면 0 으로 맞춘다. 실물의 정지 명령은 정확히 0 이다.
        settled = ~moving & (jp.linalg.norm(nxt[:3]) < 0.01)
        nxt = nxt.at[:3].set(jp.where(settled, 0.0, nxt[:3]))
        return jp.clip(nxt, self._cmd_lower, self._cmd_upper)

    def _sample_head(self, rng: jax.Array):
        """새 머리 목표, 그 목표를 유지할 스텝 수, 따라가는 1 차 지연 계수."""
        c = self._config.command_config
        pos_rng, home_rng, int_rng, tau_rng = jax.random.split(rng, 4)
        head = jax.random.uniform(
            pos_rng, (4,), minval=self._head_cmd_lower, maxval=self._head_cmd_upper
        )
        head = jp.where(jax.random.bernoulli(home_rng, c.head_home_prob), 0.0, head)
        steps = jp.round(
            jax.random.uniform(int_rng, minval=c.head_interval_range[0], maxval=c.head_interval_range[1])
            / self.dt
        ).astype(jp.int32)
        tau = jax.random.uniform(tau_rng, minval=c.head_tau_range[0], maxval=c.head_tau_range[1])
        return head, steps, 1.0 - jp.exp(-self.dt / tau)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4, rng5, rng6, rng7 = jax.random.split(rng, 7)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        # 머리 4 채널은 정지든 보행이든 항상 뽑는다 — 12 차에 정지 구간에만 줬더니
        # 정책이 이것을 "지금은 서 있다" 신호로 읽어 이동이 죽었다.
        head_cmd = jax.random.uniform(
            rng5,
            shape=(4,),
            minval=self._head_cmd_lower,
            maxval=self._head_cmd_upper,
        )

        # 전진만 · 후진만은 제 방향 범위에서 고르게 뽑는다.
        fwd_x = jax.random.uniform(rng6, minval=0.0, maxval=self._config.lin_vel_x[1])
        back_x = jax.random.uniform(rng7, minval=self._config.lin_vel_x[0], maxval=0.0)

        # 이동 명령 종류를 먼저 뽑는다: 정지, 제자리 회전, 전진만, 후진만, 옆만, 세 축 섞음 (mode_probs).
        mode = jax.random.choice(
            rng4, 6, p=jp.array(self._config.command_config.mode_probs)
        )
        zero = jp.zeros(())
        move_cmd = jp.select(
            [mode == 0, mode == 1, mode == 2, mode == 3, mode == 4],
            [
                jp.zeros(3),
                jp.stack([zero, zero, ang_vel_yaw]),
                jp.stack([fwd_x, zero, zero]),
                jp.stack([back_x, zero, zero]),
                jp.stack([zero, lin_vel_y, zero]),
            ],
            jp.stack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )

        return jp.hstack([move_cmd, head_cmd])
