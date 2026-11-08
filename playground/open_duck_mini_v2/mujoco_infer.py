import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from typing import Optional
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2 import actuation
from playground.open_duck_mini_v2.joystick import default_config
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True


class ActionDelay:
    """학습(joystick.py step)과 같은 서보 지연.

    매 스텝 정책이 낸 목표각의 이력 [이번, 1 스텝 전, 2 스텝 전] 에서 지연만큼 지난 목표각을 서보에
    보낸다. 관측의 목표각 칸에는 지연 전 목표각을 넣는다 — 실물 Runtime 이 그렇다.
    "train" 은 학습과 같다 — 기본 지연을 [action_min_delay, action_max_delay) 에서 한 번 뽑고,
    스텝마다 확률 action_delay_jitter 로 ±1 흔들되 직전 지연 + 1 을 넘지 않게 자른다(먼저 낸
    행동이 나중 행동 뒤에 닿지 않는다). 정수를 주면 그 지연에 고정한다. 직진 쏠림이 지연에
    따라 달라진다 (research/04 8장).
    """

    def __init__(self, initial: np.ndarray, mode="train", seed: int = 0):
        noise = default_config().noise_config
        self.min_delay = noise.action_min_delay
        self.max_delay = noise.action_max_delay  # 이력 길이이자 지연 상한(미포함)
        self.jitter = noise.action_delay_jitter
        if mode != "train" and not (self.min_delay <= int(mode) < self.max_delay):
            raise ValueError(f"지연은 train 이거나 {self.min_delay} ~ {self.max_delay - 1} 이다: {mode}")
        self.initial = np.asarray(initial, dtype=np.float64).copy()
        self.mode = mode
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        # 학습 reset 과 같이 이력은 홈 목표각으로 시작한다.
        self.history = np.tile(self.initial, (self.max_delay, 1))
        self.rng = np.random.default_rng(self.seed)
        if self.mode == "train":
            self.base = int(self.rng.integers(self.min_delay, self.max_delay))
        else:
            self.base = int(self.mode)
        self.delay = self.base

    def __call__(self, target: np.ndarray) -> np.ndarray:
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = target
        if self.mode == "train":
            step = 0
            if self.rng.random() < self.jitter:
                step = 1 if self.rng.random() < 0.5 else -1
            d = int(np.clip(self.base + step, self.min_delay, self.max_delay - 1))
            self.delay = min(d, self.delay + 1)
        return self.history[self.delay].copy()


class ServoProfile:
    """서보 안의 가속 상한 흉내. 학습(joystick.py _servo_step)과 같은 식이다.

    받은 목표각으로 내부 기준을 가속 상한 안에서 목표에 멈춰 설 수 있는 속도로 옮기고, 그 기준을
    ctrl 로 낸다. 물리 스텝마다 한 번 부른다. 상한은 공칭값(servo_max_accel)이다.
    """

    def __init__(self, initial: np.ndarray, accel: Optional[float] = None):
        self.initial = np.asarray(initial, dtype=np.float64).copy()
        self.accel = default_config().servo_max_accel if accel is None else accel
        self.reset()

    def reset(self) -> None:
        self.pos = self.initial.copy()
        self.vel = np.zeros_like(self.initial)

    def __call__(self, goal: np.ndarray, dt: float) -> np.ndarray:
        if self.accel <= 0:  # 흉내 끔. 목표각을 그대로 낸다
            self.pos = np.asarray(goal, dtype=np.float64).copy()
            return self.pos.copy()
        err = goal - self.pos
        want = np.sign(err) * np.sqrt(2.0 * self.accel * np.abs(err))
        self.vel = self.vel + np.clip(want - self.vel, -self.accel * dt, self.accel * dt)
        self.pos = self.pos + self.vel * dt
        return self.pos.copy()


class MjInfer(MJInferBase):
    def __init__(
        self,
        model_path: str,
        reference_data: str,
        onnx_model_path: str,
        standing: bool,
        head_direct: bool = False,
        head_from_command: bool = False,
    ):
        super().__init__(model_path)

        self.standing = standing
        # 머리를 명령 채널로 줄지 서보 목표각으로 덮어쓸지. 기본은 명령 채널이다
        # (15 차부터 걷는 중에도 듣는 유일한 경로). --head-direct 를 주면 실물
        # Runtime 이 지금 쓰는 방식으로 바뀌며 정지 중에만 반영된다.
        self.head_direct = head_direct
        # v2_12 부터 학습 조건이다. 머리 4 축 목표각을 정책 행동 대신 머리 명령으로 준다
        # (joystick.py head_from_command). v2_11 까지의 정책은 끄고 돌린다.
        self.head_from_command = head_from_command

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        # 학습과 같게 목표각을 구동기 범위(관절 범위)로 자른다 (joystick.py step).
        self.ctrl_lower = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_upper = self.model.actuator_ctrlrange[:, 1].copy()
        # 학습과 같은 함수로 유도한다. 머리 4 축은 관절 범위를 덮는다 (actuation.py).
        self.action_scale = actuation.action_scale(
            self.model, self.default_actuator, default_config().action_scale
        )

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        # 구동기마다 짝인 유격 관절의 qpos 주소(없으면 -1). 학습 관측은 관절각에 유격각을 더한다
        # (joystick.py _get_obs). 실물 서보 엔코더가 읽는 출력축 각도와 같은 정의다.
        self._backlash_qpos_adr = np.array([
            int(self.model.jnt_qposadr[self.model.joint(n + "_backlash").id])
            if n + "_backlash" in self.backlash_joint_names else -1
            for n in self.actuator_names
        ])

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)
        # v2_25 부터 관측 마지막 칸이 IMU 끊김 표시다(1 정상, 0 끊김). 입력 101 칸인 이전 정책에는 넣지 않는다.
        self.obs_imu_ok = self.policy.ort_session.get_inputs()[0].shape[-1] == 102
        self.imu_ok = 1.0
        # 서보 가속 상한 흉내. ctrl 은 물리 스텝마다 이것을 거쳐 나간다.
        self.servo = ServoProfile(self.default_actuator)

        # 4 차 학습(joystick.py)의 명령 범위와 같아야 한다. 여기 값이 크면
        # 방향키가 정책이 배운 적 없는 명령을 보내고, 그러면 뷰어에서 본 동작이
        # 정책의 실력이 아니라 외삽 결과가 된다.
        #
        # theta 만 학습 범위(±1.111)보다 좁은 ±0.8 이다. 4 차 최종 정책은
        # +0.9 rad/s 부터 3.9 초쯤에 넘어진다(+0.8 까지는 멀쩡). 학습 범위 안에서
        # 넘어지는 것 자체가 결함이지만, 뷰어는 실사용 상한을 보여 주는 쪽이 맞다.
        self.COMMANDS_RANGE_X = [-0.111, 0.222]
        self.COMMANDS_RANGE_Y = [-0.148, 0.148]
        self.COMMANDS_RANGE_THETA = [-0.8, 0.8]  # 학습 범위는 ±1.111

        # 머리 키 범위는 학습이 뽑는 범위와 같은 함수로 유도한다 (actuation.py).
        _lo, _hi = actuation.config_head_command_range(self.model, default_config())
        self.NECK_PITCH_RANGE = [float(_lo[0]), float(_hi[0])]
        self.HEAD_PITCH_RANGE = [float(_lo[1]), float(_hi[1])]
        self.HEAD_YAW_RANGE = [float(_lo[2]), float(_hi[2])]
        self.HEAD_ROLL_RANGE = [float(_lo[3]), float(_hi[3])]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        # 정지 중 머리 조작이 쓰는 서보 목표각(절대 관절각). h 키로 머리 모드를
        # 켜고 조작하면 여기에 쌓이고, 이동 명령이 0 일 때만 적용된다. 걷는 중에는
        # 정책이 머리를 무게중심 이동에 쓰므로 건드리지 않는다.
        self.head_targets = None

        # rad/s. 학습 설정에서 읽는다 — 학습과 뷰어가 다른 상한을 쓰면 뷰어에서 본
        # 동작이 학습 조건이 아니게 된다. 따로 적어 둔 상류값 5.24 가 학습 4.83 과
        # 어긋나 있었다. 평가 도구와 reference_obs 생성도 이 값을 쓴다.
        self.max_motor_velocity = default_config().max_motor_velocity

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_measured_joint_angles(self, data) -> np.ndarray:
        """서보 엔코더가 읽는 관절각. 구동 관절각에 짝인 유격각을 더한다 (joystick.py _get_obs)."""
        angles = self.get_actuator_joints_qpos(data.qpos)
        has_bl = self._backlash_qpos_adr >= 0
        angles[has_bl] += data.qpos[self._backlash_qpos_adr[has_bl]]
        return angles

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        # 학습 측(joystick.py)은 JAX 배열의 .at[].set() 반환값을 버려 이 보정이 적용되지 않으므로 여기서도 넣지 않는다

        joint_angles = self.get_measured_joint_angles(data)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                # gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase,
                [self.imu_ok] if self.obs_imu_ok else [],
            ]
        )

        return obs

    # 키 배치. 이동과 머리를 **모드 없이 동시에** 쓴다. 예전에는 h 로 모드를
    # 토글하고 같은 키를 두 벌로 쓰다 보니, 지금 어느 모드인지 모르면 방향키가
    # 걷기가 될지 고개짓이 될지 알 수 없었다.
    #
    #   W / S      전진 / 후진        A / D    좌이동 / 우이동
    #   Q / E      좌회전 / 우회전     SPACE    전부 정지 (이동 + 머리 중립)
    #   ↑ / ↓      head_pitch        ← / →    head_yaw
    #   Z / X      neck_pitch        C / V    head_roll
    #   P / M      보행 위상 주기 (+/-)
    #
    # 방향키는 상하좌우 넷이라 2 축까지만 담긴다. 남는 neck_pitch 와 head_roll
    # 은 Z/X, C/V 에 둔다.
    KEY_W, KEY_A, KEY_S, KEY_D = 87, 65, 83, 68
    KEY_Q, KEY_E = 81, 69
    KEY_Z, KEY_X, KEY_C, KEY_V = 90, 88, 67, 86
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    KEY_SPACE, KEY_P, KEY_M = 32, 80, 59

    def key_callback(self, keycode):
        # 머리 목표는 축마다 독립으로 유지한다. neck_pitch 를 건드렸다고
        # head_yaw 가 0 으로 돌아가면 네 축을 동시에 잡을 수 없다.
        if self.head_targets is None:
            self.head_targets = np.zeros(4)

        # --- 이동 (정책 명령) ---
        move = {
            self.KEY_W: (0, self.COMMANDS_RANGE_X[1]),
            self.KEY_S: (0, self.COMMANDS_RANGE_X[0]),
            self.KEY_A: (1, self.COMMANDS_RANGE_Y[1]),
            self.KEY_D: (1, self.COMMANDS_RANGE_Y[0]),
            self.KEY_Q: (2, self.COMMANDS_RANGE_THETA[1]),
            self.KEY_E: (2, self.COMMANDS_RANGE_THETA[0]),
        }
        if keycode in move:
            # 이동 축은 배타로 둔다. 전진과 좌이동을 섞으면 시범 동작 격자의
            # 대각 점을 쓰게 되는데 거기가 가장 덜 학습된 구간이다.
            self.commands[0:3] = [0.0, 0.0, 0.0]
            idx, val = move[keycode]
            self.commands[idx] = val
            return

        # --- 머리 ---
        # 경로가 둘이고 정책마다 되는 쪽이 다르다. 명령 채널(기본값)은 관측으로 들어가
        # 정책이 스스로 머리를 보내며 걷는 중에도 듣는 유일한 경로다. --head-direct 는
        # motor_targets 를 덮어써 실물 Runtime 과 같은 방식이고 정지 중에만 반영된다.
        head = {
            self.KEY_UP: (1, self.HEAD_PITCH_RANGE[1]),
            self.KEY_DOWN: (1, self.HEAD_PITCH_RANGE[0]),
            self.KEY_LEFT: (2, self.HEAD_YAW_RANGE[1]),
            self.KEY_RIGHT: (2, self.HEAD_YAW_RANGE[0]),
            self.KEY_Z: (0, self.NECK_PITCH_RANGE[1]),
            self.KEY_X: (0, self.NECK_PITCH_RANGE[0]),
            self.KEY_C: (3, self.HEAD_ROLL_RANGE[1]),
            self.KEY_V: (3, self.HEAD_ROLL_RANGE[0]),
        }
        if keycode in head:
            idx, val = head[keycode]
            if self.head_direct:
                self.head_targets[idx] = val
                if np.linalg.norm(self.commands[:3]) >= 0.01:
                    print("  (서보 직접 방식은 정지 중에만 반영된다 - SPACE 로 멈춰라)")
            else:
                self.commands[3 + idx] = val
            return

        # --- 그 밖 ---
        if keycode == self.KEY_SPACE:
            self.commands[0:3] = [0.0, 0.0, 0.0]
            self.commands[3:7] = [0.0, 0.0, 0.0, 0.0]
            self.head_targets = np.zeros(4)
            return
        if keycode == self.KEY_P:
            self.phase_frequency_factor += 0.1
            return
        if keycode == self.KEY_M:
            self.phase_frequency_factor -= 0.1
            return

    def targets_from_action(self, action):
        """정책 행동을 서보 목표각으로 바꾼다 (joystick.py step 의 앞부분)."""
        targets = self.default_actuator + action * self.action_scale
        if self.head_from_command:
            targets[5:9] = np.asarray(self.commands[3:7])
        return targets

    def run(self):
        print(
            "\n"
            "  이동   W 전진 / S 후진 / A 좌이동 / D 우이동 / Q 좌회전 / E 우회전\n"
            "  머리   ↑↓ head_pitch / ←→ head_yaw / Z X neck_pitch / C V head_roll\n"
            "  정지   SPACE (이동 정지 + 머리 중립)      위상   P / M\n"
            "\n"
            + (
                "  머리 = 서보 목표각 직접(--head-direct). 정지 중에만 반영된다.\n"
                if self.head_direct
                else "  머리 = 명령 채널. 걷는 중에도 반영된다(15 차부터).\n"
            )
            + "  S(후진)는 학습 명령 범위 밖이라 서 있기만 한다.\n"
        )
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                # MuJoCo 뷰어가 알파벳 26 자를 전부 시각화 토글로 쓰고 있고 C++ 쪽이라
                # 막을 수 없다. 시작 상태를 기억해 매 프레임 되돌린다. 되돌릴 수 있는 것은
                # mjVIS_* 뿐이라 W(와이어프레임)와 S(그림자)는 실제로 토글된다.
                # viewer.scn 은 존재하지 않으며 접근하면 뷰어가 죽는다.
                vis_flags = np.array(viewer.opt.flags, dtype=bool)
                counter = 0
                # `while True` 면 창을 닫아도 루프가 돌아 프로세스가 남는다.
                while viewer.is_running():

                    step_start = time.time()

                    self.data.ctrl = self.servo(self.motor_targets, self.sim_dt)
                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        if not self.standing:
                            self.imitation_i += 1.0 * self.phase_frequency_factor
                            self.imitation_i = (
                                self.imitation_i % self.PRM.nb_steps_in_period
                            )
                            # print(self.PRM.nb_steps_in_period)
                            # exit()
                            self.imitation_phase = np.array(
                                [
                                    np.cos(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                    np.sin(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                ]
                            )
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = self.targets_from_action(action)

                        if USE_MOTOR_SPEED_LIMITS:
                            ctrl_dt = self.sim_dt * self.decimation
                            # joystick.py 와 같은 식이다 (속도 클립만).
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity * ctrl_dt,
                                self.prev_motor_targets
                                + self.max_motor_velocity * ctrl_dt,
                            )
                        self.motor_targets = np.clip(
                            self.motor_targets, self.ctrl_lower, self.ctrl_upper
                        )
                        self.prev_motor_targets = self.motor_targets.copy()

                        # 정지 중에만 머리를 직접 잡는다. 걷는 중에는 정책이
                        # 머리를 균형에 쓰므로 덮어쓰면 성능이 떨어진다 —
                        # 실측으로 8 차 정책의 전진이 0.1649 -> 0.0492 m/s 다.
                        if (
                            self.head_direct
                            and self.head_targets is not None
                            and np.linalg.norm(self.commands[:3]) < 0.01
                        ):
                            self.motor_targets[5:9] = self.head_targets
                        # ctrl 은 다음 물리 스텝부터 self.servo 가 motor_targets 를 쫓아 낸다.

                    # 조종키가 건드린 시각화 토글을 원복한다(위 주석 참고).
                    viewer.opt.flags[:] = vis_flags

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pass
        finally:
            # 창을 닫아 정상 종료했을 때도 관측 기록을 남긴다. 예전에는
            # KeyboardInterrupt 경로에만 있어서, 창을 닫으면 그때까지 모은
            # 관측이 그냥 사라졌다.
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))
            print("뷰어를 닫았다.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    # 기본값이 비-backlash 판이었다. kp 가 13.37 대 17.11 이라 뷰어에서 본 동작이
    # 학습 조건이 아니었다 — 게인이 24 % 낮으면 같은 정책이 더 물렁하게 움직인다.
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument(
        "--head-direct",
        dest="head_direct",
        action="store_true",
        default=False,
        help="머리를 서보 목표각으로 직접 덮어쓴다(실물 Runtime 방식, 정지 중에만 반영). "
        "생략하면 명령 채널로 준다.",
    )
    parser.add_argument(
        "--head-from-command",
        dest="head_from_command",
        action="store_true",
        default=False,
        help="머리 목표각을 정책 행동 대신 머리 명령으로 준다(v2_12 부터의 학습 조건).",
    )

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path,
        args.reference_data,
        args.onnx_model_path,
        args.standing,
        args.head_direct,
        args.head_from_command,
    )
    mjinfer.run()
