"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, standing


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        # 물리 백엔드는 설정을 통해 환경으로 내려간다. base.py 가 이 값으로
        # mjx.put_model(impl=...) 을 부르고, reset 이 make_data(impl=...) 를 부른다.
        self.env_config.impl = args.impl
        overrides = {"impl": args.impl}
        self.env = self.env_file[1](task=args.task, config_overrides=overrides)
        self.eval_env = self.env_file[1](task=args.task, config_overrides=overrides)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    # 기본값은 운영 문서(HANDOFF 4절), README, scripts/setup_before_training.sh 와 같은
    # 3 억 스텝이다. 예전 기본값 1.5 억으로는 인자 없이 돌렸을 때 절반만 학습되었다.
    parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    # M-05b 미확정 상태로 진행하므로 학습 대상을 backlash 모델로 고정한다 (kp 17.11)
    parser.add_argument(
        "--task", type=str, default="flat_terrain_backlash", help="Task to run"
    )
    # 물리 백엔드. warp 가 MJWarp 이고 CUDA 장치를 요구한다. jax 는 기존 MJX 경로다.
    # 같은 A100 에서 MJWarp 이 처리율 1.68 배, 총 벽시계 2.00 배다
    # (docs/archive/research/10_mjwarp_backend.md). 맥처럼 CUDA 가 없는 곳에서는 jax 를 쓴다.
    parser.add_argument(
        "--impl",
        type=str,
        default="warp",
        choices=["jax", "warp"],
        help="Physics backend: jax (MJX) or warp (MJWarp)",
    )
    # 학습 난수 시드. 같은 설정의 결과가 시드 운인지 가릴 때 바꾼다.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
