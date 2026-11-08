# AIRookie_Playground

> Reinforcement-learning environment that trains the walking policy of AI Rookie, a small bipedal companion robot.

English | [한국어](README.ko.md)

## What it solves

AI Rookie walks with a neural-network policy, not with hand-tuned gaits. The policy has to work on a real robot whose servos lag, whose shell has play, and whose mass is only known by measurement. This repository trains that policy in MuJoCo and folds real-robot measurements back into the simulation so that the next training round starts closer to the hardware.

## Who it is for

- People building a small bipedal robot on the Open Duck Mini v2 lineage who want a working training setup with a custom exterior
- Anyone curious how sim-to-real gaps (servo delay, backlash, center-of-mass error) were handled in a student robot project

## Features

- PPO training on MuJoCo MJX or MJWarp (`--impl warp`, CUDA) with domain randomization
- Imitation reward from a reference gait produced by the bundled reference motion generator
- Robot model of the AI Rookie exterior: 10 leg joints and 4 head joints, servo backlash included
- Trained policies exported to ONNX under `policies/`
- `real_fit/`: searches center of mass, shell play, head play and floor friction that reproduce recorded real-robot walks
- Evaluation tools: posture, torque, head stress and slow-walk checks for each training round

## How it works

1. The reference motion generator (placo) records one gait cycle as a table, stored in `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`.
2. `runner.py` trains a joystick-conditioned walking policy against that reference in the `flat_terrain_backlash` scene.
3. The policy is exported to ONNX. The onboard software ([AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime)) runs it at 50 Hz with the same observation layout.
4. Real walking records are packed into `real_fit/data/records.npz`, and the fitted physical parameters feed the next round.

## Requirements and usage

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- Training: an NVIDIA GPU with CUDA for the default MJWarp backend. On machines without CUDA use `--impl jax`.

```bash
uv sync
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
uv run tensorboard --logdir=checkpoints
```

Run a trained policy in the MuJoCo viewer:

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o policies/v2/v2_48_300M.onnx
```

Tools that talk to the training server (`tools/round_eval/run.sh`) read `SERVER_USER`, `SERVER_HOST`, `SERVER_PORT` and `KEY_PATH` from a `.env` file one directory above this repository.

## Related repositories

| Repository | Role |
|---|---|
| [AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime) | Onboard software on the Jetson |
| [AIRookie_Sim](https://github.com/h-taek/AIRookie-Sim) | Simulator app that runs the onboard software |

## Credits and license

This repository is derived from [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground), the [Open Duck reference motion generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) and [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) by Antoine Pirrone, Steve Nguyen and contributors. See [NOTICE](NOTICE) for details. Parts derived from Open Duck Mini are under the Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE)).

No open-source license has been chosen yet for the AI Rookie additions; all rights reserved.
