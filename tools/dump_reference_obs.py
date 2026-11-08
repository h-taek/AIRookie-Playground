"""동등성 검증용 기준 관측값을 생성한다.

mujoco_infer.py의 run() 루프에서 뷰어만 제거한 것이다. 스텝 순서를 바꾸면
기준값이 달라지므로 원본과 대조하며 유지한다.
"""

import argparse

import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer


def dump(onnx_path, model_path, reference_data, out_path, num_steps, command):
    infer = MjInfer(model_path, reference_data, onnx_path, standing=False)
    infer.commands = list(command)

    import mujoco

    obs_log = []
    action_log = []
    target_log = []

    counter = 0
    while len(obs_log) < num_steps:
        mujoco.mj_step(infer.model, infer.data)
        counter += 1
        if counter % infer.decimation != 0:
            continue

        infer.imitation_i += 1.0 * infer.phase_frequency_factor
        infer.imitation_i = infer.imitation_i % infer.PRM.nb_steps_in_period
        infer.imitation_phase = np.array(
            [
                np.cos(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
                np.sin(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
            ]
        )

        obs = infer.get_obs(infer.data, infer.commands)
        action = infer.policy.infer(obs)

        infer.last_last_last_action = infer.last_last_action.copy()
        infer.last_last_action = infer.last_action.copy()
        infer.last_action = action.copy()

        infer.motor_targets = infer.default_actuator + action * infer.action_scale
        infer.motor_targets = np.clip(
            infer.motor_targets,
            infer.prev_motor_targets - infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
            infer.prev_motor_targets + infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
        )
        infer.motor_targets = np.clip(infer.motor_targets, infer.ctrl_lower, infer.ctrl_upper)
        infer.prev_motor_targets = infer.motor_targets.copy()
        infer.data.ctrl = infer.motor_targets.copy()

        obs_log.append(np.asarray(obs, dtype=np.float64))
        action_log.append(np.asarray(action, dtype=np.float64))
        target_log.append(np.asarray(infer.motor_targets, dtype=np.float64))

    np.savez(
        out_path,
        obs=np.array(obs_log),
        action=np.array(action_log),
        motor_targets=np.array(target_log),
        command=np.array(command, dtype=np.float64),
    )
    print(f"저장 완료: {out_path}  obs shape={np.array(obs_log).shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml",
    )
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument("--out", type=str, default="tools/reference_obs.npz")
    parser.add_argument("--num_steps", type=int, default=200)
    args = parser.parse_args()

    dump(
        args.onnx_model_path,
        args.model_path,
        args.reference_data,
        args.out,
        args.num_steps,
        command=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
