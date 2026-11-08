import numpy as np
import json
from glob import glob
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--ref_motion", type=str, default="ref_motion")
args = parser.parse_args()

all_files = glob(f"{args.ref_motion}/*.json")


def body_angular_velocity(quats_xyzw, fps):
    """연속 프레임 몸통 자세(x, y, z, w)에서 각속도를 구한다. q0^-1 * q1 의 회전 벡터 x fps.

    생성기 compute_angular_velocity 와 같은 정의(몸통 좌표계)다. 이 버전 이전의 녹화는 그 함수가
    각도를 두 번 곱해 world_angular_vel 칸이 실제의 3 % 였으므로, 그 칸을 읽지 않고 자세에서 다시 구한다.
    """
    q = np.asarray(quats_xyzw, dtype=np.float64)
    x0, y0, z0, w0 = -q[:-1, 0], -q[:-1, 1], -q[:-1, 2], q[:-1, 3]
    x1, y1, z1, w1 = q[1:].T
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    v = np.stack(
        [
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        axis=1,
    )
    v *= np.sign(w + (w == 0))[:, None]
    n = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(n, np.abs(w))
    axis = np.divide(v, n[:, None], out=np.zeros_like(v), where=n[:, None] > 1e-12)
    return axis * (angle * fps)[:, None]


def fit_ref_motion(file):
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    period = data["Placo"]["period"]
    fps = data["FPS"]
    frame_offsets = data["Frame_offset"][0]
    startend_double_support_ratio = data["Placo"]["startend_double_support_ratio"]
    start_offset = int(startend_double_support_ratio * fps)
    nb_steps_in_period = int(period * fps)
    _Y = Y_all[start_offset : start_offset + int(nb_steps_in_period)]
    joints_pos = _Y[:, frame_offsets["joints_pos"] : frame_offsets["left_toe_pos"]]
    joints_vel = _Y[:, frame_offsets["joints_vel"] : frame_offsets["left_toe_vel"]]
    foot_contacts = _Y[
        :, frame_offsets["foot_contacts"] : frame_offsets["foot_contacts"] + 2
    ]
    base_linear_vel = _Y[
        :, frame_offsets["world_linear_vel"] : frame_offsets["world_angular_vel"]
    ]
    root_quat = Y_all[:, frame_offsets["root_quat"] : frame_offsets["root_quat"] + 4]
    base_angular_vel = body_angular_velocity(
        root_quat[start_offset - 1 : start_offset + nb_steps_in_period], fps
    )

    # 학습은 정수 위상 0 ~ N-1 에서만 시범 동작을 꺼낸다. 다항식으로 맞추면 구간 끝이 튀고
    # (27 점에 15 차) 녹화와 최대 16 도 어긋났다(research/02 10장). 녹화 한 주기를 그대로 둔다.
    frames = np.concatenate(
        [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],
        axis=1,
    ).astype(np.float64)

    ret_data = {
        "frames": frames.tolist(),
        "period": period,
        "fps": fps,
        "frame_offsets": frame_offsets,
        "start_offset": start_offset,
        "nb_steps_in_period": nb_steps_in_period,
        "startend_double_support_ratio": startend_double_support_ratio,
    }

    return ret_data


all_coefficients = {}
for file in all_files:
    name = os.path.basename(file).strip(".json")
    tmp = name.split("_")
    name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"

    all_coefficients[name] = fit_ref_motion(file)


pickle.dump(all_coefficients, open("polynomial_coefficients.pkl", "wb"))