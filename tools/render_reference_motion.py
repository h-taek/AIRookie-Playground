"""시범 동작 녹화(recordings/*.json)를 학습 모델에 그대로 재생해 영상으로 뽑는다.

생성기는 물리 없이 규칙과 IK 로 만들므로, 재생성하면 눈으로 확인한다. 몸통 위치·자세와
관절각을 녹화 그대로 넣고(물리 없이 기구학만) 대표 동작 9 개를 각각 mp4 로, 그리고 3x3
한 화면으로 묶는다. 칸마다 명령 라벨과 몸통 기준 실제 속도를 적는다.

    uv run python tools/render_reference_motion.py <recordings 폴더> <출력 폴더>
"""

from pathlib import Path
import glob
import json
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
W, H = 480, 360

# 녹화 프레임: 몸통 위치 3 + 몸통 쿼터니언(x, y, z, w) 4 + 관절 16(프리셋 joints 순서)
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "left_antenna", "right_antenna",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
LEGS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]

CLIPS = [
    ("stand", (0.0, 0.0, 0.0)),
    ("forward_max", (0.06, 0.0, 0.0)),
    ("backward", (-0.03, 0.0, 0.0)),
    ("left", (0.0, 0.04, 0.0)),
    ("right", (0.0, -0.04, 0.0)),
    ("turn_left", (0.0, 0.0, 0.3)),
    ("turn_right", (0.0, 0.0, -0.3)),
    ("corner", (0.06, 0.04, 0.3)),
]


def load(folder):
    recs = {}
    for f in glob.glob(str(Path(folder) / "*.json")):
        d = json.load(open(f))
        p = d["Placo"]
        recs[(round(p["dx"], 2), round(p["dy"], 2), round(p["dtheta"], 2))] = d
    return recs


def worst(recs):
    """다리 관절 각속도가 가장 큰 점. 계산이 튄 곳을 보려고 고른다."""
    def peak(d):
        j = np.array(d["Frames"])[:, 7:23][:, LEGS]
        return np.abs(np.diff(j, axis=0)).max() / d["FrameDuration"]
    key = max(recs, key=lambda k: peak(recs[k]))
    return key, np.degrees(peak(recs[key]))


def body_velocity(F, dt, start):
    """정상 구간의 몸통 기준 평균 속도 (vx, vy, yaw rate)."""
    pos, quat = F[start:, :3], F[start:, 3:7]
    rot = R.from_quat(quat)
    v_world = np.diff(pos, axis=0) / dt
    v_body = rot[:-1].inv().apply(v_world)
    yaw = np.unwrap(rot.as_euler("xyz")[:, 2])
    return v_body[:, 0].mean(), v_body[:, 1].mean(), (yaw[-1] - yaw[0]) / (dt * (len(yaw) - 1))


def render(model, d, out, label):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -12, 0.85
    qadr = {n: model.jnt_qposadr[model.joint(n).id] for n in JOINTS if "antenna" not in n}
    font = ImageFont.truetype(FONT, 13)
    F = np.array(d["Frames"])
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(d["FPS"]), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(out)],
        stdin=subprocess.PIPE,
    )
    for frame in F:
        data.qpos[:3] = frame[:3]
        data.qpos[3:7] = np.roll(frame[3:7], 1)  # (x, y, z, w) -> (w, x, y, z)
        for i, n in enumerate(JOINTS):
            if n in qadr:
                data.qpos[qadr[n]] = frame[7 + i]
        mujoco.mj_kinematics(model, data)
        cam.lookat[:] = [frame[0], frame[1], 0.19]
        renderer.update_scene(data, cam)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).multiline_text((6, 4), label, fill=(0, 0, 0), font=font)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def main(folder, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    recs = load(folder)
    key, peak = worst(recs)
    clips = CLIPS + [("worst", key)]
    files = []
    for name, cmd in clips:
        d = recs[cmd]
        p = d["Placo"]
        dt = d["FrameDuration"]
        start = int(round(p["startend_double_support_ratio"] * p["period"] / dt)) + d["FPS"]
        vx, vy, wz = body_velocity(np.array(d["Frames"]), dt, start)
        label = (f"{name}  step {cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}\n"
                 f"body vx {vx:+.3f}  vy {vy:+.3f}  wz {wz:+.3f}")
        if name == "worst":
            label += f"\npeak {peak:.0f} deg/s"
        f = out_dir / f"{len(files) + 1}_{name}.mp4"
        render(model, d, f, label)
        files.append(f)
        print(f"{f.name:22s} 명령 {cmd}  몸통 기준 vx {vx:+.3f} vy {vy:+.3f} wz {wz:+.3f}"
              + (f"  최대 관절 각속도 {peak:.0f} deg/s" if name == "worst" else ""))
    inputs = sum([["-i", str(f)] for f in files], [])
    layout = "|".join(
        f"{'+'.join(['w0'] * (i % 3)) or '0'}_{'+'.join(['h0'] * (i // 3)) or '0'}" for i in range(9)
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex",
         f"xstack=inputs=9:layout={layout}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(out_dir / "0_grid.mp4")],
        check=True,
    )
    print("3x3 묶음", out_dir / "0_grid.mp4")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
