"""홈 자세를 시범 동작 생성기가 서 있으려는 자세로 맞춘다.

생성기 프리셋(medium.json)의 서 있는 목표 — 무게중심 높이 walk_com_height, 몸통 피치
walk_trunk_pitch, 발 간격 feet_spacing, 무게중심이 두 발 중앙 위, 발의 앞뒤·방향 기울기 0 — 를
학습 모델에서 왼다리로 풀고 오른다리는 미러로 둔다. 발의 좌우 기울기는 생성기처럼 풀어 둔다
(발목 좌우 관절이 없다). 머리 4 축은 지금 홈 그대로다.

생성기 안에는 대칭으로 멈춘 순간이 없어 녹화에서 꺼내지 않는다. 녹화 첫 프레임은 첫 걸음의
체중 이동이 이미 시작된 자세이고, 시작 로그의 자세는 덜 풀린 값이다.

각도는 서보 눈금(2π/4096 rad)에 맞춘다. 몸통 높이는 발바닥이 바닥에 1 mm 묻히는 높이다 —
딱 닿게 두면 시작 순간 한 발이 접촉으로 잡히지 않는다(22 차는 1.2 mm).
고치는 곳: 장면 XML 세 개의 home 키프레임(qpos, ctrl), Runtime src/config/home_pose.json.
home_pose.json 에는 홈과 함께 정책과 짝인 값 — 행동 배율, 관절 범위, 머리 명령 범위 — 을 학습 장면에서
학습과 같은 함수(actuation.py)로 구해 적는다. Runtime 은 XML 을 읽지 않으므로 이 파일로 받는다.

    uv run python tools/set_home_pose.py [--dry-run]
"""

from pathlib import Path
import argparse
import json
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from playground.open_duck_mini_v2 import actuation
from playground.open_duck_mini_v2.joystick import default_config

ROOT = Path(__file__).resolve().parents[1]
XMLS = ROOT / "playground/open_duck_mini_v2/xmls"
SCENES = ["scene_flat_terrain_backlash.xml", "scene_flat_terrain.xml", "scene_rough_terrain_backlash.xml"]
TRAIN_SCENE = XMLS / SCENES[0]  # runner.py 의 기본 task
HOME_JSON = ROOT.parent / "AIRookie_Runtime/src/config/home_pose.json"
TICK = 2 * np.pi / 4096
PRESET = ROOT / "tools/reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_presets/medium.json"
MIRROR = np.array([-1, -1, -1, 1, 1])  # 오른다리 = 엉덩이 세 축 부호 반대, 무릎·발목 같은 부호

LEGS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
SOLES = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]
SINK = 0.001


def stance():
    """프리셋의 서 있는 목표를 만족하는 왼다리 5 축(rad)."""
    preset = json.loads(PRESET.read_text())
    model = mujoco.MjModel.from_xml_path(str(XMLS / SCENES[1]))
    data = mujoco.MjData(model)
    key = model.keyframe("home")
    left, right = model.site("left_foot").id, model.site("right_foot").id
    base = model.body("base").id
    adr = lambda n: model.jnt_qposadr[model.joint(n).id]
    data.qpos[:] = model.qpos0
    data.qpos[:3] = 0
    mujoco.mj_kinematics(model, data)
    flat = data.site_xmat[left].reshape(3, 3).copy()  # 몸통 수평 0 자세의 발 = 평평한 발
    trunk = np.roll(R.from_euler("y", preset["walk_trunk_pitch"], degrees=True).as_quat(), 1)

    def residual(q):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = 0
        data.qpos[3:7] = trunk
        for i in range(model.nu):
            data.qpos[adr(model.actuator(i).name)] = key.ctrl[i]
        for n, v in zip(LEGS, np.concatenate([q, q * MIRROR])):
            data.qpos[adr(n)] = v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com, lf, rf = data.subtree_com[base], data.site_xpos[left], data.site_xpos[right]
        tilt = R.from_matrix(data.site_xmat[left].reshape(3, 3) @ flat.T).as_euler("xyz")
        return [com[2] - lf[2] - preset["walk_com_height"], lf[1] - rf[1] - preset["feet_spacing"],
                com[0] - (lf[0] + rf[0]) / 2, tilt[1], tilt[2]]

    sol = least_squares(residual, np.radians([0, 0, -42, 77, -38]), xtol=1e-12, ftol=1e-12)
    assert np.abs(sol.fun).max() < 1e-6, sol.fun
    return sol.x


def home_angles():
    model = mujoco.MjModel.from_xml_path(str(XMLS / SCENES[1]))
    key = model.keyframe("home")
    names = [model.actuator(i).name for i in range(model.nu)]
    angles = {n: float(key.ctrl[i]) for i, n in enumerate(names)}
    q = stance()
    for n, v in zip(LEGS, np.concatenate([q, q * MIRROR])):
        angles[n] = float(np.round(v / TICK) * TICK)
    return names, angles


def base_height(names, angles):
    """발바닥 최저점이 z=-SINK 가 되는 몸통 높이."""
    model = mujoco.MjModel.from_xml_path(str(XMLS / SCENES[1]))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]  # 몸통 원점에서 잰다
    for n in names:
        data.qpos[model.jnt_qposadr[model.joint(n).id]] = angles[n]
    mujoco.mj_kinematics(model, data)
    low = []
    for g in SOLES:
        gid = model.geom(g).id
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        low.append((data.geom_xpos[gid] + v @ data.geom_xmat[gid].reshape(3, 3).T)[:, 2].min())
    return -min(low) - SINK


def key_text(model, names, angles, z):
    """키프레임 qpos 는 모델 주소로 채운다. 유격 · 외장 유격 관절은 0 이다."""
    f = lambda v: f"{v:.9f}".rstrip("0").rstrip(".") if v else "0"
    q = np.zeros(model.nq)
    q[2], q[3] = round(z, 4), 1.0
    for n in names:
        q[model.jnt_qposadr[model.joint(n).id]] = angles[n]
    return " ".join(f(v) for v in q), " ".join(f(angles[n]) for n in names)


def write_scene(path, names, angles, z):
    text = path.read_text(encoding="utf-8")
    note = ("<!-- 홈 자세. tools/set_home_pose.py 가 쓴다 — 시범 동작 생성기 프리셋의 서 있는 목표를 푼 다리\n"
            "         (오른쪽은 왼쪽 미러, 서보 눈금 단위)와 그 전 홈의 머리 4 축. 몸통 높이는 발바닥이 1 mm 묻히는 높이다. -->")
    text, n = re.subn(r"<!--(?:(?!-->).)*-->(\s*<key name=\"home\")", lambda m: note + m.group(1), text, count=1, flags=re.S)
    assert n == 1, path
    # 키프레임 길이가 지금 모델과 다를 수 있어 장면 대신 장면이 포함하는 로봇 모델로 주소를 읽는다.
    robot = path.parent / re.search(r'<include file="([^"]+)"', re.sub(r"<!--.*?-->", "", text, flags=re.S)).group(1)
    model = mujoco.MjModel.from_xml_path(str(robot))
    qpos, ctrl = key_text(model, names, angles, z)
    text, n = re.subn(r'(<key name="home"\s+qpos=")[^"]*(")', rf"\g<1>{qpos}\g<2>", text)
    assert n == 1, path
    text, n = re.subn(r'(<key name="home"[^>]*?ctrl=")[^"]*(")', rf"\g<1>{ctrl}\g<2>", text, flags=re.S)
    assert n == 1, path
    path.write_text(text, encoding="utf-8")
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.key_qpos.shape[1] == m.nq


def pair_values(names, angles):
    """정책과 짝인 값. 학습 장면에서 학습과 같은 함수로 구한다."""
    model = mujoco.MjModel.from_xml_path(str(TRAIN_SCENE))
    assert [model.actuator(i).name for i in range(model.nu)] == names
    home = np.array([angles[n] for n in names])
    cfg = default_config()
    scale = actuation.action_scale(model, home, cfg.action_scale)
    head_lo, head_hi = actuation.config_head_command_range(model, cfg)
    r = lambda v: round(float(v), 9)
    return {
        "_짝": ("정책과 짝인 값. 목표각 = 홈 + 행동 배율 x 행동, 속도 상한 뒤 관절 범위로 자른다. "
               "머리 명령은 학습이 뽑은 범위 안에서만 낸다. 학습 장면에서 actuation.py 로 구한다"),
        "action_scale": {n: r(v) for n, v in zip(names, scale)},
        "joint_range": {n: [r(v) for v in model.actuator_ctrlrange[i]] for i, n in enumerate(names)},
        "head_command_range": {n: [r(lo), r(hi)]
                               for n, lo, hi in zip(actuation.HEAD_ACTUATORS, head_lo, head_hi)},
        "head_from_command": bool(cfg.head_from_command),
    }


def write_json(names, angles):
    data = json.loads(HOME_JSON.read_text(encoding="utf-8"))
    data["_출처"] = ("시범 동작 생성기 프리셋의 서 있는 목표를 학습 모델에서 푼 다리 10 축(오른쪽은 미러), "
                    "머리 4 축은 그 전 홈. 서보 눈금 단위. Playground tools/set_home_pose.py 가 장면 XML home 키프레임과 함께 쓴다")
    data["joints"] = {n: round(angles[n], 9) for n in data["joints"]}
    assert list(data["joints"]) == names
    data.update(pair_values(names, angles))
    HOME_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names, angles = home_angles()
    z = base_height(names, angles)
    for n in LEGS:
        print(f"{n:16s} {np.degrees(angles[n]):+8.3f} deg  {round(angles[n] / TICK):+5d} 눈금")
    print(f"몸통 높이 {z:.4f} m")
    if args.dry_run:
        return
    for s in SCENES:
        write_scene(XMLS / s, names, angles, z)
    write_json(names, angles)
    print("장면 XML 세 개와 home_pose.json 을 고쳤다")


if __name__ == "__main__":
    main()
