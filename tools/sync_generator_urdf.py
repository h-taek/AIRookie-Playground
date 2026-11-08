"""시범 동작 생성기의 로봇 모델을 학습 XML 에 맞춘다.

생성기는 자체 URDF 로 무게중심을 계산하고 관절각을 푼다. 학습 XML 이 바뀌면 이것을 돌려
생성기가 같은 몸을 쓰게 한다. 맞추는 것은 다섯이다.

1. 링크 질량·무게중심·관성 — XML 바디 값을 URDF 링크 좌표계로 옮겨 적는다.
   오른쪽 정강이·발은 URDF 링크 원점이 XML 보다 회전축 방향으로 36.9 mm 밀려 있어
   (기구학은 같다) 그대로 복사하면 안 되고 좌표 변환을 거친다.
   XML 에 없는 귀 받침 링크는 질량을 머리에 두었으므로 거의 0 으로 둔다.
2. 관절 한계 — XML 14 축의 범위.
3. 발바닥 기준점 — XML `left_foot`·`right_foot` 사이트 위치.
4. 프리셋 머리 각도 — 학습 장면의 `home` 키프레임 목·머리 4 축.
5. 프리셋 발 간격(feet_spacing) — FEET_SPACING. 다리를 곧게 편 자연 간격(0 자세에서 두 발바닥
   기준점 사이, 178.8 mm)은 참고로 출력만 한다. 상류 0.16 은 자연 간격보다 좁아 딛는 다리가
   늘 안쪽으로 기울었다. 자연 간격 그대로는 격자 모서리(전진 0.06·옆 −0.04)에서 다리가 닿지
   않아 IK 가 튄다(0.179: 104 점 중 2 점, 0.174: 1 점, 0.170: 0 점).

다시 돌려도 결과가 같다.

    uv run python tools/sync_generator_urdf.py
"""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
XMLS = ROOT / "playground/open_duck_mini_v2/xmls"
XML = XMLS / "open_duck_mini_v2.xml"
SCENE = XMLS / "scene_flat_terrain_backlash.xml"
ROBOT = (
    ROOT
    / "tools/reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2"
)
URDF = ROBOT / "open_duck_mini_v2.urdf"
PRESETS = sorted((ROBOT / "placo_presets").glob("*.json"))

BASE = "trunk_assembly"
FOOT_FRAMES = ["left_foot", "right_foot"]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
FEET_SPACING = 0.170
TINY_MASS = 1e-6
TINY_INERTIA = 1e-9


def origin_matrix(origin):
    xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def fmt(values):
    return " ".join(repr(float(v)) for v in values)


def urdf_link_frames(robot):
    """영 자세에서 각 링크 좌표계를 BASE 좌표계로 낸다."""
    frames = {BASE: np.eye(4)}
    joints = robot.findall("joint")
    while True:
        added = False
        for j in joints:
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            if parent in frames and child not in frames:
                frames[child] = frames[parent] @ origin_matrix(j.find("origin"))
                added = True
        if not added:
            return frames


def xml_in_base(model, data):
    """XML 바디 좌표계·관성 좌표계를 BASE 좌표계로 내는 함수를 돌려준다."""
    b = model.body(BASE).id
    T_base = np.eye(4)
    T_base[:3, :3] = data.xmat[b].reshape(3, 3)
    T_base[:3, 3] = data.xpos[b]
    inv = np.linalg.inv(T_base)
    return lambda p: (inv @ np.r_[p, 1.0])[:3], inv[:3, :3]


def set_inertial(link, mass, com, inertia):
    inertial = link.find("inertial")
    inertial.find("origin").set("xyz", fmt(com))
    inertial.find("origin").set("rpy", "0 0 0")
    inertial.find("mass").set("value", repr(float(mass)))
    el = inertial.find("inertia")
    for key, (i, k) in {
        "ixx": (0, 0), "iyy": (1, 1), "izz": (2, 2),
        "ixy": (0, 1), "ixz": (0, 2), "iyz": (1, 2),
    }.items():
        el.set(key, repr(float(inertia[i, k])))


def sync_urdf():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    to_base, rot_base = xml_in_base(model, data)

    tree = ET.parse(URDF, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    robot = tree.getroot()
    frames = urdf_link_frames(robot)
    body_names = {model.body(i).name for i in range(model.nbody)}

    total = 0.0
    for link in robot.findall("link"):
        name = link.get("name")
        if link.find("inertial") is None:
            continue
        if name not in body_names:
            set_inertial(link, TINY_MASS, np.zeros(3), np.eye(3) * TINY_INERTIA)
            total += TINY_MASS
            continue
        b = model.body(name).id
        R_link = frames[name][:3, :3]
        com = R_link.T @ (to_base(data.xipos[b]) - frames[name][:3, 3])
        R_i = R_link.T @ rot_base @ data.ximat[b].reshape(3, 3)
        inertia = R_i @ np.diag(model.body_inertia[b]) @ R_i.T
        set_inertial(link, model.body_mass[b], com, inertia)
        total += model.body_mass[b]

    for j in robot.findall("joint"):
        name = j.get("name")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j.get("type") == "revolute" and jid >= 0:
            lo, hi = model.jnt_range[jid]
            j.find("limit").set("lower", repr(float(lo)))
            j.find("limit").set("upper", repr(float(hi)))
        if j.find("child").get("link") in FOOT_FRAMES:
            site = model.site(j.find("child").get("link")).id
            parent = frames[j.find("parent").get("link")]
            p = np.linalg.inv(parent) @ np.r_[to_base(data.site_xpos[site]), 1.0]
            j.find("origin").set("xyz", fmt(p[:3]))

    tree.write(URDF, encoding="utf-8", xml_declaration=True)
    xml_total = float(sum(model.body_mass))
    print(f"URDF 질량 합 {total:.4f} kg / XML {xml_total:.4f} kg")
    assert abs(total - xml_total) < 1e-4


def sync_presets():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    home = model.keyframe("home")
    names = [model.actuator(i).name for i in range(model.nu)]
    head = {n: round(float(np.degrees(home.ctrl[names.index(n)])), 2) for n in HEAD_JOINTS}
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    natural = float(np.linalg.norm(data.site_xpos[model.site(FOOT_FRAMES[0]).id]
                                   - data.site_xpos[model.site(FOOT_FRAMES[1]).id]))
    spacing = FEET_SPACING
    for path in PRESETS:
        preset = json.loads(path.read_text())
        preset["joint_angles"].update(head)
        preset["feet_spacing"] = spacing
        path.write_text(json.dumps(preset, indent=2) + "\n")
    print("프리셋 머리 각도", head, f"발 간격 {spacing} (자연 간격 {natural:.4f})")


if __name__ == "__main__":
    sync_urdf()
    sync_presets()
