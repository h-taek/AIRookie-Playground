# 외장 유격 관절(`*_play`)이 있던 모델을 전제로 한다. 지금 모델에는 외장 유격이 없으므로 다시 쓸 때 고친다 (2026-10-21).
"""맞춤용 작업 모델. 학습 장면에 머리 외장 유격 관절 둘을 더한다.

학습 XML 은 건드리지 아니한다. 만든 XML 은 `real_fit/xmls/` 에 쓰고 환경이 그 경로를 읽는다. 정책마다 다른 짝 홈
자세는 키프레임이 아니라 환경마다 넘긴다(`home_qpos`). 키프레임은 학습 장면의 홈 그대로다.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XMLS = ROOT / "playground" / "open_duck_mini_v2" / "xmls"
TRAIN_SCENE = XMLS / "scene_flat_terrain_backlash.xml"
OUT = Path(__file__).resolve().parent / "xmls"
SOLES = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]
SINK = 0.001  # set_home_pose.py 와 같다: 발바닥이 바닥에 1 mm 묻히는 몸통 높이

# 머리 외장 유격. 다리의 `*_play` 와 같이 구동 관절 바로 뒤에 붙는 자유 관절이고 엔코더에 보이지 않는다.
HEAD_PLAY = {"neck_pitch_play": "neck_pitch_assembly", "head_pitch_play": "head_pitch_to_yaw"}
TRUNK = "trunk_assembly"
HEAD = "head_assembly"  # 머리 질량 369.7 g 의 바디. 머리 무게중심 탐색은 이 바디의 ipos 를 옮긴다


def _spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(TRAIN_SCENE))
    for name, body in HEAD_PLAY.items():
        b = spec.body(body)
        drive = b.joints[0]  # 그 바디의 구동 관절(neck_pitch · head_pitch)
        # 다리 외장 유격과 같은 기본값(class shell_play: damping · armature 0.01, frictionloss 0)을 쓴다.
        j = b.add_joint(spec.find_default("shell_play"))
        j.name = name
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.pos = drive.pos
        j.axis = drive.axis
        j.range = [-1e-4, 1e-4]  # 반각은 환경마다 넣는다
    return spec


def _base_height(model: mujoco.MjModel, names: list[str], home: np.ndarray) -> float:
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    for n, v in zip(names, home):
        data.qpos[model.jnt_qposadr[model.joint(n).id]] = v
    mujoco.mj_kinematics(model, data)
    low = []
    for g in SOLES:
        gid = model.geom(g).id
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        low.append((data.geom_xpos[gid] + v @ data.geom_xmat[gid].reshape(3, 3).T)[:, 2].min())
    return -min(low) - SINK


def home_qpos(model: mujoco.MjModel, home: np.ndarray) -> np.ndarray:
    """짝 홈 자세 `home`(구동 14 축, 구동기 순서)의 전체 qpos. 몸통 높이는 발바닥이 1 mm 묻히는 높이다."""
    names = [model.actuator(i).name for i in range(model.nu)]
    q = np.zeros(model.nq)
    q[2], q[3] = _base_height(model, names, home), 1.0
    for n, v in zip(names, home):
        q[model.jnt_qposadr[model.joint(n).id]] = v
    return q


def build(tag: str = "fit", home: np.ndarray | None = None) -> Path:
    """작업 장면을 쓰고 경로를 낸다. `home` 을 주면 키프레임을 그 자세로 바꾼다(확인용)."""
    spec = _spec()
    if home is None:
        path = OUT / f"scene_{tag}.xml"
        OUT.mkdir(exist_ok=True)
        path.write_text(spec.to_xml(), encoding="utf-8")
        return path
    for k in list(spec.keys):
        spec.delete(k)
    model = spec.compile()
    q = home_qpos(model, home)
    key = spec.add_key()
    key.name = "home"
    key.qpos = q.tolist()
    key.ctrl = [float(v) for v in home]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"scene_{tag}.xml"
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path


def load(path: Path) -> mujoco.MjModel:
    """환경과 같은 방식(학습 자산 묶음 + 문자열)으로 읽는다. 메시 경로가 학습 XML 폴더 기준이다."""
    from playground.open_duck_mini_v2.base import get_assets
    return mujoco.MjModel.from_xml_string(Path(path).read_text(encoding="utf-8"), assets=get_assets())


def head_play_joints(model: mujoco.MjModel) -> list[int]:
    return [model.joint(n).id for n in HEAD_PLAY]


def leg_play_joints(model: mujoco.MjModel) -> list[int]:
    return [j for j in range(model.njnt) if model.joint(j).name.endswith("_play") and model.joint(j).name not in HEAD_PLAY]


def backlash_joints(model: mujoco.MjModel) -> list[int]:
    return [j for j in range(model.njnt) if model.joint(j).name.endswith("_backlash")]


if __name__ == "__main__":
    m0 = mujoco.MjModel.from_xml_path(str(TRAIN_SCENE))
    home = m0.keyframe("home").ctrl.copy()
    p = build("check", home)
    m = load(p)
    print(p.name, "nq", m0.nq, "→", m.nq, "njnt", m0.njnt, "→", m.njnt)
    print("머리 유격", [m.joint(j).name for j in head_play_joints(m)], "다리 유격", len(leg_play_joints(m)), "서보 유격", len(backlash_joints(m)))
    print("키 z", m.keyframe("home").qpos[2], "학습 장면 z", m0.keyframe("home").qpos[2])
    print("몸통 무게중심", m.body_ipos[m.body(TRUNK).id], "질량", m.body_mass.sum(), m0.body_mass.sum())
