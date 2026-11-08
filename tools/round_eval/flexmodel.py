"""학습 모델에 몸통 휨(숨은 스프링 관절)을 넣은 장면을 만든다. 다리 뿌리(hip_roll_assembly)마다 좌우(x) · 앞뒤(y) 경첩 둘.
엔코더 관측에는 들어가지 않는다. make(kr, dr, kp, dp) -> 장면 파일 경로. base.get_assets 를 바꿔 끼운다."""
import re, tempfile
from pathlib import Path
import mujoco, numpy as np
from playground.open_duck_mini_v2 import base, constants
X = constants.ROOT_PATH / "xmls"
_orig_assets = base.get_assets
_extra = {}
def _assets():
    a = _orig_assets(); a.update(_extra); return a
base.get_assets = _assets
OUT = Path(tempfile.gettempdir()) / "flexscene"

def make(kr, dr, kp, dp, tag=None):
    body = (X / "open_duck_mini_v2_backlash.xml").read_text()
    for side in ("left", "right"):
        j = (f'<joint name="{side}_flex_roll" type="hinge" pos="0 0 0" axis="1 0 0" stiffness="{kr}" damping="{dr}" armature="0.0005" frictionloss="0"/>\n'
             f'          <joint name="{side}_flex_pitch" type="hinge" pos="0 0 0" axis="0 1 0" stiffness="{kp}" damping="{dp}" armature="0.0005" frictionloss="0"/>\n          ')
        key = f'<joint name="{side}_hip_yaw" '
        assert body.count(key) == 1; body = body.replace(key, j + key)
    _extra["open_duck_mini_v2_flex.xml"] = body.encode()
    scene0 = (X / "scene_flat_terrain_backlash.xml").read_text()
    scene = scene0.replace("open_duck_mini_v2_backlash.xml", "open_duck_mini_v2_flex.xml")
    old = mujoco.MjModel.from_xml_string(scene0, assets=_assets())
    nokey = re.sub(r"<keyframe>.*?</keyframe>", "", scene, flags=re.S)
    new = mujoco.MjModel.from_xml_string(nokey, assets=_assets())
    keys = []
    for k in range(old.nkey):
        q = new.qpos0.copy()
        for jid in range(old.njnt):
            n = old.joint(jid).name; a0 = old.jnt_qposadr[jid]; a1 = new.jnt_qposadr[new.joint(n).id]
            w = 7 if old.jnt_type[jid] == 0 else (4 if old.jnt_type[jid] == 1 else 1)
            q[a1:a1 + w] = old.key_qpos[k][a0:a0 + w]
        keys.append(f'<key name="{old.key(k).name}" qpos="{" ".join(f"{v:.9g}" for v in q)}" ctrl="{" ".join(f"{v:.9g}" for v in old.key_ctrl[k])}"/>')
    scene = nokey.replace("</mujoco>", "<keyframe>" + "".join(keys) + "</keyframe>\n</mujoco>")
    OUT.mkdir(exist_ok=True)
    p = OUT / f"scene_flex_{tag or f'{kr}_{dr}_{kp}_{dp}'}.xml"; p.write_text(scene)
    return str(p)
