"""학습 XML 을 뷰어로 띄운다. 홈 키프레임에서 시작하고 관절 목표를 홈에 묶은 채 물리를 돌린다.

`python -m mujoco.viewer` 는 이 맥(mujoco 3.13.0)에서 창을 만들다 죽는다(_Simulate 예외).
launch_passive 는 돈다. mjpython 을 직접 실행하지 않는다.

    .venv/bin/python .venv/bin/mjpython tools/view_model.py [장면 XML]

뷰어 왼쪽 Rendering 의 Geom group 3 을 켜면 충돌 geom(발바닥)이 보인다.
"""

import sys
import time

import mujoco
import mujoco.viewer

SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"

model = mujoco.MjModel.from_xml_path(sys.argv[1] if len(sys.argv) > 1 else SCENE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
home = data.ctrl.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.time()
        data.ctrl[:] = home
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))
