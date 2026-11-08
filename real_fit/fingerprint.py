"""저장 지점 지문. 데이터 · 코드 · 정책 · 학습 환경 · 라이브러리 판 · 실행 설정을 해시한다.

jax 를 불러오지 않는다. 서버에서 탐색이 돌고 있을 때 따로 불러도 GPU 를 잡지 않는다.
    python -m real_fit.fingerprint model.py,policy.py,score.py 16384
"""
import hashlib
import importlib.metadata as im
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACK = HERE / "data" / "records.npz"
CODE = ("env.py", "model.py", "policy.py", "score.py", "search.py")
TRAIN = ("joystick.py", "base.py", "actuation.py", "constants.py", "data/polynomial_coefficients.pkl",
         "xmls/scene_flat_terrain_backlash.xml", "xmls/open_duck_mini_v2_backlash.xml",
         "xmls/joints_properties.xml", "xmls/sensors.xml")
LIBS = ("jax", "mujoco", "mujoco-mjx", "warp-lang", "mujoco-playground", "cma")


def digest(code=CODE, smoke=False, impl="warp", max_envs=8192) -> str:
    h = hashlib.sha256()
    h.update(PACK.read_bytes())
    for f in code:
        h.update((HERE / f).read_bytes())
    od = ROOT / "playground" / "open_duck_mini_v2"
    for f in TRAIN:
        if (od / f).exists():
            h.update((od / f).read_bytes())
    for pkg in LIBS:
        try:
            h.update(f"{pkg}={im.version(pkg)}".encode())
        except im.PackageNotFoundError:
            h.update(f"{pkg}=없음".encode())
    for p in np.load(PACK)["policies"]:
        h.update((ROOT / "policies" / "v2" / f"{p}_300M.onnx").read_bytes())
    h.update(repr((smoke, impl, max_envs)).encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    print(digest(tuple(sys.argv[1].split(",")), False, "warp", int(sys.argv[2])))
