#!/usr/bin/env bash
set -Eeuo pipefail

# 비공개 저장소다. 서버에서 clone 하려면 자격증명이 필요하다.
# 업스트림 원본은 https://github.com/apirrone/Open_Duck_Playground.git 이다.
REPO_URL="${REPO_URL:-https://github.com/h-taek/AIRookie-Playground.git}"
REPO_REF="${REPO_REF:-}"
TARGET_DIR="${TARGET_DIR:-AIRookie_Playground}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PLAYGROUND_VERSION="${PLAYGROUND_VERSION:-0.2.0}"
JAX_VERSION="${JAX_VERSION:-0.9.2}"
JAXLIB_VERSION="${JAXLIB_VERSION:-0.9.2}"
# 물리 백엔드. jax 가 기존 MJX 경로, warp 가 MJWarp 다.
TRAIN_IMPL="${TRAIN_IMPL:-warp}"
TRAIN_TASK="${TRAIN_TASK:-flat_terrain_backlash}"
TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-300000000}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

log() {
  printf '\n[setup] %s\n' "$*"
}

die() {
  printf '\n[setup:error] %s\n' "$*" >&2
  exit 1
}

clone_or_enter_repo() {
  if [[ -f "${REPO_ROOT}/pyproject.toml" && -d "${REPO_ROOT}/.git" ]]; then
    cd "${REPO_ROOT}"
    return
  fi

  cd "${PWD}"
  command -v git >/dev/null 2>&1 || die "git is required to clone ${REPO_URL}."

  if [[ -d "${TARGET_DIR}/.git" ]]; then
    log "Using existing repository: ${TARGET_DIR}"
    cd "${TARGET_DIR}"
    return
  fi

  if [[ -e "${TARGET_DIR}" ]]; then
    die "TARGET_DIR exists but is not a git repository: ${TARGET_DIR}"
  fi

  log "Cloning ${REPO_URL} into ${TARGET_DIR}"
  if [[ -n "${REPO_REF}" ]]; then
    git clone --branch "${REPO_REF}" --single-branch "${REPO_URL}" "${TARGET_DIR}"
  else
    git clone "${REPO_URL}" "${TARGET_DIR}"
  fi

  cd "${TARGET_DIR}"
}

clone_or_enter_repo

log "Repository: $(pwd)"

if ! command -v uv >/dev/null 2>&1; then
  log "uv was not found. Installing uv..."
  command -v curl >/dev/null 2>&1 || die "curl is required to install uv."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

command -v uv >/dev/null 2>&1 || die "uv is still unavailable after install."

log "Creating/updating .venv with Python ${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}"

# pyproject.toml 이 의존성의 유일한 명세(SSOT)다. 스크립트는 그것을 설치만 한다.
#
# 예전에는 여기서 grep 으로 핀이 박혀 있는지 검사하고 아니면 `uv add` 로 고쳐
# 넣었는데, 검사 패턴 "jax==0.6.2" 가 실제 줄 "jax==0.6.2; sys_platform != 'linux'"
# 와 닫는 따옴표에서 어긋나 한 번도 일치하지 않았다. 그래서 매번 `uv add` 분기로
# 빠져 플랫폼 분기 줄은 그대로 둔 채 마커 없는 "jax==0.6.2" 를 덧붙였고, 추적
# 파일이 더러워져 이후 git pull 에서 충돌했다. 검사 자체를 없앤다.
log "Installing dependencies from pyproject.toml"
uv lock
uv sync

log "Verifying Open Duck runtime prerequisites"
uv run python - <<'PY'
from pathlib import Path
import importlib.metadata as md
import importlib.util
import inspect

expected = {
    "playground": "0.2.0",
    "jax": "0.9.2",
    "jaxlib": "0.9.2",
    "mujoco": "3.13.0",
    "mujoco-warp": "3.13.0",
    "warp-lang": "1.17.0",
}

for package, version in expected.items():
    installed = md.version(package)
    if installed != version:
        raise SystemExit(f"{package}=={installed} installed, expected {version}")
    print(f"{package}=={installed}")

from mujoco_playground._src import mjx_env

# 0.0.3 의 mjx_env.init 은 0.2.0 에서 make_data 로 바뀌었고 impl 인자가 붙었다.
# 이름만 보지 않고 인자까지 확인한다.
if not hasattr(mjx_env, "make_data"):
    raise SystemExit("mujoco_playground._src.mjx_env.make_data is missing")
params = inspect.signature(mjx_env.make_data).parameters
for name in ("impl", "naconmax", "njmax"):
    if name not in params:
        raise SystemExit(f"mjx_env.make_data has no parameter: {name}")

# 0.0.3 의 _src.collision 은 삭제됐다. 남아 있으면 구버전이 깔린 것이다.
if importlib.util.find_spec("mujoco_playground._src.collision") is not None:
    raise SystemExit("mujoco_playground._src.collision still present; 0.0.3 installed")

import jax
if not hasattr(jax, "device_put_replicated"):
    raise SystemExit("jax.device_put_replicated is missing; Brax PPO will fail")

required_files = [
    Path("playground/open_duck_mini_v2/runner.py"),
    Path("playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"),
    Path("playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"),
]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise SystemExit("Missing required files:" + ", ".join(missing))

# 접촉 판정이 기하 쌍 질의에서 MJCF 접촉 센서로 바뀌었다. 씬에 센서가 없으면
# 학습은 돌지만 발 접촉이 항상 거짓이 되어 보상이 조용히 망가진다.
import mujoco
from playground.open_duck_mini_v2 import constants

model = mujoco.MjModel.from_xml_path(
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"
)
for geom in constants.FEET_GEOMS:
    sensor_name = f"{geom}_floor_found"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) < 0:
        raise SystemExit(f"contact sensor missing from scene xml: {sensor_name}")
print(f"contact sensors verified: {len(constants.FEET_GEOMS)}")

# Warp 백엔드는 임포트만 확인한다. 커널 실행은 CUDA 장치를 요구하므로
# 여기서 판정하지 아니한다.
import mujoco_warp  # noqa: F401
import warp  # noqa: F401

print("runtime imports and required files verified")
PY

log "Setup complete. Training was not started."
cat <<EOF

Repository is ready at:

  $(pwd)

Run training when ready:

  uv run playground/open_duck_mini_v2/runner.py \\
    --task ${TRAIN_TASK} \\
    --num_timesteps ${TRAIN_TIMESTEPS} \\
    --impl ${TRAIN_IMPL}

If Ctrl+C does not stop a JAX compile/run immediately, stop the process group:

  pkill -TERM -f 'playground/open_duck_mini_v2/runner.py'

EOF
