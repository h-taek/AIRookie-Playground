#!/usr/bin/env bash
# 실물 기록 맞춤 자동 실행 (plan/01). 서버 tmux duck 에서 /data/work/playground 를 작업 폴더로 돌린다.
# 죽으면 30 초 뒤 다시 띄우고 real_fit/out/state.pkl 저장 지점부터 잇는다. real_fit/out/DONE 이 생기면 끝난다.
#   bash real_fit/run.sh && echo ALL_DONE
set -u
set -o pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=/data/work/playground/.uv_cache UV_PYTHON_INSTALL_DIR=/data/work/playground/.uv_python
export PATH="$HOME/.local/bin:$PATH"
# cma 는 uv.lock 밖이다. lock 을 다시 풀면 jax 고정 버전이 흔들려서 venv 에만 넣는다. 실패하면 바로 멈춘다.
uv pip install --python .venv/bin/python "cma==4.0.0" >/dev/null || { echo "cma 설치 실패"; exit 1; }
.venv/bin/python -c "import cma; assert cma.__version__.startswith('4.0.0')" || { echo "cma 확인 실패"; exit 1; }
mkdir -p real_fit/out
tries=0
# 파이썬을 먼저 돌린다. 지문이 바뀌었으면 파이썬이 옛 출력(옛 DONE 포함)을 옮기고 새로 시작한다.
while true; do
    tries=$((tries + 1))
    .venv/bin/python -m real_fit.search 2>&1 | grep -v "^actuators\|^joints\|^backlash joints\|^actuator joints\|^floating\|^xml:" >> real_fit/run.log
    [ -f real_fit/out/DONE ] && break
    echo "$(date '+%F %T') 프로세스가 끝났는데 DONE 이 없다 — ${tries} 번째 재시작" >> real_fit/out/progress.log
    if [ "$tries" -ge 20 ]; then
        echo "$(date '+%F %T') 재시작 20 번 — 멈춘다. real_fit/run.log 를 본다" >> real_fit/out/progress.log
        exit 1
    fi
    sleep 30
done
