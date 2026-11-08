"""학습 체크포인트마다 정지 3 s → 명령 10 s 를 걸어 전진 · 후진 · 옆 · 회전이 자리 잡는 시점을 본다.
몸통 휨 강성은 인자로 준다(v2_43 은 학습 내내 200). 가속 흉내 0, 시드 2.
사용: PYTHONPATH=. python tools/round_eval/gait_timeline.py <강성 좌우> <onnx>..."""
import sys
from pathlib import Path
import numpy as np
from tools.round_eval import flex_site as F
CMDS = [("전진 0.05", 0.05, 0, 0), ("전진 0.111", 0.111, 0, 0), ("전진 0.222", 0.222, 0, 0), ("후진 0.111", -0.111, 0, 0),
        ("좌 0.148", 0, 0.148, 0), ("우 0.148", 0, -0.148, 0), ("회전 1.11", 0, 0, 1.11)]
kr = float(sys.argv[1]); kp = kr * 4.0 / 5.1; d = 0.03 * np.sqrt(kr / 5.1)
s = F.make(0, kr, d, kp, d)
for pol in sys.argv[2:]:
    out = []
    for name, vx, vy, wz in CMDS:
        _, _, fell, _, (bx, by, bw) = F.sway(s, pol, 0.0, vx, ns=2, vy=vy, wz=wz)
        v = bw if wz else (by if vy else bx)
        out.append(f"{name} {v:+.3f}{f' 넘어짐 {fell}' if fell else ''}")
    print(f"{Path(pol).stem} | " + " | ".join(out), flush=True)
