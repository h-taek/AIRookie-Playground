"""맥에서 기록을 묶어 서버로 보낼 파일 하나를 만든다: real_fit/data/records.npz.

서버에는 Runtime 기록이 없다. 시뮬에 필요한 것(명령 · 짝 홈 · 위상 · 접촉 고정 · 길이)과 실물 지표만 담는다.
"""
import json
from pathlib import Path

import numpy as np

from real_fit import model as M, records as R

OUT = Path(__file__).resolve().parent / "data" / "records.npz"


def main():
    h = json.loads((R.RUNTIME / "src" / "config" / "home_pose.json").read_text(encoding="utf-8"))
    scale = np.array(list(h["action_scale"].values()))
    head_home = np.array(list(h["joints"].values()))[5:9]
    joint_range = np.array(list(h["joint_range"].values()))
    recs = R.load_all(scale, head_home, joint_range)
    tmax = max(r.end for r in recs)
    cmd = np.zeros((len(recs), tmax + 1, 7), np.float32)
    acc = np.zeros((len(recs), tmax, 3), np.float32)
    knee = np.zeros((len(recs), tmax, 2), np.float32)
    ktg = np.zeros((len(recs), tmax, 2), np.float32)
    for i, r in enumerate(recs):
        n = min(len(r.cmd), r.end + 1)
        cmd[i, :n] = r.cmd[:n]
        cmd[i, n:] = r.cmd[n - 1]
        acc[i, :r.end] = r.accel[:r.end]
        knee[i, :r.end] = r.angle[:r.end][:, list(R.KNEES)]
        ktg[i, :r.end] = r.target[:r.end][:, list(R.KNEES)]
    policies = sorted(set(r.policy for r in recs), key=lambda p: int(p.split("_")[1]))
    work = M.load(M.build())
    init_q = np.stack([M.home_qpos(work, r.home) for r in recs])
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, name=np.array([r.name for r in recs]), floor=np.array([r.floor for r in recs]),
             policy=np.array([r.policy for r in recs]), policies=np.array(policies),
             pid=np.array([policies.index(r.policy) for r in recs]), init_q=init_q,
             cmd=cmd, home=np.stack([r.home for r in recs]),
             phase0=np.array([r.phase0 for r in recs]), contact_fixed=np.array([r.contact_fixed for r in recs]),
             move=np.array([r.move for r in recs]), end=np.array([r.end for r in recs]),
             fall_at_end=np.array([r.name in R.FALL_AT_END for r in recs]),
             raw_imu=np.array([r.name.startswith("log_") for r in recs]),
             real_acc=acc, real_knee=knee, real_tg=ktg,
             metrics=np.array([json.dumps(r.metrics, ensure_ascii=False) for r in recs]))
    print(OUT, len(recs), "기록", "최장", tmax, "주기", "합", int(sum(r.end for r in recs)), "주기")


if __name__ == "__main__":
    main()
