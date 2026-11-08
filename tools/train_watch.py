"""학습 진행 화면. 서버에서 checkpoints/ 이름(시각 · 스텝)과 train_<회차>.log 의 STEP 줄을 5 초마다 읽어 막대로 보여 준다.
연쇄 학습(../rounds/chain.log)이 있으면 회차별 줄과 전체 남은 시간도 보인다. 표준 라이브러리만 쓴다.

서버: python3 tools/train_watch.py [총 스텝]   (Playground 루트에서)
맥:   ssh -t <서버> 'cd /data/work/playground/AIRookie_Playground && python3 tools/train_watch.py'
"""
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
ROUNDS = WORK / "rounds"
# --num_timesteps 3 억 의 실제 끝 스텝(평가 간격 21,463,040 × 14). 다른 길이면 첫 인자로 준다
TOTAL = next((int(a) for a in sys.argv[1:] if a.isdigit()), 300_482_560)
CKPT = re.compile(r"^(\d{4}_\d\d_\d\d_\d{6})_(\d+)$")


def _ckpts(d: Path) -> list:
    """[(시각, 스텝)] 스텝 순. 체크포인트 폴더 이름이 '저장 시각_스텝' 이다."""
    out = []
    if d.is_dir():
        for p in d.iterdir():
            m = CKPT.match(p.name)
            if m and p.is_dir():
                out.append((datetime.strptime(m.group(1), "%Y_%m_%d_%H%M%S").timestamp(), int(m.group(2))))
    return sorted(out, key=lambda x: x[1])


def _rewards(v: str) -> list:
    try:
        text = (WORK / f"train_{v}.log").read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    return [float(x) for x in re.findall(r"^STEP: \d+ reward: ([-\d.e]+)", text, re.M)]


def _chain() -> tuple:
    """연쇄 기록 → (회차 순서, {회차: 시작 시각}, {회차: 끝 상태}). 없으면 빈 값."""
    order, start, end = [], {}, {}
    try:
        lines = (ROUNDS / "chain.log").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    for ln in lines:
        m = re.match(r"(\S+ \S+) (v\d+_\w+) (시작|끝|실패|이미 끝)", ln)
        if not m:
            continue
        t, v, what = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp(), m.group(2), m.group(3)
        if v not in order:
            order.append(v)
        if what == "시작":
            start[v] = t
        else:
            end[v] = (what, t)
    # 아직 시작하지 않은 회차도 줄을 세운다. 가장 최근 연쇄 스크립트(chain*.sh)의 회차 목록을 쓴다
    shs = sorted(ROUNDS.glob("chain*.sh"), key=lambda p: p.stat().st_mtime)
    if shs:
        m = re.search(r"for [VP] in ([^;]+);", shs[-1].read_text(encoding="utf-8"))
        if m:
            order = re.findall(r"v\d+_\w+", m.group(1))
    return order, start, end


def bar(frac: float, width: int) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "░" * (width - n)


def hm(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}"


def render() -> str:
    w = max(40, min(shutil.get_terminal_size().columns, 100))
    now = time.time()
    order, start, end = _chain()
    live = (REPO / "checkpoints").is_dir()
    cur = next((v for v in order if v in start and v not in end), None)
    if cur is None and live:
        logs = sorted(WORK.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
        cur = logs[-1].stem[len("train_"):] if logs else "?"
    ck = _ckpts(REPO / "checkpoints") if live else []
    step = ck[-1][1] if ck else 0
    frac = step / TOTAL

    # 처리량은 마지막 두 평가 사이, 컴파일(시작 → 스텝 0 저장)은 따로 센다.
    rate = (ck[-1][1] - ck[-2][1]) / max(ck[-1][0] - ck[-2][0], 1) if len(ck) >= 2 else None
    t0 = start.get(cur)
    compile_s = ck[0][0] - t0 if ck and t0 else None
    left = (TOTAL - step) / rate if rate else None

    # 끝난 회차의 걸린 시간 평균으로 남은 회차를 어림한다. 없으면 이번 회차 어림을 쓴다.
    took = [end[v][1] - start[v] for v in order if v in start and v in end and end[v][0] == "끝"]
    per = sum(took) / len(took) if took else ((compile_s or 0) + TOTAL / rate if rate else None)
    todo = [v for v in order if v not in end and v != cur]
    total_left = (left or 0) + per * len(todo) if per and left is not None else None

    bw = w - 18
    rows = [f"학습  {cur or '-'}" + (f"  ({order.index(cur) + 1}/{len(order)})" if cur in order else "")]
    if not live and not cur:
        rows.append("학습 중 아님")
    rows += ["", f"회차  {bar(frac, bw)} {100 * frac:6.2f}%"]
    if order:
        done = sum(1 for v in order if v in end) + (frac if cur in order else 0)
        rows.append(f"전체  {bar(done / len(order), bw)} {100 * done / len(order):6.2f}%")
    rows.append("")
    rows.append(f"스텝       {step:,} / {TOTAL:,}")
    if rate:
        rows.append(f"처리량     {rate:,.0f} 스텝/s")
    if left is not None:
        rows.append(f"회차 남은  {hm(left)}  ({time.strftime('%H:%M', time.localtime(now + left))})")
    if total_left is not None and todo:
        rows.append(f"전체 남은  {hm(total_left)}  ({time.strftime('%m-%d %H:%M', time.localtime(now + total_left))})")
    if compile_s is not None:
        rows.append(f"컴파일     {compile_s:.0f} s")
    rw = _rewards(cur) if cur else []
    if rw:
        rows.append(f"평가 보상  {rw[-1]:.1f}  (최고 {max(rw):.1f}, 평가 {len(rw)}/15)")

    if order:
        rows += ["", "회차      상태        시작   걸림   마지막 보상"]
        for v in order:
            r = _rewards(v)
            if v in end:
                st = end[v][0]
                dur = hm(end[v][1] - start[v]) if v in start else "-"
            elif v == cur:
                st, dur = f"진행 {100 * frac:3.0f}%", hm(now - start[v]) if v in start else "-"
            else:
                st, dur = "대기", "-"
            s0 = time.strftime("%H:%M", time.localtime(start[v])) if v in start else "-"
            rows.append(f"{v:<9} {st:<10} {s0:>6} {dur:>6}   {r[-1]:.1f}" if r else f"{v:<9} {st:<10} {s0:>6} {dur:>6}   -")

    age = now - ck[-1][0] if ck else None
    if age is not None:
        stale = rate and age > 3 * (ck[-1][1] - ck[-2][1]) / rate
        rows += ["", f"마지막 저장 {age:.0f}초 전" + ("  (정지 의심)" if stale else "")]
    return "\n".join(rows)


def main():
    once = "--once" in sys.argv
    try:
        while True:
            out = render()
            if once:
                print(out)
                return
            sys.stdout.write("\033[H\033[2J" + out + "\n")
            sys.stdout.flush()
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
