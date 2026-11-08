"""진행 화면. real_fit/out/status.json 을 2 초마다 읽어 막대로 보여 준다. 표준 라이브러리만 쓴다.

서버: python -m real_fit.watch
맥:   ssh -t <서버> 'cd /data/work/playground && .venv/bin/python -m real_fit.watch'
"""
import ast
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / ("out_smoke" if "--smoke" in sys.argv else "out")


SEARCH = Path(__file__).resolve().parent / "search.py"
PACK = Path(__file__).resolve().parent / "data" / "records.npz"


def _consts() -> dict:
    """search.py 의 맨 위 상수를 문법 분석으로 읽는다. 모듈을 불러오지 않는다 — 불러오면 JAX 가 GPU 메모리를
    잡아 돌고 있는 탐색과 부딪힐 수 있다. `if SMOKE:` 안의 축소 값은 읽지 않는다."""
    c = {}
    for node in ast.parse(SEARCH.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        names = [t.id for t in tgt.elts] if isinstance(tgt, ast.Tuple) else [getattr(tgt, "id", None)]
        try:
            val = ast.literal_eval(node.value)
        except ValueError:
            if names == ["SPACE"]:
                c["DIM"] = len(node.value.elts)
            continue
        vals = val if len(names) > 1 else [val]
        c.update(zip(names, vals))
    return c


def _plan(c: dict, K: int, K2: int) -> list:
    """단계별 계획 환경-스텝. search.Progress.plan 과 같은 식이다."""
    end = np.load(PACK)["end"]
    per = float(sum(next(b for b in c["BUCKETS"] if e <= b) for e in end))
    return [c["N_PILOT"] * c["K_PILOT"] * per,
                (2 * c["N_SOBOL"] + 2 * c["TOP_RECHECK"] * 2) * K * per,
                c["CMA_BUDGET"] * K2 * per,
                c["DIM"] * c["PROFILE_LEVELS"] * c["PROFILE_LAMBDA"] * c["PROFILE_ITERS"] * K2 * per,
                (11 * 4 + 2 * c["DIM"] * 2 + 2 + 2 + c["DIM"] * c["PROFILE_LEVELS"] * 4) * K2 * per]


def _from_log() -> tuple:
    """progress.log 에서 탐색이 평가마다 직접 잰 처리량의 최신값과 K2 를 읽는다. 0 단계 값은 컴파일이 섞여 낮다."""
    try:
        text = (OUT / "progress.log").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None, None
    rate, pilot = None, None
    for line in text.splitlines():
        m = re.search(r"처리량 ([\d,]+)(?: 환경-스텝)?/초", line)
        if m:
            v = float(m.group(1).replace(",", ""))
            if "[0]" in line:
                pilot = v
            else:
                rate = v
    k2 = re.findall(r"K2 = (\d+)", text)
    return rate, pilot, int(k2[-1]) if k2 else None


def bar(frac: float, width: int) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "░" * (width - n)


def render() -> str:
    w = max(40, min(shutil.get_terminal_size().columns, 100))
    try:
        s = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "real_fit  대기 중"
    done = (OUT / "DONE").exists()
    age = time.time() - s["time"]
    rate, pilot, k2 = _from_log()
    r = rate or pilot
    try:
        c = _consts()
        plan = _plan(c, s.get("K") or 4, k2 or 8)
        chunk = int(os.environ.get("FIT_MAX_ENVS", "16384")) * max(c["BUCKETS"])
    except Exception:  # noqa: BLE001 — 화면은 죽지 않는다
        plan, chunk = None, 0
    # 탐색은 GPU 묶음(1 ~ 4 분)마다 상태를 고친다. 그 사이는 실측 처리량 × 지난 시간으로 어림하되 묶음 하나를 넘지 않는다.
    if r and plan and not done and s["stage"] < len(plan):
        add = min(age * r, chunk)
        s["stage_frac"] = min(s["stage_frac"] + add / plan[s["stage"]], 0.999)
        s["overall"] = min(s["overall"] + add / sum(plan), 0.999)
    overall = 1.0 if done else s["overall"]
    bw = w - 18

    if done:
        ok = s.get("checks") and all(s["checks"].values())
        head = "real_fit  끝 — " + ("검증 통과" if ok else "검증 실패")
    else:
        head = f"real_fit  {s['stage_name']} ({min(s['stage'] + 1, 5)}/5)"
    rows = [head, "",
            f"전체  {bar(overall, bw)} {100 * overall:6.2f}%"]
    if not done:
        rows.append(f"단계  {bar(s['stage_frac'], bw)} {100 * s['stage_frac']:6.2f}%")
    rows.append("")

    if not done and r and plan:
        h = (1 - overall) * sum(plan) / r / 3600
        end = time.strftime("%m-%d %H:%M", time.localtime(time.time() + h * 3600))
        rows.append(f"남은 시간  {'약 ' if not k2 else ''}{h:.1f} h  ({end})")
    if r:
        rows.append(f"처리량     {r:,.0f} /s")
    rows.append(f"복제       K {s.get('K') or '-'} · K2 {k2 or '-'}")
    rows.append(f"가속 흉내  {'-' if s.get('mode') is None else ('켬' if s['mode'] else '끔')}")
    best = s.get("f_best")
    if best is None:
        m = re.search(r"최저 ([\d.]+)", s.get("note", ""))
        best = float(m.group(1)) if m else None
    rows.append(f"최저 손실  {best:.2f}" if best is not None else "최저 손실  -")
    note = re.sub(r",?\s*(이 모드 |이 세대 )?최저 [\d.]+", "", s.get("note", "")).strip(" ,")
    if note and not done:
        rows.append(f"진행       {note}")
    if done and s.get("checks"):
        rows.append("검증       " + "  ".join(f"{k} {'O' if v else 'X'}" for k, v in s["checks"].items()))

    if s.get("best"):
        b = s["best"]
        rows += ["", "최종점" if done else "최저점",
                 f"  몸통 무게중심  x {b['dx_mm']:+6.2f}  y {b['dy_mm']:+6.2f} mm",
                 f"  머리 무게중심  x {b['hx_mm']:+6.2f}  y {b['hy_mm']:+6.2f} mm",
                 f"  유격           다리 ±{b['leg_play_deg']:.2f}°  머리 ±{b['head_play_deg']:.2f}°  서보 {b['backlash_deg']:.2f}°",
                 f"  마찰           마루 {b['mu_maru']:.2f}  타일 {b['mu_tile']:.2f}"]
    rows += ["", f"갱신 {age:.0f}초 전" + ("  (정지 의심)" if age > 300 and not done else "")]
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
