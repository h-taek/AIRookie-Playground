"""head_stress.py 출력 파일을 머리 방식별 넘어짐 · 최대 기울기 · 흔들림으로 줄인다. 사용: python tools/round_eval/stress_summary.py <출력 파일>"""
import sys,re,collections
lines=open(sys.argv[1]).read().splitlines(); mv=None
fall=collections.Counter(); n=collections.Counter(); tmax=collections.defaultdict(float); rms=collections.defaultdict(list); where=[]
for l in lines:
    if l.startswith("== "): mv=l[3:]; continue
    m=re.match(r"\s+(\S+(?: \S+)*?)(?: s\d)?\s{2,}(넘어짐|기울기)",l)
    if not m: continue
    h=m.group(1).strip(); n[h]+=1
    if "넘어짐" in l: fall[h]+=1; where.append(f"{mv}/{l.strip()}")
    else:
        a=re.search(r"기울기\s+([\d.]+)/\s*([\d.]+)\s+흔들림 ([\d.]+)",l)
        if mv!="정지": tmax[h]=max(tmax[h],float(a.group(2))); rms[h].append(float(a.group(3)))
print("총 넘어짐", sum(fall.values()),"/",sum(n.values()), where)
for h in n: print(f"  {h:14s} 넘어짐 {fall[h]}/{n[h]}  최대 기울기 {tmax[h]:4.1f}  흔들림 {sum(rms[h])/len(rms[h]):.2f}")
