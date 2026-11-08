"""lag_fit.py 결과(jsonl)를 조건별로 묶어 넘어짐과 다리 10 축 지연을 실물 값과 나란히 찍는다."""
import json, sys, numpy as np, collections
LEG = ['L_hy', 'L_hr', 'L_hp', 'L_kn', 'L_an', 'R_hy', 'R_hr', 'R_hp', 'R_kn', 'R_an']
REAL = [4, 4, 5, 7, 6, 4, 4, 5, 9, 5]
G = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); G[(r['acc'], r['tau'], r['rsc'])].append(r)
print('실물 지연', dict(zip(LEG, REAL)))
for key in sorted(G):
    rs = G[key]; S = np.sum([r['sums'] for r in rs], 0); C = sum(r['cnt'] for r in rs)
    d = S.argmin(0); mae = S.min(0) / max(C, 1)
    fl = [r['fell'] for r in rs if r['fell']]
    print(f"가속 {key[0]:>4} T {key[1]:>3.0f}ms 전압 {key[2]:.0f} | 넘어짐 {len(fl)}/{len(rs)} {''.join(sorted(x[1] for x in fl))} {sorted(x[0] for x in fl)[:3]} | 지연 " +
          ' '.join(f'{n}{x}' for n, x in zip(LEG, d)) + f" | 무릎 MAE {mae[3]:.3f}/{mae[8]:.3f}")
