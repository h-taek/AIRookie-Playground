"""실물 기록의 관측을 정책에 다시 넣는다(열린 고리). 관측 묶음 하나를 5 Hz 위 성분을 뺀 값으로 바꿔 무릎 행동 거칠기 변화를 본다.
사용: python obs_replay.py <onnx> <log.npz> <home_pose.json>"""
import sys, json
import numpy as np, onnxruntime as ort
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from cmd_rough import rough
sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
d = np.load(sys.argv[2], allow_pickle=True); t = d["t"]; O = d["obs"].astype(np.float32); names = list(d["names"])
pair = json.load(open(sys.argv[3])); scale = np.array([pair["action_scale"][n] for n in d["names"]], float)
G = {"gyro": slice(0, 3), "accel": slice(3, 6), "joint_ang": slice(13, 27), "joint_vel": slice(27, 41),
     "last_act x3": slice(41, 83), "motor_tgt": slice(83, 97), "contact": slice(97, 99)}
K = [names.index("left_knee"), names.index("right_knee")]
w = (t >= 3.5) & (t < 6.5)

def lowpass(x, fc=5.0, dt=0.02):
    X = np.fft.rfft(x, axis=0); f = np.fft.rfftfreq(len(x), dt); X[f > fc] = 0
    return np.fft.irfft(X, n=len(x), axis=0)

def act(o):
    return np.concatenate([sess.run(None, {inp: o[i:i + 1]})[0] for i in range(len(o))])

A = act(O)
chk = np.abs(A[:-1] - O[1:, 41:55]).max()
print(f"다시 넣은 행동 vs 다음 관측의 last_action 최대 차 {chk:.2e}")
sc = scale[K]
def show(lbl, A_):
    r = [rough(np.degrees(A_[w, k] * s)) for k, s in zip(K, sc)]
    print(f"{lbl:14s} 왼 {r[0][0]:.2f}° {100*r[0][1]:.0f}% {100*r[0][2]:.0f}%   오른 {r[1][0]:.2f}° {100*r[1][1]:.0f}% {100*r[1][2]:.0f}%")
show("기록 그대로", A)
for g, s in G.items():
    O2 = O.copy(); O2[:, s] = lowpass(O[:, s]).astype(np.float32); show(g + " 매끄럽게", act(O2))
O2 = O.copy()
for s in G.values(): O2[:, s] = lowpass(O[:, s]).astype(np.float32)
show("전부 매끄럽게", act(O2))
O2 = O.copy()
for g in ("gyro", "accel", "joint_ang", "joint_vel", "contact"): O2[:, G[g]] = lowpass(O[:, G[g]]).astype(np.float32)
show("센서만 매끄럽게", act(O2))
O2 = O.copy()
for g in ("last_act x3", "motor_tgt"): O2[:, G[g]] = lowpass(O[:, G[g]]).astype(np.float32)
show("이력만 매끄럽게", act(O2))
# 센서 묶음별 5 Hz 위 성분 RMS(관측 단위)
for g in ("gyro", "accel", "joint_ang", "joint_vel"):
    x = O[w][:, G[g]]; hf = x - lowpass(x)
    print(f"  {g:10s} 5 Hz 위 RMS {np.sqrt(np.mean(hf ** 2)):.4f}  전체 표준편차 {x.std(0).mean():.4f}")
