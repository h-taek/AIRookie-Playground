"""flexmodel 장면에서 툭 치기. 사용: PYTHONPATH=.:tools/round_eval python tools/round_eval/flex_tap.py <kr> <dr> <kp> <dp> <축 0|1> <힘 N> [가속 흉내=38.8]"""
import sys
import mujoco, numpy as np
import flexmodel
from playground.open_duck_mini_v2.mujoco_infer import MjInfer, ServoProfile
from tools.eval_policy_behavior import DEFAULT_REFERENCE
kr, dr, kp, dp, AX, FN = map(float, sys.argv[1:7]); AX = int(AX); ACC = float(sys.argv[7]) if len(sys.argv) > 7 else 38.8
p = flexmodel.make(kr, dr, kp, dp)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    infer = MjInfer(p, DEFAULT_REFERENCE, "policies/v2/v2_15_300M.onnx", standing=False, head_from_command=True)
m, d = infer.model, infer.data; home = infer.default_actuator.copy()
mujoco.mj_resetData(m, d); mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
servo = ServoProfile(home, accel=ACC); trunk = m.body("trunk_assembly").id
T, G, TL = [], [], []
for k in range(int(4.0 / infer.sim_dt)):
    t = k * infer.sim_dt; d.xfrc_applied[trunk, :] = 0
    if 2.0 <= t < 2.02: d.xfrc_applied[trunk, 1 - AX] = FN
    d.ctrl = servo(home, infer.sim_dt); mujoco.mj_step(m, d)
    if k % 2 == 0:
        T.append(t); G.append(np.degrees(infer.get_gyro(d)[AX]))
        R = d.site_xmat[m.site("imu").id].reshape(3, 3); TL.append(np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(R[2, 0])])[AX])
T, G, TL = map(np.array, (T, G, TL))
up = (d.site_xmat[m.site("imu").id].reshape(3, 3)[2, 2])
w = T >= 2.0; tt, gg = T[w] - 2.0, G[w]; j = np.argmax(np.abs(gg[:100])); tt, gg = tt[j:] - tt[j], gg[j:]
gg = gg[tt < 1.0]; tt = tt[tt < 1.0]
zc = np.flatnonzero(np.diff(np.sign(gg)) != 0)
f = (len(zc) - 1) / (2 * (tt[zc[-1]] - tt[zc[0]])) if len(zc) > 2 else np.nan
pk = [np.abs(gg[zc[i]:zc[i + 1] + 1]).max() for i in range(len(zc) - 1)]
dec = [np.log(pk[i] / pk[i + 2]) for i in range(len(pk) - 2) if pk[i + 2] > 0.3]
zeta = np.median(dec) / np.sqrt(4 * np.pi ** 2 + np.median(dec) ** 2) if dec else np.nan
print(f"k {kr if AX == 0 else kp} d {dr if AX == 0 else dp} {'좌우' if AX == 0 else '앞뒤'} {FN} N  서기 기울기 {TL[T < 2].mean():+.2f}°  첫 봉우리 {gg[0]:+.1f}°/s  주파수 {f:.1f} Hz  반주기 봉우리 {np.round(pk[:6],1)}  감쇠비 {zeta:.2f}  끝 자세 up {up:.2f}")
