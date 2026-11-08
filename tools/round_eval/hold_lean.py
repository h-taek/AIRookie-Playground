"""실물 정책 없는 홈 유지 기록의 관절각을 시뮬에서 맞추고 몸통 실제 앞숙임을 잰다. 사용: python tools/round_eval/hold_lean.py"""
import sys, numpy as np, mujoco
sys.argv = ["x", "policies/v2/v2_15_300M.onnx", "0", "0", "1"]
exec(open("tools/round_eval/com_shift.py").read().split("for deg in IMU_DEGS:")[0])
names = [model.actuator(i).name for i in range(model.nu)]
z = np.load("../AIRookie_Runtime/logs/jetson/2026-10-10/log_imu_floor_hold253.npz")
rn = list(z["names"]); o = z["obs"]
real_dev = o[100:900, 13:27].mean(0)              # 실물 관절 − 홈 (rad), 실물 순서
dev = np.array([real_dev[rn.index(n)] for n in names])
a = o[100:900, 3:6]; print(f"실물 정책 없이 홈 유지: IMU 앞숙임 {np.degrees(np.arctan2(-a[:,0].mean(), a[:,2].mean())):+.2f}°")
def settle(target, dx):
    model.body_ipos[trunk] = ipos0 + np.array([dx / 1000, 0, 0])
    mujoco.mj_resetData(model, data); mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    L, Q = [], []
    for k in range(int(4.0 / model.opt.timestep)):
        data.ctrl = target; mujoco.mj_step(model, data)
        if k * model.opt.timestep > 2.0:
            R = np.zeros(9); mujoco.mju_quat2Mat(R, data.qpos[3:7]); L.append(lean(R.reshape(3, 3)))
            Q.append(infer.get_measured_joint_angles(data) - infer.default_actuator)
    return np.mean(L), np.mean(Q, 0)

for dx in (0, 30, 45, 60):
    target = infer.default_actuator + dev
    for it in range(6):
        l, q = settle(target, dx)
        target = target + (dev - q)
    l, q = settle(target, dx)
    print(f"시뮬 실물 관절각 일치 · 몸통 +{dx:2d}mm  몸통 실제 앞숙임 {l:+.2f}°  관절각 차(실물−시뮬) 최대 {np.degrees(np.abs(dev - q).max()):.2f}°")
