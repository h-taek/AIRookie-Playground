"""시범 동작을 d 스텝 늦춰 비교한 imitation 점수와 학습 환경의 관절 지연을 잰다(research/04 41장).
학습 환경(joystick.Joystick)으로 정책을 굴리며, 매 스텝 같은 몸 상태를 imitation_i − d 의 시범 동작과 견준다.
사용 (Playground 루트에서): python tools/round_eval/imit_delay.py <onnx> <impl> <시드 수> <가속 배율,...>
"""
import sys
import numpy as np
import jax
import jax.numpy as jp
sys.path.insert(0, ".")
from playground.open_duck_mini_v2 import joystick
from playground.open_duck_mini_v2.custom_rewards import reward_imitation
from playground.common.onnx_infer import OnnxInfer

POL, IMPL, NS = sys.argv[1], sys.argv[2], int(sys.argv[3])
SCALES = [float(x) for x in sys.argv[4].split(",")]
D = np.arange(0, 13)
WIN = (4, 10)
env = joystick.Joystick(task="flat_terrain_backlash", config_overrides={"impl": IMPL})
pol = OnnxInfer(POL, awd=True)
reset, step = jax.jit(env.reset), jax.jit(env.step)
dt = env.dt
TQ = env._config.reward_config.scales.torques
ACC = env._config.servo_max_accel
LEG = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])


@jax.jit
def imit_d(data, info):
    contact = data.sensordata[env._feet_floor_found_adr] > 0
    c = info["command"]

    def one(d):
        ref = env.PRM.get_reference_motion(c[0], c[1], c[2], info["imitation_i"] - d)
        return reward_imitation(env.get_floating_base_qpos(data.qpos), env.get_floating_base_qvel(data.qvel),
                                env.get_actuator_joints_qpos(data.qpos), env.get_actuator_joints_qvel(data.qvel),
                                contact, ref, c, True)
    return jax.vmap(one)(jp.asarray(D))


for scale in SCALES:
    for vx in (0.05, 0.08):
        sc, tg, q = [], [], []
        falls = 0
        for seed in range(NS):
            st = reset(jax.random.PRNGKey(seed))
            st = st.replace(info={**st.info, "torque_scale": jp.asarray(TQ, dtype=jp.float32),
                                  "servo_accel": jp.asarray(ACC * scale, dtype=jp.float32)})
            n = int(round(13.0 / dt))
            T, Q = [], []
            for i in range(n):
                t = i * dt
                c = jp.array([0.0 if t < 3.0 else vx, 0, 0, 0, 0, 0, 0], dtype=jp.float32)
                st = st.replace(info={**st.info, "command": c, "command_target": c, "step": 0})
                a = pol.infer(np.asarray(st.obs["state"]))
                st = step(st, jp.asarray(a))
                if bool(st.done): falls += 1; break
                T.append(np.asarray(st.info["motor_targets"])[LEG]); Q.append(np.asarray(env.get_actuator_joints_qpos(st.data.qpos))[LEG])
                if 5.0 <= t < 13.0:
                    sc.append(np.asarray(imit_d(st.data, st.info)))
            T, Q = np.array(T), np.array(Q)
            w = np.arange(len(T)) * dt >= 5.0
            tg.append((T, Q, w))
        sc = np.array(sc)
        # 관절 지연: mean|목표[k-1-d] - 관절각[k]| 최소 d (실물 기록과 같은 정의)
        lag = []
        for j in range(len(LEG)):
            s = np.zeros(13)
            for T, Q, w in tg:
                n = len(T)
                for d in range(13):
                    e = np.full(n, np.nan); e[1 + d:] = T[:n - 1 - d, j] - Q[1 + d:, j]
                    m = w & ~np.isnan(e); s[d] += np.abs(e[m]).sum()
            lag.append(int(np.argmin(s)))
        best = sc[:, WIN[0]:WIN[1] + 1].max(1).mean() if len(sc) else np.nan
        print(f"== 가속 ×{scale} 전진 {vx}  넘어짐 {falls}/{NS}  지연(hy hr hp kn an × 왼 · 오른) {lag}", flush=True)
        print("   d   " + " ".join(f"{d:6d}" for d in D), flush=True)
        print("   imit" + " ".join(f"{v:+6.3f}" for v in sc.mean(0)), flush=True)
        print(f"   창 {WIN[0]}~{WIN[1]} 스텝마다 최대 {best:+.3f}", flush=True)
