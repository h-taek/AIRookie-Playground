"""몸통 휨 스프링을 다리의 다른 마디로 옮겨, 툭 치기로 실물 주파수에 맞춘 뒤 걷기 흔들림 · 넘어짐을 잰다.
위치 L0 몸통↔엉덩이 yaw(학습 모델) · L1 yaw→roll · L2 roll→pitch · L3 무릎 · L4 발목. 경첩 축은 홈 자세의 몸통 x(좌우) · y(앞뒤)다.
사용: PYTHONPATH=. python tools/round_eval/flex_site.py <위치 0~4> <정책 onnx>..."""
import contextlib, io, re, sys, tempfile
from pathlib import Path
import mujoco, numpy as np
from playground.open_duck_mini_v2 import base, constants
from playground.open_duck_mini_v2.mujoco_infer import ActionDelay, MjInfer, ServoProfile
from tools.eval_policy_behavior import DEFAULT_REFERENCE
X = constants.ROOT_PATH / "xmls"
_orig_assets = base.get_assets
_extra = {}
def _assets():
    a = _orig_assets(); a.update(_extra); return a
base.get_assets = _assets
OUT = Path(tempfile.gettempdir()) / "flexsite"
SITES = [("hip_roll_assembly", "hip_roll_assembly_2"), ("left_roll_to_pitch_assembly", "right_roll_to_pitch_assembly"),
         ("knee_and_ankle_assembly", "knee_and_ankle_assembly_3"), ("knee_and_ankle_assembly_2", "knee_and_ankle_assembly_4"),
         ("foot_assembly", "foot_assembly_2")]
F_TARGET, ZETA = (4.2, 2.6), 0.14   # 실물 툭 치기 좌우 · 앞뒤 (Runtime research/10 26-12)


def _load(scene, body, name="open_duck_mini_v2_flexsite.xml"):
    # 장면마다 몸 XML 자산 이름을 따로 둔다. 하나를 덮어쓰면 먼저 만든 장면도 나중 몸으로 돈다.
    _extra[name] = body.encode()
    return mujoco.MjModel.from_xml_string(scene.replace("open_duck_mini_v2_backlash.xml", name), assets=_assets())


def make(loc, kr, dr, kp, dp):
    body0 = re.sub(r'\n\s*<joint name="(left|right)_flex_(roll|pitch)"[^>]*/>', "", (X / "open_duck_mini_v2_backlash.xml").read_text())
    scene0 = (X / "scene_flat_terrain_backlash.xml").read_text()
    nokey = re.sub(r"<keyframe>.*?</keyframe>", "", scene0, flags=re.S)
    old = mujoco.MjModel.from_xml_string(scene0, assets=_assets()); od = mujoco.MjData(old)
    mujoco.mj_resetDataKeyframe(old, od, old.keyframe("home").id); mujoco.mj_forward(old, od)
    trunk = od.xmat[old.body("trunk_assembly").id].reshape(3, 3)
    body = body0
    for side, bname in zip(("left", "right"), SITES[loc]):
        R = od.xmat[old.body(bname).id].reshape(3, 3)
        ax = [R.T @ trunk[:, i] for i in (0, 1)]   # 몸통 x · y 를 이 마디 좌표로
        j = "".join(f'<joint name="{side}_flex_{n}" type="hinge" pos="0 0 0" axis="{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}" stiffness="{k}" damping="{c}" armature="0.0005" frictionloss="0"/>\n          '
                    for n, a, k, c in (("roll", ax[0], kr, dr), ("pitch", ax[1], kp, dp)))
        m = re.search(rf'<body name="{bname}"[^>]*>.*?(<joint )', body, flags=re.S)
        body = body[:m.start(1)] + j + body[m.start(1):]
    tag = f"L{loc}_{kr:.3g}_{dr:.3g}_{kp:.3g}_{dp:.3g}"
    asset = f"open_duck_mini_v2_flexsite_{tag}.xml"
    new = _load(nokey, body, asset)
    keys = []
    for k in range(old.nkey):
        q = new.qpos0.copy()
        for jid in range(old.njnt):
            n = old.joint(jid).name; a0 = old.jnt_qposadr[jid]; a1 = new.jnt_qposadr[new.joint(n).id]
            w = 7 if old.jnt_type[jid] == 0 else (4 if old.jnt_type[jid] == 1 else 1)
            q[a1:a1 + w] = old.key_qpos[k][a0:a0 + w]
        keys.append(f'<key name="{old.key(k).name}" qpos="{" ".join(f"{v:.9g}" for v in q)}" ctrl="{" ".join(f"{v:.9g}" for v in old.key_ctrl[k])}"/>')
    OUT.mkdir(exist_ok=True)
    p = OUT / f"scene_{tag}.xml"
    p.write_text(nokey.replace("open_duck_mini_v2_backlash.xml", asset).replace("</mujoco>", "<keyframe>" + "".join(keys) + "</keyframe>\n</mujoco>"))
    return str(p)


def _infer(scene, pol):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return MjInfer(scene, DEFAULT_REFERENCE, pol, standing=False, head_from_command=True)


def tap(scene, ax, fn, acc=38.8):
    """flex_tap.py 와 같은 툭 치기. (주파수, 감쇠비, 서기 기울기°)"""
    infer = _infer(scene, "policies/v2/v2_15_300M.onnx")
    m, d = infer.model, infer.data; home = infer.default_actuator.copy()
    mujoco.mj_resetData(m, d); mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
    servo = ServoProfile(home, accel=acc); trunk = m.body("trunk_assembly").id
    T, G, TL = [], [], []
    for k in range(int(4.0 / infer.sim_dt)):
        t = k * infer.sim_dt; d.xfrc_applied[trunk, :] = 0
        if 2.0 <= t < 2.02: d.xfrc_applied[trunk, 1 - ax] = fn
        d.ctrl = servo(home, infer.sim_dt); mujoco.mj_step(m, d)
        if k % 2 == 0:
            T.append(t); G.append(np.degrees(infer.get_gyro(d)[ax]))
            R = d.site_xmat[m.site("imu").id].reshape(3, 3); TL.append(np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(R[2, 0])])[ax])
    T, G, TL = map(np.array, (T, G, TL))
    w = T >= 2.0; tt, gg = T[w] - 2.0, G[w]; j = np.argmax(np.abs(gg[:100])); tt, gg = tt[j:] - tt[j], gg[j:]
    gg = gg[tt < 1.0]; tt = tt[tt < 1.0]
    zc = np.flatnonzero(np.diff(np.sign(gg)) != 0)
    f = (len(zc) - 1) / (2 * (tt[zc[-1]] - tt[zc[0]])) if len(zc) > 2 else np.nan
    pk = [np.abs(gg[zc[i]:zc[i + 1] + 1]).max() for i in range(len(zc) - 1)]
    dec = [np.log(pk[i] / pk[i + 2]) for i in range(len(pk) - 2) if pk[i + 2] > 0.3]
    zeta = np.median(dec) / np.sqrt(4 * np.pi ** 2 + np.median(dec) ** 2) if dec else np.nan
    return f, zeta, TL[T < 2].mean()


def fit(loc):
    k, c = [5.1, 4.0], [0.03, 0.03]
    for it in range(8):
        s = make(loc, k[0], c[0], k[1], c[1])
        r = [tap(s, 0, 4.0), tap(s, 1, 8.0)]
        print(f"L{loc} 맞춤 {it}: 좌우 k {k[0]:.3g} d {c[0]:.3g} → {r[0][0]:.2f} Hz ζ {r[0][1]:.2f} 기울기 {r[0][2]:+.2f}° | "
              f"앞뒤 k {k[1]:.3g} d {c[1]:.3g} → {r[1][0]:.2f} Hz ζ {r[1][1]:.2f} 기울기 {r[1][2]:+.2f}°", flush=True)
        ok = all(abs(r[i][0] - F_TARGET[i]) < 0.1 and abs(r[i][1] - ZETA) < 0.02 for i in (0, 1))
        if ok: break
        for i in (0, 1):
            if np.isfinite(r[i][0]): k[i] *= (F_TARGET[i] / r[i][0]) ** 2
            else: k[i] *= 2
            if np.isfinite(r[i][1]) and r[i][1] > 0: c[i] *= np.clip(ZETA / r[i][1], 0.5, 2.0)
    return s


def sway(scene, pol, acc, vx, ns=4, kick=0.0, vy=0.0, wz=0.0):
    """sim_sway.py 와 같은 걷기. kick 초는 Runtime real_walk.py --kick-vx 0.111 처럼 출발만 0.111 로 준다.
    (걷기 0.5~3.5 s 자이로 RMS x, y 평균, 넘어짐 수, 10 s 평균 전진 속도, 걷기 0.5~10 s 몸 기준 평균 (vx, vy, yaw 속도))"""
    infer = _infer(scene, pol)
    m, d = infer.model, infer.data
    dt = infer.sim_dt * infer.decimation; home = infer.default_actuator.copy()
    R = []; fell = 0; V = []; B = []
    for seed in range(ns):
        np.random.seed(seed)
        mujoco.mj_resetData(m, d); mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
        for a in ("last_action", "last_last_action", "last_last_last_action"): setattr(infer, a, np.zeros(infer.num_dofs))
        infer.motor_targets = home.copy(); infer.prev_motor_targets = home.copy(); infer.imitation_i = 0
        delayed = ActionDelay(home, "train", seed=seed); infer.servo = ServoProfile(home, accel=acc); goal = home.copy()
        T, W = [], []; k = 0; x0 = None
        for _ in range(int(16.0 / dt) * infer.decimation):
            d.ctrl = infer.servo(goal, infer.sim_dt); mujoco.mj_step(m, d); k += 1
            if k % infer.decimation: continue
            t = k * infer.sim_dt
            if x0 is None and t >= 3.0: x0 = d.qpos[0]
            if abs(t - 13.0) < dt / 2: V.append((d.qpos[0] - x0) / 10.0)
            walk = 3.0 <= t < 13.0
            infer.commands = [(0.111 if t < 3.0 + kick else vx) if walk else 0.0, vy if walk else 0.0, wz if walk else 0.0, 0, 0, 0, 0]
            if 3.5 <= t < 13.0:
                Rb = d.xmat[1].reshape(3, 3); B.append([*(Rb.T @ d.qvel[0:3])[:2], d.qvel[5]])
            infer.imitation_i = (infer.imitation_i + infer.phase_frequency_factor) % infer.PRM.nb_steps_in_period
            ph = infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi
            infer.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
            act = infer.policy.infer(infer.get_obs(d, infer.commands))
            infer.last_last_last_action = infer.last_last_action.copy(); infer.last_last_action = infer.last_action.copy(); infer.last_action = act.copy()
            tg = infer.targets_from_action(act)
            tg = np.clip(tg, infer.prev_motor_targets - infer.max_motor_velocity * dt, infer.prev_motor_targets + infer.max_motor_velocity * dt)
            tg = np.clip(tg, infer.ctrl_lower, infer.ctrl_upper)
            infer.motor_targets = tg.copy(); infer.prev_motor_targets = tg.copy(); goal = delayed(tg)
            T.append(t); W.append(np.degrees(np.asarray(infer.get_gyro(d))))
        fell += (d.xmat[1].reshape(3, 3)[2, 2]) < 0.5
        T, W = np.array(T), np.array(W); w1 = (T >= 3.5) & (T < 6.5)
        R.append([np.sqrt(np.mean(W[w1, 0] ** 2)), np.sqrt(np.mean(W[w1, 1] ** 2))])
    R = np.array(R)
    return R[:, 0].mean(), R[:, 1].mean(), int(fell), float(np.mean(V)) if V else np.nan, tuple(np.mean(B, axis=0))


if __name__ == "__main__":
    loc = int(sys.argv[1]); s = fit(loc)
    for pol in sys.argv[2:]:
        for vx, kick in ((0.05, 0.0), (0.05, 0.5), (0.111, 0.0)):
            for acc in (0.0, 38.8):
                x, y, f, v, _ = sway(s, pol, acc, vx, kick=kick)
                print(f"L{loc} {Path(pol).stem} vx {vx}{' 출발 0.111 0.5 s' if kick else ''} 가속 {acc}: 자이로 RMS x {x:.1f} y {y:.1f}  넘어짐 {f}/4  월드 전진 {v:+.3f}", flush=True)
