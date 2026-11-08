"""학습 환경(`Joystick`)을 그대로 굴리되 실물 기록의 명령을 넣는다 (plan/01 4장).

학습 `step` 을 그대로 부른다. 바꾸는 것은 넷이다.
- 명령: 스텝마다 기록의 명령으로 덮어쓴다. 관측의 명령 칸도 같은 값이다
- 시작: 짝 홈 자세에서 흔들기 없이 서 있는 채로 시작하고, 걸음 위상을 기록과 맞춘다
- 종료: 넘어짐으로 끝내지 않는다. 넘어짐은 기록과 같은 가속도 식으로 밖에서 판정한다
- 지연: 학습의 스텝 단위 무작위 지연 대신 실측 지연(`HOLD_SUBSTEPS`)을 물리 스텝 단위로 넣는다
- 관측: 실물이 정책에 넣은 값과 같게 만든다. 관절 속도는 엔코더 각도의 20 ms 차분이다(학습은 순간 qvel,
  실물은 Runtime `tools/real_walk.py` · `real/servo.py` 의 차분). 걷기 시험 도구 기록(`log_*`)은 IMU 장착각을
  되돌리지 않은 자이로 · 가속도로 추론했으므로 그 기록의 정책 입력도 장착각만큼 돌린다

환경마다 다른 값(몸통 · 머리 무게중심 x · y, 다리 · 머리 외장 유격, 서보 유격, 바닥 마찰)은 학습의
`BraxDomainRandomizationVmapWrapper` 와 같이 모델 필드를 환경 축으로 늘려 넣는다.
"""
import contextlib

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from playground.open_duck_mini_v2 import base as B
from playground.open_duck_mini_v2.joystick import Joystick, default_config
from real_fit import model as M

PERIOD = 27  # 걸음 주기 스텝(실물 기록 위상과 학습 시범 동작이 같다)
KNEES = (3, 12)
# 실물 서보 가속 레지스터 253 을 흉내 내는 서보 내부 가속 상한(rad/s²). default_config 주석의 동정값이다.
SERVO_ACCEL_253 = 38.8
# 센서를 읽기 시작해 목표각이 서보에 닿기까지 15.3 ~ 15.5 ms 다(실물 진입점 기록 cycle_ms 중앙, 14 축 읽기
# 6.8 ms · 쓰기 0.6 ms, Runtime research/10 2 · 18장). 물리 스텝 2 ms 로 8 스텝(16 ms) 동안 직전 목표각을 둔다.
HOLD_SUBSTEPS = 8
# IMU 가 몸통 수평에서 보고하는 앞숙임 −0.74°(Runtime real/config.json mount_pitch_deg). 몸통 좌표 = R_MOUNT @ IMU 좌표.
_c, _s = np.cos(np.radians(-0.74)), np.sin(np.radians(-0.74))
R_MOUNT = np.array([[_c, 0.0, _s], [0.0, 1.0, 0.0], [-_s, 0.0, _c]])


def fit_config(impl: str, n_envs: int, servo_accel: float, voltage_drop: bool = False):
    c = default_config()
    c.impl = impl
    c.naconmax = 8 * n_envs
    c.noise_config.level = 0.0
    # 학습의 스텝 단위 지연은 끈다(기본 지연 0, 흔들림 0). 지연은 _servo_step 의 HOLD_SUBSTEPS 다.
    c.noise_config.action_min_delay = 0
    c.noise_config.action_max_delay = 1
    c.noise_config.action_delay_jitter = 0.0
    # 발 접촉 관측 지연도 끈다. 학습은 잡음 수준과 상관없이 0 ~ 2 스텝을 뽑는다. 실물 발 스위치는 주기마다 바로 읽는다.
    c.noise_config.contact_min_delay = 0
    c.noise_config.contact_max_delay = 1
    c.push_config.enable = False
    c.episode_length = 10 ** 9
    c.servo_max_accel = float(servo_accel)
    # 전압 강하는 끈다(걷기에 영향 없음으로 판정). 최종 점 확인에서만 켠다(plan/01 1장).
    c.voltage_drop.enable = bool(voltage_drop)
    return c


class FitEnv(Joystick):
    def __init__(self, xml_path, config):
        B.OpenDuckMiniV2Env.__init__(self, xml_path=str(xml_path), config=config)
        self._post_init()
        self._train_frac = 0.0

    def _servo_step(self, data: mjx.Data, goal: jax.Array, info: dict):
        """학습의 `_servo_step` 과 같되, 처음 HOLD_SUBSTEPS 물리 스텝은 직전 목표각을 쓴다."""
        prev = info["fit_prev_goal"]
        accel = info["servo_accel"]
        dt = self._config.sim_dt
        emulate = self._config.servo_max_accel > 0
        vd = self._config.voltage_drop

        def substep(carry, i):
            d, pos, vel, scale = carry
            g = jp.where(i < HOLD_SUBSTEPS, prev, goal)
            if emulate:
                err = g - pos
                want = jp.sign(err) * jp.sqrt(2.0 * accel * jp.abs(err))
                vel = vel + jp.clip(want - vel, -accel * dt, accel * dt)
                pos = pos + vel * dt
            else:
                pos = g
            ctrl = pos
            if vd.enable:  # 학습 원본과 같은 식
                ext = (d.actuator_force - self._act_damping * d.qvel[self._act_dof]
                       - self._act_armature * d.qacc[self._act_dof])
                current = vd.cur_gain * jp.maximum(jp.abs(ext) - vd.cur_hold, 0.0)
                port_current = jp.zeros(2).at[self._actuator_port].add(current)
                volt = vd.v_nominal - port_current @ info["port_r"]
                scale = jp.clip(volt / vd.v_nominal, vd.min_scale, 1.0)
                q = self.get_actuator_joints_qpos(d.qpos)
                reach = self._reach_line * scale
                ctrl = q + jp.clip((ctrl - q) * scale, -reach, reach)
            d = mjx.step(self.mjx_model, d.replace(ctrl=ctrl))
            return (d, pos, vel, scale), None

        (data, pos, vel, scale), _ = jax.lax.scan(
            substep, (data, info["servo_pos"], info["servo_vel"], info["force_scale"]), jp.arange(self.n_substeps))
        info["force_scale"] = scale
        info["fit_prev_goal"] = goal
        return data, pos, vel

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        """보상은 맞춤에 쓰지 않는다. 학습 원본은 info 를 읽기만 하므로 0 으로 바꿔도 시뮬 상태가 같다."""
        return {k: jp.zeros(()) for k in self._config.reward_config.scales}

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        # 넘어져도 info 를 첫 상태로 돌리지 않게 한다. 실물에는 그런 되돌림이 없다.
        return jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()


@contextlib.contextmanager
def _swap(env: FitEnv, mjx_model, home, init_q):
    """환경 하나의 모델과 짝 홈 자세를 끼운다. 학습 환경은 홈을 `_default_actuator` · `_init_q` 로 쥔다.
    행동 배율은 다리가 상수이고 머리는 명령 직결이라 짝마다 바꿀 필요가 없다."""
    old = env._mjx_model, env._default_actuator, env._init_q
    try:
        env._mjx_model, env._default_actuator, env._init_q = mjx_model, home, init_q
        yield env
    finally:
        env._mjx_model, env._default_actuator, env._init_q = old


def batched_model(env: FitEnv, p: dict):
    """p: 환경 축 배열. dx · dy 몸통 무게중심, hx · hy 머리 무게중심(m, 몸통 기준 + 앞 · + 왼),
    leg · head 외장 유격 반각, backlash 서보 유격 반각(rad, 10 축 같은 값), floor 바닥 마찰."""
    m = env.mjx_model
    mj = env.mj_model
    trunk = mj.body(M.TRUNK).id
    floor = mj.geom("floor").id
    leg = jp.array(M.leg_play_joints(mj))
    head = jp.array(M.head_play_joints(mj))
    bl = jp.array(M.backlash_joints(mj))
    hb = mj.body(M.HEAD).id
    # 머리 바디 축은 몸통과 다르다. 홈 자세에서 몸통 좌표 → 머리 바디 좌표 회전을 구해 둔다.
    import mujoco
    d0 = mujoco.MjData(mj)
    d0.qpos[:] = mj.keyframe("home").qpos
    mujoco.mj_kinematics(mj, d0)
    to_head = jp.asarray(d0.xmat[hb].reshape(3, 3).T @ d0.xmat[trunk].reshape(3, 3))

    def one(dx, dy, hx, hy, lp, hp, fr, blh):
        ipos = m.body_ipos.at[trunk, 0].add(dx).at[trunk, 1].add(dy)
        ipos = ipos.at[hb].add(to_head @ jp.array([hx, hy, 0.0]))
        jr = m.jnt_range
        jr = jr.at[leg, 0].set(-lp).at[leg, 1].set(lp)
        jr = jr.at[head, 0].set(-hp).at[head, 1].set(hp)
        jr = jr.at[bl, 0].set(-blh).at[bl, 1].set(blh)
        gf = m.geom_friction.at[floor, 0].set(fr)
        return ipos, jr, gf

    ipos, jr, gf = jax.vmap(one)(p["dx"], p["dy"], p["hx"], p["hy"], p["leg"], p["head"], p["floor"], p["backlash"])
    model_v = m.tree_replace({"body_ipos": ipos, "jnt_range": jr, "geom_friction": gf})
    in_axes = jax.tree_util.tree_map(lambda x: None, m)
    in_axes = in_axes.tree_replace({"body_ipos": 0, "jnt_range": 0, "geom_friction": 0})
    return model_v, in_axes


def _real_obs(env, st, prev_ang, raw):
    """학습 관측을 실물 관측식으로 바꾼다. 관절 속도 = 엔코더 각도 차분, raw 면 IMU 를 장착각만큼 돌린다."""
    ang = st[13:27]
    st = st.at[27:41].set((ang - prev_ang) / env.dt * env._config.dof_vel_scale)
    rt = jp.asarray(R_MOUNT.T, jp.float32)
    st = st.at[0:3].set(jp.where(raw, rt @ st[0:3], st[0:3]))
    st = st.at[3:6].set(jp.where(raw, rt @ st[3:6], st[3:6]))
    return st, ang


def make_rollout(env: FitEnv, act_fn, cmd_tab: jax.Array, n_steps: int, debug: bool = False):
    """cmd_tab: (기록, T+1, 7). act_fn(obs, pid). 돌려주는 함수는
    (model_v, in_axes, rec, pid, home, init_q, phase0, contact_fixed, raw_imu, rng) 를 받아 — rng 밖은 환경 축 —
    주기마다 (가속도 3, 무릎 관절각 − 홈 2, 무릎 목표각 2) 를 낸다. 첫 값은 출발 관측(주기 0)이다.
    정책 · 짝 홈 자세가 환경마다 달라 정책 여럿의 기록을 한 번에 굴린다."""

    def reset_one(model_i, home, init_q, rng, rec, i0, raw):
        with _swap(env, model_i, home, init_q) as e:
            s = e.reset(rng)
            data = mjx_env.make_data(
                e.mj_model, qpos=e._init_q, qvel=jp.zeros(e.mjx_model.nv), ctrl=e._default_actuator,
                impl=e.mjx_model.impl.value, naconmax=e._config.naconmax, njmax=e._config.njmax)
            data = mjx.forward(e.mjx_model, data)
            info = dict(s.info)
            c0 = cmd_tab[rec, 0]
            info.update(command=c0, command_target=c0, command_init=c0, imitation_i=i0, fit_prev_goal=e._default_actuator,
                        imitation_phase=jp.array([jp.cos(i0 / PERIOD * 2 * jp.pi), jp.sin(i0 / PERIOD * 2 * jp.pi)]))
            contact = data.sensordata[e._feet_floor_found_adr] > 0
            obs = e._get_obs(data, info, contact)
            st, ang = _real_obs(e, obs["state"], obs["state"][13:27], raw)  # 첫 관측의 관절 속도는 0
            info["fit_prev_ang"] = ang
            return s.replace(data=data, obs=dict(obs, state=st), info=info)

    def step_one(model_i, home, init_q, s, a, c_now, c_next, cfix, raw):
        info = dict(s.info)
        info["command"] = c_now  # 이번 스텝 머리 목표각(명령 직결)에 쓰인다
        s = s.replace(info=info)
        with _swap(env, model_i, home, init_q) as e:
            s2 = e.step(s, a)
        info = dict(s2.info)
        info["command"] = c_next
        st, info["fit_prev_ang"] = _real_obs(env, s2.obs["state"], s.info["fit_prev_ang"], raw)
        st = st.at[6:13].set(c_next)
        st = jp.where(cfix, st.at[97:99].set(1.0), st)
        return s2.replace(info=info, obs=dict(s2.obs, state=st))

    def rollout(model_v, in_axes, rec, pid, home, init_q, phase0, cfix, raw, rng):
        n = rec.shape[0]
        # 난수는 기록 번호에서 만든다. 같은 기록 · 같은 값이면 묶음 구성과 상관없이 같은 결과가 나온다.
        keys = jax.vmap(lambda r: jax.random.fold_in(rng, r))(rec)
        s = jax.vmap(reset_one, in_axes=[in_axes, 0, 0, 0, 0, 0, 0])(model_v, home, init_q, keys, rec, phase0, raw)
        s = s.replace(obs=dict(s.obs, state=jp.where(cfix[:, None], s.obs["state"].at[:, 97:99].set(1.0), s.obs["state"])))
        vstep = jax.vmap(step_one, in_axes=[in_axes, 0, 0, 0, 0, 0, 0, 0, 0])
        r_mount = jp.asarray(R_MOUNT, jp.float32)

        def body(s, k):
            st = s.obs["state"]
            a = act_fn(st, pid)
            s = vstep(model_v, home, init_q, s, a, cmd_tab[rec, k], cmd_tab[rec, k + 1], cfix, raw)
            tg = s.info["motor_targets"][:, jp.array(KNEES)]
            # 점수는 몸통 좌표 가속도로 낸다. raw 환경의 관측은 IMU 좌표라 되돌린다.
            acc = jp.where(raw[:, None], st[:, 3:6] @ r_mount.T, st[:, 3:6])
            out = (acc, st[:, 13:27][:, jp.array(KNEES)], tg)
            return s, (out + (st[:, 6:13], st[:, 97:99]) if debug else out)

        _, out = jax.lax.scan(body, s, jp.arange(n_steps))
        return out

    return rollout

