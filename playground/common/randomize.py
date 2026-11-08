# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for the Open Duck Mini V2 environment. (based on Berkeley Humanoid)"""

import jax
import numpy as np
from mujoco import mjx
import jax.numpy as jp

# 바닥 지오메트리와 몸통 바디는 이름으로 찾는다.
#
# 원본은 FLOOR_GEOM_ID = 0, TORSO_BODY_ID = 1 로 상수를 박아두었는데 현재 모델에서는
# 둘 다 다른 대상을 가리켰다. 0 번 지오메트리는 몸통 껍데기 메시이고, 1 번 바디는
# 질량 0 인 `base` 래퍼다. 그 결과 바닥 마찰은 한 번도 랜덤화되지 않고 0.6 에 고정되어
# 있었고, 몸통 무게중심 흔들기는 질량 0 인 바디에 적용되어 아무 효과가 없었다.
# 상수를 고쳐도 모델이 바뀌면 같은 사고가 반복되므로 이름 조회로 바꾼다.
FLOOR_GEOM_NAME = "floor"
TORSO_BODY_NAME = "trunk_assembly"

# 몸통 무게중심 흔들기 폭(m). 축마다 독립으로 +-이 범위에서 뽑는다.
#
# 원본은 0.05 였으나 위에 적은 이유로 한 번도 적용된 적이 없어 승계할 근거가 없다.
# 0.02 는 다음 근거로 정했다. 몸통은 전체 질량의 41.8 % 이므로 몸통 무게중심 20 mm 는
# 전체 무게중심 8.4 mm 에 해당하고, 이는 홈 자세 전후 여유 31.8 mm 의 26 % 다.
# 실제 미반영분(총질량 잔차 -1.67 %, 링크별 평균 오차 4.1 %, 배선과 체결부품,
# 강압기 장착 위치 미확정분 2.3 mm)을 덮으면서 정지 평형을 깨지 않는 크기다.
# 근거는 docs/archive/research/03_hardware_spec_derivation.md M-19 참조.
TORSO_IPOS_JITTER_M = 0.02

# 관절 유격 (M-06) 랜덤화 범위. 전체 유격 기준 도 단위.
# 데이터시트 0.5도(서보 단품 신품), 제3자 실측 0.87도(서보 단품).
# 조립 관절은 출력혼과 브래킷 유격이 더해져 단품보다 크므로 상한을 넉넉히 잡는다.
# 실측값이 확보되면 그 값을 중심으로 폭을 좁힌다.
BACKLASH_TOTAL_MIN_DEG = 0.6
BACKLASH_TOTAL_MAX_DEG = 1.8

# 외장 유격(`*_play` 관절) 반각 범위, 도. 다리 엉덩이 피치 · 무릎 · 발목에만 있고 관측에 보이지 않는다.
# 실측 몸통 밀기 약 5° 를 세 관절에 나눈 0.85° 가 가운데다(Runtime research/10 17-5, Playground research/04 28장).
SHELL_PLAY_HALF_MIN_DEG = 0.7
SHELL_PLAY_HALF_MAX_DEG = 1.0
SHELL_PLAY_SUFFIX = "_play"

_MJ_JNT_HINGE = 3


def _find_backlash_joints(model: mjx.Model):
    """유격 관절 id 목록. 힌지이면서 한계가 설정되어 있고 마찰이 0인 관절이다.

    유격 관절은 구동되지 않는 자유 관절이라 frictionloss 를 0 으로 둔다.
    구동 관절은 전부 frictionloss 가 0 이 아니므로 이 조건으로 갈린다.
    """
    return [
        j
        for j in range(model.njnt)
        if model.jnt_type[j] == _MJ_JNT_HINGE
        and model.jnt_limited[j]
        and model.dof_frictionloss[model.jnt_dofadr[j]] == 0
    ]


def _joint_name(model: mjx.Model, j: int) -> str:
    names = bytes(model.names)
    adr = int(np.asarray(model.name_jntadr)[j])
    return names[adr : names.index(b"\x00", adr)].decode()


def _name_to_id(model: mjx.Model, name_adr, name: str, kind: str) -> int:
    """이름으로 바디/지오메트리 인덱스를 찾는다.

    mjx.Model 은 mj_name2id 를 쓸 수 없으므로 `names` 버퍼를 직접 뒤진다.
    못 찾으면 조용히 0 을 돌려주지 않고 예외를 낸다. 잘못된 인덱스로 엉뚱한 대상을
    랜덤화하는 것이 이 함수가 생긴 이유이므로 실패는 드러나야 한다.
    """
    names = bytes(model.names)
    target = name.encode()
    for idx, adr in enumerate(np.asarray(name_adr)):
        adr = int(adr)
        if names[adr : names.index(b"\x00", adr)] == target:
            return idx
    raise ValueError(f"모델에서 {kind} '{name}' 을 찾지 못했다")


def domain_randomize(model: mjx.Model, rng: jax.Array):

    floor_geom_id = _name_to_id(model, model.name_geomadr, FLOOR_GEOM_NAME, "지오메트리")
    torso_body_id = _name_to_id(model, model.name_bodyadr, TORSO_BODY_NAME, "바디")

    # _dof_addr=jp.array([6,8,10,12,14,16,18,20,22,24])
    # _joint_addr=jp.array([7,9,11,13,15,17,19,21,23,25])

    # 유격 관절은 frictionloss 가 0 이다. 그것으로 구동 자유도를 가린다.
    #
    # 원본은 model.dof_hasfrictionloss 를 읽었는데 그 필드는 Warp 모델에 없고
    # jax 모델에서도 model._impl 아래로 옮겨져 폐기 예정이다. dof_frictionloss 는
    # 두 백엔드 모두 갖고 있고 값이 같다(현재 모델에서 동치 확인).
    dof_id = jp.array(
        [idx for idx, fr in enumerate(np.asarray(model.dof_frictionloss)) if fr != 0]
    )
    jnt_id = model.dof_jntid[dof_id]

    # 원본은 `[jadd for jadd in model.jnt_dofadr if jadd in dof_id]` 였다. 자유도 "주소"와
    # 자유도 "인덱스"를 비교하는 코드인데, 지금은 모든 관절이 1 자유도 힌지라 두 값이
    # 우연히 같아서 맞는 결과가 나왔다. 2 자유도 관절이 하나라도 생기면 조용히 어긋난다.
    dof_addr = model.jnt_dofadr[jnt_id]
    joint_addr = model.jnt_qposadr[jnt_id]

    free_jnt = _find_backlash_joints(model)
    play_jnt = [j for j in free_jnt if _joint_name(model, j).endswith(SHELL_PLAY_SUFFIX)]
    backlash_jnt = [j for j in free_jnt if j not in play_jnt]
    n_backlash = len(backlash_jnt)
    n_play = len(play_jnt)
    play_id = jp.array(play_jnt, dtype=jp.int32)
    backlash_id = jp.array(backlash_jnt, dtype=jp.int32)
    # 전체 유격을 반각(라디안)으로 환산한다. range 는 -반각 ~ +반각이다.
    backlash_half_min = jp.deg2rad(BACKLASH_TOTAL_MIN_DEG / 2.0)
    backlash_half_max = jp.deg2rad(BACKLASH_TOTAL_MAX_DEG / 2.0)

    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.5, 1.0). XML 공칭값 0.6 은 이 범위의 아래쪽이다.
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[floor_geom_id, 0].set(
            jax.random.uniform(key, minval=0.5, maxval=1.0)  # was 0.4, 1.0
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[dof_addr] * jax.random.uniform(
            key, shape=(model.nu,), minval=0.9, maxval=1.1
        )
        dof_frictionloss = model.dof_frictionloss.at[dof_addr].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[dof_addr] * jax.random.uniform(
            key, shape=(model.nu,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[dof_addr].set(armature)

        # Jitter center of mass positiion: +U(-TORSO_IPOS_JITTER_M, +TORSO_IPOS_JITTER_M).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(
            key, (3,), minval=-TORSO_IPOS_JITTER_M, maxval=TORSO_IPOS_JITTER_M
        )
        body_ipos = model.body_ipos.at[torso_body_id].set(
            model.body_ipos[torso_body_id] + dpos
        )

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(-0.2, 0.2).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-0.1, maxval=0.1)  # was -0.2, 0.2
        body_mass = body_mass.at[torso_body_id].set(body_mass[torso_body_id] + dmass)

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[joint_addr].set(
            qpos0[joint_addr]
            + jax.random.uniform(
                key, shape=(model.nu,), minval=-0.03, maxval=0.03
            )  # was -0.05 0.05
        )

        # # Randomize KP
        rng, key = jax.random.split(rng)
        factor = jax.random.uniform(
            key, shape=(model.nu,), minval=0.9, maxval=1.1
        )  # was 0.8, 1.2
        current_kp = model.actuator_gainprm[:, 0]
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(current_kp * factor)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-current_kp * factor)

        # 관절 유격: 전체 U(BACKLASH_TOTAL_MIN_DEG, BACKLASH_TOTAL_MAX_DEG) 도.
        # 관절마다 독립으로 뽑는다. 좌우 유격이 같을 근거가 없다.
        rng, key = jax.random.split(rng)
        half = jax.random.uniform(
            key,
            shape=(n_backlash,),
            minval=backlash_half_min,
            maxval=backlash_half_max,
        )
        jnt_range = model.jnt_range.at[backlash_id, 0].set(-half)
        jnt_range = jnt_range.at[backlash_id, 1].set(half)

        # 외장 유격: 반각 U(SHELL_PLAY_HALF_MIN_DEG, SHELL_PLAY_HALF_MAX_DEG) 도, 관절마다 독립.
        if n_play > 0:
            rng, key = jax.random.split(rng)
            half_play = jax.random.uniform(
                key,
                shape=(n_play,),
                minval=jp.deg2rad(SHELL_PLAY_HALF_MIN_DEG),
                maxval=jp.deg2rad(SHELL_PLAY_HALF_MAX_DEG),
            )
            jnt_range = jnt_range.at[play_id, 0].set(-half_play)
            jnt_range = jnt_range.at[play_id, 1].set(half_play)

        return (
            geom_friction,
            body_ipos,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
            actuator_gainprm,
            actuator_biasprm,
            jnt_range,
        )

    (
        friction,
        body_ipos,
        frictionloss,
        armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
        jnt_range,
    ) = rand_dynamics(rng)

    randomized = {
        "geom_friction": friction,
        "body_ipos": body_ipos,
        "dof_frictionloss": frictionloss,
        "dof_armature": armature,
        "body_mass": body_mass,
        "qpos0": qpos0,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    }
    # 유격 관절이 없는 모델(비 backlash task)에서는 jnt_range 를 건드리지 아니한다.
    if n_backlash > 0 or n_play > 0:
        randomized["jnt_range"] = jnt_range

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({k: 0 for k in randomized})

    model = model.tree_replace(randomized)

    return model, in_axes
