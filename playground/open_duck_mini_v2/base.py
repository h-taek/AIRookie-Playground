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
"""Base classes for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from . import constants


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, constants.ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, constants.ROOT_PATH / "xmls" / "assets")
    path = constants.ROOT_PATH
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


class OpenDuckMiniV2Env(mjx_env.MjxEnv):
    """Base class for Open Duck Mini V2 environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        print(f"xml: {xml_path}")
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt
        # 실물 무릎은 목표각 진폭의 33 ~ 66 % 만 따라간다(시뮬 91 ~ 137 %). 무릎 두 축의 kp 에 이 배율을 곱한다.
        # 도메인 무작위화의 kp 배율은 이 값 위에 곱해진다.
        knee_kp_scale = self._config.get("knee_kp_scale", 1.0)
        if knee_kp_scale != 1.0:
            for name in ("left_knee", "right_knee"):
                i = self._mj_model.actuator(name).id
                self._mj_model.actuator_gainprm[i, 0] *= knee_kp_scale
                self._mj_model.actuator_biasprm[i, 1] *= knee_kp_scale

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        # 물리 백엔드 선택. "jax" 가 기존 MJX 경로이고 "warp" 가 MJWarp 다.
        # 설정에 없으면 기존 동작을 유지한다.
        self._impl = self._config.get("impl", "warp")
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._impl)
        if self._impl == "warp":
            # 모델이 iterations=1 / ls_iterations=5 라 솔버가 수렴 전에 끊긴다.
            # MJWarp 은 그때마다 커널에서 경고를 찍는데, 학습 규모에서는 로그가
            # 분당 10 GB 로 불고 커널이 호스트로 출력을 내보내느라 동기화가 걸려
            # 속도가 물리 연산이 아니라 로그 출력에 지배된다.
            #
            # MJX 는 같은 설정에서 같은 상태인데 경고를 내지 않는다. 두 백엔드를
            # 같은 조건으로 두기 위해 경고만 끈다. 수렴 문제 자체는 그대로 남아
            # 있으므로 솔버 반복 수를 올릴지는 따로 판단한다.
            self._mjx_model = self._mjx_model.tree_replace(
                {"opt._impl.warn_overflow": 0}
            )
        self._xml_path = xml_path
        self.floating_base_name= [self._mj_model.jnt(k).name for k in range(0, self._mj_model.njnt) if self._mj_model.jnt(k).type == 0][0] #assuming only one floating object!
        self.actuator_names = [
            self._mj_model.actuator(k).name for k in range(0, self._mj_model.nu)
        ]  # will be useful to get only the actuators we care about
        self.joint_names = [ #njnt = all joints (including floating base, actuators and backlash joints)
            self._mj_model.jnt(k).name for k in range(0, self._mj_model.njnt)
        ]  # all the joint (including the backlash joints)
        # 엔코더에 보이는 서보 유격 관절만이다. 외장 유격(`*_play`)은 관측에 더하지 않는다.
        self.backlash_joint_names = [
            j for j in self.joint_names if j.endswith("_backlash")
        ]  # only the dummy backlash joint
        self.all_joint_ids = [self.get_joint_id_from_name(n) for n in self.joint_names]
        self.all_joint_qpos_addr = [self.get_joint_addr_from_name(n) for n in self.joint_names]

        self.actuator_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.actuator_names
        ]

        self.backlash_joint_ids=[
            self.get_joint_id_from_name(n) for n in self.backlash_joint_names
        ]

        self.backlash_joint_qpos_addr=[
            self.get_joint_addr_from_name(n) for n in self.backlash_joint_names
        ]

        self.all_qvel_addr=jp.array([self._mj_model.jnt_dofadr[jad] for jad in self.all_joint_ids])
        self.actuator_qvel_addr=jp.array([self._mj_model.jnt_dofadr[jad] for jad in self.actuator_joint_ids])

        self.actuator_joint_dict = {
            n: self.get_joint_id_from_name(n) for n in self.actuator_names
        }

        self._floating_base_qpos_addr = self._mj_model.jnt_qposadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][
            0
        ]  # Assuming there is only one floating base! the jnt_type==0 is a floating joint. 3 is a hinge

        self._floating_base_qvel_addr = self._mj_model.jnt_dofadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][
            0
        ]  # Assuming there is only one floating base! the jnt_type==0 is a floating joint. 3 is a hinge

        self._floating_base_id = self._mj_model.joint(self.floating_base_name).id

        # self.all_joint_no_backlash_ids=jp.zeros(7+self._mj_model.nu)
        # all_idx=self.actuator_joint_ids+list(range(self._floating_base_qpos_addr,self._floating_base_qpos_addr+7))
        # all_idx=jp.array(all_idx).sort()
        # 원본은 "trunk_assembly_freejoint" 를 이름으로 박아두었으나 모델에 그런 관절이
        # 없어 mj_name2id 가 -1 을 돌려주었고, 음수 인덱싱이 몸통 대신 마지막 유격 관절
        # (right_ankle_backlash)을 가리키고 있었다. 부유 바디 이름은 이미 위에서
        # jnt_type==0 으로 찾아 두었으므로 그것을 쓴다.
        all_idx=self.actuator_joint_ids+list([self.get_joint_id_from_name(self.floating_base_name)])
        all_idx=jp.array(all_idx).sort()
        # self.all_joint_no_backlash_ids=[idx for idx in self.all_joint_ids if idx not in self.backlash_joint_ids]+list(range(self._floating_base_add,self._floating_base_add+7))
        self.all_joint_no_backlash_ids=[idx for idx in all_idx]
        # print(f"ALL: {self.all_joint_no_backlash_ids} back_id: {self.backlash_joint_ids} base_id: {list(range(self._floating_base_qpos_addr,self._floating_base_qpos_addr+7))}")

        self.backlash_idx_to_add = []

        for i, actuator_name in enumerate(self.actuator_names):
            if actuator_name + "_backlash" not in self.backlash_joint_names:
                self.backlash_idx_to_add.append(i)

        print(f"actuators: {self.actuator_names}")
        print(f"joints: {self.joint_names}")
        print(f"backlash joints: {self.backlash_joint_names}")
        print(f"actuator joints ids: {self.actuator_joint_ids}")
        print(f"actuator joints dict: {self.actuator_joint_dict}")
        print(f"floating qpos addr: {self._floating_base_qpos_addr} qvel addr: {self._floating_base_qvel_addr}")



    def get_actuator_id_from_name(self, name: str) -> int:
        """Return the id of a specified actuator"""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint_id_from_name(self, name: str) -> int:
        """Return the id of a specified joint

        mj_name2id 는 이름을 못 찾으면 -1 을 돌려주고, 파이썬 음수 인덱싱은 그것을
        마지막 관절로 조용히 해석한다. 그러면 엉뚱한 관절을 읽으면서 테스트는 전부
        통과한다. 실패는 여기서 드러나야 한다.
        """
        jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"모델에서 관절 '{name}' 을 찾지 못했다")
        return jid


    def get_joint_addr_from_name(self, name: str) -> int:
        """Return the address of a specified joint"""
        return self._mj_model.joint(name).qposadr

    def get_dof_id_from_name(self, name: str) -> int:
        """Return the id of a specified dof"""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_DOF, name)


    def get_actuator_joint_qpos_from_name(self, data: jax.Array, name: str) -> jax.Array:
        """Return the qpos of a given actual joint"""
        addr = self._mj_model.jnt_qposadr[self.actuator_joint_dict[name]]
        return data[addr]

    def get_actuator_joints_qpos_addr(self) -> jax.Array:
        """Return the all the idx of actual joints"""
        addr = jp.array(
            [self._mj_model.jnt_qposadr[idx] for idx in self.actuator_joint_ids]
        )
        return addr

    def get_floating_base_qpos(self, data:jax.Array) -> jax.Array:
        # 끝 주소에 qvel 주소를 섞어 쓰고 있었다. 지금은 두 주소가 모두 0 이라 우연히
        # [0:7] 로 맞지만, 부유 바디 앞에 관절이 하나라도 생기면 두 값이 갈라지면서
        # reset 과 모방 보상이 조용히 어긋난다.
        return data[self._floating_base_qpos_addr:self._floating_base_qpos_addr+7]

    def get_floating_base_qvel(self, data:jax.Array) -> jax.Array:
        return data[self._floating_base_qvel_addr:self._floating_base_qvel_addr+6]


    def set_floating_base_qpos(self, new_qpos:jax.Array, qpos:jax.Array) -> jax.Array:
        return qpos.at[self._floating_base_qpos_addr:self._floating_base_qpos_addr+7].set(new_qpos)

    def set_floating_base_qvel(self, new_qvel:jax.Array, qvel:jax.Array) -> jax.Array:
        return qvel.at[self._floating_base_qvel_addr:self._floating_base_qvel_addr+6].set(new_qvel)


    def exclude_backlash_joints_addr(self) -> jax.Array:
        """Return the all the idx of actual joints and floating base"""
        addr = jp.array(
            [self._mj_model.jnt_qposadr[idx] for idx in self.all_joint_no_backlash_ids]
        )
        return addr


    def get_all_joints_addr(self) -> jax.Array:
        """Return the all the idx of all joints"""
        addr = jp.array([self._mj_model.jnt_qposadr[idx] for idx in self.all_joint_ids])
        return addr

    def get_actuator_joints_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of actual joints"""
        return data[self.get_actuator_joints_qpos_addr()]

    def set_actuator_joints_qpos(self, new_qpos: jax.Array, qpos: jax.Array) -> jax.Array:
        """Set the qpos only for the actual joints (omit the backlash joint)"""
        return qpos.at[self.get_actuator_joints_qpos_addr()].set(new_qpos)

    def get_actuator_backlash_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of backlash joints"""
        if self.backlash_joint_qpos_addr == []:
            return jp.array([])
        return data[jp.array(self.backlash_joint_qpos_addr)]


    def get_actuator_joints_qvel(self, data: jax.Array) -> jax.Array:
        """Return the all the qvel of actual joints"""
        return data[self.actuator_qvel_addr]

    def set_actuator_joints_qvel(self, new_qvel: jax.Array, qvel: jax.Array) -> jax.Array:
        """Set the qvel only for the actual joints (omit the backlash joint)"""
        return qvel.at[self.actuator_qvel_addr].set(new_qvel)

    def get_all_joints_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of all joints"""
        return data[self.get_all_joints_addr()]

    def get_all_joints_qvel(self, data: jax.Array) -> jax.Array:
        """Return the all the qvel of all joints"""
        return data[self.all_qvel_addr]

    def get_joints_nobacklash_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of actual joints with the floating base"""
        return data[self.exclude_backlash_joints_addr()]

    def set_complete_qpos_from_joints(self, no_backlash_qpos: jax.Array, full_qpos: jax.Array) -> jax.Array:
        """In the case of backlash joints, we want to ignore them (remove them) but we still need to set the complete state incuding them"""
        # JAX 배열은 불변이라 .at[].set() 의 반환값을 받지 않으면 아무 일도 일어나지
        # 않는다. 원본은 반환값을 버려 이 함수가 통째로 no-op 이었다.
        full_qpos = full_qpos.at[self.exclude_backlash_joints_addr()].set(no_backlash_qpos)
        return jp.array(full_qpos)

    # Sensor readings.
    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Return the gravity vector in the world frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, constants.GRAVITY_SENSOR)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        """Return the angular velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        """Return the accelerometer readings in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        """Return the gyroscope readings in the local frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, constants.GYRO_SENSOR)

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        """Return the position of the feet in the world frame."""
        return jp.vstack(
            [
                mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
                for sensor_name in constants.FEET_POS_SENSOR
            ]
        )

    # Accessors.

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
