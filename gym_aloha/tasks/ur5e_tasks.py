import collections

import numpy as np
from dm_control.suite import base


CAMERA_LIST = [
    "teleoperator_pov",
    "collaborator_pov",
    "overhead_cam",
    "worms_eye_cam",
]

GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.7821

class BimanualUr5ETask(base.Task):
    def __init__(self, camera_list=CAMERA_LIST, random=None):
        super().__init__(random=random)
        self.camera_list = camera_list

    def before_step(self, action, physics):
        super().before_step(action, physics)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_gripper_actuator = physics.data.actuator_length[physics.model.actuator("left/fingers_actuator").id]
        right_gripper_actuator = physics.data.actuator_length[physics.model.actuator("right/fingers_actuator").id]

        left_qpos_raw = qpos_raw[:14]
        right_qpos_raw = qpos_raw[14:]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_normalized = [(left_gripper_actuator - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)]
        right_gripper_normalized = [(right_gripper_actuator - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)]
        return np.concatenate([left_arm_qpos, left_gripper_normalized, right_arm_qpos, right_gripper_normalized])

    @staticmethod
    def get_gripper_vel(physics):
        left_gripper_id = physics.model.actuator("left/fingers_actuator").id
        right_gripper_id = physics.model.actuator("right/fingers_actuator").id
        return physics.data.actuator_velocity[left_gripper_id], physics.data.actuator_velocity[right_gripper_id]

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_vel, right_gripper_vel = BimanualUr5ETask.get_gripper_vel(physics)
        return np.concatenate([left_arm_qvel, [left_gripper_vel], right_arm_qvel, [right_gripper_vel]])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        for camera_id in self.camera_list:
            obs["images"][camera_id] = physics.render(height=480, width=640, camera_id=camera_id)

        return obs

class EmptyTask(BimanualUr5ETask):
    def get_reward(self, physics):
        return 0

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state
