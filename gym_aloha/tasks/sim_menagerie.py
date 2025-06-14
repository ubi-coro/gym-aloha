import collections

import numpy as np
from dm_control.suite import base

from gym_aloha.constants import (
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)

BOX_POSE = [None]  # to be changed from outside
CAMERA_LIST = [
    "wrist_cam_right",
    "wrist_cam_left",
    "teleoperator_pov",
    "collaborator_pov",
    "overhead_cam",
    "worms_eye_cam",
]
"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""


class BimanualViperXTask(base.Task):
    def __init__(self, camera_list=CAMERA_LIST, random=None):
        super().__init__(random=random)
        self.camera_list = camera_list

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = unnormalize_puppet_gripper_position(normalized_left_gripper_action)
        right_gripper_action = unnormalize_puppet_gripper_position(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action]
        full_right_gripper_action = [right_gripper_action]

        env_action = np.concatenate(
            [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

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

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, camera_list=CAMERA_LIST, random=None):
        super().__init__(camera_list=camera_list, random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            start_pose = START_ARM_POSE[:7] + START_ARM_POSE[8:15]
            np.copyto(physics.data.ctrl, start_pose)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0][:7]
            physics.model.body_pos[physics.model.name2id("goal_pos", "body")] = BOX_POSE[0][7:10]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_goal = ("goal_pos", "red_box") in all_contact_pairs

        in_left_gripper = (
            ("left/left_g0", "red_box") in all_contact_pairs
            or ("left/left_g1", "red_box") in all_contact_pairs
            or ("left/right_g0", "red_box") in all_contact_pairs
            or ("left/right_g1", "red_box") in all_contact_pairs
        )
        in_right_gripper = (
            ("right/left_g0", "red_box") in all_contact_pairs
            or ("right/left_g1", "red_box") in all_contact_pairs
            or ("right/right_g0", "red_box") in all_contact_pairs
            or ("right/right_g1", "red_box") in all_contact_pairs
        )
        in_gripper = in_left_gripper or in_right_gripper

        touch_table = ("table", "red_box") in all_contact_pairs

        reward = 0
        if in_gripper and touch_table:
            reward = 1
        if not touch_table:
            reward = 2
        if touch_goal and in_gripper:
            reward = 3
        if touch_goal and not in_gripper:  # successful placement
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, camera_list=CAMERA_LIST, random=None):
        super().__init__(camera_list=camera_list, random=random)
        self.max_reward = 4  # TODO(jzilke)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            start_pose = START_ARM_POSE[:7] + START_ARM_POSE[8:15]
            np.copyto(physics.data.ctrl, start_pose)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]
            # physics.named.data.qpos[-7:] = BOX_POSE[0][:7]
            # physics.model.body_pos[physics.model.name2id("goal_pos", "body")] = BOX_POSE[0][7:10]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            all_contact_pairs.append((name_geom_1, name_geom_2))
            all_contact_pairs.append((name_geom_2, name_geom_1))

        touch_right_gripper = ("red_peg", "right/left_g0") in all_contact_pairs or (
            "red_peg",
            "right/left_g1",
        ) in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "left/left_g0") in all_contact_pairs
            or ("socket-2", "left/left_g0") in all_contact_pairs
            or ("socket-3", "left/left_g0") in all_contact_pairs
            or ("socket-4", "left/left_g0") in all_contact_pairs
            or ("socket-1", "left/left_g1") in all_contact_pairs
            or ("socket-2", "left/left_g1") in all_contact_pairs
            or ("socket-3", "left/left_g1") in all_contact_pairs
            or ("socket-4", "left/left_g1") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward


class StackingTask(BimanualViperXTask):
    def __init__(self, camera_list=CAMERA_LIST, random=None):
        super().__init__(camera_list=camera_list, random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            start_pose = START_ARM_POSE[:7] + START_ARM_POSE[8:15]
            np.copyto(physics.data.ctrl, start_pose)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 3 :] = BOX_POSE[0]
            # physics.named.data.qpos[-7:] = BOX_POSE[0][:7]
            # physics.model.body_pos[physics.model.name2id("goal_pos", "body")] = BOX_POSE[0][7:10]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            all_contact_pairs.append((name_geom_1, name_geom_2))
            all_contact_pairs.append((name_geom_2, name_geom_1))

        stack_1 = ("box_1", "box_2") in all_contact_pairs and ("box_2", "table") not in all_contact_pairs

        stack_2 = (
            stack_1
            and ("box_2", "box_3") in all_contact_pairs
            and ("box_3", "table") not in all_contact_pairs
        )

        reward = 0
        if stack_1:
            reward = 1
        if stack_2:
            reward = 2

        box3_pos = physics.named.data.xpos["box_3"]
        grip_right_pos = physics.named.data.xpos["right/right_finger_link"]
        grip_left_pos = physics.named.data.xpos["left/left_finger_link"]
        distance_right = np.linalg.norm(grip_right_pos - box3_pos)
        distance_left = np.linalg.norm(grip_left_pos - box3_pos)

        if stack_2 and min(distance_right, distance_left) > 0.4:
            reward = self.max_reward
        return reward
