import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_aloha.constants import (
    ACTIONS,
    DT,
    JOINTS,
    MENAGERIE_ASSETS_DIR,
    START_ARM_POSE,
)
from gym_aloha.tasks.sim_menagerie import BOX_POSE, CAMERA_LIST, InsertionTask, TransferCubeTask
from gym_aloha.utils import sample_insertion_pose, sample_transfer_box_pose


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array", "rgb_array_list"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array_list",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        camera_list=CAMERA_LIST,
        stage=0,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_list = camera_list
        self.stage = stage
        self._env = self._make_env_task(self.task)

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            self.observation_space = self.get_cams()
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": self.get_cams(),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

    def get_cams(self):
        cam_dict = {}
        for cam in self.camera_list:
            cam_dict[cam] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
        return spaces.Dict(cam_dict)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode in ["rgb_array", "rgb_array_list"]
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        if self.render_mode == "rgb_array_list":
            images = [
                self._env.physics.render(height=height, width=width, camera_id=cam)
                for cam in self.camera_list
            ]
            return images
        else:  # self.render_mode == "rgb_array"
            image = self._env.physics.render(height=height, width=width, camera_id="wrist_cam_right")
            return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = MENAGERIE_ASSETS_DIR / "aloha_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask(self.camera_list)
        elif task_name == "insertion":
            xml_path = MENAGERIE_ASSETS_DIR / "aloha_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask(self.camera_list)
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"wrist_cam_right": raw_obs["images"]["wrist_cam_right"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {cam: raw_obs["images"][cam].copy() for cam in self.camera_list},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = np.concatenate(sample_transfer_box_pose(seed, self.stage))  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        for _ in range(1000):
            self._env.physics.step()  # hotfix to prevent the gripper from getting stuck when the leader gripper is closed at the beginning TODO(jzilke)

        raw_obs = self._env._task.get_observation(self._env.physics)
        observation = self._format_raw_obs(raw_obs)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        # global START_ARM_POSE
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        START_ARM_POSE[:16] = self._env.physics.named.data.qpos[:16]
        return observation, reward, terminated, truncated, info

    def get_ncams(self):
        cams = self.get_cams()
        return len(cams.spaces)

    def close(self):
        pass

    def set_stage(self, stage):
        self.stage = stage
