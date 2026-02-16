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
from gym_aloha.tasks.sim_menagerie import BOX_POSE, CAMERA_LIST, InsertionTask, StackingTask, TransferCubeTask
from gym_aloha.tasks.ur5e_tasks import EmptyTask


class Ur5eEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "rgb_array_list"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array_list",
        observation_width=640,
        observation_height=480,
        camera_list=["teleoperator_pov"],
    ):
        super().__init__()
        if camera_list is None:
            camera_list = []
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.camera_list = camera_list
        self._env = self._make_env_task(self.task)

        if self.obs_type == "pixels":
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

    def _render(self):
        assert self.render_mode in ["rgb_array", "rgb_array_list"]
        width, height = self.observation_width, self.observation_height
        images = [
            self._env.physics.render(height=height, width=width, camera_id=cam)
            for cam in self.camera_list
        ]
        return images

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "empty":
            xml_path = "/home/jzilke/ws/gym-aloha/assets/ur5e_gripper/scene.xml"
            print("Loading XML from:", xml_path)
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = EmptyTask(self.camera_list)
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
            obs = {"teleoperator_pov": raw_obs["images"]["teleoperator_pov"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {cam: raw_obs["images"][cam].copy() for cam in self.camera_list},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        if self.task == "TODO":
            raise ValueError(self.task)

        raw_obs = self._env._task.get_observation(self._env.physics)
        observation = self._format_raw_obs(raw_obs)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        _, reward, _, raw_obs = self._env.step(action)

        terminated = is_success = False

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
