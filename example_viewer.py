import importlib
import os

import gymnasium as gym
import mujoco.viewer

gym_aloha = importlib.import_module("gym_aloha")
os.environ["MUJOCO_GL"] = "egl"

env = gym.make("gym_aloha/AlohaStacking-v0")
observation, info = env.reset()


physics = env.unwrapped._env.physics
model = physics.model.ptr
data = physics.data.ptr

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
    viewer.sync()
    while viewer.is_running():
        action = env.action_space.sample()
        env.step(action)
        # mujoco.mj_step(model, data)
        viewer.sync()


env.close()
