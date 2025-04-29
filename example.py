import os
os.environ["MUJOCO_GL"] = "egl"
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v1")
observation, info = env.reset()
frames_dict = {key: [] for key in observation["pixels"].keys()}



def add_frames(frames_dict, frames):
    for key, value in frames.items():
        frames_dict[key].append(value)

import time

start_time = time.time()

for step in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    add_frames(frames_dict, observation["pixels"])


    if terminated or truncated:
        observation, info = env.reset()

end_time = time.time()
print(f"Execution time: {round(end_time - start_time, 2)} seconds")

env.close()

video_dir = "example"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

for key, value in frames_dict.items():
    imageio.mimsave(f"{video_dir}/{key}.mp4", np.stack(value), fps=10)
