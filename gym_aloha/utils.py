import numpy as np


def sample_box_pose(seed=None):
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_transfer_box_pose(seed=None):
    x_range = [-0.25, 0.25]
    y_range = [-0.2, 0.2]
    z_range = [0.1, 0.1]

    rng = np.random.RandomState(seed)
    ranges = np.vstack([x_range, y_range, z_range])

    # Box
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    yaw = rng.uniform(0, 2 * np.pi)
    cube_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    box_pose = np.concatenate([cube_position, cube_quat])

    # Goal
    distance = 0.0
    while distance < 0.1:
        goal_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        distance = np.linalg.norm(cube_position - goal_position)  # distance

    goal_position[2] = 0.001
    goal_quat = np.array([1, 0, 0, 0])
    goal_pose = np.concatenate([goal_position, goal_quat])

    return box_pose, goal_pose


def sample_insertion_pose(seed=None):
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose
