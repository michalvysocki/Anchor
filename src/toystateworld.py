import math
from itertools import pairwise

import numpy as np
import torch
from torch import nn

from config import ObservabilityMode


def step(z: tuple[float, float], a: float = 1.4, b: float = 0.3) -> tuple[float, float]:
    """
    Compute the step in the Henon map environment.

    Args:
        z (tuple[float, float]): initial point
        a (float): controls whether the map exhibits chaotic behavior
        b (float): controls whether the map exhibits chaotic behavior

    Returns:
        tuple[float, float]: next point in trajectory
    """

    x, y = z
    x_next = 1 - a * x**2 + y
    y_next = b * x

    return (x_next, y_next)


def rollout(z0: tuple[float, float], K: int) -> list[tuple[float, float]]:
    """
    Recursively computes a trajectory using previous step as an input
    for generating another point.

    Args:
        z0 (tuple[float, float]): Initial point of the trajectory
        K (int): number of steps

    Returns:
        list[tuple[float, float]]: trajectory - a list of 2D points
    """

    traj = [z0]
    z = z0
    for _ in range(K):
        z = step(z)
        traj.append(z)

    return traj


def compute_drifts(
    traj_a: list[tuple[float, float]], traj_b: list[tuple[float, float]]
) -> list[float]:
    """
    Compute drifts between two trajectories. Drifts are distances
    between two points on a same position in compared trajectories.

    Args:
        traj_a (list[tuple[float, float]]): trajectory of points
        traj_b (list[tuple[float, float]]): trajectory of points

    Returns:
        list[float]: List of drifts
    """
    drifts = []
    for point_a, point_b in zip(traj_a, traj_b):
        distance = math.dist(point_a, point_b)
        drifts.append(distance)

    return drifts


def is_trajectory_stable(
    traj: list[tuple[float, ...]], threshold: float = 10.0
) -> bool:
    """
    Checks if a trajectory is stable. Trajectory is unstable when
    one of it's coordinates is bigger than threshold or is a NaN value.

    Args:
        traj (list[tuple[float, ...]]): trajectory, list of N-Dimensional points
        threshold (float): number a coordinate must not exceed

    Returns:
        bool: True if trajectory is stable or False if it isn't
    """

    return not any(
        abs(coord) > threshold or np.isnan(coord) for point in traj for coord in point
    )


def observe(
    traj: list[tuple[float, float]], mode: ObservabilityMode
) -> list[tuple[float, ...]]:
    """
    Makes a projection for a trajectory implementing partial or full observability.

    Args:
        traj (list[tuple[float, float]]): trajectory of 2D points
        mode (ObservabilityMode): Observability mode. It can be either
            sufficient - full point in known or insufficient - only the x coordinate is shown

    Returns:
        list[tuple[float, ...]]: A trajectory in either sufficient or insufficient observability mode.
    """
    if mode == "insufficient":
        output = [(p[0],) for p in traj]

    elif mode == "sufficient":
        output = list(pairwise(p[0] for p in traj))

    else:
        raise ValueError()

    return output


def observation_dim(mode: ObservabilityMode) -> int:
    """
    Returns the dimension of observation's points based on observability mode.

    Args:
        mode (ObservabilityMode): Sufficient or Insufficient observation mode

    Returns:
        int: observation's dimensions
    """
    traj = [(1.5, 1.9), (1.1, 2.1)]
    z = observe(traj, mode)

    return len(z[0])


def generate_dataset(
    n_traj: int, len_traj: int, burn: int, mode: ObservabilityMode
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset for training model on Henon map. Creates n
    trajectories of a given length, checks if they're stable,
    deletes a given number of first points, because they are not
    on attractor yet. Makes a projection for observability. For every
    point as an input there is a target point.

    Args:
        n_traj (int): number of trajectories to generate
        len_traj (int): length of each single trajectory
        burn (int): number of points which will be deleted
        mode (ObservabilityMode): Observability mode. It can be either
            sufficient - full point in known or insufficient - only the x coordinate is shown

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of inputs and targets
    """

    inputs = []
    targets = []

    for _ in range(n_traj):
        z0 = np.random.uniform(-1, 1, 2)
        traj = rollout(z0, len_traj)
        if not is_trajectory_stable(traj):
            continue

        traj = traj[burn:]
        proj_traj = observe(traj, mode)
        for ipt, target in pairwise(proj_traj):
            inputs.append(ipt)
            targets.append(target)

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(
        targets, dtype=torch.float32
    )


def build_horizon_input(
    point: tuple[float, float], k: int, k_max: int
) -> tuple[float, ...]:
    """
    Combines a Henon map point with a list made with one-hot encoding
    the value of k.

    Args:
        point (tuple[float, float]): A point coordinates from the Henon map
        k (int): k value that represents a step in the rollout
        k_max (int): maximum step

    Returns:
        tuple[float, ...]: a tuple of point coordinates and one-hot encoding list
    """

    k -= 1
    k = torch.tensor(k, dtype=torch.int64)
    one_hot = nn.functional.one_hot(k, k_max).tolist()

    return (*point, *one_hot)


def generate_horizon_dataset(
    n_traj: int, len_traj: int, burn: int, k_max: int, mode: ObservabilityMode
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset for training direct prediction model on
    Henon map. Creates n trajectories of a given length, checks
    if they're stable, deletes a given number of first points, because
    they are not on attractor yet. Makes a projection for observability.
    In the dataset there are many horizons for many different starting points.

    Args:
        n_traj (int): number of trajectories to generate
        len_traj (int): length of each single trajectory
        burn (int): number of points which will be deleted
        k_max (int): maximum step
        mode (ObservabilityMode): Observability mode. It can be either
            sufficient - full point in known or insufficient - only the x coordinate is shown


    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of point
        coordinates and one-hot encoded steps
    """

    inputs = []
    targets = []

    for _ in range(n_traj):
        z0 = np.random.uniform(-1, 1, 2)
        traj = rollout(z0, len_traj)
        if not is_trajectory_stable(traj):
            continue

        traj = traj[burn:]
        proj_traj = observe(traj, mode)
        T = len(proj_traj)

        for t in range(T - 1):
            steps_till_end = T - 1 - t
            real_k_max = min(k_max, steps_till_end)

            for k in range(1, real_k_max + 1):
                ipt = build_horizon_input(proj_traj[t], k, k_max)
                target = proj_traj[t + k]
                inputs.append(ipt)
                targets.append(target)

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(
        targets, dtype=torch.float32
    )
