"""
Hénon map (a = 1.4, b = 0.3): one step, rollouts, rejection of unstable
trajectories, observation modes (sufficient: (x_{t-1}, x_t), insufficient:
x_t), windows of trajectories and a sample of the attractor. Used by e01,
e01b and, with a learned latent, e03.
"""

from itertools import pairwise

import numpy as np
import torch

from anchor.config import ObservabilityMode


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


def generate_windows(
    n_traj: int,
    len_traj: int,
    burn: int,
    k_max: int,
    mode: ObservabilityMode,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Generates trajectory windows for training with predictive paths.
    Each window is a starting point followed by k_max true points, so
    a direct target, any intermediate observation and any decomposition
    k = p + q can be read from the same window. Uses the same trajectories
    and starting points as the horizon dataset of the pre-v0 pipeline.

    Args:
        n_traj (int): number of trajectories to generate
        len_traj (int): length of each single trajectory
        burn (int): number of points which will be deleted
        k_max (int): maximum step
        mode (ObservabilityMode): Observability mode. It can be either
            sufficient - full point in known or insufficient - only the x coordinate is shown
        rng (np.random.Generator): generator for initial points

    Returns:
        torch.Tensor: windows of size (n_windows, k_max + 1, point_dim)
    """

    windows = []

    for _ in range(n_traj):
        z0 = rng.uniform(-1, 1, 2)
        traj = rollout(z0, len_traj)
        if not is_trajectory_stable(traj):
            continue

        traj = traj[burn:]
        proj_traj = observe(traj, mode)
        T = len(proj_traj)
        if T <= k_max:
            raise ValueError(
                f"Trajectory too short: {T} points after burn-in, need more than {k_max}"
            )

        for t in range(T - k_max):
            windows.append(proj_traj[t : t + k_max + 1])

    return torch.tensor(windows, dtype=torch.float32)


def attractor_points(mode: ObservabilityMode, n_points: int = 10000) -> np.ndarray:
    """Reference sample of the observed attractor, used to detect blurred predictions."""
    traj = rollout((0.1, 0.1), n_points + 1000)[1000:]

    return np.array(observe(traj, mode))
