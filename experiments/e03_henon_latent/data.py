"""Data of e03: Hénon x observed with noise, windows of past and future (README, Setup)."""

import numpy as np
import torch
from design import EVAL_HORIZON, LEN_TRAJ, X_MEAN, X_VAR, rng

from anchor.envs.henon import add_noise, x_trajectories


def observations(n_traj: int, kappa: float, tag: int, *key: int):
    """
    Clean x and noisy y of shape (n_traj, LEN_TRAJ), centred and in units
    of sqrt(Var(y)). The trajectories depend only on (tag, key), the noise
    also on kappa, so configurations with the same seed share trajectories.
    """

    x = x_trajectories(n_traj, LEN_TRAJ, rng(tag, *key))
    return add_noise(x, kappa, X_MEAN, X_VAR, rng(tag, *key, 99, round(100 * kappa)))


def windows(series: np.ndarray, past: int) -> torch.Tensor:
    """All windows of past + EVAL_HORIZON consecutive values, (n, past + EVAL_HORIZON)."""
    span = past + EVAL_HORIZON
    idx = np.arange(series.shape[1] - span + 1)[:, None] + np.arange(span)
    return torch.tensor(series[:, idx].reshape(-1, span), dtype=torch.float32)
