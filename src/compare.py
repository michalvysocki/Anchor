from math import dist

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import RESULTS_DIR, ObservabilityMode, K_MAX, HIDDEN_SIZE, DP_HIDDEN_SIZE
from direct_predictor import DirectPredictor
from evaluate import model_rollout
from model import Predictor
from toystateworld import (
    build_horizon_input,
    is_trajectory_stable,
    observe,
    observation_dim,
    rollout,
)


def sample_paired_errors(
    rollout_model: Predictor,
    direct_model: DirectPredictor,
    n_samples: int,
    k_max: int,
    burn_in: int,
    mode: ObservabilityMode,
) -> tuple:
    """
    Samples errors of rollout and direct predictor for correlation mesaurement.

    Args:
        rollout_model (Predictor): one-step prediction model
        direct_model (DirectPredictor): direct prediction model
        n_samples (int): number of trajectories
        k_max (int): maximum prediction horizon
        burn_in (int): number of discarded points from each trajectory
        mode (ObservabilityMode): insufficient or sufficient projection

    Returns:
        np.ndarray: matrix of error measurement of the direct model of size (n_valid, k_max)
        np.ndarray: matrix of error measurement of the rollout model of size (n_valid, k_max)
        int: number of unstable trajectories
    """
    dir_rows = []
    roll_rows = []
    n_unstable = 0

    for _ in range(n_samples):
        z0_raw = np.random.uniform(-1, 1, 2)
        full_traj = rollout(z0_raw, burn_in + k_max)
        if not is_trajectory_stable(full_traj):
            n_unstable += 1
            continue

        proj_traj = observe(full_traj, mode)
        offset = len(full_traj) - len(proj_traj)
        start_idx = burn_in - offset
        z_start = proj_traj[start_idx]

        row_dir = []
        row_roll = []
        with torch.inference_mode():
            roll_traj = model_rollout(rollout_model, z_start, k_max)
            for k in range(1, k_max + 1):
                target = proj_traj[start_idx + k]
                e_roll = dist(roll_traj[k], target)
                direct_input = torch.tensor(
                    build_horizon_input(z_start, k, k_max), dtype=torch.float32
                )
                direct_input = direct_input.unsqueeze(0)
                predicted_state = tuple(direct_model(direct_input)[0].tolist())

                e_dir = dist(predicted_state, target)
                row_dir.append(e_dir)
                row_roll.append(e_roll)

            dir_rows.append(row_dir)
            roll_rows.append(row_roll)

    return np.array(dir_rows), np.array(roll_rows), n_unstable

for mode in ("insufficient", "sufficient"):
    rollout_model = Predictor(observation_dim(mode), HIDDEN_SIZE)
    direct_model = DirectPredictor(K_MAX, observation_dim(mode), DP_HIDDEN_SIZE)
    dirct, roll, n_unstable = sample_paired_errors(rollout_model, direct_model, n_samples=200, k_max=K_MAX, burn_in=5, mode=mode)

    print(dirct.shape, roll.shape, "\n")
    print(dirct.shape[0] + n_unstable, roll.shape[0] + n_unstable, "\n")
    print(dirct[:, 0], "\n\n", roll[:, 0])
    print(np.median(dirct, axis=0), "\n\n", np.median(roll, axis=0))




# baseline_means = np.load(RESULTS_DIR / "baseline_drift_means.npy")
# direct_means = np.load(RESULTS_DIR / "direct_diff_means.npy")

# plt.semilogy(range(1, len(baseline_means) + 1), baseline_means, label="Mean Drift")
# plt.semilogy(range(1, len(direct_means) + 1), direct_means, label="Mean Difference")
# plt.xlabel("Prediction step k")
# plt.ylabel("Drift or Difference value")
# plt.title("Comparison of one-step and direct predictors")
# plt.legend()
# plt.grid()
# plt.savefig(RESULTS_DIR / "model_comparison_plot.png", dpi=150)
# plt.show()
