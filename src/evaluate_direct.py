import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    DP_HIDDEN_SIZE,
    K_MAX,
    RESULTS_DIR,
    SEED,
    ObservabilityMode,
    direct_model_path,
)
from direct_predictor import DirectPredictor
from toystateworld import (
    build_horizon_input,
    is_trajectory_stable,
    observation_dim,
    observe,
    rollout,
)

np.random.seed(seed=SEED)


def average_diff(
    model: DirectPredictor,
    n_samples: int,
    burn_in: int,
    k_max: int,
    mode: ObservabilityMode,
) -> tuple[list[np.ndarray], int, int, int]:
    """
    Compute average difference between direct prediction and true
    state from the trajectory rollout from a given number of
    trajectories.

    Args:
        model (DirectPredictor): direct predictor object
        n_samples (int): number of trajectories
        burn_in (int): number of discarded points from trajectory
        k_max (int): maximum number of steps to predict to
        mode (ObservabilityMode): insufficient or sufficient observability

    Returns:
        tuple[list[np.ndarray], int, int]: list of average differences
        for each k, number of unstable trajectories, number of
        unstable projected trajectories.
    """

    diffs = []
    unstable_traj = 0
    unstable_proj_traj = 0

    for k in range(1, k_max + 1):
        diffs_k = []
        for i in range(n_samples):
            z0_raw = np.random.uniform(-1, 1, 2)
            full_traj = rollout(z0_raw, burn_in + k)
            if not is_trajectory_stable(full_traj):
                unstable_traj += 1
                continue

            proj_traj = observe(full_traj, mode)[-(k + 1) :]
            if not is_trajectory_stable(proj_traj):
                unstable_proj_traj += 1
                continue

            true_state = proj_traj[-1]
            model_input = torch.tensor(
                build_horizon_input(proj_traj[0], k, k_max), dtype=torch.float32
            )
            model_input = model_input.unsqueeze(0)
            predicted_state = tuple(model(model_input)[0].tolist())
            diff = math.dist(true_state, predicted_state)
            diffs_k.append(diff)

        diffs.append(np.mean(diffs_k, axis=0))

    return (diffs, unstable_traj, unstable_proj_traj)


def evaluate_direct(n_samples: int, k_max: int, burn_in: int):
    """
    Evaluates the direct predictor model in both sufficient and insufficient
    projection modes. Computes and measures unstability. Saves metrics to a .npy
    file and creates a plot for each mode.

    Args:
        n_samples (int): number of trajectories in the dataset
        k_max (int): maximum length of the trajectory
        burn_in (int): number of burned samples from the beginning of the trajectory
    """
    for mode in ("insufficient", "sufficient"):
        weights = torch.load(direct_model_path(mode), weights_only=True)
        model = DirectPredictor(
            k_max, point_dim=observation_dim(mode), hidden_size=DP_HIDDEN_SIZE
        ) 
        model.load_state_dict(weights)
        model.eval()

        with torch.inference_mode():
            means, n_unstable, n_unstable_true = average_diff(
                model, n_samples, burn_in, k_max, mode
            )

        print(f"Unstable warmup trajectories: {n_unstable}/{k_max * n_samples}")
        print(f"Unstable projected trajectories: {n_unstable_true}/{k_max * n_samples}")

        np.save(RESULTS_DIR / f"direct_diff_means_{mode}.npy", means)

        plt.cla()
        plt.semilogy(list(range(1, k_max + 1)), means)
        plt.xlabel("Prediction step k")
        plt.ylabel("Difference ||true - predicted||")
        plt.title(f"Average differences, direct prediction, {mode} mode (n={n_samples}, k_max={k_max})")
        plt.grid()
        plt.savefig(RESULTS_DIR / f"direct_diff_plot_{mode}.png", dpi=150)
        plt.show()

        

if __name__ == "__main__":
    evaluate_direct(1000, K_MAX, 5)