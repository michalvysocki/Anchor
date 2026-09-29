import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    HIDDEN_SIZE,
    RESULTS_DIR,
    SEED,
    ObservabilityMode,
    model_path,
)
from model import Predictor
from toystateworld import (
    compute_drifts,
    is_trajectory_stable,
    observation_dim,
    observe,
    rollout,
)

np.random.seed(seed=SEED)


def model_rollout(
    model: Predictor, z0: tuple[float, ...], K: int
) -> list[tuple[float, ...]]:
    """
    Create a rollout trajectory autoregressively using
    one-step predictor model.

    Args:
        model (Predictor): one-step predictor object
        z0 (tuple[float, ...]): initial N-dimensional point
        K (int): number of steps

    Returns:
        list[tuple[float, ...]]: final predicted trajectory
    """

    traj = [z0]
    z0 = torch.tensor(z0, dtype=torch.float32)
    z0 = z0.unsqueeze(0)
    z = z0
    for _ in range(K):
        z = model(z)
        z_new = tuple(z[0].tolist())
        traj.append(z_new)

    return traj


def average_drift(
    model: Predictor, n_samples: int, K: int, burn_in: int, mode: ObservabilityMode
) -> tuple[np.ndarray, int, int, int]:
    """
    Compute average drift from a given number of trajectories.

    Args:
        model (Predictor): one-step predictor object
        n_samples (int): number of trajectories
        K (int): number of steps for each trajectory
        burn_in (int): number of discarded point from trajectory
        mode (ObservabilityMode): insufficient or sufficient observability

    Returns:
        tuple[np.ndarray, int, int, int]: mean drift, number of unstable
        trajectories, number of unstable true trajectories, number
        of unstable predicted trajectories
    """

    drifts = []
    unstable_traj = 0
    unstable_proj_traj = 0
    unstable_model_traj = 0

    for i in range(n_samples):
        z0_raw = np.random.uniform(-1, 1, 2)
        full_traj = rollout(z0_raw, burn_in + K)
        if not is_trajectory_stable(full_traj):
            unstable_traj += 1
            continue

        proj_traj = observe(full_traj, mode)[-(K + 1) :]
        if not is_trajectory_stable(proj_traj):
            unstable_proj_traj += 1
            continue

        model_traj = model_rollout(model, proj_traj[0], K)
        if not is_trajectory_stable(model_traj):
            unstable_model_traj += 1
            continue

        drift = compute_drifts(proj_traj, model_traj)
        drifts.append(drift)

    return (
        np.mean(drifts, axis=0),
        unstable_traj,
        unstable_proj_traj,
        unstable_model_traj,
    )


def evaluate(n_samples: int, K: int, burn_in: int):
    """
    Evaluates the predictor model in both sufficient and insufficient
    projection modes. Computes and measures unstability. Saves metrics to a .npy
    file and creates a plot for each mode.

    Args:
        n_samples (int): number of trajectories in the dataset
        K (int): length of the trajectory
        burn_in (int): number of burned samples from the beginning of the trajectory
    """
    for mode in ("insufficient", "sufficient"):
        weights = torch.load(model_path(mode), weights_only=True)
        model = Predictor(point_dim=observation_dim(mode), hidden_size=HIDDEN_SIZE)
        model.load_state_dict(weights)
        model.eval()

        with torch.inference_mode():
            means, n_unstable, n_unstable_true, n_unstable_model = average_drift(
                model, n_samples, K, burn_in, mode
            )

        print(f"Unstable warmup trajectories: {n_unstable}/{n_samples}")
        print(f"Unstable true trajectories: {n_unstable_true}/{n_samples}")
        print(f"Unstable model rollouts: {n_unstable_model}/{n_samples}")

        np.save(RESULTS_DIR / f"baseline_drift_means_{mode}.npy", means)

        plt.cla()
        plt.semilogy(list(range(1, K+1)), means[1:])
        plt.xlabel("Rollout step k")
        plt.ylabel("Drift ||true - predicted||")
        plt.title(f"Average drift, baseline, {mode} mode (n={n_samples}, K={K})")
        plt.grid()
        plt.savefig(RESULTS_DIR / f"baseline_drift_plot_{mode}.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    evaluate(20, 50, 5)
