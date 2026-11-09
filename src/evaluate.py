import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    BURN_IN,
    HIDDEN_SIZE,
    RESULTS_DIR,
    TEST_DATA_SEED,
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
    model: Predictor,
    n_samples: int,
    K: int,
    burn_in: int,
    mode: ObservabilityMode,
    rng: np.random.Generator,
    threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute median drift from a given number of trajectories. Model rollouts
    that diverge are kept as failures: from the step where a coordinate
    exceeds the threshold (or becomes NaN) their drift is infinite.

    Args:
        model (Predictor): one-step predictor object
        n_samples (int): number of trajectories
        K (int): number of steps for each trajectory
        burn_in (int): number of discarded point from trajectory
        mode (ObservabilityMode): insufficient or sufficient observability
        rng (np.random.Generator): generator for initial points
        threshold (float): coordinate value above which a rollout is diverged

    Returns:
        tuple[np.ndarray, np.ndarray, int]: median drift for each step,
        fraction of diverged rollouts for each step, number of unstable
        true trajectories
    """

    drifts = []
    diverged = []
    unstable_traj = 0

    for i in range(n_samples):
        z0_raw = rng.uniform(-1, 1, 2)
        full_traj = rollout(z0_raw, burn_in + K)
        if not is_trajectory_stable(full_traj):
            unstable_traj += 1
            continue

        proj_traj = observe(full_traj, mode)[-(K + 1) :]
        model_traj = model_rollout(model, proj_traj[0], K)

        step_diverged = np.array(
            [not is_trajectory_stable([point], threshold) for point in model_traj]
        )
        step_diverged = np.logical_or.accumulate(step_diverged)

        drift = np.array(compute_drifts(proj_traj, model_traj))
        drift[step_diverged] = np.inf
        drifts.append(drift)
        diverged.append(step_diverged)

    return (
        np.median(drifts, axis=0),
        np.mean(diverged, axis=0),
        unstable_traj,
    )


def evaluate(n_samples: int, K: int, burn_in: int):
    """
    Evaluates the predictor model in both sufficient and insufficient
    projection modes. Computes median drift and the fraction of diverged
    rollouts. Saves metrics to .npy files and creates a plot for each mode.

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
            medians, diverged_rate, n_unstable = average_drift(
                model, n_samples, K, burn_in, mode, np.random.default_rng(TEST_DATA_SEED)
            )

        print(f"Unstable true trajectories: {n_unstable}/{n_samples}")
        print(f"Diverged model rollouts at k={K}: {diverged_rate[-1]:.1%}")

        np.save(RESULTS_DIR / f"baseline_drift_medians_{mode}.npy", medians)
        np.save(RESULTS_DIR / f"baseline_diverged_rate_{mode}.npy", diverged_rate)

        plt.cla()
        plt.semilogy(list(range(1, K + 1)), medians[1:])
        plt.xlabel("Rollout step k")
        plt.ylabel("Median drift ||true - predicted||")
        plt.title(f"Median drift, baseline, {mode} mode (n={n_samples}, K={K})")
        plt.grid()
        plt.savefig(RESULTS_DIR / f"baseline_drift_plot_{mode}.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    evaluate(1000, 50, BURN_IN)
