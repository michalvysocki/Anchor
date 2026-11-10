from collections.abc import Callable
from math import dist

import numpy as np
import torch

from config import (
    BURN_IN,
    DP_HIDDEN_SIZE,
    HIDDEN_SIZE,
    K_MAX,
    RESULTS_DIR,
    SEED,
    TEST_DATA_SEED,
    TRAINING_SEEDS,
    ObservabilityMode,
    direct_model_path,
    model_path,
)
from direct_predictor import DirectPredictor
from evaluate import model_rollout
from model import Predictor
from toystateworld import (
    build_horizon_input,
    is_trajectory_stable,
    observation_dim,
    observe,
    rollout,
)


def sample_paired_errors(
    rollout_model: Predictor,
    direct_model: DirectPredictor,
    n_samples: int,
    k_max: int,
    burn_in: int,
    mode: ObservabilityMode,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Samples errors of rollout and direct predictor for correlation measurement.

    Args:
        rollout_model (Predictor): one-step prediction model
        direct_model (DirectPredictor): direct prediction model
        n_samples (int): number of valid trajectories
        k_max (int): maximum prediction horizon
        burn_in (int): number of discarded points from each trajectory
        mode (ObservabilityMode): insufficient or sufficient projection
        seed (int): seed for randomization

    Returns:
        np.ndarray: matrix of error measurement of the direct model of size (n_valid, k_max)
        np.ndarray: matrix of error measurement of the rollout model of size (n_valid, k_max)
        int: number of discarded unstable trajectories
    """
    rng = np.random.default_rng(seed)

    dir_rows = []
    roll_rows = []
    n_unstable = 0

    while len(dir_rows) < n_samples:
        z0_raw = rng.uniform(-1, 1, 2)
        full_traj = rollout(z0_raw, burn_in + k_max)
        if not is_trajectory_stable(full_traj):
            n_unstable += 1
            if n_unstable > n_samples:
                raise ValueError("More than half of initial points are unstable")
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


def pearson_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson per column. a, b: (n, k_max). Returns (k_max,)."""
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)

    num = np.sum(a_c * b_c, axis=0)
    den = np.sqrt(np.sum(a_c**2, axis=0) * np.sum(b_c**2, axis=0))

    return np.divide(
        num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0
    )


def average_ranks(x: np.ndarray) -> np.ndarray:
    """
    Ranks of a 1D array, starting from 1. Tied values get the mean of
    their ranks, so equal errors (e.g. two infinite ones) are not ordered
    arbitrarily.

    Args:
        x (np.ndarray): values of size (n,)

    Returns:
        np.ndarray: ranks of size (n,)
    """
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)

    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, weights=ranks)

    return (rank_sums / counts)[inverse]


def spearman_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman per column. a, b: (n, k_max). Returns (k_max,)."""
    a_ranks = np.column_stack([average_ranks(col) for col in a.T])
    b_ranks = np.column_stack([average_ranks(col) for col in b.T])

    return pearson_columns(a_ranks, b_ranks)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Percentile CI of stat_fn(a, b) from row resampling. Rows are resampled
    together in a and b, so the pairing of errors is kept.

    Args:
        a (np.ndarray): first error matrix of size (n, k_max)
        b (np.ndarray): second error matrix of size (n, k_max)
        stat_fn (Callable): statistic computed per column
        n_boot (int): number of bootstrap resamples
        ci (float): confidence level, e.g. 0.95
        rng (np.random.Generator): generator for resampling

    Returns:
        tuple[np.ndarray, np.ndarray]: lower and upper bound for each column
    """
    n = a.shape[0]
    stats = np.stack(
        [stat_fn(a[idx], b[idx]) for idx in rng.integers(0, n, size=(n_boot, n))]
    )
    alpha = (1 - ci) / 2

    return (
        np.nanpercentile(stats, 100 * alpha, axis=0),
        np.nanpercentile(stats, 100 * (1 - alpha), axis=0),
    )


def paired_error_correlation(
    dir_errors: np.ndarray,
    roll_errors: np.ndarray,
    n_boot: int,
    ci: float,
    seed: int,
    eps: float,
) -> dict[str, np.ndarray]:
    """
    Pearson on logs and Spearman, with CI, for each k.

    Pearson is computed only on rows where both models have finite errors
    for every k, because a diverged rollout has an infinite log error.
    Spearman uses all rows, a diverged rollout simply gets the highest rank.

    Args:
        dir_errors (np.ndarray): direct model errors of size (n, k_max)
        roll_errors (np.ndarray): rollout model errors of size (n, k_max)
        n_boot (int): number of bootstrap resamples
        ci (float): confidence level, e.g. 0.95
        seed (int): seed for bootstrap resampling
        eps (float): added to errors before the log, guards against log(0)

    Returns:
        dict[str, np.ndarray]: correlations and CI bounds of size (k_max,),
        and the number of rows excluded from Pearson
    """
    rng = np.random.default_rng(seed)

    finite = np.isfinite(dir_errors).all(axis=1) & np.isfinite(roll_errors).all(
        axis=1
    )
    log_dir = np.log(dir_errors[finite] + eps)
    log_roll = np.log(roll_errors[finite] + eps)

    pearson = pearson_columns(log_dir, log_roll)
    pearson_lo, pearson_hi = bootstrap_ci(
        log_dir, log_roll, pearson_columns, n_boot, ci, rng
    )
    spearman = spearman_columns(dir_errors, roll_errors)
    spearman_lo, spearman_hi = bootstrap_ci(
        dir_errors, roll_errors, spearman_columns, n_boot, ci, rng
    )

    return {
        "pearson": pearson,
        "pearson_lo": pearson_lo,
        "pearson_hi": pearson_hi,
        "spearman": spearman,
        "spearman_lo": spearman_lo,
        "spearman_hi": spearman_hi,
        "n_excluded": np.array(int((~finite).sum())),
    }


def main(seed: int):
    for mode in ("insufficient", "sufficient"):
        rollout_weights = torch.load(model_path(mode, seed), weights_only=True)
        direct_weights = torch.load(direct_model_path(mode, seed), weights_only=True)
        rollout_model = Predictor(observation_dim(mode), HIDDEN_SIZE)
        direct_model = DirectPredictor(K_MAX, observation_dim(mode), DP_HIDDEN_SIZE)
        rollout_model.load_state_dict(rollout_weights)
        direct_model.load_state_dict(direct_weights)
        rollout_model.eval()
        direct_model.eval()
        dirct, roll, n_unstable = sample_paired_errors(
            rollout_model,
            direct_model,
            n_samples=1000,
            k_max=K_MAX,
            burn_in=BURN_IN,
            mode=mode,
            seed=TEST_DATA_SEED,
        )
        corr = paired_error_correlation(
            dirct, roll, n_boot=1000, ci=0.95, seed=SEED, eps=1e-12
        )

        np.savez(
            RESULTS_DIR / f"paired_errors_{mode}_seed{seed}.npz",
            dir_errors=dirct,
            roll_errors=roll,
            n_unstable=n_unstable,
            seed=TEST_DATA_SEED,
            burn_in=BURN_IN,
            k_max=K_MAX,
            mode=mode,
            **corr,
        )

        print(f"\n{mode} seed {seed} | unstable initial points: {n_unstable}")
        print(f"rows excluded from Pearson: {corr['n_excluded']}")
        print(" k | med dir | med roll | Pearson log [CI] | Spearman [CI]")
        for k in range(K_MAX):
            print(
                f"{k + 1:2d} | {np.median(dirct[:, k]):.4f} | "
                f"{np.median(roll[:, k]):.4f} | "
                f"{corr['pearson'][k]:+.2f} "
                f"[{corr['pearson_lo'][k]:+.2f}, {corr['pearson_hi'][k]:+.2f}] | "
                f"{corr['spearman'][k]:+.2f} "
                f"[{corr['spearman_lo'][k]:+.2f}, {corr['spearman_hi'][k]:+.2f}]"
            )


if __name__ == "__main__":
    for seed in TRAINING_SEEDS:
        main(seed)


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
