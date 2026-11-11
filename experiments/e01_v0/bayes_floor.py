"""
Post-hoc analysis for e01: how close is the direct predictor to the best
possible prediction from x_t alone in the insufficient mode?

E[x_{t+k} | x_t] and the conditional median are estimated by nearest-neighbour
regression on a large sample of windows from a separate generator, then
evaluated on the e01 test windows with the e01 metric (median absolute error
per horizon).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from config import BURN_IN, K_MAX, TEST_DATA_SEED, TRAINING_SEEDS
from evaluate_variants import load_model, mean_ci, path_errors
from toystateworld import generate_windows
from train_variants import LEN_TRAJ, N_TRAJ, VARIANT_RESULTS_DIR

# not used by train (0), test (1) or validation (2)
REFERENCE_DATA_SEED = 3
N_REFERENCE_TRAJ = 200_000
NEIGHBOURS = (50, 200)


def nearest_neighbours(
    reference_x: np.ndarray, query_x: np.ndarray, n_neighbours: int
) -> np.ndarray:
    """
    Indices of the n nearest reference points of each query point in 1D.

    Args:
        reference_x (np.ndarray): sorted reference points of size (m,)
        query_x (np.ndarray): query points of size (n,)
        n_neighbours (int): number of neighbours

    Returns:
        np.ndarray: indices into reference_x of size (n, n_neighbours)
    """

    # the n nearest points of a sorted 1D array lie within n positions of
    # the insertion point
    pos = np.searchsorted(reference_x, query_x)
    offsets = np.arange(-n_neighbours, n_neighbours)
    candidates = np.clip(pos[:, None] + offsets, 0, len(reference_x) - 1)
    dist = np.abs(reference_x[candidates] - query_x[:, None])
    nearest = np.argsort(dist, axis=1)[:, :n_neighbours]

    return np.take_along_axis(candidates, nearest, axis=1)


def floor_errors(n_neighbours: int, reference: np.ndarray, test: np.ndarray) -> dict:
    """Median absolute test error per horizon of the conditional mean and median."""
    order = np.argsort(reference[:, 0])
    reference = reference[order]
    idx = nearest_neighbours(reference[:, 0], test[:, 0], n_neighbours)
    futures = reference[idx][:, :, 1:]  # (n_test, n_neighbours, k_max)
    truth = test[:, 1:]

    return {
        "mean": np.median(np.abs(futures.mean(axis=1) - truth), axis=0),
        "median": np.median(np.abs(np.median(futures, axis=1) - truth), axis=0),
        "mse_of_mean": ((futures.mean(axis=1) - truth) ** 2).mean(axis=0),
    }


def direct_mse(mode: str, windows: np.ndarray) -> np.ndarray:
    """Test MSE per horizon of the e01 direct baseline, mean over seeds."""
    windows = torch.from_numpy(windows[..., None]).float()
    mse = []
    for seed in TRAINING_SEEDS:
        _, preds = path_errors(
            load_model("direct", mode, seed, "direct"), windows, "direct"
        )
        mse.append(((preds[..., 0] - windows[:, 1:, 0].numpy()) ** 2).mean(axis=0))

    return np.mean(mse, axis=0)


if __name__ == "__main__":
    mode = "insufficient"
    reference = generate_windows(
        N_REFERENCE_TRAJ,
        LEN_TRAJ,
        BURN_IN,
        K_MAX,
        mode,
        np.random.default_rng(REFERENCE_DATA_SEED),
    )[..., 0].numpy()
    test = generate_windows(
        N_TRAJ, LEN_TRAJ, BURN_IN, K_MAX, mode, np.random.default_rng(TEST_DATA_SEED)
    )[..., 0].numpy()

    with open(VARIANT_RESULTS_DIR / "test_results.json") as f:
        e01 = json.load(f)
    direct, _ = mean_ci(np.array(e01[mode]["direct"]["direct"]["median"]))
    rollout, _ = mean_ci(np.array(e01[mode]["rollout"]["onestep"]["median"]))

    floors = {n: floor_errors(n, reference, test) for n in NEIGHBOURS}
    mse_direct = direct_mse(mode, test)
    print(
        f"reference windows: {len(reference)}, test windows: {len(test)}, "
        f"e01 models: mean over {len(TRAINING_SEEDS)} seeds"
    )
    print(
        " k | kNN mean (50) | kNN mean (200) | kNN median (200) "
        "| direct (e01) | onestep rollout (e01)"
    )
    for k in range(K_MAX):
        print(
            f"{k + 1:2d} | {floors[50]['mean'][k]:13.4f} | "
            f"{floors[200]['mean'][k]:14.4f} | {floors[200]['median'][k]:16.4f} | "
            f"{direct[k]:12.4f} | {rollout[k]:21.4f}"
        )
    print("\n k | MSE kNN mean (200) | MSE direct (e01)")
    for k in range(K_MAX):
        print(
            f"{k + 1:2d} | {floors[200]['mse_of_mean'][k]:18.4f} | {mse_direct[k]:16.4f}"
        )
    print(
        "mean over k | "
        + " | ".join(
            f"{name} {values.mean():.4f}"
            for name, values in [
                ("kNN mean (200)", floors[200]["mean"]),
                ("kNN median (200)", floors[200]["median"]),
                ("direct", direct),
                ("onestep rollout", rollout),
            ]
        )
    )

    with open(VARIANT_RESULTS_DIR / "bayes_floor_insufficient.json", "w") as f:
        json.dump(
            {
                str(n): {key: v.tolist() for key, v in floors[n].items()}
                for n in NEIGHBOURS
            },
            f,
        )
