"""Test-set evaluation of the shelved limited-horizon experiment, see README.md in this directory."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_limited_horizon import FAMILIES, LH_MODELS_DIR, LH_RESULTS_DIR, best_setting

from config import (
    BURN_IN,
    DP_HIDDEN_SIZE,
    K_MAX,
    TEST_DATA_SEED,
    TRAINING_SEEDS,
)
from direct_predictor import DirectPredictor
from evaluate_variants import (
    COLORS,
    MODES,
    N_OFF_WINDOWS,
    attractor_points,
    mean_ci,
    off_attractor,
    path_errors,
    sharpness,
    to_lists,
)
from toystateworld import generate_windows, observation_dim
from train_variants import LEN_TRAJ, N_TRAJ, run_name

BASELINE = "onestep"
VARIANTS = [BASELINE] + [v for family in FAMILIES.values() for v in family]


def setting(variant: str) -> tuple[int | None, float]:
    """(K_A, lambda) of the final runs, (None, 0) for the baseline."""
    if variant == BASELINE:
        return None, 0.0

    return best_setting(variant)


def load_model(variant: str, mode: str, seed: int) -> DirectPredictor:
    """Loads the rollout checkpoint of one final run."""
    max_horizon, lam = setting(variant)
    name = run_name(variant, mode, lam, seed, max_horizon)
    model = DirectPredictor(K_MAX, observation_dim(mode), DP_HIDDEN_SIZE)
    model.load_state_dict(
        torch.load(LH_MODELS_DIR / f"{name}_rollout.pt", weights_only=True)
    )
    model.eval()

    return model


def evaluate_mode(mode: str) -> dict:
    """
    Per-seed rollout metrics of every variant in one observation mode.

    Returns:
        dict: results[variant] with arrays over seeds: "median", "diverged",
        "off" and "sharpness" of size (n_seeds, k_max) and "mean_median"
        of size (n_seeds,)
    """

    windows = generate_windows(
        N_TRAJ, LEN_TRAJ, BURN_IN, K_MAX, mode, np.random.default_rng(TEST_DATA_SEED)
    )
    reference = attractor_points(mode)
    results = {}

    for variant in VARIANTS:
        median, diverged, off, sharp = [], [], [], []
        for seed in TRAINING_SEEDS:
            errors, preds = path_errors(
                load_model(variant, mode, seed), windows, "rollout"
            )
            median.append(np.median(errors, axis=0))
            diverged.append(np.isinf(errors).mean(axis=0))
            off.append(
                np.median(off_attractor(preds[:N_OFF_WINDOWS], reference), axis=0)
            )
            sharp.append(sharpness(preds, windows))
        median = np.array(median)
        results[variant] = {
            "median": median,
            "diverged": np.array(diverged),
            "off": np.array(off),
            "sharpness": np.array(sharp),
            "mean_median": median.mean(axis=1),
        }

    return results


def print_summary(mode: str, results: dict):
    """
    One line per variant: error averaged over horizons, its paired difference
    to the baseline, and the paired difference of sharpness at k = 20, all
    with 95% CI over seeds.
    """

    print(f"\n=== {mode} (rollout path, baseline: {BASELINE}) ===")
    print(
        "variant | K_A | lambda | mean median error | diff vs baseline "
        "| sharpness k=20 | sharpness diff | off-attractor k=20 | diverged k=20"
    )
    base = results[BASELINE]
    for variant, r in results.items():
        max_horizon, lam = setting(variant)
        m, h = mean_ci(r["mean_median"][:, None])
        d, dh = mean_ci((r["mean_median"] - base["mean_median"])[:, None])
        s, sh = mean_ci(r["sharpness"][:, -1:])
        sd, sdh = mean_ci((r["sharpness"] - base["sharpness"])[:, -1:])
        off, _ = mean_ci(r["off"])
        div, _ = mean_ci(r["diverged"])
        print(
            f"{variant:7s} | {max_horizon or '-'!s:>3s} | {lam:6.3f} | "
            f"{m[0]:.4f} ± {h[0]:.4f} | {d[0]:+.4f} ± {dh[0]:.4f} | "
            f"{s[0]:.3f} ± {sh[0]:.3f} | {sd[0]:+.3f} ± {sdh[0]:.3f} | "
            f"{off[-1]:.4f} | {div[-1]:.2f}"
        )


def plot(all_results: dict[str, dict]):
    """Median rollout error and sharpness vs horizon, one column per mode."""
    k = np.arange(1, K_MAX + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for col, mode in enumerate(MODES):
        for row, (key, label) in enumerate(
            (("median", "Median rollout error (log scale)"), ("sharpness", "Sharpness"))
        ):
            ax = axes[row, col]
            for variant, r in all_results[mode].items():
                mean, half = mean_ci(r[key])
                if not np.isfinite(mean).all():
                    continue
                ax.plot(k, mean, color=COLORS[variant], linewidth=2, label=variant)
                ax.fill_between(
                    k,
                    np.maximum(mean - half, 1e-6) if row == 0 else mean - half,
                    mean + half,
                    color=COLORS[variant],
                    alpha=0.15,
                )
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(f"{mode}")
            else:
                ax.axhline(1.0, color="#9e9e9e", linewidth=0.8, linestyle=":")
                ax.set_xlabel("Prediction horizon k")
                ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(True, which="major", color="#e0e0e0", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Limited-horizon rollout path on the test set, mean ± 95% CI over {len(TRAINING_SEEDS)} seeds"
    )
    fig.tight_layout()
    fig.savefig(LH_RESULTS_DIR / "limited_horizon_comparison.png", dpi=150)


if __name__ == "__main__":
    all_results = {mode: evaluate_mode(mode) for mode in MODES}
    for mode in MODES:
        print_summary(mode, all_results[mode])
    plot(all_results)
    with open(LH_RESULTS_DIR / "test_results.json", "w") as f:
        json.dump(to_lists(all_results), f)
