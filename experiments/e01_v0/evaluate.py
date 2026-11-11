"""Test-set evaluation of e01, see README.md in this directory."""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from train import MODELS_DIR, RESULTS_DIR, best_lambda

from anchor.config import (
    BURN_IN,
    K_MAX,
    TEST_DATA_SEED,
    TRAINING_SEEDS,
    ObservabilityMode,
)
from anchor.envs.henon import attractor_points, generate_windows, observation_dim
from anchor.metrics import mean_ci, off_attractor, path_errors, to_lists
from anchor.models import HorizonMLP
from anchor.plotting import VARIANT_COLORS
from anchor.training import HIDDEN_SIZE, LEN_TRAJ, N_TRAJ, VARIANTS, run_name

MODES = ("insufficient", "sufficient")
# exact nearest-neighbour search is costly; a median needs only a subset
N_OFF_WINDOWS = 1000
# the one-step family never trains F(z, k) for k > 1
DIRECT_PATH_VARIANTS = [
    v for v, (family, _, _) in VARIANTS.items() if family == "direct"
]
ROLLOUT_PATH_VARIANTS = list(VARIANTS)


def variant_lambda(variant: str, mode: ObservabilityMode) -> float:
    """Lambda used by the final runs of a variant, 0 for the baselines."""
    if VARIANTS[variant][1] is None:
        return 0.0

    return best_lambda(variant, mode, TRAINING_SEEDS[0])


def load_model(
    variant: str, mode: ObservabilityMode, seed: int, path: str
) -> HorizonMLP:
    """Loads the checkpoint of one run selected for one path."""
    name = run_name(variant, mode, variant_lambda(variant, mode), seed)
    model = HorizonMLP(K_MAX, observation_dim(mode), HIDDEN_SIZE)
    model.load_state_dict(
        torch.load(MODELS_DIR / f"{name}_{path}.pt", weights_only=True)
    )
    model.eval()

    return model


def evaluate_mode(mode: ObservabilityMode) -> dict:
    """
    Per-seed test metrics of every variant and path in one observation mode.

    Returns:
        dict: results[path][variant] with arrays over seeds: "median" of size
        (n_seeds, k_max), "diverged" of size (n_seeds, k_max), "off" of size
        (n_seeds, k_max) and "mean_median" of size (n_seeds,)
    """

    windows = generate_windows(
        N_TRAJ, LEN_TRAJ, BURN_IN, K_MAX, mode, np.random.default_rng(TEST_DATA_SEED)
    )
    reference = attractor_points(mode)
    results = {}

    for path, variants in (
        ("direct", DIRECT_PATH_VARIANTS),
        ("rollout", ROLLOUT_PATH_VARIANTS),
    ):
        results[path] = {}
        for variant in variants:
            median, diverged, off = [], [], []
            for seed in TRAINING_SEEDS:
                model = load_model(variant, mode, seed, path)
                errors, preds = path_errors(model, windows, path)
                median.append(np.median(errors, axis=0))
                diverged.append(np.isinf(errors).mean(axis=0))
                off.append(
                    np.median(off_attractor(preds[:N_OFF_WINDOWS], reference), axis=0)
                )
            median = np.array(median)
            results[path][variant] = {
                "median": median,
                "diverged": np.array(diverged),
                "off": np.array(off),
                "mean_median": median.mean(axis=1),
            }

    return results


def print_summary(mode: ObservabilityMode, results: dict):
    """
    Prints one line per variant and path: error averaged over horizons and
    its paired difference to the baseline of the same seeds, both with 95% CI.
    """

    print(f"\n=== {mode} ===")
    for path, baseline in (("direct", "direct"), ("rollout", "onestep")):
        print(f"\n{path} path (baseline: {baseline})")
        print(
            "variant | lambda | mean median error | diff vs baseline "
            "| off-attractor k=20 | diverged k=20"
        )
        base = results[path][baseline]["mean_median"]
        for variant, r in results[path].items():
            m, h = mean_ci(r["mean_median"][:, None])
            d, dh = mean_ci((r["mean_median"] - base)[:, None])
            off, _ = mean_ci(r["off"])
            div, _ = mean_ci(r["diverged"])
            print(
                f"{variant:7s} | {variant_lambda(variant, mode):6.1f} | "
                f"{m[0]:.4f} ± {h[0]:.4f} | {d[0]:+.4f} ± {dh[0]:.4f} | "
                f"{off[-1]:.4f} | {div[-1]:.2f}"
            )


def plot(all_results: dict[str, dict]):
    """Median test error vs horizon for every variant, one panel per mode and path."""
    k = np.arange(1, K_MAX + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for row, mode in enumerate(MODES):
        for col, path in enumerate(("direct", "rollout")):
            ax = axes[row, col]
            for variant, r in all_results[mode][path].items():
                mean, half = mean_ci(r["median"])
                if not np.isfinite(mean).all():
                    continue
                ax.plot(
                    k, mean, color=VARIANT_COLORS[variant], linewidth=2, label=variant
                )
                ax.fill_between(
                    k,
                    np.maximum(mean - half, 1e-6),
                    mean + half,
                    color=VARIANT_COLORS[variant],
                    alpha=0.15,
                )
            ax.set_yscale("log")
            ax.set_title(f"{mode}, {path} path")
            ax.grid(True, which="major", color="#e0e0e0", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 1:
                ax.set_xlabel("Prediction horizon k")
                ax.set_xticks([1, 5, 10, 15, 20])
            if col == 0:
                ax.set_ylabel("Median test error (log scale)")
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Anchor variants on the test set, mean ± 95% CI over {len(TRAINING_SEEDS)} seeds"
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "variants_comparison.png", dpi=150)


if __name__ == "__main__":
    all_results = {mode: evaluate_mode(mode) for mode in MODES}
    for mode in MODES:
        print_summary(mode, all_results[mode])
    plot(all_results)
    with open(RESULTS_DIR / "test_results.json", "w") as f:
        json.dump(to_lists(all_results), f)
