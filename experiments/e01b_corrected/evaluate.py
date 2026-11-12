"""Test-set evaluation of e01b, see README.md in this directory."""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from run import (
    BASELINES,
    EVALUATION_SEEDS,
    LAMBDAS,
    MODELS_DIR,
    MODES,
    RESULTS_DIR,
    SUFFICIENT_TOLERANCE,
    VAL_HORIZON,
    best_lambda,
    family_variants,
    relative_validation_error,
    run_file,
)

from anchor.config import (
    BURN_IN,
    K_MAX,
    REFERENCE_DATA_SEED,
    TEST_DATA_SEED,
)
from anchor.envs.henon import attractor_points, generate_windows, observation_dim
from anchor.metrics import (
    mean_ci,
    mse_per_horizon,
    neighbour_futures,
    off_attractor,
    path_errors,
    sharpness,
    to_lists,
)
from anchor.models import HorizonMLP
from anchor.plotting import VARIANT_COLORS
from anchor.training import HIDDEN_SIZE, LEN_TRAJ, N_TRAJ, VARIANTS

# primary metric and guards are read at the end of the predictable range
EVAL_HORIZON = VAL_HORIZON
# exact nearest-neighbour search is costly; a median needs only a subset
N_OFF_WINDOWS = 1000
# nearest-neighbour estimate of E[x_{t+k} | x_t] in the insufficient mode,
# the best prediction under MSE, as in e01's floor analysis
N_REFERENCE_TRAJ = 200_000
N_NEIGHBOURS = 200
# a variant is harmless in the sufficient mode when the upper end of the 95%
# interval of its paired relative MSE difference to the baseline is below this
EQUIVALENCE_MARGIN = 0.10
# variants entering the decision criteria; A and B are reported as secondary
MAIN_VARIANTS = ("C", "D", "E")
# the one-step family never trains F(z, k) for k > 1
PATH_VARIANTS = {
    "direct": [v for v, (family, _, _) in VARIANTS.items() if family == "direct"],
    "rollout": list(VARIANTS),
}


def variant_lambda(variant: str) -> float:
    """Lambda of the final runs, 0 for the baselines."""
    if VARIANTS[variant][1] is None:
        return 0.0

    return best_lambda(variant)


def load_model(variant: str, mode: str, seed: int, path: str) -> HorizonMLP:
    """Loads the checkpoint of one final run selected for one path."""
    name = run_file(variant, mode, variant_lambda(variant), seed)
    model = HorizonMLP(K_MAX, observation_dim(mode), HIDDEN_SIZE)
    model.load_state_dict(
        torch.load(MODELS_DIR / f"{name}_{path}.pt", weights_only=True)
    )
    model.eval()

    return model


def conditional_mean_reference(windows: torch.Tensor) -> dict:
    """
    Test MSE and sharpness per horizon of the nearest-neighbour estimate of
    the conditional mean from x_t (insufficient mode). Under MSE this is the
    best possible prediction, so it is the reference for MSE and sharpness.
    """

    reference = generate_windows(
        N_REFERENCE_TRAJ,
        LEN_TRAJ,
        BURN_IN,
        K_MAX,
        "insufficient",
        np.random.default_rng(REFERENCE_DATA_SEED),
    )[..., 0].numpy()
    test = windows[..., 0].numpy()
    preds = neighbour_futures(reference, test[:, 0], N_NEIGHBOURS).mean(axis=1)

    return {
        "mse": ((preds - test[:, 1:]) ** 2).mean(axis=0),
        "sharpness": sharpness(preds[..., None], windows),
    }


def evaluate_mode(mode: str) -> dict:
    """
    Per-seed test metrics of every variant and path in one observation mode.

    Returns:
        dict: results[path][variant] with arrays over seeds of size
        (n_seeds, k_max): "mse", "median", "sharpness", "off", "diverged";
        in the insufficient mode also results["floor"] with "mse" and
        "sharpness" of the conditional mean, each of size (k_max,)
    """

    windows = generate_windows(
        N_TRAJ, LEN_TRAJ, BURN_IN, K_MAX, mode, np.random.default_rng(TEST_DATA_SEED)
    )
    point_dim = observation_dim(mode)
    reference = attractor_points(mode)
    results = {}

    for path, variants in PATH_VARIANTS.items():
        results[path] = {}
        for variant in variants:
            metrics = {
                key: [] for key in ("mse", "median", "sharpness", "off", "diverged")
            }
            for seed in EVALUATION_SEEDS:
                errors, preds = path_errors(
                    load_model(variant, mode, seed, path), windows, path
                )
                metrics["mse"].append(mse_per_horizon(errors, point_dim))
                metrics["median"].append(np.median(errors, axis=0))
                metrics["sharpness"].append(sharpness(preds, windows))
                metrics["off"].append(
                    np.median(off_attractor(preds[:N_OFF_WINDOWS], reference), axis=0)
                )
                metrics["diverged"].append(np.isinf(errors).mean(axis=0))
            results[path][variant] = {key: np.array(v) for key, v in metrics.items()}

    if mode == "insufficient":
        results["floor"] = conditional_mean_reference(windows)

    return results


def verdict(
    mode: str,
    variant: str,
    diff_ci: tuple,
    rel_ci: tuple,
    sharpness_ci: tuple | None = None,
) -> str:
    """
    Pre-registered reading of one main variant on the rollout path. In the
    insufficient mode the intervals are of the difference to onestep: "better"
    when the MSE difference lies below 0. In the sufficient mode they are of
    the difference to the same family at lambda = 0, which isolates the
    auxiliary term: "blurred" when the sharpness difference lies below 0,
    "harmless" when the upper end of the relative MSE difference is below
    EQUIVALENCE_MARGIN, "worse" when the MSE difference lies above 0,
    otherwise "inconclusive".
    """

    if variant not in MAIN_VARIANTS:
        return "secondary"
    (d, dh), (r, rh) = diff_ci, rel_ci
    if mode == "insufficient":
        return "better" if d + dh < 0 else "not better"
    if sharpness_ci is not None and sum(sharpness_ci) < 0:
        return "blurred"
    if r + rh < EQUIVALENCE_MARGIN:
        return "harmless"

    return "worse" if d - dh > 0 else "inconclusive"


def paired_differences(primary: np.ndarray, reference: np.ndarray) -> tuple:
    """Mean and 95% half-width of the absolute and relative paired differences."""
    d, dh = mean_ci((primary - reference)[:, None])
    rel, relh = mean_ci((primary / reference - 1)[:, None])

    return (d[0], dh[0]), (rel[0], relh[0])


def print_summary(mode: str, results: dict):
    """
    One line per variant and path: primary MSE over k = 1..EVAL_HORIZON, its
    paired absolute and relative difference to the baseline, sharpness at
    EVAL_HORIZON and its paired difference, guards, secondary metrics and,
    for the main variants on the rollout path, the pre-registered verdict.
    All intervals are 95% over the evaluation seeds.
    """

    h = EVAL_HORIZON
    print(f"\n=== {mode} ===")
    for path, baseline in BASELINES.items():
        print(f"\n{path} path (baseline: {baseline})")
        print(
            f"variant | lambda | MSE k<={h} | diff vs baseline | relative diff "
            f"| sharpness k={h} | sharpness diff | off-attractor k={h} "
            f"| diverged k={h} | MSE k<=20 | median error k<=20 (e01 metric) | verdict"
        )
        base = results[path][baseline]["mse"][:, :h].mean(axis=1)
        for variant, r in results[path].items():
            primary = r["mse"][:, :h].mean(axis=1)
            m, mh = mean_ci(primary[:, None])
            (d, dh), (rel, relh) = paired_differences(primary, base)
            s, sh = mean_ci(r["sharpness"][:, h - 1 : h])
            sd, sdh = mean_ci(
                (r["sharpness"] - results[path][baseline]["sharpness"])[:, h - 1 : h]
            )
            off, _ = mean_ci(r["off"][:, h - 1 : h])
            div, _ = mean_ci(r["diverged"][:, h - 1 : h])
            full, _ = mean_ci(r["mse"].mean(axis=1)[:, None])
            med, _ = mean_ci(r["median"].mean(axis=1)[:, None])
            reading = ""
            if path == "rollout" and variant != baseline:
                family = VARIANTS[variant][0]
                if mode == "sufficient" and variant in MAIN_VARIANTS:
                    family_r = results[path][family]
                    family_mse = family_r["mse"][:, :h].mean(axis=1)
                    harm, harm_rel = paired_differences(primary, family_mse)
                    sharp, sharp_h = mean_ci(
                        (r["sharpness"] - family_r["sharpness"])[:, h - 1 : h]
                    )
                    reading = (
                        f"{verdict(mode, variant, harm, harm_rel, (sharp[0], sharp_h[0]))} "
                        f"(vs {family}: MSE {harm_rel[0]:+.3f} ± {harm_rel[1]:.3f}, "
                        f"sharpness {sharp[0]:+.3f} ± {sharp_h[0]:.3f})"
                    )
                else:
                    reading = verdict(mode, variant, (d, dh), (rel, relh))
            print(
                f"{variant:7s} | {variant_lambda(variant):6.3f} | "
                f"{m[0]:.4f} ± {mh[0]:.4f} | {d:+.4f} ± {dh:.4f} | "
                f"{rel:+.3f} ± {relh:.3f} | "
                f"{s[0]:.3f} ± {sh[0]:.3f} | {sd[0]:+.3f} ± {sdh[0]:.3f} | "
                f"{off[0]:.4f} | {div[0]:.2f} | {full[0]:.4f} | {med[0]:.4f} | "
                f"{reading}"
            )
        if "floor" in results:
            floor = results["floor"]
            print(
                f"floor (conditional mean, kNN) | MSE k<={h}: "
                f"{floor['mse'][:h].mean():.4f} | sharpness k={h}: "
                f"{floor['sharpness'][h - 1]:.3f}"
            )


def print_frontier():
    """
    Validation errors of every lambda on the variant's selection path,
    relative to the same family at lambda = 0 with the same seed and averaged
    over the selection seeds, in both modes, with the lambda chosen for the
    final runs. Shows the trade-off between gain in the insufficient mode and
    harm in the sufficient mode.
    """

    print(
        f"\n=== lambda trade-off (selection seeds, validation, relative to the same family "
        f"at lambda = 0; "
        f"admissible: sufficient <= {SUFFICIENT_TOLERANCE}) ==="
    )
    print("variant | lambda | insufficient | sufficient | admissible | chosen")
    for family in ("direct", "onestep"):
        for variant in family_variants(family):
            chosen = best_lambda(variant)
            for lam in LAMBDAS:
                rel = {m: relative_validation_error(variant, m, lam) for m in MODES}
                print(
                    f"{variant:7s} | {lam:6.3f} | {rel['insufficient']:12.3f} | "
                    f"{rel['sufficient']:10.3f} | "
                    f"{'yes' if rel['sufficient'] <= SUFFICIENT_TOLERANCE else 'no':10s} | "
                    f"{'*' if lam == chosen else ''}"
                )


def plot(all_results: dict[str, dict]):
    """Test MSE of both paths and rollout sharpness vs horizon, one row per mode."""
    k = np.arange(1, K_MAX + 1)
    panels = (
        ("direct", "mse", "MSE, direct path (log scale)"),
        ("rollout", "mse", "MSE, rollout path (log scale)"),
        ("rollout", "sharpness", "Sharpness, rollout path"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for row, mode in enumerate(MODES):
        for col, (path, key, label) in enumerate(panels):
            ax = axes[row, col]
            for variant, r in all_results[mode][path].items():
                mean, half = mean_ci(r[key])
                if not np.isfinite(mean).all():
                    continue
                ax.plot(
                    k, mean, color=VARIANT_COLORS[variant], linewidth=2, label=variant
                )
                low = np.maximum(mean - half, 1e-8) if key == "mse" else mean - half
                ax.fill_between(
                    k, low, mean + half, color=VARIANT_COLORS[variant], alpha=0.15
                )
            if "floor" in all_results[mode] and path == "rollout":
                ax.plot(
                    k,
                    all_results[mode]["floor"][key],
                    color="#1f1f1f",
                    linewidth=1.5,
                    linestyle="--",
                    label="conditional mean",
                )
            if key == "mse":
                ax.set_yscale("log")
            else:
                ax.axhline(1.0, color="#9e9e9e", linewidth=0.8, linestyle=":")
            ax.axvline(EVAL_HORIZON, color="#9e9e9e", linewidth=0.8, linestyle="--")
            ax.set_title(f"{mode}: {label}", fontsize=10)
            ax.grid(True, which="major", color="#e0e0e0", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 1:
                ax.set_xlabel("Prediction horizon k")
                ax.set_xticks([1, 5, 10, 15, 20])
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"e01b on the test set, mean ± 95% CI over {len(EVALUATION_SEEDS)} evaluation seeds "
        f"(dashed: end of the primary range k = {EVAL_HORIZON})"
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "e01b_comparison.png", dpi=150)


if __name__ == "__main__":
    all_results = {mode: evaluate_mode(mode) for mode in MODES}
    for mode in MODES:
        print_summary(mode, all_results[mode])
    print_frontier()
    plot(all_results)
    with open(RESULTS_DIR / "test_results.json", "w") as f:
        json.dump(to_lists(all_results), f)
