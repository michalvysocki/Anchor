import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from anchor_loss import UnitStep, predict
from config import (
    BURN_IN,
    DP_HIDDEN_SIZE,
    K_MAX,
    TEST_DATA_SEED,
    TRAINING_SEEDS,
    ObservabilityMode,
)
from direct_predictor import DirectPredictor
from toystateworld import generate_windows, observation_dim, observe, rollout
from train_variants import (
    LEN_TRAJ,
    N_TRAJ,
    VARIANT_MODELS_DIR,
    VARIANT_RESULTS_DIR,
    VARIANTS,
    best_lambda,
    run_name,
)

# two-sided 95% Student t quantile for 4 degrees of freedom (5 seeds)
T_95_DF4 = 2.776
DIVERGENCE_THRESHOLD = 10.0
MODES = ("insufficient", "sufficient")
# exact nearest-neighbour search is costly; a median needs only a subset
N_OFF_WINDOWS = 1000
# fixed categorical order, one color per variant on every plot
COLORS = {
    "direct": "#2a78d6",
    "A": "#eb6834",
    "B": "#1baf7a",
    "E": "#eda100",
    "onestep": "#e87ba4",
    "C": "#008300",
    "D": "#7a5cd6",
}
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
) -> DirectPredictor:
    """Loads the checkpoint of one run selected for one path."""
    name = run_name(variant, mode, variant_lambda(variant, mode), seed)
    model = DirectPredictor(K_MAX, observation_dim(mode), DP_HIDDEN_SIZE)
    model.load_state_dict(
        torch.load(VARIANT_MODELS_DIR / f"{name}_{path}.pt", weights_only=True)
    )
    model.eval()

    return model


def path_errors(
    model: torch.nn.Module, windows: torch.Tensor, path: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean errors of one prediction path against true future points.
    A rollout counts as diverged from the step where a coordinate exceeds
    DIVERGENCE_THRESHOLD or becomes NaN, and its error is infinite from then on.

    Args:
        model (torch.nn.Module): horizon-conditioned model F(z, k)
        windows (torch.Tensor): test windows of size (n, k_max + 1, point_dim)
        path (str): "direct" for F(z, k) or "rollout" for F(., 1)^k

    Returns:
        tuple[np.ndarray, np.ndarray]: errors of size (n, k_max) and the
        predictions of size (n, k_max, point_dim)
    """

    n, length, _ = windows.shape
    k_max = length - 1
    z = windows[:, 0]

    with torch.no_grad():
        if path == "direct":
            preds = torch.stack(
                [
                    predict(model, z, torch.full((n,), k, dtype=torch.int64), k_max)
                    for k in range(1, k_max + 1)
                ],
                dim=1,
            )
        else:
            unit = UnitStep(model, k_max)
            states = []
            for _ in range(k_max):
                z = unit(z)
                states.append(z)
            preds = torch.stack(states, dim=1)

    preds = preds.numpy()
    errors = np.linalg.norm(preds - windows[:, 1:].numpy(), axis=-1)
    diverged = ~np.isfinite(preds).all(axis=-1) | (
        np.abs(preds) > DIVERGENCE_THRESHOLD
    ).any(axis=-1)
    diverged = np.logical_or.accumulate(diverged, axis=1)
    errors[diverged] = np.inf

    return errors, preds


def attractor_points(mode: ObservabilityMode, n_points: int = 10000) -> np.ndarray:
    """Reference sample of the observed attractor, used to detect blurred predictions."""
    traj = rollout((0.1, 0.1), n_points + 1000)[1000:]

    return np.array(observe(traj, mode))


def off_attractor(preds: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Distance from each prediction to the nearest reference attractor point.
    A prediction of the mean of possible futures lies off the attractor, so
    this separates sharp predictions from blurred ones at equal error.

    Args:
        preds (np.ndarray): predictions of size (n, k_max, point_dim)
        reference (np.ndarray): attractor points of size (m, point_dim)

    Returns:
        np.ndarray: distances of size (n, k_max)
    """

    flat = preds.reshape(-1, preds.shape[-1])
    finite = np.isfinite(flat).all(axis=1)
    dist = np.full(len(flat), np.inf)
    ref = torch.from_numpy(reference).float()
    for start in range(0, len(flat), 4096):
        chunk = flat[start : start + 4096]
        ok = finite[start : start + 4096]
        if ok.any():
            d = (
                torch.cdist(
                    torch.from_numpy(chunk[ok]).float(),
                    ref,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                .min(dim=1)
                .values
            )
            dist[start : start + 4096][ok] = d.numpy()

    return dist.reshape(preds.shape[:2])


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds (first axis) and half-width of the 95% t-interval."""
    half = T_95_DF4 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])

    return values.mean(axis=0), half


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
                ax.plot(k, mean, color=COLORS[variant], linewidth=2, label=variant)
                ax.fill_between(
                    k,
                    np.maximum(mean - half, 1e-6),
                    mean + half,
                    color=COLORS[variant],
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
    fig.savefig(VARIANT_RESULTS_DIR / "variants_comparison.png", dpi=150)


def to_lists(results: dict) -> dict:
    """Converts numpy arrays to lists for JSON."""
    if isinstance(results, dict):
        return {k: to_lists(v) for k, v in results.items()}

    return results.tolist()


if __name__ == "__main__":
    all_results = {mode: evaluate_mode(mode) for mode in MODES}
    for mode in MODES:
        print_summary(mode, all_results[mode])
    plot(all_results)
    with open(VARIANT_RESULTS_DIR / "test_results.json", "w") as f:
        json.dump(to_lists(all_results), f)
