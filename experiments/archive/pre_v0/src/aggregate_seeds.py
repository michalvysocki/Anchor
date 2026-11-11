import matplotlib.pyplot as plt
import numpy as np
from config import K_MAX, RESULTS_DIR, TRAINING_SEEDS

# two-sided 95% Student t quantile for 4 degrees of freedom (5 seeds)
T_95_DF4 = 2.776
DIRECT_COLOR = "#2a78d6"
ROLLOUT_COLOR = "#eb6834"


def load_seed(mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads paired test errors and Spearman correlation of one trained seed.

    Args:
        mode (str): insufficient or sufficient observability
        seed (int): training seed

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: direct errors (n, k_max),
        rollout errors (n, k_max), Spearman correlation (k_max,)
    """
    data = np.load(RESULTS_DIR / f"paired_errors_{mode}_seed{seed}.npz")

    return data["dir_errors"], data["roll_errors"], data["spearman"]


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean over seeds and half-width of the 95% t-interval.

    Args:
        values (np.ndarray): per-seed statistic of size (n_seeds, k_max)

    Returns:
        tuple[np.ndarray, np.ndarray]: mean and half-width, each (k_max,)
    """
    n = values.shape[0]
    half = T_95_DF4 * values.std(axis=0, ddof=1) / np.sqrt(n)

    return values.mean(axis=0), half


def aggregate(mode: str) -> dict[str, np.ndarray]:
    """
    Aggregates per-seed medians, win rates and correlations for one mode.

    Args:
        mode (str): insufficient or sufficient observability

    Returns:
        dict[str, np.ndarray]: per-seed arrays of size (n_seeds, k_max)
    """
    med_dir, med_roll, win, spearman = [], [], [], []
    for seed in TRAINING_SEEDS:
        dir_errors, roll_errors, rho = load_seed(mode, seed)
        med_dir.append(np.median(dir_errors, axis=0))
        med_roll.append(np.median(roll_errors, axis=0))
        win.append(np.mean(dir_errors < roll_errors, axis=0))
        spearman.append(rho)

    return {
        "med_dir": np.array(med_dir),
        "med_roll": np.array(med_roll),
        "win": np.array(win),
        "spearman": np.array(spearman),
    }


def print_table(mode: str, agg: dict[str, np.ndarray]):
    """Prints mean over seeds with 95% CI for each horizon."""
    d, d_h = mean_ci(agg["med_dir"])
    r, r_h = mean_ci(agg["med_roll"])
    w, w_h = mean_ci(agg["win"])
    s, _ = mean_ci(agg["spearman"])

    print(
        f"\n{mode} (median test error, mean over {len(TRAINING_SEEDS)} seeds, 95% CI)"
    )
    print(" k | direct            | rollout           | P(direct<rollout) | Spearman")
    for k in range(K_MAX):
        print(
            f"{k + 1:2d} | {d[k]:.4f} ± {d_h[k]:.4f} | {r[k]:.4f} ± {r_h[k]:.4f} | "
            f"{w[k]:.2f} ± {w_h[k]:.2f}       | {s[k]:+.2f}"
        )


def plot(results: dict[str, dict[str, np.ndarray]]):
    """Plots median error vs horizon for direct and rollout, one panel per mode."""
    k = np.arange(1, K_MAX + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (mode, agg) in zip(axes, results.items()):
        for key, label, color, style in [
            ("med_dir", "direct", DIRECT_COLOR, "-"),
            ("med_roll", "rollout", ROLLOUT_COLOR, "--"),
        ]:
            mean, half = mean_ci(agg[key])
            ax.plot(k, mean, style, color=color, linewidth=2, label=label)
            ax.fill_between(
                k, np.maximum(mean - half, 1e-6), mean + half, color=color, alpha=0.2
            )
        ax.set_yscale("log")
        ax.set_title(f"{mode} observation")
        ax.set_xlabel("Prediction horizon k")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(True, which="major", color="#e0e0e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Median test error (log scale)")
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Direct vs rollout, median over test points, mean ± 95% CI over "
        f"{len(TRAINING_SEEDS)} seeds"
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "direct_vs_rollout.png", dpi=150)


if __name__ == "__main__":
    results = {mode: aggregate(mode) for mode in ("insufficient", "sufficient")}
    for mode, agg in results.items():
        print_table(mode, agg)
    plot(results)
