"""
Evaluation of e02 against the pre-registered criteria, see README.md.
Every trained model is evaluated exactly from the known covariance.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from design import (
    ABLATION_OF,
    ABLATIONS,
    EVALUATION_SEEDS,
    HORIZON,
    LAMBDAS,
    SELECTION_SEEDS,
    SYSTEMS,
    VARIANTS,
    is_markov,
    n_windows,
)
from run import RUNS_DIR, second_moment, train_set

from anchor import linear
from anchor.envs.lgssm import observation_covariance
from anchor.plotting import VARIANT_COLORS

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
T_95_DF29 = 2.045
MARGIN = np.log(1.02)
GAP_MIN = 0.02
PREDICTED_MIN = 0.01
AGREEMENT_TOL = 0.01
SPEARMAN_MIN = 0.8
CRITIC_MAX, RESIDUAL_MIN = 1e-3, 1e-2
COLORS = {
    "C": VARIANT_COLORS["C"],
    "D": VARIANT_COLORS["D"],
    "M": "#7f7f7f",
    "M*": "#000000",
}


def ci(values: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of the 95% t-interval over evaluation seeds."""
    assert len(values) == len(EVALUATION_SEEDS)
    return float(values.mean()), float(
        T_95_DF29 * values.std(ddof=1) / np.sqrt(len(values))
    )


def verdict(mean: float, half: float) -> str:
    """Non-inferiority verdict of a log ratio to onestep with margin MARGIN (README)."""
    if mean + half < 0:
        return "better"
    if mean - half > MARGIN:
        return "material harm"
    if mean - half > 0:
        return "small increase"
    if mean + half < MARGIN:
        return "harmless"
    return "inconclusive"


def prediction_key(variant: str, arm: str) -> str:
    """Name of a variant in predictions.json; the oracle ablation is M* there."""
    if variant.startswith("M*"):
        return f"{variant.replace('_oracle', '')}_{arm}"
    return variant


def load_runs() -> dict:
    """runs[(system, window, arm)][variant] = run summary."""
    runs = defaultdict(dict)
    for path in RUNS_DIR.glob("*.json"):
        r = json.loads(path.read_text())
        runs[(r["system"], r["window"], r["arm"])][r["variant"]] = r
    return runs


def mean_mse(theta: np.ndarray, cov: np.ndarray) -> float:
    return float(linear.rollout_mse(theta, cov, HORIZON).mean())


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_config(
    system, window: int, arm: str, by_variant: dict, prediction: dict
) -> dict:
    cov = observation_covariance(system, linear.max_horizon_size(window, HORIZON))
    base = by_variant["onestep"]
    theta_base = {s: np.array(t) for s, t in zip(base["seeds"], base["theta"])}
    base_mse = {s: mean_mse(t, cov) for s, t in theta_base.items()}
    out = {"markov": is_markov(system, window), "onestep": base_mse, "variants": {}}

    for variant in VARIANTS + ABLATIONS:
        r = by_variant[variant]
        seeds, lams = r["seeds"], r["lambdas"]
        theta = np.array(r["theta"])
        val = np.array(r["val_mse"])
        selection = {
            lam: np.mean(
                [
                    val[i]
                    for i in range(len(seeds))
                    if lams[i] == lam and seeds[i] in SELECTION_SEEDS
                ]
            )
            for lam in LAMBDAS
        }
        chosen = min(selection, key=selection.get)
        key = prediction_key(variant, arm)
        per_lam = {}
        for lam in LAMBDAS:
            idx = [
                i
                for i in range(len(seeds))
                if lams[i] == lam and seeds[i] in EVALUATION_SEEDS
            ]
            log_ratio = np.array(
                [np.log(mean_mse(theta[i], cov) / base_mse[seeds[i]]) for i in idx]
            )
            # without stop-gradient the fixed point may not exist (predictions)
            star = None
            if variant != "D_nosg":
                star = linear.fixed_point(
                    variant.replace("_oracle", ""),
                    lam,
                    cov,
                    window,
                    n_eff=n_windows(arm, window),
                )["theta"]
            predicted = float(
                np.log(
                    np.mean(prediction[f"{key}_lam{lam}"]["mse"])
                    / np.mean(prediction["onestep"])
                )
            )
            mean, half = ci(log_ratio)
            per_lam[lam] = {
                "log_ratio": (mean, half),
                "per_seed": log_ratio.tolist(),
                "predicted": predicted,
                "distance": None
                if star is None
                else float(np.mean([np.linalg.norm(theta[i] - star) for i in idx])),
                "grad_ratio_end": float(
                    np.median([r["grad_ratio_end"][i] for i in idx])
                ),
            }
        out["variants"][variant] = {"lambda": chosen, "per_lambda": per_lam}

    # Q5: critic and raw residual of D's targets at the trained onestep models
    if arm == "large":
        system_idx = SYSTEMS.index(system)
        stats = []
        for s in EVALUATION_SEEDS:
            moments = second_moment(train_set(system_idx, system, arm, s, window))
            stats.append(linear.critic_and_residual(theta_base[s], moments))
        out["critic"], out["residual"] = np.mean(stats, axis=0).tolist()
    return out


def verdicts(results: dict, predictions: dict) -> dict:
    v = {}
    large = {k: r for k, r in results.items() if k[2] == "large"}
    small = {k: r for k, r in results.items() if k[2] == "small"}

    def gap(key):
        """Relative onestep -> direct improvement, selects the gap configurations."""
        p = predictions[f"{key[0]}_L{key[1]}"]
        return 1 - np.mean(p["direct"]) / np.mean(p["onestep"])

    def absolute_gap(key):
        """onestep - direct in units of Var(y), the scale of the critic (H5)."""
        p = predictions[f"{key[0]}_L{key[1]}"]
        return np.mean(p["onestep"]) - np.mean(p["direct"])

    # H1: agreement and "better" where the predicted improvement >= 1%
    h1 = {}
    for variant in ("C", "M", "M*"):
        checks = []
        for key, r in large.items():
            if r["markov"] or gap(key) < GAP_MIN:
                continue
            d = r["variants"][variant]
            p = d["per_lambda"][d["lambda"]]
            if -p["predicted"] < np.log(1 / (1 - PREDICTED_MIN)):
                continue
            mean, half = p["log_ratio"]
            agree = abs(mean - p["predicted"]) <= max(half, AGREEMENT_TOL)
            checks.append(agree and mean + half < 0)
        h1[variant] = (sum(checks), len(checks))
    v["H1"] = {
        "counts": h1,
        "holds": all(n > 0 and k / n >= 0.8 for k, n in h1.values()),
    }

    # H2 and H3: equivalence at every lambda in Markov configurations
    def classify(arm_results, variants=VARIANTS):
        counts = defaultdict(int)
        for r in arm_results.values():
            if not r["markov"]:
                continue
            for d in (r["variants"][x] for x in variants):
                for p in d["per_lambda"].values():
                    counts[verdict(*p["log_ratio"])] += 1
        return dict(counts)

    v["H2"] = classify(large)
    v["H2"]["holds"] = v["H2"].get("material harm", 0) == 0
    v["H3"] = classify(small)

    # secondary ablations: Markov verdicts at every lambda, and the paired
    # log(ablation / the variant it modifies) at the selected lambdas elsewhere
    for ablation in ABLATIONS:
        entry = {
            "markov_large": classify(large, (ablation,)),
            "markov_small": classify(small, (ablation,)),
        }
        for arm_name, arm_results in (("large", large), ("small", small)):
            diffs = []
            for r in arm_results.values():
                if r["markov"]:
                    continue
                a, m = r["variants"][ablation], r["variants"][ABLATION_OF[ablation]]
                diff = np.array(a["per_lambda"][a["lambda"]]["per_seed"]) - np.array(
                    m["per_lambda"][m["lambda"]]["per_seed"]
                )
                diffs.append(ci(diff))
            entry[f"vs_{ABLATION_OF[ablation]}_{arm_name}"] = {
                "mean_of_means": float(np.mean([d[0] for d in diffs]))
                if diffs
                else None,
                "worse": sum(d[0] - d[1] > 0 for d in diffs),
                "better": sum(d[0] + d[1] < 0 for d in diffs),
                "configs": len(diffs),
            }
        v[ablation] = entry

    # H4: in every non-Markov configuration of an arm, the upper end of the CI
    # of the paired log(M* / better of C and D) is below the margin
    for arm_name, arm_results in (("large", large), ("small", small)):
        passed, excess = [], []
        for r in arm_results.values():
            if r["markov"]:
                continue
            chosen = {}
            for variant in ("C", "D", "M*"):
                d = r["variants"][variant]
                chosen[variant] = d["per_lambda"][d["lambda"]]
            better = min(("C", "D"), key=lambda x: chosen[x]["log_ratio"][0])
            diff = np.array(chosen["M*"]["per_seed"]) - np.array(
                chosen[better]["per_seed"]
            )
            mean, half = ci(diff)
            passed.append(mean + half < MARGIN)
            excess.append(mean + half)
        v[f"H4_{arm_name}"] = {
            "passed": f"{sum(passed)}/{len(passed)}",
            "max_upper_end": float(max(excess)) if excess else None,
            "holds": bool(passed) and all(passed),
        }

    # H5: critic tracks the gap, raw residual does not
    nm = [(k, r) for k, r in large.items() if not r["markov"]]
    rho_critic = spearman(
        [r["critic"] for _, r in nm], [absolute_gap(k) for k, _ in nm]
    )
    rho_resid = spearman(
        [r["residual"] for _, r in nm], [absolute_gap(k) for k, _ in nm]
    )
    markov_ok = all(
        r["critic"] < CRITIC_MAX and r["residual"] > RESIDUAL_MIN
        for r in large.values()
        if r["markov"]
    )
    v["H5"] = {
        "spearman_critic": rho_critic,
        "spearman_residual": rho_resid,
        "markov_ok": markov_ok,
        "holds": rho_critic >= SPEARMAN_MIN and markov_ok,
    }
    return v


def plot(results: dict, predictions: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for variant in VARIANTS:
        xs, ys, es = [], [], []
        for key, r in results.items():
            if key[2] != "large" or r["markov"]:
                continue
            d = r["variants"][variant]
            p = d["per_lambda"][d["lambda"]]
            xs.append(p["predicted"])
            ys.append(p["log_ratio"][0])
            es.append(p["log_ratio"][1])
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            fmt="o",
            ms=4,
            color=COLORS[variant],
            label=variant,
            alpha=0.8,
        )
    lim = ax.get_xlim()
    ax.plot(lim, lim, "--", color="gray", lw=1)
    ax.set_xlabel("predicted log(MSE / onestep), population fixed point")
    ax.set_ylabel("trained, mean ± 95% CI over 30 seeds")
    ax.set_title("Large arm, non-Markov configurations, selected λ")
    ax.legend()

    ax = axes[1]
    for key, r in results.items():
        if key[2] != "large":
            continue
        p = predictions[f"{key[0]}_L{key[1]}"]
        g = np.mean(p["onestep"]) - np.mean(p["direct"])
        ax.scatter(max(g, 1e-6), r["critic"], color="black", s=14)
        ax.scatter(max(g, 1e-6), r["residual"], color="gray", s=14, marker="x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("population gap: onestep − direct, units of Var(y) (Markov at 1e-6)")
    ax.set_ylabel("estimated at trained onestep")
    ax.scatter([], [], color="black", label="critic")
    ax.scatter([], [], color="gray", marker="x", label="raw residual")
    ax.legend()
    ax.set_title("Q5: critic vs residual")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "e02_summary.png", dpi=150)


def main():
    predictions = json.loads((RESULTS_DIR / "predictions.json").read_text())
    runs = load_runs()
    systems = {s.name: s for s in SYSTEMS}
    results = {}
    for (name, window, arm), by_variant in sorted(runs.items()):
        results[(name, window, arm)] = evaluate_config(
            systems[name], window, arm, by_variant, predictions[f"{name}_L{window}"]
        )

    print("selected λ and log(MSE / onestep), mean ± 95% CI over 30 seeds; [predicted]")
    for key, r in results.items():
        cells = []
        for variant in VARIANTS:
            d = r["variants"][variant]
            p = d["per_lambda"][d["lambda"]]
            m, h = p["log_ratio"]
            cells.append(
                f"{variant} λ={d['lambda']:<5} {m:+.4f}±{h:.4f} [{p['predicted']:+.4f}]"
            )
        flag = "M" if r["markov"] else " "
        print(f"{key[0]:<16} L{key[1]} {key[2]:<5} {flag} " + " | ".join(cells))

    v = verdicts(results, predictions)
    print("\nverdicts")
    for k, val in v.items():
        print(f"  {k}: {val}")

    plot(results, predictions)
    serial = {f"{k[0]}_L{k[1]}_{k[2]}": r for k, r in results.items()}
    with open(RESULTS_DIR / "test_results.json", "w") as f:
        json.dump({"results": serial, "verdicts": v}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
