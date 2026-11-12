"""
Theory predictions for e02, computed before any training: population
stop-gradient fixed points of every variant on every system and window,
their rollout errors, the critic and the raw residual. See README.md.
"""

import json
from pathlib import Path

from design import HORIZON, LAMBDAS, N_TRAJ, SYSTEMS, WINDOWS, is_markov, n_windows

from anchor import linear
from anchor.envs.lgssm import observation_covariance, optimal_mse
from anchor.metrics import to_lists

EXPERIMENT_DIR = Path(__file__).resolve().parent
# (name in the output, variant, data size that sets M*'s effective sample
# size); the oracle ablation of M* is M* itself in the population
ARMS = (
    [("C", "C", None), ("D", "D", None), ("M", "M", None)]
    + [
        (f"{variant}_{data}", variant, data)
        for variant in ("M*", "M*_p1")
        for data in N_TRAJ
    ]
    + [("D_nosg", "D_nosg", None)]
)


def predict(system, window: int) -> dict:
    cov = observation_covariance(system, linear.max_horizon_size(window, HORIZON))
    base = linear.one_step_solution(cov, window)
    critic, residual = linear.critic_and_residual(base, cov)
    out = {
        "markov": is_markov(system, window),
        "kalman": optimal_mse(system, HORIZON),
        "direct": linear.direct_mse(cov, window, HORIZON),
        "onestep": linear.rollout_mse(base, cov, HORIZON),
        "critic": critic,
        "residual": residual,
    }
    for arm, variant, data in ARMS:
        n_eff = n_windows(data, window) if data else None
        for lam in LAMBDAS:
            point = linear.fixed_point(variant, lam, cov, window, n_eff=n_eff)
            weights = point["weights"] or {}
            out[f"{arm}_lam{lam}"] = {
                "mse": linear.rollout_mse(point["theta"], cov, HORIZON),
                "residual": point["residual"],
                "stable": point["stable"],
                "weights": {str(k): w for k, w in weights.items()},
            }

    return out


def main():
    results = {}
    for system in SYSTEMS:
        for window in WINDOWS:
            results[f"{system.name}_L{window}"] = predict(system, window)

    arms = [arm for arm, _, _ in ARMS]
    print("best over lambda of the mean rollout MSE over k = 1..10, relative to")
    print("onestep (population); unstable fixed points listed last")
    print(
        f"{'config':<22} markov  onestep  direct  kalman  "
        + "  ".join(f"{a:<12}" for a in arms)
        + "  critic   resid  unstable"
    )
    for name, r in results.items():
        base = r["onestep"].mean()
        best = [
            min(r[f"{a}_lam{lam}"]["mse"].mean() for lam in LAMBDAS) / base - 1
            for a in arms
        ]
        keys = [f"{a}_lam{lam}" for a in arms for lam in LAMBDAS]
        unstable = [key for key in keys if not r[key]["stable"]]
        # the main variants must converge; an ablation that does not (D_nosg
        # can lose its fixed point) is flagged, not an error
        main_keys = [k for k in keys if not k.startswith(("M*_p1", "D_nosg"))]
        worst = max(r[key]["residual"] for key in main_keys)
        assert worst < 1e-8, f"{name}: fixed point not converged ({worst})"
        unstable += [k + "(no fixed point)" for k in keys if r[k]["residual"] > 1e-8]
        print(
            f"{name:<22} {r['markov']!s:<7} {base:7.4f} "
            f"{r['direct'].mean() / base - 1:+7.3f} "
            f"{r['kalman'].mean() / base - 1:+7.3f}  "
            + "  ".join(f"{x:+12.3f}" for x in best)
            + f"  {r['critic']:.1e} {r['residual']:.1e}  {','.join(unstable) or '-'}"
        )

    with open(EXPERIMENT_DIR / "results" / "predictions.json", "w") as f:
        json.dump(to_lists(results), f, indent=1)


if __name__ == "__main__":
    main()
