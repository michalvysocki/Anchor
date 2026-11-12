"""
Exploratory check after the e02 evaluation, not part of the pre-registered
protocol (README: Exploratory analysis). In the Markov configurations of the
large arm, the main variants at lambda = 10 are retrained with ten times as
many steps. If the "small increase" and "material harm" verdicts of H2 come
from optimization that had not converged, they vanish and the trained
models move to the predicted fixed point (the one-step optimum).
"""

import json

import numpy as np
import run
from design import EVALUATION_SEEDS, VARIANTS, is_markov
from evaluate import RESULTS_DIR, ci, mean_mse, verdict

from anchor import linear
from anchor.envs.lgssm import observation_covariance

LAMBDA = 10.0
N_STEPS = 30_000


def main():
    run.LAMBDAS = (LAMBDA,)
    run.SEEDS = EVALUATION_SEEDS
    run.N_STEPS = N_STEPS
    before = json.loads((RESULTS_DIR / "test_results.json").read_text())["results"]

    out = {}
    for system_idx, system, window in run.configs("large"):
        if not is_markov(system, window):
            continue
        name = f"{system.name}_L{window}_large"
        cov = observation_covariance(system, linear.max_horizon_size(window, 10))
        star = linear.one_step_solution(cov, window)
        base = {int(s): float(v) for s, v in before[name]["onestep"].items()}
        for variant in VARIANTS:
            trained = run.train_config(system_idx, system, window, variant, "large")
            theta = np.array(trained["theta"])
            log_ratio = np.array(
                [
                    np.log(mean_mse(t, cov) / base[s])
                    for t, s in zip(theta, trained["seeds"])
                ]
            )
            prev = before[name]["variants"][variant]["per_lambda"][str(LAMBDA)]
            mean, half = ci(log_ratio)
            out[f"{name}_{variant}"] = {
                "before": {
                    "log_ratio": prev["log_ratio"],
                    "verdict": verdict(*prev["log_ratio"]),
                    "distance": prev["distance"],
                },
                "after": {
                    "log_ratio": (mean, half),
                    "verdict": verdict(mean, half),
                    "distance": float(
                        np.mean([np.linalg.norm(t - star) for t in theta])
                    ),
                },
            }
            print(
                f"{name:<24} {variant:<3} "
                f"{prev['log_ratio'][0]:+.4f} ({verdict(*prev['log_ratio'])}, "
                f"d={prev['distance']:.3f}) -> {mean:+.4f} ({verdict(mean, half)}, "
                f"d={out[f'{name}_{variant}']['after']['distance']:.3f})",
                flush=True,
            )

    with open(RESULTS_DIR / "convergence_check.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
