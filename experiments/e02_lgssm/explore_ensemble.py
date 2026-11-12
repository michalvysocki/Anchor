"""
Exploratory analysis after the e02 conclusion, not part of the
pre-registered protocol: does the disagreement of an ensemble of one-step
models see the Markov gap, or only the estimation error? See "Exploratory
ensemble check" in README.md.

Prediction, written before the first run: every member converges to the
same population one-step solution, so the ensemble disagreement falls like
1/N, while the excess rollout error over the best direct prediction and the
critic stay at their population values, positive with a gap and zero for a
Markov latent. An ensemble is blind to the Markov gap; the critic is not.

Ensemble: M members fitted by least squares on bootstrap resamples of the
same training windows (as in PETS). Without resampling the members of a
linear least-squares ensemble coincide and the disagreement is exactly zero.
All quantities are exact population moments in units of Var(y):

- disagreement at k: mean over members of E[(g_m^k(z_t) - mean_m g_m^k(z_t))^2]
- excess at k: rollout MSE of the ensemble mean minus the best linear direct MSE
- critic: E[(E[R | z_t])^2] of the ensemble mean, as in e02
"""

import json
from pathlib import Path

import numpy as np
import torch
from design import HORIZON, LEN_TRAJ

from anchor.envs.lgssm import (
    ar1,
    ar2,
    observation_covariance,
    sample_observations,
    windows,
)
from anchor.linear import (
    critic_and_residual,
    direct_mse,
    embed,
    max_horizon_size,
    one_step_solution,
    quadratic,
    rollout_mse,
    rollout_row,
)
from anchor.stats import mean_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MEMBERS = 5
SEEDS = range(10)
N_TRAJ = (5, 20, 200, 2000)
CONFIGS = [
    ("ar1_a0.9_k1.0_L1", ar1(0.9, 1.0), 1),
    ("ar1_a0.9_k0.8_L1", ar1(0.9, 0.8), 1),
    ("ar1_a0.9_k0.5_L1", ar1(0.9, 0.5), 1),
    ("ar2_k0.8_L2", ar2(0.9, np.pi / 6, 0.8), 2),
]


def fit(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(z, y, rcond=None)[0]


def disagreement(thetas: list[np.ndarray], cov: np.ndarray) -> np.ndarray:
    """Mean spread of the members' rollouts at k = 1..HORIZON, units of Var(y)."""
    size = cov.shape[0]
    cov_n = torch.as_tensor(cov / cov[0, 0], dtype=torch.float64)
    out = []
    for k in range(1, HORIZON + 1):
        rows = [
            embed(rollout_row(torch.as_tensor(t, dtype=torch.float64), k), 0, size)
            for t in thetas
        ]
        mean = torch.stack(rows).mean(0)
        out.append(np.mean([quadratic(r - mean, cov_n).item() for r in rows]))

    return np.array(out)


def main():
    results = {}
    for name, system, window in CONFIGS:
        cov = observation_covariance(system, max_horizon_size(window, HORIZON))
        theta_pop = one_step_solution(cov, window)
        direct = direct_mse(cov, window, HORIZON)
        population = {
            "excess_k10": float(rollout_mse(theta_pop, cov, HORIZON)[-1] - direct[-1]),
            "critic": critic_and_residual(theta_pop, cov)[0],
        }
        rows = {}
        for n_traj in N_TRAJ:
            stats = {"disagreement_k10": [], "excess_k10": [], "critic": []}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                data = windows(
                    sample_observations(system, n_traj, LEN_TRAJ, rng), window, 1
                )
                z, y = data[:, :window], data[:, window]
                thetas = []
                for _ in range(MEMBERS):
                    idx = rng.integers(0, len(z), len(z))
                    thetas.append(fit(z[idx], y[idx]))
                mean = np.mean(thetas, axis=0)
                stats["disagreement_k10"].append(disagreement(thetas, cov)[-1])
                stats["excess_k10"].append(
                    rollout_mse(mean, cov, HORIZON)[-1] - direct[-1]
                )
                stats["critic"].append(critic_and_residual(mean, cov)[0])
            rows[n_traj] = {
                key: [float(x) for x in mean_ci(v)] for key, v in stats.items()
            }
        results[name] = {"population": population, "by_n_traj": rows}

        print(
            f"\n{name}  population: excess_k10 {population['excess_k10']:.4f}"
            f"  critic {population['critic']:.5f}"
        )
        print(
            f"{'n_traj':>7} {'windows':>8} {'disagree_k10':>14} {'excess_k10':>12} {'critic':>10}"
        )
        for n_traj, r in rows.items():
            n_win = n_traj * (LEN_TRAJ - window)
            print(
                f"{n_traj:>7} {n_win:>8} {r['disagreement_k10'][0]:>14.2e}"
                f" {r['excess_k10'][0]:>12.4f} {r['critic'][0]:>10.5f}"
            )

    (RESULTS_DIR / "explore_ensemble.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
