"""
Exploratory analysis after the e02 conclusion, not part of the
pre-registered protocol: does a one-step model that distinguishes encoded
from predicted latents remove the Markov-gap ceiling of the rollout? See
"Exploratory flag check" in README.md.

Flagged model: the first step maps the window to a prediction latent,
u_1 = G0 z_t (dimension n), every later step applies G1, u_{j+1} = G1 u_j,
and a readout predicts y_{t+j} = c . u_j. The unflagged one-step model of
e02 is the special case in which the prediction latent is the window itself
and the first step equals every later step.

Prediction, written before the first run: for any linear-Gaussian system and
any window, E[x_{t+k} | z_t] = A^{k-1} E[x_{t+1} | z_t] (tower property on the
Markov state), so with n at least the state dimension the flagged rollout can
equal the best linear direct prediction at every horizon. Then the one-step
and the multi-step terms share their minimizer, so training on horizons
{1, 3, 4} reaches the direct optimum at k = 1..10 for every lambda, with no
trade-off at k = 1.

Added after the first run: AR(2) with a one-observation window failed with
{1, 3, 4} (the trained G1 is not identified and diverges beyond k = 4), so
every configuration is also trained on {1, 2, 3, 4}, the first 2n horizons
(a state-space model of order n is determined by 2n consecutive impulse
responses, as in Ho-Kalman realization).

Training: population losses (exact quadratic forms in the covariance), the
one-step loss plus lambda times the Monte Carlo loss at AUX_HORIZONS, all
parameters trained jointly, L-BFGS from several random initializations; the
best final loss is kept. Reported in units of Var(y): the largest excess of
the rollout MSE over the direct MSE across k = 1..10, for the flagged model
and for the e02 one-step model.
"""

import json
from pathlib import Path

import numpy as np
import torch
from design import HORIZON

from anchor.envs.lgssm import ar1, ar2, observation_covariance
from anchor.linear import (
    AUX_HORIZONS,
    direct_mse,
    embed,
    max_horizon_size,
    one_step_solution,
    quadratic,
    rollout_mse,
    truth,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LAMBDAS = (0.01, 1.0, 100.0)
TRAIN_HORIZONS = (AUX_HORIZONS, (2, 3, 4))
RESTARTS = 8
CONFIGS = [
    ("ar1_a0.9_k1.0_L1", ar1(0.9, 1.0), 1),
    ("ar1_a0.9_k0.8_L1", ar1(0.9, 0.8), 1),
    ("ar1_a0.99_k0.5_L1", ar1(0.99, 0.5), 1),
    ("ar1_a0.9_k0.5_L4", ar1(0.9, 0.5), 4),
    ("ar2_k0.8_L1", ar2(0.9, np.pi / 6, 0.8), 1),
    ("ar2_k0.8_L2", ar2(0.9, np.pi / 6, 0.8), 2),
    ("ar2_k0.5_L4", ar2(0.9, np.pi / 6, 0.5), 4),
]


def flagged_row(g0, g1, c, k):
    """Row r with r . z_t = flagged prediction of y_{t+k}."""
    u = g0
    for _ in range(k - 1):
        u = g1 @ u
    return c @ u


def flagged_mse(params, cov_n, window, horizon):
    g0, g1, c = params
    size = cov_n.shape[0]
    return np.array(
        [
            quadratic(
                embed(flagged_row(g0, g1, c, k), 0, size) - truth(window, k, size),
                cov_n,
            ).item()
            for k in range(1, horizon + 1)
        ]
    )


def train(cov_n, window, n, lam, seed, horizons=AUX_HORIZONS):
    gen = torch.Generator().manual_seed(seed)
    g0 = (
        0.5 * torch.randn(n, window, generator=gen, dtype=torch.float64)
    ).requires_grad_()
    g1 = (0.5 * torch.randn(n, n, generator=gen, dtype=torch.float64)).requires_grad_()
    c = (0.5 * torch.randn(n, generator=gen, dtype=torch.float64)).requires_grad_()
    size = cov_n.shape[0]

    def loss_fn():
        def term(k):
            return quadratic(
                embed(flagged_row(g0, g1, c, k), 0, size) - truth(window, k, size),
                cov_n,
            )

        multi = sum(term(k) for k in horizons) / len(horizons)
        return term(1) + lam * multi

    opt = torch.optim.LBFGS(
        [g0, g1, c],
        max_iter=5000,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)

    return loss_fn().item(), (g0.detach(), g1.detach(), c.detach())


def main():
    results = {}
    for name, system, window in CONFIGS:
        n = len(system.phi)
        cov = observation_covariance(system, max_horizon_size(window, HORIZON))
        cov_n = torch.as_tensor(cov / cov[0, 0], dtype=torch.float64)
        direct = direct_mse(cov, window, HORIZON)
        onestep = rollout_mse(one_step_solution(cov, window), cov, HORIZON)
        row = {"onestep_max_excess": float((onestep - direct).max()), "flagged": {}}
        for horizons in TRAIN_HORIZONS:
            key = "1," + ",".join(map(str, horizons))
            row["flagged"][key] = {}
            for lam in LAMBDAS:
                runs = [
                    train(cov_n, window, n, lam, s, horizons) for s in range(RESTARTS)
                ]
                _, params = min(runs, key=lambda r: r[0])
                excess = flagged_mse(params, cov_n, window, HORIZON) - direct
                row["flagged"][key][lam] = {
                    "max_excess": float(excess.max()),
                    "excess_k1": float(excess[0]),
                    "excess": excess.tolist(),
                    "eigenvalues_g1": [
                        str(e) for e in np.linalg.eigvals(params[1].numpy())
                    ],
                }
            worst = max(r["max_excess"] for r in row["flagged"][key].values())
            print(
                f"{name:20s} onestep {row['onestep_max_excess']:.4f}"
                f" | flagged on {{{key}}}: worst over lambda {worst:.1e}"
            )
        results[name] = row

    (RESULTS_DIR / "explore_flag.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
