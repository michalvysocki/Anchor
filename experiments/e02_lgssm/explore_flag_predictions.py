"""
Exploratory analysis after the e02 conclusion, not part of the
pre-registered protocol: population stop-gradient fixed points of the
flagged variants C+F, D+F, M+F and M*+F in all 48 configurations of e02.
See "Exploratory flag predictions" in README.md.

The flagged model is the one of explore_flag.py: u_1 = G0 z_t, u_{j+1} =
G1 u_j, y_{t+j} = c . u_j, with n = AR order. Every target is a mixture over
the anchor points p = 2..k as in e02 (C: p = k only; D: uniform on
p = 2..k-1; M: uniform; M*: bias-variance weights from the flagged model's
critics, N of the large arm), and the bootstrapped targets are rollouts of
the same flagged model from the window at t + p, under stop-gradient.

Prediction, written before the first run: with the training horizons
{1, 2, 3, 4} (the first 2n for every system) the flagged rollout can be
optimal at every horizon, every target is then unbiased, and all four
variants share this fixed point at every lambda, so each closes the whole
gap between the one-step model and the best direct prediction (fraction
1.0), against at most 0.89 for the e02 variants. With the e02 horizons
{1, 3, 4} the same holds except where G1 is not identified (AR(2), L = 1).

Changed after a test on four configurations, before the full run: the
prediction fails for D and M. A bootstrapped target averages the prediction
from the window at t + p, and a window at t + p does not contain the window
at t, so E[E[y_{t+k} | z_{t+p}] | z_t] differs from E[y_{t+k} | z_t] (the
tower property needs nested information). The flag removes the bias of the
model, not this bias of the target, which vanishes only for a Markov
latent. C has no such bias, and M* detects it with its critics and moves to
C. Revised prediction: C+F and M*+F close the whole gap (fraction 1.0); D+F
and M+F do not; all four are exact for Markov latents. The first solver
(undamped stop-gradient iteration from random initializations) diverged for
D; the bootstrapped variants now start from the C+F solution with a damped
target.

Fixed points are computed by stop-gradient iteration: the target is frozen
at the current parameters, the loss is minimized by L-BFGS, and this is
repeated until the predictions stop changing (the parameters themselves are
defined only up to an invertible change of basis of u). Each variant starts
from RESTARTS random initializations; the run with the lowest final loss is
kept.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from design import HORIZON, LAMBDAS, SYSTEMS, WINDOWS, is_markov, n_windows

from anchor.envs.lgssm import observation_covariance
from anchor.linear import (
    direct_mse,
    embed,
    max_horizon_size,
    one_step_solution,
    quadratic,
    rollout_mse,
    simplex_minimizer,
    truth,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
HORIZON_SETS = {"1,2,3,4": (2, 3, 4), "1,3,4": (3, 4)}
VARIANTS = ("C", "D", "M", "M*")
RESTARTS = 3
OUTER = 400
DAMPING = 0.3


def row(params, k):
    """Row r with r . z = flagged prediction of y k steps ahead."""
    g0, g1, c = params
    u = g0
    for _ in range(k - 1):
        u = g1 @ u
    return c @ u


def target(params, k, p, window, size):
    if p == k:
        return truth(window, k, size)
    return embed(row(params, k - p), p, size)


def mixture_weights(variant, params, cov_n, window, horizons, n_eff):
    weights = {}
    for k in horizons:
        points = list(range(2, k + 1))
        if variant == "C" or k == 2:
            w = np.eye(len(points))[-1]
        elif variant == "D":
            w = np.r_[np.full(len(points) - 1, 1 / (len(points) - 1)), 0.0]
        elif variant == "M":
            w = np.full(len(points), 1 / len(points))
        else:
            w = simplex_minimizer(moments(params, cov_n, window, k, n_eff))
        weights[k] = w
    return weights


def moments(params, cov_n, window, k, n_eff):
    """B + V / n_eff of the anchor targets of horizon k, from two regressions on z_t."""
    size = cov_n.shape[0]
    c = cov_n.numpy()
    rows = np.stack(
        [target(params, k, p, window, size).numpy() for p in range(2, k + 1)]
    )
    past = c[:window, :window]
    betas = np.linalg.solve(past, (rows @ c)[:, :window].T).T
    bias = betas - betas[-1]
    resid = rows.copy()
    resid[:, :window] -= betas
    return bias @ past @ bias.T + resid @ c @ resid.T / n_eff


def loss(params, frozen, weights, cov_n, window, lam):
    size = cov_n.shape[0]
    out = quadratic(embed(row(params, 1), 0, size) - truth(window, 1, size), cov_n)
    aux = []
    for k, w in weights.items():
        tgt = sum(
            float(wp) * target(frozen, k, p, window, size)
            for wp, p in zip(w, range(2, k + 1))
        )
        aux.append(quadratic(embed(row(params, k), 0, size) - tgt, cov_n))
    return out + lam * torch.stack(aux).mean()


def minimize(params, frozen, weights, cov_n, window, lam):
    """Minimizes the loss over params with the target frozen, by L-BFGS."""
    params = [p.detach().clone().requires_grad_() for p in params]
    opt = torch.optim.LBFGS(
        params,
        max_iter=500,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        value = loss(params, frozen, weights, cov_n, window, lam)
        value.backward()
        return value

    opt.step(closure)
    return params


def field_norm(params, weights, cov_n, window, lam):
    """Norm of the stop-gradient update field: gradient with the target at params."""
    live = [p.detach().clone().requires_grad_() for p in params]
    frozen = [p.detach() for p in params]
    grads = torch.autograd.grad(loss(live, frozen, weights, cov_n, window, lam), live)
    return torch.cat([g.flatten() for g in grads]).norm().item()


def fixed_point(variant, lam, cov_n, window, n, horizons, n_eff, seed, start=None):
    """
    C is minimized from a random initialization. The bootstrapped variants
    start from the C+F solution (the branch connected to the optimum, as e02
    continues from the one-step optimum) and iterate with a damped target,
    frozen <- frozen + DAMPING (params - frozen), until the update field
    vanishes.
    """

    if variant == "C" or start is None:
        gen = torch.Generator().manual_seed(seed)
        params = [
            0.5 * torch.randn(n, window, generator=gen, dtype=torch.float64),
            0.5 * torch.randn(n, n, generator=gen, dtype=torch.float64),
            0.5 * torch.randn(n, generator=gen, dtype=torch.float64),
        ]
    else:
        params = [p.clone() for p in start]
    frozen = [p.detach().clone() for p in params]
    weights = mixture_weights(variant, frozen, cov_n, window, horizons, n_eff)
    for _ in range(OUTER):
        weights = mixture_weights(variant, frozen, cov_n, window, horizons, n_eff)
        params = minimize(params, frozen, weights, cov_n, window, lam)
        params = [p.detach() for p in params]
        if variant == "C":
            break
        frozen = [f + DAMPING * (p - f) for f, p in zip(frozen, params)]
        if field_norm(frozen, weights, cov_n, window, lam) < 1e-10:
            break
    params = frozen if variant != "C" else params
    final = loss(params, params, weights, cov_n, window, lam).item()
    residual = field_norm(params, weights, cov_n, window, lam)
    return final, params, weights, residual


def rollout(params, cov_n, window):
    size = cov_n.shape[0]
    return np.array(
        [
            quadratic(
                embed(row(params, k), 0, size) - truth(window, k, size), cov_n
            ).item()
            for k in range(1, HORIZON + 1)
        ]
    )


def main():
    predictions = json.loads((RESULTS_DIR / "predictions.json").read_text())
    results = {}
    for system in SYSTEMS:
        for window in WINDOWS:
            name = f"{system.name}_L{window}"
            n = len(system.phi)
            cov = observation_covariance(system, max_horizon_size(window, HORIZON))
            cov_n = torch.as_tensor(cov / cov[0, 0], dtype=torch.float64)
            direct = direct_mse(cov, window, HORIZON)
            onestep = rollout_mse(one_step_solution(cov, window), cov, HORIZON)
            gap = onestep.mean() - direct.mean()
            e02 = predictions[name]
            out = {
                "markov": is_markov(system, window),
                "gap": float(gap),
                "e02_best_fraction": {
                    v: float(
                        max(
                            (onestep.mean() - np.mean(e02[f"{key}_lam{lam}"]["mse"]))
                            / gap
                            for lam in LAMBDAS
                        )
                    )
                    if gap > 1e-12
                    else None
                    for v, key in (
                        ("C", "C"),
                        ("D", "D"),
                        ("M", "M"),
                        ("M*", "M*_large"),
                    )
                },
                "flagged": {},
            }
            for hname, horizons in HORIZON_SETS.items():
                for lam in LAMBDAS:
                    start = None
                    for variant in VARIANTS:
                        runs = [
                            fixed_point(
                                variant,
                                lam,
                                cov_n,
                                window,
                                n,
                                horizons,
                                n_windows("large", window),
                                seed,
                                start,
                            )
                            for seed in range(RESTARTS if variant == "C" else 1)
                        ]
                        _, params, weights, residual = min(runs, key=lambda r: r[0])
                        if variant == "C":
                            start = params
                        mse = rollout(params, cov_n, window)
                        out["flagged"][f"{variant}+F_{hname}_lam{lam}"] = {
                            "max_excess": float((mse - direct).max()),
                            "excess_k1": float(mse[0] - direct[0]),
                            "fraction": float((onestep.mean() - mse.mean()) / gap)
                            if gap > 1e-12
                            else None,
                            "residual": residual,
                            "weights": {k: w.tolist() for k, w in weights.items()},
                        }
            results[name] = out
            worst = {
                h: max(
                    r["max_excess"]
                    for key, r in out["flagged"].items()
                    if f"_{h}_" in key
                )
                for h in HORIZON_SETS
            }
            e02_best = out["e02_best_fraction"]
            e02_text = (
                " ".join(f"{v} {f:.2f}" for v, f in e02_best.items())
                if gap > 1e-12
                else "Markov"
            )
            print(
                f"{name:22s} gap {gap:.4f} | e02 best fraction: {e02_text} | "
                + " | ".join(f"flag {{{h}}} worst {w:.1e}" for h, w in worst.items()),
                flush=True,
            )
            sys.stdout.flush()

    (RESULTS_DIR / "explore_flag_predictions.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()
