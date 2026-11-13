"""
Exploratory analysis after the e02 conclusion, not part of the
pre-registered protocol: does the telescoped Debiased Bootstrap (TDB) beat
the Monte Carlo target C with finite data? See "Exploratory TDB check" in
README.md.

All variants use the flagged model of explore_flag.py (u_1 = G0 z_t,
u_{j+1} = G1 u_j, y_{t+j} = c . u_j, n = AR order), the training horizons
{1, 2, 3, 4}, lambda = 1, and are trained on the empirical second moments
of the same windows as the e02 small arm (evaluation seeds 3..32), plus a
tiny arm of 5 trajectories. Targets for k in {2, 3, 4}, under
stop-gradient, anchor points p = 2..k-1 with equal weights (k = 2: the true
future):

- C+F: the true future y_{t+k}
- D+F: T_{k,p}, the flagged rollout of k - p steps from the window at t + p
- DB+F: T_{k,p} + delta_{k,p}(z_t), delta the regression of y_{t+k} - T_{k,p}
  on z_t, one per (k, p)
- TDB+F: c G1^{k-p-1} (G0 z_{t+p} + delta_p(z_t)), one correction delta_p
  (n x L) per anchor point, fitted by least squares on the true futures
  pooled over k = p+1..10 through the model's own dynamics

Critics are fitted on the same windows. Evaluation is exact, on the
population covariance: mean rollout MSE over k = 1..10, as log ratio to C+F
on the same seed (paired, 30 seeds, 95% t-interval).

Prediction, written before the first run: DB+F equals C+F up to numerical
error, because a critic linear in z_t fitted on the same data makes the two
losses differ by a constant (the identity derived on 2026-11-13). D+F is
worse than C+F (target information bias). For TDB+F there is no directional
prediction: it can be better than C+F only through the restriction that one
delta_p serves all horizons, and in e02 the variance of the targets was
negligible even in the small arm, so the difference may be below the noise.
"""

import json
from pathlib import Path

import numpy as np
import torch
from design import EVALUATION_SEEDS, HORIZON, LEN_TRAJ
from run import SMALL_CONFIGS, second_moment, train_set

from anchor.envs.lgssm import observation_covariance, sample_observations, windows
from anchor.linear import embed, max_horizon_size, quadratic, truth
from anchor.stats import mean_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TRAIN_HORIZONS = (2, 3, 4)
LAM = 1.0
VARIANTS = ("D", "DB", "TDB")
RESTARTS = 3
OUTER = 300
DAMPING = 0.3
TINY_TAG, TINY_TRAJ = 7, 5


def tiny_set(system_idx, system, seed, window):
    rng = np.random.default_rng([TINY_TAG, system_idx, seed])
    obs = sample_observations(system, TINY_TRAJ, LEN_TRAJ, rng)
    return windows(obs, window, HORIZON) / np.sqrt(
        observation_covariance(system, 1)[0, 0]
    )


def row(params, k):
    g0, g1, c = params
    u = g0
    for _ in range(k - 1):
        u = g1 @ u
    return c @ u


def project(alpha, cov, window):
    """Functional of the regression of alpha . Y on z_t (fitted under cov)."""
    beta = np.linalg.solve(cov[:window, :window], (cov @ alpha)[:window])
    return embed(torch.as_tensor(beta), 0, len(alpha)).numpy()


def tdb_corrections(params, cov, window, p):
    """delta_p as an n x L matrix, least squares on the true futures k = p+1..HORIZON."""
    g0, g1, c = [x.numpy() for x in params]
    n, size = g1.shape[0], cov.shape[0]
    a, b = np.zeros((n * window, n * window)), np.zeros(n * window)
    feats = {}
    for k in range(p + 1, HORIZON + 1):
        lead = c @ np.linalg.matrix_power(g1, k - p - 1)
        feat = np.zeros((n * window, size))
        for i in range(n):
            feat[i * window : (i + 1) * window, :window] = lead[i] * np.eye(window)
        td = embed(torch.as_tensor(lead @ g0), p, size).numpy()
        resid = truth(window, k, size).numpy() - td
        a += feat @ cov @ feat.T
        b += feat @ cov @ resid
        feats[k] = (feat, td)
    d = np.linalg.lstsq(a, b, rcond=None)[0]
    return {k: td + feat.T @ d for k, (feat, td) in feats.items()}


def targets(variant, frozen, cov, window):
    size = cov.shape[0]
    out = {}
    for k in TRAIN_HORIZONS:
        y = truth(window, k, size).numpy()
        if variant == "C" or k == 2:
            out[k] = y
            continue
        rows = []
        for p in range(2, k):
            td = embed(row(frozen, k - p), p, size).numpy()
            if variant == "D":
                rows.append(td)
            elif variant == "DB":
                rows.append(td + project(y - td, cov, window))
            else:
                rows.append(tdb_corrections(frozen, cov, window, p)[k])
        out[k] = np.mean(rows, axis=0)
    return {k: torch.as_tensor(v) for k, v in out.items()}


def loss(params, tgt, cov_t, window):
    size = cov_t.shape[0]
    one = quadratic(embed(row(params, 1), 0, size) - truth(window, 1, size), cov_t)
    aux = [quadratic(embed(row(params, k), 0, size) - t, cov_t) for k, t in tgt.items()]
    return one + LAM * torch.stack(aux).mean()


def minimize(params, tgt, cov_t, window):
    params = [p.detach().clone().requires_grad_() for p in params]
    opt = torch.optim.LBFGS(
        params,
        max_iter=500,
        tolerance_grad=1e-13,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        value = loss(params, tgt, cov_t, window)
        value.backward()
        return value

    opt.step(closure)
    return [p.detach() for p in params]


def fit(variant, cov, window, n, start=None, seed=0):
    cov_t = torch.as_tensor(cov)
    if start is None:
        gen = torch.Generator().manual_seed(seed)
        params = [
            0.5 * torch.randn(n, window, generator=gen, dtype=torch.float64),
            0.5 * torch.randn(n, n, generator=gen, dtype=torch.float64),
            0.5 * torch.randn(n, generator=gen, dtype=torch.float64),
        ]
    else:
        params = [p.clone() for p in start]
    if variant == "C":
        params = minimize(params, targets("C", params, cov, window), cov_t, window)
        return params, loss(
            params, targets("C", params, cov, window), cov_t, window
        ).item()
    frozen = params
    for _ in range(OUTER):
        new = minimize(frozen, targets(variant, frozen, cov, window), cov_t, window)
        step = max((a - b).abs().max().item() for a, b in zip(new, frozen))
        frozen = [f + DAMPING * (p - f) for f, p in zip(frozen, new)]
        if step < 1e-10:
            break
    return frozen, step


def rollout_mse(params, cov_pop, window):
    size = cov_pop.shape[0]
    cov_t = torch.as_tensor(cov_pop)
    return np.array(
        [
            quadratic(
                embed(row(params, k), 0, size) - truth(window, k, size), cov_t
            ).item()
            for k in range(1, HORIZON + 1)
        ]
    )


def main():
    from design import SYSTEMS

    results = {}
    for system, window in SMALL_CONFIGS:
        idx = SYSTEMS.index(system)
        n = len(system.phi)
        size = max_horizon_size(window, HORIZON)
        cov_pop = observation_covariance(system, size)
        cov_pop = cov_pop / cov_pop[0, 0]
        name = f"{system.name}_L{window}"
        results[name] = {}
        for arm in ("small", "tiny"):
            ratios = {v: [] for v in VARIANTS}
            unconverged = {v: 0 for v in VARIANTS}
            for seed in EVALUATION_SEEDS:
                data = (
                    train_set(idx, system, "small", seed, window)
                    if arm == "small"
                    else tiny_set(idx, system, seed, window)
                )
                cov = second_moment(data)
                runs = [
                    fit("C", cov, window, n, seed=100 * seed + r)
                    for r in range(RESTARTS)
                ]
                c_params, _ = min(runs, key=lambda r: r[1])
                base = rollout_mse(c_params, cov_pop, window).mean()
                for v in VARIANTS:
                    params, step = fit(v, cov, window, n, start=c_params)
                    unconverged[v] += step >= 1e-6
                    ratios[v].append(
                        np.log(rollout_mse(params, cov_pop, window).mean() / base)
                    )
            row_out = {
                v: {
                    "log_ratio_vs_CF": [float(x) for x in mean_ci(np.array(r))],
                    "unconverged": int(unconverged[v]),
                }
                for v, r in ratios.items()
            }
            results[name][arm] = row_out
            print(
                f"{name:20s} {arm:5s} "
                + "  ".join(
                    f"{v}+F vs C+F {r['log_ratio_vs_CF'][0]:+.2e} ± {r['log_ratio_vs_CF'][1]:.1e}"
                    f" (unconv {r['unconverged']})"
                    for v, r in row_out.items()
                ),
                flush=True,
            )
    (RESULTS_DIR / "explore_tdb_small.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
