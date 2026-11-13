"""
Exploratory pilot, not pre-registered and not part of e03 (whose code it
imports but does not change): DB*, a multi-anchor control-variate target.
See README.md in this directory.

For a horizon k, every bootstrapped anchor is debiased by the critics,
T_{k,p} + h_y(z_t) - h_p(z_t), where h_p regresses T_{k,p} and h_y regresses
y_{t+k} on sg[phi(z_t)] (the critic heads of M*). With exact critics every
candidate (the debiased anchors and y_{t+k} itself) has the same conditional
mean E[y_{t+k} | z_t], so the weights minimize the variance alone,
w = argmin over the simplex of w'Vw, V the covariance of the candidates
around h_y; no bias term, no division by N.

Variants, all with the flag: C+F, D+F, DB+F (e03's variants) and DB*+F.
Configurations: κ ∈ {0.8, 0.5, 1}, window L = 1. Five seeds (100-104, no
draw shared with e03), 30 000 steps, Adam 1e-3 (critics 3e-3) reduced
tenfold for the last 20%, λ = 1, horizons {2, 3, 4}; DB*'s weights stay on C
for the first 1000 steps, then are recomputed every 100 steps.

Predictions, written before the first run:
R1  the variance of the DB* target around h_y is at most that of DB and
    below that of C
R2  the weights of DB* lie near the boundary of the simplex, with little
    weight on y_{t+k}, where the observation noise is large (κ = 0.5)
R3  no prediction on the rollout MSE (k = 1..4); it is measured per seed
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

E03 = Path(__file__).resolve().parents[1] / "e03_henon_latent"
sys.path.insert(0, str(E03))

from design import SIZES
from run import Data

from anchor.ensemble import EnsembleAdam
from anchor.latent import WorldModels
from anchor.linear import simplex_minimizer

OUT = Path(
    os.environ.get("PILOT_OUT", str(Path(__file__).resolve().parent / "results"))
)
CONFIGS = ((0.8, 1), (0.5, 1), (1.0, 1))
VARIANTS = ("C+F", "D+F", "DB+F", "DB*+F")
SEEDS = (100, 101, 102, 103, 104)
HORIZONS = (2, 3, 4)
STEPS = int(os.environ.get("PILOT_STEPS", "30000"))
BATCH, LR, CRITIC_FACTOR, LAM = 512, 1e-3, 3.0, 1.0
WARMUP, EVERY = 1000, 100


class DBStarModels(WorldModels):
    """M*'s critic heads, with debiased anchors weighted by their variance alone."""

    def __init__(self, window, seeds):
        super().__init__("M*+F", window, HORIZONS, list(seeds), SIZES)

    def candidates(self, w, out, boot, k):
        """Debiased anchors for p = 2..k-1 and y_{t+k}, stacked as (P, M, B)."""
        h_y = out[..., len(self.pairs) + self.horizons.index(k)]
        rows = [
            boot[k, p] + h_y - out[..., self.pairs.index((k, p))] for p in range(2, k)
        ]
        return torch.stack(rows + [self.y(w, k)]), h_y

    @torch.no_grad()
    def targets(self, w, z_t, boot):
        out = self.critic_out(z_t)
        tg = {}
        for k in self.horizons:
            if k == 2:
                tg[k] = self.y(w, k)
                continue
            cand, _ = self.candidates(w, out, boot, k)
            wts = self.weights[k].to(w.dtype)  # (M, P)
            tg[k] = torch.einsum("mp,pmb->mb", wts, cand)
        return tg

    @torch.no_grad()
    def update_weights(self, w, n_windows):
        z_t = self.encode(w, 0)
        out = self.critic_out(z_t)
        boot = self.bootstraps(w)
        for k in self.horizons:
            if k == 2:
                continue
            cand, h_y = self.candidates(w, out, boot, k)
            resid = cand - h_y
            resid = resid - resid.mean(-1, keepdim=True)
            v = torch.einsum("pmb,qmb->mpq", resid, resid) / (w.shape[1] - 1)
            self.weights[k] = torch.tensor(
                np.stack([simplex_minimizer(q) for q in v.double().numpy()]),
                dtype=w.dtype,
            )


def make(variant, window):
    if variant == "DB*+F":
        return DBStarModels(window, SEEDS)
    return WorldModels(variant, window, HORIZONS, list(SEEDS), SIZES)


def train(variant, kappa, window):
    data = Data(kappa, window, "large", list(SEEDS))
    m = make(variant, window)
    n_models = len(SEEDS)
    opt = EnsembleAdam(m.model_parameters(), torch.full((n_models,), LR))
    crit = (
        EnsembleAdam(m.critic_parameters(), torch.full((n_models,), LR * CRITIC_FACTOR))
        if m.critic_parameters()
        else None
    )
    gen = torch.Generator().manual_seed(11)
    lam = torch.full((n_models,), LAM)
    n = data.train.shape[1]
    for step in range(1, STEPS + 1):
        if step == int(0.8 * STEPS):
            opt.lr = opt.lr * 0.1
            if crit is not None:
                crit.lr = crit.lr * 0.1
        idx = torch.randint(0, n, (n_models, BATCH), generator=gen)
        w = data.train[torch.arange(n_models)[:, None], idx]
        if variant == "DB*+F" and step > WARMUP and step % EVERY == 1:
            m.update_weights(w, n)
        loss, closs = m.losses(w, lam)
        params = m.model_parameters()
        g = torch.autograd.grad(
            loss.sum(), params, allow_unused=True, retain_graph=crit is not None
        )
        for p, gi in zip(params, g):
            p.grad = gi
        if crit is not None:
            cg = torch.autograd.grad(closs.sum(), crit.params, allow_unused=True)
            for p, gi in zip(crit.params, cg):
                p.grad = gi
            crit.step()
            crit.zero_grad()
        opt.step()
        opt.zero_grad()
    return m, data


@torch.no_grad()
def measure(variant, m, data):
    w = data.val[None].expand(len(SEEDS), -1, -1)
    us = m.rollout(m.encode(w, 0), 4)
    preds = torch.stack([m.decode(u) for u in us])
    ys = torch.stack([m.y(w, k) for k in range(1, 5)])
    out = {"mse_per_seed": (preds - ys).pow(2).mean((0, 2)).tolist()}
    if variant == "DB*+F":
        # R1, R2: variance of each target around h_y and the weights, at k = 3, 4
        z_t = m.encode(w, 0)
        crit = m.critic_out(z_t)
        boot = m.bootstraps(w)
        for k in (3, 4):
            cand, h_y = m.candidates(w, crit, boot, k)
            r = cand - h_y
            debiased_mean = r[:-1].mean(0)
            star = torch.einsum("mp,pmb->mb", m.weights[k], r)
            out[f"k{k}"] = {
                "var_C": float(r[-1].var(-1).mean()),
                "var_DB_equal_weights": float(debiased_mean.var(-1).mean()),
                "var_DBstar": float(star.var(-1).mean()),
                "weights_mean": m.weights[k].mean(0).tolist(),
                "weight_on_C_per_seed": m.weights[k][:, -1].tolist(),
            }
    return out


def main():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    path = OUT / "pilot.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for kappa, window in CONFIGS:
        for v in VARIANTS:
            key = f"k{kappa}_L{window}_{v}"
            if only not in key or key in results:
                continue
            m, data = train(v, kappa, window)
            results[key] = measure(v, m, data)
            r = results[key]
            line = f"{key:16s} MSE k1..4 per seed {np.round(r['mse_per_seed'], 4).tolist()} mean {np.mean(r['mse_per_seed']):.4f}"
            for k in (3, 4):
                if f"k{k}" in r:
                    x = r[f"k{k}"]
                    line += (
                        f"\n   k={k}: var C {x['var_C']:.4f}  DB {x['var_DB_equal_weights']:.4f}"
                        f"  DB* {x['var_DBstar']:.4f}  weights {np.round(x['weights_mean'], 3).tolist()}"
                    )
            print(line, flush=True)
            path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
