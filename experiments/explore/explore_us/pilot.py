"""
Exploratory pilot, not pre-registered and not part of e03 (whose code it
imports but does not change): a latent split into u (compared with
encodings) and s (free memory of the rollout). See README.md.

The last S_DIMS of the 16 latent dimensions are reserved for s: encodings
have s = 0, the decoder reads u only, and the latent consistency compares u
only, so s is trained only by helping the dynamics predict later u. No new
parameters and no change of shapes: a mask in the encoder output, the
decoder input and the consistency loss.

Variants, all with latent multi-step targets (CL: u_k ~ sg[phi_u(z_{t+k})],
k = 1..4, plus the observation term at k = 1):
- CL: no flag, no s
- CL+F: the one-hot flag of e03 (encoded / predicted input of the dynamics)
- CL+s: the masked split, no flag
- CL12: a plain latent of 12 dimensions, no flag and no s: the capacity
  control for CL+s (added by a check before the run)

Predictions, written before the first run:
S1  in the gap configurations CL+s has a rollout MSE (k = 1..4) at most that
    of CL12 (same capacity of u) and close to CL+F's or below
S2  s is used: its norm grows with k, and across quantile bins of y_t it
    correlates positively with the variance of the u-encodings k steps ahead
S3  no prediction for CL+s against CL+F beyond S1
"""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

E03 = Path(__file__).resolve().parents[1] / "e03_henon_latent"
sys.path.insert(0, str(E03))

from design import SIZES
from run import Data

from anchor.ensemble import EnsembleAdam
from anchor.latent import WorldModels

OUT = Path(
    os.environ.get("PILOT_OUT", str(Path(__file__).resolve().parent / "results"))
)
CONFIGS = ((0.8, 1), (0.5, 1), (1.0, 1))
VARIANTS = ("CL", "CL+F", "CL+s", "CL12")
SEEDS = (200, 201, 202, 203, 204)
HORIZONS = (2, 3, 4)
S_DIMS = 4
STEPS = int(os.environ.get("PILOT_STEPS", "30000"))
BATCH, LR, LAM, N_BINS = 512, 1e-3, 1.0, 20


class SplitModels(WorldModels):
    """e03's world model trained with latent targets, optionally with the flag or the u/s split."""

    def __init__(self, variant, window, seeds):
        self.split = variant == "CL+s"
        # CL12: a plain latent of 16 - S_DIMS dimensions, the capacity control
        # for CL+s, whose u has that many dimensions
        sizes = (
            replace(SIZES, latent=SIZES.latent - S_DIMS) if variant == "CL12" else SIZES
        )
        target = "C+F" if variant == "CL+F" else "C"
        super().__init__(target, window, HORIZONS, list(seeds), sizes)
        self.d_u = sizes.latent - (S_DIMS if self.split else 0)

    def encode(self, w, start):
        z = super().encode(w, start)
        if self.split:
            z = torch.cat(
                [z[..., : self.d_u], torch.zeros_like(z[..., self.d_u :])], -1
            )
        return z

    def decode(self, u):
        if self.split:
            u = torch.cat(
                [u[..., : self.d_u], torch.zeros_like(u[..., self.d_u :])], -1
            )
        return super().decode(u)

    def u_part(self, z):
        return z[..., : self.d_u]

    def losses(self, w, lam):
        z_t = self.encode(w, 0)
        us = self.rollout(z_t, max(HORIZONS))
        with torch.no_grad():
            z_next = [self.encode(w, k) for k in range(1, max(HORIZONS) + 1)]

        def consistency(k):
            return (
                (self.u_part(us[k - 1]) - self.u_part(z_next[k - 1]))
                .pow(2)
                .mean((-2, -1))
            )

        one = (self.decode(us[0]) - self.y(w, 1)).pow(2).mean(-1) + consistency(1)
        aux = torch.stack([consistency(k) for k in HORIZONS]).mean(0)
        loss = one + lam * aux
        return loss, torch.zeros_like(loss)


def train(variant, kappa, window):
    data = Data(kappa, window, "large", list(SEEDS))
    m = SplitModels(variant, window, SEEDS)
    n_models = len(SEEDS)
    opt = EnsembleAdam(m.model_parameters(), torch.full((n_models,), LR))
    gen = torch.Generator().manual_seed(13)
    lam = torch.full((n_models,), LAM)
    n = data.train.shape[1]
    for step in range(1, STEPS + 1):
        if step == int(0.8 * STEPS):
            opt.lr = opt.lr * 0.1
        idx = torch.randint(0, n, (n_models, BATCH), generator=gen)
        w = data.train[torch.arange(n_models)[:, None], idx]
        loss, _ = m.losses(w, lam)
        loss.sum().backward()
        opt.step()
        opt.zero_grad()
    return m, data


@torch.no_grad()
def measure(variant, m, data):
    w = data.val[None].expand(len(SEEDS), -1, -1)
    us = m.rollout(m.encode(w, 0), 10)
    preds = torch.stack([m.decode(u) for u in us])
    ys = torch.stack([m.y(w, k) for k in range(1, 11)])
    mse = (preds - ys).pow(2).mean(2)  # (K, M)
    out = {
        "mse_1_4_per_seed": mse[:4].mean(0).tolist(),
        "mse_1_10_per_seed": mse.mean(0).tolist(),
    }
    if variant == "CL+s":
        # S2: norm of s along the rollout, and its relation to the variance
        # of the next u-encodings across quantile bins of y_t
        y_t = w[0, :, 0].numpy()
        edges = np.quantile(y_t, np.linspace(0, 1, N_BINS + 1))
        b = np.clip(np.searchsorted(edges, y_t, side="right") - 1, 0, N_BINS - 1)
        s_norm_k, rho_k = [], []
        for k in range(1, 7):
            s = us[k - 1][..., m.d_u :].norm(dim=-1).mean(0).numpy()  # (B,)
            z_k = m.u_part(m.encode(w, k))
            var_bins, s_bins = [], []
            for i in range(N_BINS):
                sel = b == i
                var_bins.append(
                    float(z_k[:, torch.as_tensor(sel)].var(1).sum(-1).mean())
                )
                s_bins.append(float(s[sel].mean()))
            rv, rs = np.argsort(np.argsort(var_bins)), np.argsort(np.argsort(s_bins))
            s_norm_k.append(float(s.mean()))
            rho_k.append(float(np.corrcoef(rv, rs)[0, 1]))
        out["s_norm_k"] = s_norm_k
        out["s_variance_spearman_k"] = rho_k
    return out


def main():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    OUT.mkdir(parents=True, exist_ok=True)
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
            line = (
                f"{key:14s} MSE k1..4 per seed {np.round(r['mse_1_4_per_seed'], 4).tolist()}"
                f" mean {np.mean(r['mse_1_4_per_seed']):.4f}"
            )
            if "s_norm_k" in r:
                line += (
                    f"\n   |s_k| {np.round(r['s_norm_k'], 3).tolist()}"
                    f"  rho(|s|, variance)_k {np.round(r['s_variance_spearman_k'], 2).tolist()}"
                )
            print(line, flush=True)
            path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
