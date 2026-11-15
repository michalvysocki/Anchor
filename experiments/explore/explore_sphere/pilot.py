"""
Exploratory pilot, not pre-registered and not part of e03 (whose code it
imports but does not change): spherical encodings as a source of an implicit
flag and of horizon-dependent blur. See README.md in this directory.

A loss mu * (||phi(z_t)||^2 - 1)^2 keeps encodings near the unit sphere;
predictions are left free. With latent consistency, a predicted latent
approaches the mean of the future encodings, and for unit vectors
||E[phi]||^2 = 1 - tr Cov(phi), so its norm carries the uncertainty.

Variants, each without and with the sphere (suffix "+S"):
- onestep: the e03 one-step loss (observation + latent consistency at k = 1)
- C: plus lambda * observation targets y_{t+k}, k = 2..4 (e03's C)
- CL: plus lambda * latent consistency u_k ~ sg[phi(z_{t+k})], k = 2..4 (as TD-MPC2)

Predictions, written before the first run:
P1  for CL+S, the mean ||u_k|| falls with k in the gap configurations and stays
    near 1 for the first steps in the Markov control
P2  for the "+S" variants, 1 - ||u_1||^2 tracks the variance of the next
    encoding given z_t (binned on y_t, window 1): rank correlation over bins
    positive and, for CL+S, the two of similar size
P3  the sphere does not raise the rollout MSE over k = 1..4, and brings the
    sharpness of the rollout closer to that of the direct reference (e03's
    gap-stage direct model, seed 0)
All measurements are on the validation set; e03's test set is not touched.
Check: encodings of the "+S" variants have norm close to 1.
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
from anchor.latent import DirectModels, WorldModels

OUT = Path(os.environ.get("PILOT_OUT", Path(__file__).resolve().parent / "results"))
CONFIGS = ((0.8, 1), (1.0, 1), (1.0, 2))
VARIANTS = ("onestep", "C", "CL", "onestep+S", "C+S", "CL+S")
SEEDS = (0, 1, 2)
HORIZONS = (2, 3, 4)
STEPS = int(os.environ.get("PILOT_STEPS", "30000"))
BATCH, LR, LAM, MU = 512, 1e-3, 1.0, 1.0
N_BINS = 20


class SphereModels(WorldModels):
    """e03's world model with optional spherical encodings and latent multi-step targets."""

    def __init__(self, variant, window, seeds):
        self.kind = variant.split("+")[0]
        self.sphere = variant.endswith("+S")
        target = "onestep" if self.kind == "onestep" else "C"
        super().__init__(target, window, HORIZONS, list(seeds), SIZES)

    def losses(self, w, lam):
        z_t = self.encode(w, 0)
        steps = 1 if self.kind == "onestep" else max(HORIZONS)
        us = self.rollout(z_t, steps)
        with torch.no_grad():
            z_next = [self.encode(w, k) for k in range(1, steps + 1)]
        one = (self.decode(us[0]) - self.y(w, 1)).pow(2).mean(-1)
        one = one + (us[0] - z_next[0]).pow(2).mean((-2, -1))
        aux = torch.zeros_like(one)
        if self.kind == "C":
            aux = torch.stack(
                [
                    (self.decode(us[k - 1]) - self.y(w, k)).pow(2).mean(-1)
                    for k in HORIZONS
                ]
            ).mean(0)
        elif self.kind == "CL":
            aux = torch.stack(
                [(us[k - 1] - z_next[k - 1]).pow(2).mean((-2, -1)) for k in HORIZONS]
            ).mean(0)
        loss = one + lam * aux
        if self.sphere:
            loss = loss + MU * (z_t.pow(2).sum(-1) - 1).pow(2).mean(-1)
        return loss, torch.zeros_like(loss)


def train(variant, kappa, window):
    data = Data(kappa, window, "large", list(SEEDS))
    m = SphereModels(variant, window, SEEDS)
    opt = EnsembleAdam(m.model_parameters(), torch.full((len(SEEDS),), LR))
    gen = torch.Generator().manual_seed(7)
    lam = torch.full((len(SEEDS),), LAM)
    n = data.train.shape[1]
    for step in range(STEPS):
        if step == int(0.8 * STEPS):
            opt.lr = opt.lr * 0.1
        idx = torch.randint(0, n, (len(SEEDS), BATCH), generator=gen)
        w = data.train[torch.arange(len(SEEDS))[:, None], idx]
        loss, _ = m.losses(w, lam)
        loss.sum().backward()
        opt.step()
        opt.zero_grad()
    return m, data


@torch.no_grad()
def measure(m, data, window):
    # validation, not test: e03's test set is reserved for its final stage
    w = data.val[None].expand(len(SEEDS), -1, -1)
    z = m.encode(w, 0)
    us = m.rollout(z, 10)
    preds = torch.stack([m.decode(u) for u in us])  # (K, M, B)
    ys = torch.stack([m.y(w, k) for k in range(1, 11)])
    out = {
        "encoding_norm": [float(z.norm(dim=-1).mean()), float(z.norm(dim=-1).std())],
        "pred_norm_k": [float(u.norm(dim=-1).mean()) for u in us],
        "mse_k": (preds - ys).pow(2).mean((1, 2)).tolist(),
        "sharpness_k": (preds.std(-1).mean(-1) / ys.std(-1).mean(-1)).tolist(),
    }
    if window == 1:
        # P2: variance of the next encoding given y_t, by quantile bins of y_t
        y_t = w[0, :, 0].numpy()
        z_next = m.encode(w, 1)
        deficit = (1 - us[0].norm(dim=-1).pow(2)).mean(0).numpy()
        edges = np.quantile(y_t, np.linspace(0, 1, N_BINS + 1))
        b = np.clip(np.searchsorted(edges, y_t, side="right") - 1, 0, N_BINS - 1)
        var_bins, def_bins = [], []
        for i in range(N_BINS):
            sel = b == i
            zn = z_next[:, torch.as_tensor(sel)]  # (M, n_i, d)
            var_bins.append(float(zn.var(1).sum(-1).mean()))
            def_bins.append(float(deficit[sel].mean()))
        rv, rd = np.argsort(np.argsort(var_bins)), np.argsort(np.argsort(def_bins))
        out["p2_spearman"] = float(np.corrcoef(rv, rd)[0, 1])
        out["p2_mean_variance"] = float(np.mean(var_bins))
        out["p2_mean_deficit"] = float(np.mean(def_bins))
    return out


@torch.no_grad()
def direct_sharpness(kappa, window, data):
    """Sharpness at every k of e03's direct reference (gap stage, seed 0), on validation."""
    path = E03 / "results" / "runs" / "gap" / f"k{kappa}_L{window}_direct_main.pt"
    m = DirectModels(window, [0], SIZES)
    m.load_state_dict(torch.load(path)["state"])
    w = data.val[None]
    preds = m.predict_all(w)  # (K, 1, B)
    ys = torch.stack([w[..., m.past + k] for k in range(10)])
    return (preds.std(-1)[:, 0] / ys.std(-1)[:, 0]).tolist()


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    results = {}
    path = OUT / "pilot.json"
    if path.exists():
        results = json.loads(path.read_text())
    for kappa, window in CONFIGS:
        for v in VARIANTS:
            key = f"k{kappa}_L{window}_{v}"
            if only not in key or key in results:
                continue
            m, data = train(v, kappa, window)
            results[key] = measure(m, data, window)
            ref = f"k{kappa}_L{window}_direct_sharpness"
            if ref not in results:
                results[ref] = direct_sharpness(kappa, window, data)
            r = results[key]
            print(
                f"{key:18s} |phi| {r['encoding_norm'][0]:.3f}±{r['encoding_norm'][1]:.3f}"
                f"  |u_k| {np.round(r['pred_norm_k'][:6], 3).tolist()}"
                f"  mse1..4 {np.mean(r['mse_k'][:4]):.4f}"
                + (
                    f"  P2 rho {r['p2_spearman']:+.2f} var {r['p2_mean_variance']:.3f} def {r['p2_mean_deficit']:.3f}"
                    if "p2_spearman" in r
                    else ""
                ),
                flush=True,
            )
            path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
