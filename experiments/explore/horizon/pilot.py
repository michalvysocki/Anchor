"""
Exploratory pilot, not pre-registered and not part of e03 (whose code it
imports but does not change): does the horizon reversal between observation
and latent targets hold up? See README.md in this directory.

In the first sphere pilot (3 seeds), observation targets (C) had a lower
rollout MSE than latent targets (CL) over the trained horizons k = 1..4 in
every configuration, while at κ = 1, L = 1 CL was about 20% lower over
k = 1..10. This pilot repeats the comparison with 10 fresh seeds and
measures every k up to 20.

Variants, as in the first sphere pilot (no flag, no sphere):
- C: one-step loss of e03 (observation + latent consistency at k = 1) plus
  lambda * observation targets y_{t+k}, k = 2..4
- CL: the same one-step loss plus lambda * latent consistency
  u_k ~ sg[phi(z_{t+k})], k = 2..4 (as TD-MPC2)

Measured on validation trajectories of e03 (windows extended to k = 20;
e03's test set is not touched), per seed and per k: rollout MSE, sharpness
(standard deviation of the predictions over that of y) and mean norm of the
predicted latent.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

E03 = Path(__file__).resolve().parents[2] / "e03_henon_latent"
sys.path.insert(0, str(E03))

from data import observations
from design import N_VAL_TRAJ, SIZES, VAL_TAG
from run import Data

from anchor.ensemble import EnsembleAdam
from anchor.latent import WorldModels

OUT = Path(
    os.environ.get("PILOT_OUT", str(Path(__file__).resolve().parent / "results"))
)
CONFIGS = ((1.0, 1), (0.8, 1))
VARIANTS = ("C", "CL")
SEEDS = tuple(range(300, 310))
HORIZONS = (2, 3, 4)
K_MAX = 20
STEPS = int(os.environ.get("PILOT_STEPS", "30000"))
BATCH, LR, LAM = 512, 1e-3, 1.0


class TargetModels(WorldModels):
    """e03's world model with observation (C) or latent (CL) multi-step targets."""

    def __init__(self, variant, window, seeds):
        self.kind = variant
        super().__init__("C", window, HORIZONS, list(seeds), SIZES)

    def losses(self, w, lam):
        z_t = self.encode(w, 0)
        us = self.rollout(z_t, max(HORIZONS))
        with torch.no_grad():
            z_next = [self.encode(w, k) for k in range(1, max(HORIZONS) + 1)]
        one = (self.decode(us[0]) - self.y(w, 1)).pow(2).mean(-1)
        one = one + (us[0] - z_next[0]).pow(2).mean((-2, -1))
        if self.kind == "C":
            terms = [
                (self.decode(us[k - 1]) - self.y(w, k)).pow(2).mean(-1)
                for k in HORIZONS
            ]
        else:
            terms = [
                (us[k - 1] - z_next[k - 1]).pow(2).mean((-2, -1)) for k in HORIZONS
            ]
        loss = one + lam * torch.stack(terms).mean(0)
        return loss, torch.zeros_like(loss)


def train(variant, kappa, window):
    data = Data(kappa, window, "large", list(SEEDS))
    m = TargetModels(variant, window, SEEDS)
    n_models = len(SEEDS)
    opt = EnsembleAdam(m.model_parameters(), torch.full((n_models,), LR))
    gen = torch.Generator().manual_seed(17)
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


def long_validation(kappa, past):
    """e03's validation trajectories as windows of past + K_MAX values."""
    _, y = observations(N_VAL_TRAJ, kappa, VAL_TAG)
    span = past + K_MAX
    idx = np.arange(y.shape[1] - span + 1)[:, None] + np.arange(span)
    return torch.tensor(y[:, idx].reshape(-1, span), dtype=torch.float32)


@torch.no_grad()
def measure(m, data):
    v = long_validation(data.kappa, data.past)
    w = v[None].expand(len(SEEDS), -1, -1)
    z = m.encode(w, 0)
    us = m.rollout(z, K_MAX)
    preds = torch.stack([m.decode(u) for u in us])  # (K, M, B)
    ys = torch.stack([w[..., data.past + k - 1] for k in range(1, K_MAX + 1)])
    return {
        "mse_k_seed": (preds - ys).pow(2).mean(-1).T.tolist(),  # (M, K)
        # reference: the MSE of predicting the mean of y at each k
        "var_y_k": ys[:, 0].var(-1).tolist(),
        "sharpness_k_seed": (preds.std(-1) / ys.std(-1)).T.tolist(),
        "latent_norm_k_seed": torch.stack(
            [u.norm(dim=-1).mean(-1) for u in us]
        ).T.tolist(),
        # H3: range of the encodings' norm per seed (5th, 50th, 95th percentile)
        "encoding_norm_q_seed": torch.quantile(
            z.norm(dim=-1), torch.tensor([0.05, 0.5, 0.95]), dim=-1
        ).T.tolist(),
        "n_windows": v.shape[0],
    }


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
            results[key] = measure(m, data)
            mse = np.array(results[key]["mse_k_seed"])
            print(
                f"{key:12s} MSE k1-4 {mse[:, :4].mean():.4f}  k5-10 {mse[:, 4:10].mean():.4f}"
                f"  k11-20 {mse[:, 10:].mean():.4f}"
                f"  per seed k5-10 {np.round(mse[:, 4:10].mean(1), 3).tolist()}",
                flush=True,
            )
            path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
