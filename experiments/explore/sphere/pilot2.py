"""
Exploratory pilot 2 (see README.md): separates the effect of the sphere from
a plain normalization of the latent scale, and tests a centred sphere. Only
CL (latent multi-step targets), since pilot 1 showed that the mechanism
needs latent targets. Same configurations, seeds, training and measurements
as pilot 1 (on the validation set).

Variants (penalties added to CL's loss, weight MU each):
- CL: none
- CL+N: (mean over the batch of ||phi||^2 - 1)^2, the scale alone: the mean
  squared norm is 1, single encodings are free
- CL+S: mean over the batch of (||phi||^2 - 1)^2, the sphere (pilot 1)
- CL+SC: the sphere plus ||mean over the batch of phi||^2, a centred sphere

Predictions, written before the first run:
Q1  if CL+N matches CL+S in rollout MSE (k = 1..4), the sphere's effect on
    the error in pilot 1 came from normalizing the scale, not from the
    belief mechanism
Q2  the deficit 1 - ||u_1||^2 matches the binned variance of the next
    encoding only for the sphere variants (CL+S, CL+SC), not for CL+N
Q3  centring makes the norm track the growing uncertainty along the rollout
    more faithfully: for k = 1..6 (window 1), the mean absolute difference
    between the deficit 1 - ||u_k||^2 and the binned variance of phi(z_{t+k})
    given y_t is smaller for CL+SC than for CL+S (redefined by a check before
    the first full run: a steeper fall of the norm alone is not evidence)
"""

import json
import sys

import numpy as np
import pilot
import torch

VARIANTS = ("CL", "CL+N", "CL+S", "CL+SC")
TRACK_K = 6


class Pilot2Models(pilot.SphereModels):
    def __init__(self, variant, window, seeds):
        self.penalty = variant.split("+")[1] if "+" in variant else ""
        super().__init__("CL", window, seeds)

    def losses(self, w, lam):
        loss, extra = super().losses(w, lam)  # CL, no sphere (variant "CL")
        z = self.encode(w, 0)
        sq = z.pow(2).sum(-1)  # (M, B)
        if self.penalty == "N":
            loss = loss + pilot.MU * (sq.mean(-1) - 1).pow(2)
        elif self.penalty in ("S", "SC"):
            loss = loss + pilot.MU * (sq - 1).pow(2).mean(-1)
        if self.penalty == "SC":
            loss = loss + pilot.MU * z.mean(1).pow(2).sum(-1)
        return loss, extra


def train(variant, kappa, window):
    data = pilot.Data(kappa, window, "large", list(pilot.SEEDS))
    m = Pilot2Models(variant, window, pilot.SEEDS)
    opt = pilot.EnsembleAdam(
        m.model_parameters(), torch.full((len(pilot.SEEDS),), pilot.LR)
    )
    gen = torch.Generator().manual_seed(7)
    lam = torch.full((len(pilot.SEEDS),), pilot.LAM)
    n = data.train.shape[1]
    for step in range(pilot.STEPS):
        if step == int(0.8 * pilot.STEPS):
            opt.lr = opt.lr * 0.1
        idx = torch.randint(0, n, (len(pilot.SEEDS), pilot.BATCH), generator=gen)
        w = data.train[torch.arange(len(pilot.SEEDS))[:, None], idx]
        loss, _ = m.losses(w, lam)
        loss.sum().backward()
        opt.step()
        opt.zero_grad()
    return m, data


@torch.no_grad()
def identity_by_horizon(m, data):
    """
    For k = 1..TRACK_K (window 1): the mean deficit 1 - ||u_k||^2 and the
    mean variance of phi(z_{t+k}) within quantile bins of y_t.
    """

    w = data.val[None].expand(len(pilot.SEEDS), -1, -1)
    y_t = w[0, :, 0].numpy()
    us = m.rollout(m.encode(w, 0), TRACK_K)
    edges = np.quantile(y_t, np.linspace(0, 1, pilot.N_BINS + 1))
    b = np.clip(np.searchsorted(edges, y_t, side="right") - 1, 0, pilot.N_BINS - 1)
    deficit, variance = [], []
    for k in range(1, TRACK_K + 1):
        z_k = m.encode(w, k)
        d = (1 - us[k - 1].norm(dim=-1).pow(2)).mean(0).numpy()
        dk, vk = [], []
        for i in range(pilot.N_BINS):
            sel = b == i
            vk.append(float(z_k[:, torch.as_tensor(sel)].var(1).sum(-1).mean()))
            dk.append(float(d[sel].mean()))
        deficit.append(float(np.mean(dk)))
        variance.append(float(np.mean(vk)))
    gap = float(np.mean(np.abs(np.array(deficit) - np.array(variance))))
    return {"deficit_k": deficit, "variance_k": variance, "mean_abs_difference": gap}


def main():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    pilot.OUT.mkdir(exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    path = pilot.OUT / "pilot2.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for kappa, window in pilot.CONFIGS:
        for v in VARIANTS:
            key = f"k{kappa}_L{window}_{v}"
            if only not in key or key in results:
                continue
            m, data = train(v, kappa, window)
            r = pilot.measure(m, data, window)
            with torch.no_grad():
                z = m.encode(data.val[None].expand(len(pilot.SEEDS), -1, -1), 0)
                r["encoding_mean_norm"] = float(z.mean(1).norm(dim=-1).mean())
            if window == 1:
                r["identity_by_horizon"] = identity_by_horizon(m, data)
            results[key] = r
            print(
                f"{key:16s} |phi| {r['encoding_norm'][0]:.3f}±{r['encoding_norm'][1]:.3f}"
                f" |mean phi| {r['encoding_mean_norm']:.3f}"
                f"  |u_k| {np.round(r['pred_norm_k'][:8], 3).tolist()}"
                f"  mse1..4 {np.mean(r['mse_k'][:4]):.4f}"
                + (
                    f"  P2 rho {r['p2_spearman']:+.2f} var {r['p2_mean_variance']:.3f}"
                    f" def {r['p2_mean_deficit']:.3f}"
                    if "p2_spearman" in r
                    else ""
                )
                + (
                    f"\n{'':16s} deficit_k {np.round(r['identity_by_horizon']['deficit_k'], 3).tolist()}"
                    f"  variance_k {np.round(r['identity_by_horizon']['variance_k'], 3).tolist()}"
                    f"  mean |diff| {r['identity_by_horizon']['mean_abs_difference']:.3f}"
                    if "identity_by_horizon" in r
                    else ""
                ),
                flush=True,
            )
            path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
