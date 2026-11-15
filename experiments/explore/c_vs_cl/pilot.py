"""
Exploratory pilot, not pre-registered and not part of e03 (whose code it
imports but does not change): which property of latent targets (CL) keeps
information in the rollout beyond the trained horizon, where observation
targets (C) lose it? See README.md in this directory.

Information horizon k*: the first k at which the correlation of the
predictions with y_{t+k} falls below 0.3. In the off-curve pilot (Hénon,
κ = 1, L = 1, horizons 2..4) it was 6 for C in 10/10 seeds and 8-9 for CL.

Arms, Hénon at κ = 1, L = 1, seeds and training of the horizon pilot:
- C8: C with horizons 2..8 (specialization to the trained horizon)
- CL8: CL with horizons 2..8 (reference for C8)
- C_sgdec: C with horizons 2..4 whose terms at k >= 2 do not train the
  decoder, which then learns from k = 1 only, as in CL (co-adaptation of
  the decoder with the rollout)
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name, folder):
    spec = importlib.util.spec_from_file_location(
        name, HERE.parent / folder / "pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


horizon = _load("horizon", "horizon")
offcurve = _load("offcurve", "offcurve")

from run import Data

from anchor.ensemble import EnsembleAdam
from anchor.latent import WorldModels

OUT = Path(os.environ.get("PILOT_OUT", str(HERE / "results")))
SEEDS = horizon.SEEDS
PAST = 1
LONG = tuple(range(2, 9))
ARMS = {
    "C8": ("C", LONG, False),
    "CL8": ("CL", LONG, False),
    "C_sgdec": ("C", horizon.HORIZONS, True),
}
CORR_LOST = 0.3


class ArmModels(WorldModels):
    """The horizon pilot's C and CL with given horizons, optionally with a decoder trained at k = 1 only."""

    def __init__(self, kind, horizons, frozen_decoder, seeds):
        self.kind, self.frozen_decoder = kind, frozen_decoder
        super().__init__("C", PAST, horizons, list(seeds), horizon.SIZES)

    def decode_frozen(self, u):
        """The decoder with its parameters cut from the graph: the gradient reaches u only."""
        layers = self.dec.layers
        for i, layer in enumerate(layers):
            u = torch.einsum("mbi,mio->mbo", u, layer.weight.detach())
            u = u + layer.bias.detach()[:, None]
            if i < len(layers) - 1:
                u = torch.tanh(u)
        return u[..., 0]

    def losses(self, w, lam):
        z_t = self.encode(w, 0)
        us = self.rollout(z_t, max(self.horizons))
        with torch.no_grad():
            z_next = [self.encode(w, k) for k in range(1, max(self.horizons) + 1)]
        one = (self.decode(us[0]) - self.y(w, 1)).pow(2).mean(-1)
        one = one + (us[0] - z_next[0]).pow(2).mean((-2, -1))
        if self.kind == "C":
            dec = self.decode_frozen if self.frozen_decoder else self.decode
            terms = [
                (dec(us[k - 1]) - self.y(w, k)).pow(2).mean(-1) for k in self.horizons
            ]
        else:
            terms = [
                (us[k - 1] - z_next[k - 1]).pow(2).mean((-2, -1)) for k in self.horizons
            ]
        loss = one + lam * torch.stack(terms).mean(0)
        return loss, torch.zeros_like(loss)


def train(m, data_train):
    """The horizon pilot's training loop, for a given model."""
    n_models = len(SEEDS)
    opt = EnsembleAdam(m.model_parameters(), torch.full((n_models,), horizon.LR))
    gen = torch.Generator().manual_seed(17)
    lam = torch.full((n_models,), horizon.LAM)
    n = data_train.shape[1]
    for step in range(1, horizon.STEPS + 1):
        if step == int(0.8 * horizon.STEPS):
            opt.lr = opt.lr * 0.1
        idx = torch.randint(0, n, (n_models, horizon.BATCH), generator=gen)
        w = data_train[torch.arange(n_models)[:, None], idx]
        loss, _ = m.losses(w, lam)
        loss.sum().backward()
        opt.step()
        opt.zero_grad()
    return m


def k_star(corr):
    """First k with correlation below CORR_LOST, per seed; K_MAX + 1 if it never falls."""
    corr = np.asarray(corr)
    lost = corr < CORR_LOST
    return np.where(lost.any(1), lost.argmax(1) + 1, corr.shape[1] + 1).tolist()


def main():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    path = OUT / "pilot.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    data_train = Data(1.0, PAST, "large", list(SEEDS)).train
    v = horizon.long_validation(1.0, PAST)
    reference = HERE.parent / "offcurve" / "results" / "pilot.json"
    if reference.exists():
        ref = json.loads(reference.read_text())
        for key in ("henon_C", "henon_CL"):
            print(f"{key:9s} (off-curve pilot) k* {k_star(ref[key]['corr_k_seed'])}")
    for arm, (kind, horizons, frozen) in ARMS.items():
        if only not in arm or arm in results:
            continue
        m = train(ArmModels(kind, horizons, frozen, SEEDS), data_train)
        r = offcurve.measure(m, v)
        r["k_star_seed"] = k_star(r["corr_k_seed"])
        results[arm] = r
        mse = np.array(r["mse_k_seed"])
        corr = np.median(np.array(r["corr_k_seed"]), 0)
        print(
            f"{arm:9s} MSE k1-4 {mse[:, :4].mean():.4f}  k5-8 {mse[:, 4:8].mean():.4f}"
            f"  k9-14 {mse[:, 8:14].mean():.4f}  k* {r['k_star_seed']}"
            f"\n   correlation k=1..12 {np.round(corr[:12], 2).tolist()}",
            flush=True,
        )
        path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
