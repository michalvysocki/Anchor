"""
Exploratory pilot, not pre-registered and not part of e03 (whose code it
imports but does not change): what do observation targets (C) do to the
rollout latent beyond the trained horizon that latent targets (CL) do not?
See README.md in this directory.

With a window of L = 1 the encoder sees a single observation, so all
encodings lie on a curve E(y) in the 16-dimensional latent. The pilot
measures, per seed and per k up to 20, how far the rollout latent u_k is
from that curve, and whether the component off the curve carries
information (memory) or not.

Systems, all at κ = 1, L = 1, variants C and CL of the horizon pilot
(same training, same seeds):
- henon: the horizon pilot's configuration, retrained identically (its MSE
  must match the stored one)
- ar2: AR(2), radius 0.9, angle π/6 (as in e02), not Markov at L = 1
- ar1: AR(1), a = 0.99, Markov at L = 1 (the control)
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "horizon", HERE.parent / "horizon" / "pilot.py"
)
horizon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(horizon)

from design import LEN_TRAJ, N_TRAJ, N_VAL_TRAJ
from run import Data

from anchor.ensemble import EnsembleAdam
from anchor.envs.lgssm import (
    ar1,
    ar2,
    observation_autocovariance,
    optimal_mse,
    sample_observations,
)

OUT = Path(os.environ.get("PILOT_OUT", str(HERE / "results")))
SYSTEMS = (
    ("henon", None),
    ("ar2", ar2(0.9, np.pi / 6, 1.0)),
    ("ar1", ar1(0.99, 1.0)),
)
VARIANTS = horizon.VARIANTS
SEEDS = horizon.SEEDS
K_MAX = horizon.K_MAX
PAST = 1
N_GRID, CHUNK = 2001, 1000
# generator tag for the AR data: new, no draw shared with e02 or e03
AR_TAG = 50


def spans(series: np.ndarray, span: int) -> torch.Tensor:
    """All windows of `span` consecutive values, (n, span)."""
    idx = np.arange(series.shape[1] - span + 1)[:, None] + np.arange(span)
    return torch.tensor(series[:, idx].reshape(-1, span), dtype=torch.float32)


def ar_data(index: int, system):
    """Training windows per seed and long validation windows, in units of sqrt(Var(y))."""
    scale = np.sqrt(observation_autocovariance(system, 1)[0])

    def draw(n, *key):
        rng = np.random.default_rng([AR_TAG, index, *key])
        return sample_observations(system, n, LEN_TRAJ, rng) / scale

    train = torch.stack(
        [
            spans(draw(N_TRAJ["large"], 0, s), PAST + horizon.SIZES.eval_horizon)
            for s in SEEDS
        ]
    )
    return train, spans(draw(N_VAL_TRAJ, 1), PAST + K_MAX)


def train(variant, data_train):
    """The horizon pilot's training loop, on given training windows."""
    m = horizon.TargetModels(variant, PAST, SEEDS)
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


def nearest(u, curve):
    """Index of the nearest curve point and that point, for every latent in u (M, B, D)."""
    idx = torch.cat(
        [
            torch.cdist(u[:, i : i + CHUNK], curve).argmin(-1)
            for i in range(0, u.shape[1], CHUNK)
        ],
        1,
    )
    on = torch.gather(curve, 1, idx[..., None].expand(-1, -1, curve.shape[-1]))
    return idx, on


def r2(x, t, half):
    """Out-of-sample R^2 of a least-squares fit of t on x, per model: fit on the first half."""
    x = torch.cat([x, torch.ones_like(x[..., :1])], -1).double()
    t = t.double()
    beta = torch.linalg.lstsq(x[:, :half], t[:, :half, None]).solution
    res = t[:, half:] - (x[:, half:] @ beta)[..., 0]
    return 1 - res.var(-1) / t[:, half:].var(-1)


@torch.no_grad()
def measure(m, v):
    n_models = len(SEEDS)
    w = v[None].expand(n_models, -1, -1)
    z = m.encode(w, 0)
    us = m.rollout(z, K_MAX)
    preds = torch.stack([m.decode(u) for u in us])  # (K, M, B)
    ys = torch.stack([w[..., PAST + k - 1] for k in range(1, K_MAX + 1)])
    # the curve of encodings over the range of the observations
    grid = torch.linspace(float(v.min()), float(v.max()), N_GRID)
    curve = m.encode(grid[None, :, None].expand(n_models, -1, 1), 0)  # (M, G, D)
    spread = (z - z.mean(1, keepdim=True)).pow(2).sum(-1).mean(-1).sqrt()  # (M,)
    # windows are ordered by trajectory, so the halves split trajectories
    half = v.shape[0] // 2
    dist, dec_shift, gain_prev, gain_next = [], [], [], []
    for k, u in enumerate(us, start=1):
        idx, on = nearest(u, curve)
        r = u - on
        dist.append(r.norm(dim=-1).mean(-1) / spread)
        dec_shift.append((m.decode(u) - m.decode(on)).pow(2).mean(-1))
        # memory: does the off-curve component add to the position on the curve?
        y_star = grid[idx][..., None]
        a = torch.cat([y_star, y_star**2, y_star**3], -1)
        b = torch.cat([a, r], -1)
        prev = w[..., PAST + k - 2]
        gain_prev.append(r2(b, prev, half) - r2(a, prev, half))
        if k < K_MAX:
            nxt = w[..., PAST + k]
            gain_next.append(r2(b, nxt, half) - r2(a, nxt, half))
    _, on_z = nearest(z, curve)
    centred_p = preds - preds.mean(-1, keepdim=True)
    centred_y = ys - ys.mean(-1, keepdim=True)
    corr = (centred_p * centred_y).mean(-1) / (preds.std(-1) * ys.std(-1))

    def per_seed(rows):
        return torch.stack(rows).T.tolist()  # (M, K)

    return {
        "mse_k_seed": (preds - ys).pow(2).mean(-1).T.tolist(),
        "var_y_k": ys[:, 0].var(-1).tolist(),
        "corr_k_seed": corr.T.tolist(),
        "bias_k_seed": (preds.mean(-1) - ys.mean(-1)).T.tolist(),
        "sharpness_k_seed": (preds.std(-1) / ys.std(-1)).T.tolist(),
        # distance of u_k to the curve, in units of the spread of the encodings
        "curve_dist_k_seed": per_seed(dist),
        # the same for the encodings themselves (resolution of the grid)
        "encoding_dist_seed": ((z - on_z).norm(dim=-1).mean(-1) / spread).tolist(),
        # how much the decoder sees of the off-curve component
        "decoder_shift_k_seed": per_seed(dec_shift),
        # incremental out-of-sample R^2 of the off-curve component, for
        # y_{t+k-1} (the state's second coordinate) and y_{t+k+1}
        "memory_gain_prev_k_seed": per_seed(gain_prev),
        "memory_gain_next_k_seed": per_seed(gain_next),
        "n_windows": v.shape[0],
    }


def main():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    path = OUT / "pilot.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    stored = HERE.parent / "horizon" / "results" / "pilot.json"
    for index, (name, system) in enumerate(SYSTEMS):
        if system is None:
            data_train = Data(1.0, PAST, "large", list(SEEDS)).train
            v = horizon.long_validation(1.0, PAST)
        else:
            data_train, v = ar_data(index, system)
        for variant in VARIANTS:
            key = f"{name}_{variant}"
            if only not in key or key in results:
                continue
            m = train(variant, data_train)
            r = measure(m, v)
            if system is not None:
                r["floor_window_k"] = optimal_mse(system, K_MAX, history=PAST).tolist()
                r["floor_kalman_k"] = optimal_mse(system, K_MAX).tolist()
            results[key] = r
            mse = np.array(r["mse_k_seed"])
            d = np.median(np.array(r["curve_dist_k_seed"]), 0)
            g = np.median(np.array(r["memory_gain_prev_k_seed"]), 0)
            line = (
                f"{key:9s} MSE k1-4 {mse[:, :4].mean():.4f}  k5-10 {mse[:, 4:10].mean():.4f}"
                f"  k11-20 {mse[:, 10:].mean():.4f}"
                f"\n   curve distance k=1,4,7,10,20 {np.round(d[[0, 3, 6, 9, 19]], 3).tolist()}"
                f"  memory gain (prev) {np.round(g[[0, 3, 6, 9, 19]], 3).tolist()}"
            )
            if name == "henon" and stored.exists():
                old = json.loads(stored.read_text()).get(f"k1.0_L1_{variant}")
                if old is not None:
                    diff = np.abs(mse - np.array(old["mse_k_seed"])).max()
                    line += f"\n   max |MSE - horizon pilot| {diff:.2e}"
            print(line, flush=True)
            path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
