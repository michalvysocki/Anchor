"""
Exploratory pilot, not pre-registered and not part of e03 (whose code it
imports but does not change): does the edge of the information horizon move
with the trained horizon K, and is what breaks beyond K calibration (the
amplitude of the predictions outliving their information)? See README.md in
this directory.

Calibration gap: sharpness minus correlation, per seed and k. Both are equal
for the prediction that minimizes MSE given any information.

Arms, κ = 1, L = 1, seeds and training of the horizon pilot, model class of
c_vs_cl:
- henon_C6, henon_CL6: horizons 2..6 (K = 4 is the off-curve pilot, K = 8
  c_vs_cl)
- ar2_C8, ar2_CL8: horizons 2..8 on the off-curve pilot's AR(2) data (K = 4
  is the off-curve pilot)
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
c_vs_cl = _load("c_vs_cl", "c_vs_cl")

from run import Data

from anchor.envs.lgssm import observation_autocovariance, optimal_mse

OUT = Path(os.environ.get("PILOT_OUT", str(HERE / "results")))
SEEDS = horizon.SEEDS
PAST = 1
# index of AR(2) in the off-curve pilot's systems: its data generator key
AR2_INDEX, AR2 = 1, offcurve.SYSTEMS[1][1]
ARMS = {
    "henon_C6": ("henon", "C", 6),
    "henon_CL6": ("henon", "CL", 6),
    "ar2_C8": ("ar2", "C", 8),
    "ar2_CL8": ("ar2", "CL", 8),
}
# the same quantities for the arms already run: (file, key, K)
EXISTING = {
    "henon_C4": ("offcurve", "henon_C", 4),
    "henon_CL4": ("offcurve", "henon_CL", 4),
    "henon_C8": ("c_vs_cl", "C8", 8),
    "henon_CL8": ("c_vs_cl", "CL8", 8),
    "ar2_C4": ("offcurve", "ar2_C", 4),
    "ar2_CL4": ("offcurve", "ar2_CL", 4),
}
INSIDE_MAX, OUTSIDE_MIN = 0.1, 0.3


def empirical_floor(v):
    """MSE of the best prediction from y_t, rho_k * y_t, on the same windows, per k."""
    g = observation_autocovariance(AR2, offcurve.K_MAX + 1)
    rho = torch.tensor(g[1:] / g[0], dtype=v.dtype)
    preds = rho[:, None] * v[None, :, PAST - 1]
    ys = v[:, PAST : PAST + offcurve.K_MAX].T
    return (preds - ys).pow(2).mean(-1).tolist()


def gaps(r, K):
    """Inside gap (median |gap| over k = 1..K) and outside gap (median gap over K+2..K+6), per seed."""
    gap = np.array(r["sharpness_k_seed"]) - np.array(r["corr_k_seed"])
    return np.median(np.abs(gap[:, :K]), 1), np.median(gap[:, K + 1 : K + 6], 1)


def report(name, r, K):
    mse = np.array(r["mse_k_seed"])
    inside, outside = gaps(r, K)
    line = (
        f"{name:10s} K={K}  MSE k1-4 {np.median(mse[:, :4].mean(1)):.3f}"
        f"  k{K + 1}-{K + 6} {np.median(mse[:, K : K + 6].mean(1)):.3f}"
        f"\n   gap medians: inside {np.median(inside):.2f}, outside {np.median(outside):.2f}"
    )
    if "_CL" in name:
        n = int((np.abs(outside) < OUTSIDE_MIN).sum())
        line += f"\n   P2 (CL): |outside gap| < {OUTSIDE_MIN} in {n}/10 seeds"
    else:
        n = int(((inside <= INSIDE_MAX) & (outside >= OUTSIDE_MIN)).sum())
        line += (
            f"\n   P2 (C): inside gap <= {INSIDE_MAX} and outside gap >= {OUTSIDE_MIN}"
            f" in the same seed: {n}/10"
        )
    if name.startswith("henon"):
        line += f"\n   k* {c_vs_cl.k_star(r['corr_k_seed'])}"
    if "floor_empirical_k" in r:
        excess = mse - np.array(r["floor_empirical_k"])
        line += (
            f"\n   MSE - floor: k1-{K} {np.median(excess[:, :K].mean(1)):.3f}"
            f"  k9-14 {np.median(excess[:, 8:14].mean(1)):.3f}"
        )
    print(line, flush=True)


def main():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    path = OUT / "pilot.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    ar2_floor = empirical_floor(offcurve.ar_data(AR2_INDEX, AR2)[1])
    for name, (folder, key, K) in EXISTING.items():
        stored = HERE.parent / folder / "results" / "pilot.json"
        if stored.exists():
            r = json.loads(stored.read_text())[key]
            if name.startswith("ar2"):
                r["floor_empirical_k"] = ar2_floor
            report(name + " (ref)", r, K)
    data = {}
    for arm, (system, kind, K) in ARMS.items():
        if only not in arm or arm in results:
            continue
        if system not in data:
            if system == "henon":
                data[system] = (
                    Data(1.0, PAST, "large", list(SEEDS)).train,
                    horizon.long_validation(1.0, PAST),
                )
            else:
                data[system] = offcurve.ar_data(AR2_INDEX, AR2)
        data_train, v = data[system]
        m = c_vs_cl.ArmModels(kind, tuple(range(2, K + 1)), False, SEEDS)
        r = offcurve.measure(c_vs_cl.train(m, data_train), v)
        r["k_star_seed"] = c_vs_cl.k_star(r["corr_k_seed"])
        if system == "ar2":
            r["floor_window_k"] = optimal_mse(
                AR2, offcurve.K_MAX, history=PAST
            ).tolist()
            r["floor_kalman_k"] = optimal_mse(AR2, offcurve.K_MAX).tolist()
            r["floor_empirical_k"] = empirical_floor(v)
        results[arm] = r
        report(arm, r, K)
        path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
