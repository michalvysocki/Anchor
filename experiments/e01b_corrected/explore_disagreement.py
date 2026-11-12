"""
Exploratory analysis after the e01b conclusion, not part of the
pre-registered protocol: does the path disagreement of the baseline models
separate the two observability modes, and does it track how much the
rollout loses to the direct prediction? See "Exploratory analysis" in
README.md.

All quantities are computed on the test windows for the baselines at
lambda = 0 on the evaluation seeds, at the auxiliary horizons k = 2..4,
averaged over the splits k = p + q, and divided by the variance of the true
point at horizon k so that the two modes are on the same scale:

- direct vs rollout: |F_direct(z_t, k) - g^k(z_t)|^2, two models
- bootstrap residual: |g^k(z_t) - g^q(z_{t+p})|^2, the D loss of the one-step model
- composition: |F(F(z_t, p), q) - F(z_t, k)|^2, the A loss of the direct model
- rollout gap: MSE of g^k(z_t) minus MSE of F_direct(z_t, k), the error the
  disagreement should predict
"""

import json

import numpy as np
import torch
from evaluate import load_model
from run import EVALUATION_SEEDS, MAX_HORIZON, MODES, RESULTS_DIR

from anchor.config import BURN_IN, K_MAX, TEST_DATA_SEED
from anchor.envs.henon import generate_windows
from anchor.metrics import mean_ci, to_lists
from anchor.models import UnitStep, predict
from anchor.training import LEN_TRAJ, N_TRAJ

HORIZONS = range(2, MAX_HORIZON + 1)
MEASURES = ("direct_vs_rollout", "bootstrap_residual", "composition", "rollout_gap")


def rollout(model: UnitStep, z: torch.Tensor, steps: int) -> torch.Tensor:
    """Applies the one-step model `steps` times."""
    for _ in range(steps):
        z = model(z)

    return z


def direct(model: torch.nn.Module, z: torch.Tensor, k: int) -> torch.Tensor:
    """Direct prediction F(z, k)."""
    horizons = torch.full((len(z),), k, dtype=torch.int64)

    return predict(model, z, horizons, K_MAX)


def squared(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared difference per coordinate."""
    return float(((a - b) ** 2).mean())


@torch.no_grad()
def seed_measures(mode: str, seed: int, windows: torch.Tensor) -> dict:
    """Normalized measures of one seed, each of size (len(HORIZONS),)."""
    onestep = UnitStep(load_model("onestep", mode, seed, "rollout"), K_MAX)
    direct_model = load_model("direct", mode, seed, "direct")
    z = windows[:, 0]
    out = {m: [] for m in MEASURES}

    for k in HORIZONS:
        truth = windows[:, k]
        variance = float(truth.var(dim=0).mean())
        rolled = rollout(onestep, z, k)
        straight = direct(direct_model, z, k)

        bootstrap, composition = [], []
        for p in range(1, k):
            q = k - p
            bootstrap.append(squared(rolled, rollout(onestep, windows[:, p], q)))
            composition.append(
                squared(direct(direct_model, direct(direct_model, z, p), q), straight)
            )

        out["direct_vs_rollout"].append(squared(straight, rolled) / variance)
        out["bootstrap_residual"].append(np.mean(bootstrap) / variance)
        out["composition"].append(np.mean(composition) / variance)
        out["rollout_gap"].append(
            (squared(rolled, truth) - squared(straight, truth)) / variance
        )

    return {m: np.array(v) for m, v in out.items()}


def main():
    results = {}
    for mode in MODES:
        windows = generate_windows(
            N_TRAJ,
            LEN_TRAJ,
            BURN_IN,
            K_MAX,
            mode,
            np.random.default_rng(TEST_DATA_SEED),
        )
        per_seed = [seed_measures(mode, seed, windows) for seed in EVALUATION_SEEDS]
        results[mode] = {m: np.stack([s[m] for s in per_seed]) for m in MEASURES}

    print("normalized by the variance of the true point at horizon k;")
    print("mean ± 95% t-interval over evaluation seeds\n")
    for m in MEASURES:
        print(m)
        for mode in MODES:
            mean, half = mean_ci(results[mode][m])
            cells = "  ".join(
                f"k={k}: {a:.2e} ± {h:.1e}" for k, a, h in zip(HORIZONS, mean, half)
            )
            print(f"  {mode:<12} {cells}")
        ratio = results["insufficient"][m].mean(axis=0) / results["sufficient"][m].mean(
            axis=0
        )
        print(
            "  insufficient / sufficient: "
            + "  ".join(f"k={k}: {r:.1e}" for k, r in zip(HORIZONS, ratio))
        )
        print()

    with open(RESULTS_DIR / "explore_disagreement.json", "w") as f:
        json.dump(to_lists(results), f, indent=2)


if __name__ == "__main__":
    main()
