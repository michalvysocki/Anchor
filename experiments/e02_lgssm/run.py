"""
Training of e02, see README.md. All lambdas and seeds of one configuration,
variant and data arm are trained together as a batch of independent linear
models; each seed has its own training set, shared by every variant and
lambda of that seed so that comparisons are paired.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from design import (
    ABLATIONS,
    EVALUATION_SEEDS,
    HORIZON,
    LAMBDAS,
    LEN_TRAJ,
    N_TRAJ,
    SELECTION_SEEDS,
    SYSTEMS,
    VARIANTS,
    WINDOWS,
    n_windows,
)

from anchor import linear
from anchor.envs.lgssm import (
    observation_autocovariance,
    observation_covariance,
    sample_observations,
    windows,
)
from anchor.tracking import RunLogger

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "results" / "runs"
SEEDS = SELECTION_SEEDS + EVALUATION_SEEDS
LEARNING_RATE = 1e-2
N_STEPS = 3000
BATCH_WINDOWS = 256
WEIGHT_EVERY = 100
N_VAL_TRAJ = 200
# small-data arm (README: Setup)
SMALL_CONFIGS = [
    (s, w)
    for s in SYSTEMS
    if s.name.startswith(("ar1_a0.9_", "ar1_a0.99_", "ar2")) and s.kappa in (1.0, 0.5)
    for w in (1, 2)
]
# data generators are tagged so that training, validation and seeds never share draws
TRAIN_TAG, VAL_TAG = 0, 1


def scaled_windows(system, n_traj: int, window: int, rng) -> np.ndarray:
    """Windows of window + HORIZON observations in units of sqrt(Var(y))."""
    obs = sample_observations(system, n_traj, LEN_TRAJ, rng)
    scale = np.sqrt(observation_autocovariance(system, 1)[0])

    return windows(obs / scale, window, HORIZON)


def train_set(system_idx: int, system, arm: str, seed: int, window: int) -> np.ndarray:
    rng = np.random.default_rng([TRAIN_TAG, system_idx, list(N_TRAJ).index(arm), seed])
    return scaled_windows(system, N_TRAJ[arm], window, rng)


def val_set(system_idx: int, system, window: int) -> np.ndarray:
    rng = np.random.default_rng([VAL_TAG, system_idx])
    return scaled_windows(system, N_VAL_TRAJ, window, rng)


def second_moment(x: np.ndarray) -> np.ndarray:
    """Empirical E[Y Y^T] of the windows: every loss and regression is a quadratic form in it."""
    # see sample_observations: spurious Accelerate warnings on finite inputs
    with np.errstate(all="ignore"):
        out = x.T @ x / len(x)
    assert np.isfinite(out).all()

    return out


def batched_rows(theta: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """rows[k] of size (B, L): rows[k] . z_t is the rollout prediction of y_{t+k}."""
    n_models, window = theta.shape
    shift = torch.zeros(window, window, dtype=theta.dtype)
    shift[:-1, 1:] = torch.eye(window - 1, dtype=theta.dtype)
    m = torch.cat([shift[:-1].expand(n_models, -1, -1), theta[:, None]], dim=1)
    row = torch.zeros(n_models, window, dtype=theta.dtype)
    row[:, -1] = 1.0
    rows = [row]
    for _ in range(steps):
        rows.append(torch.einsum("bi,bij->bj", rows[-1], m))

    return rows


def predict(rows: list[torch.Tensor], x: torch.Tensor, k: int, start: int = 0):
    """Rollout of k steps from the window starting at position start, per model."""
    window = rows[0].shape[1]
    return (x[:, :, start : start + window] * rows[k][:, None]).sum(-1)


def aux_loss(
    variant: str, theta: torch.Tensor, x: torch.Tensor, weights: list[dict] | None
) -> torch.Tensor:
    """Per-model auxiliary loss on a minibatch x of size (B, n, L + HORIZON)."""
    window = theta.shape[1]
    rows = batched_rows(theta, max(linear.AUX_HORIZONS))
    # the ablation D_nosg lets the gradient flow through the targets
    target_rows = (
        rows
        if variant == "D_nosg"
        else batched_rows(theta.detach(), max(linear.AUX_HORIZONS))
    )
    if variant == "C":
        terms = [
            ((predict(rows, x, k) - x[:, :, window + k - 1]) ** 2).mean(1)
            for k in linear.AUX_HORIZONS
        ]
        return torch.stack(terms).mean(0)
    if variant in ("D", "D_nosg"):
        terms = [
            w * ((predict(rows, x, k) - predict(target_rows, x, k - p, p)) ** 2).mean(1)
            for (k, p), w in zip(linear.bootstrap_pairs(), linear.pair_weights())
        ]
        return torch.stack(terms).sum(0)
    # mixtures: weights[b][k] over the anchor points first..k
    terms = []
    for k in linear.AUX_HORIZONS:
        w = torch.tensor(np.stack([wb[k] for wb in weights]), dtype=theta.dtype)
        first = linear.first_point(k, weights[0][k])
        targets = torch.stack(
            [
                x[:, :, window + k - 1] if p == k else predict(target_rows, x, k - p, p)
                for p in linear.anchor_points(k, first)
            ],
            dim=-1,
        )
        target = (targets * w[:, None]).sum(-1)
        terms.append(((predict(rows, x, k) - target) ** 2).mean(1))

    return torch.stack(terms).mean(0)


def one_step(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    rows = batched_rows(theta, 1)
    return ((predict(rows, x, 1) - x[:, :, theta.shape[1]]) ** 2).mean(1)


def grad_ratio(variant, theta, x, lams, weights) -> np.ndarray:
    """Per model |grad of lam * aux| / |grad of the one-step loss| on one minibatch."""
    t = theta.detach().clone().requires_grad_(True)
    base = torch.autograd.grad(one_step(t, x).sum(), t)[0].norm(dim=1)
    aux = torch.autograd.grad((lams * aux_loss(variant, t, x, weights)).sum(), t)[0]

    return (aux.norm(dim=1) / base.clamp_min(1e-30)).numpy()


def train_config(system_idx: int, system, window: int, variant: str, arm: str) -> dict:
    """Trains every (seed, lambda) model of one configuration, variant and arm."""
    lams_list = [0.0] if variant == "onestep" else list(LAMBDAS)
    pairs = [(s, lam) for s in SEEDS for lam in lams_list]
    data = {s: train_set(system_idx, system, arm, s, window) for s in SEEDS}
    moments = {s: second_moment(d) for s, d in data.items()}
    x_all = torch.tensor(np.stack([data[s] for s in SEEDS]))
    seed_index = torch.tensor([SEEDS.index(s) for s, _ in pairs])
    lams = torch.tensor([lam for _, lam in pairs], dtype=torch.float64)
    n_eff = n_windows(arm, window)
    population = observation_covariance(system, window + HORIZON)

    torch.manual_seed(system_idx * 1000 + window)
    theta = torch.zeros(len(pairs), window, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    # cosine decay removes the jitter of a constant Adam step around the fixed point
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_STEPS)

    def current_weights():
        if variant == "M":
            return [linear.uniform_weights()] * len(pairs)
        if variant.startswith("M*"):
            # the oracle ablation uses the true moments instead of the data's
            first = 1 if variant == "M*_p1" else 2
            th = theta.detach().numpy()
            return [
                linear.bias_variance_weights(
                    th[b],
                    population if variant == "M*_oracle" else moments[s],
                    n_eff,
                    first,
                )
                for b, (s, _) in enumerate(pairs)
            ]
        return None

    def minibatch():
        idx = torch.randint(0, x_all.shape[1], (len(pairs), BATCH_WINDOWS))
        return x_all[seed_index[:, None], idx]

    weights = current_weights()
    ratio_start = (
        None
        if variant == "onestep"
        else grad_ratio(variant, theta, minibatch(), lams, weights)
    )
    curve = []
    for step in range(N_STEPS):
        if variant.startswith("M*") and step % WEIGHT_EVERY == 0:
            weights = current_weights()
        x = minibatch()
        loss = one_step(theta, x)
        if variant != "onestep":
            loss = loss + lams * aux_loss(variant, theta, x, weights)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        schedule.step()
        if (step + 1) % 100 == 0:
            curve.append(loss.detach().mean().item())
    ratio_end = (
        None
        if variant == "onestep"
        else grad_ratio(variant, theta, minibatch(), lams, weights)
    )

    th = theta.detach().numpy()
    val = second_moment(val_set(system_idx, system, window))
    return {
        "system": system.name,
        "window": window,
        "variant": variant,
        "arm": arm,
        "seeds": [s for s, _ in pairs],
        "lambdas": [lam for _, lam in pairs],
        "theta": th,
        "val_mse": np.array(
            [linear.rollout_mse(th[b], val, HORIZON).mean() for b in range(len(pairs))]
        ),
        "weights": [{str(k): w for k, w in wb.items()} for wb in (weights or [])],
        "grad_ratio_start": ratio_start,
        "grad_ratio_end": ratio_end,
        "curve": np.array(curve),
    }


def configs(arm: str):
    if arm == "small":
        return [(SYSTEMS.index(s), s, w) for s, w in SMALL_CONFIGS]
    return [(i, s, w) for i, s in enumerate(SYSTEMS) for w in WINDOWS]


def to_json(value):
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    parser = argparse.ArgumentParser(description="Train the e02 models.")
    parser.add_argument("arm", choices=list(N_TRAJ))
    parser.add_argument(
        "--only", help="run only configurations whose name contains this"
    )
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for system_idx, system, window in configs(args.arm):
        name = f"{system.name}_L{window}_{args.arm}"
        if args.only and args.only not in name:
            continue
        for variant in ("onestep", *VARIANTS, *ABLATIONS):
            run = f"{name}_{variant.replace('*', 'star')}"
            tracker = RunLogger(
                run,
                RUNS_DIR,
                {
                    "system": system.name,
                    "window": window,
                    "variant": variant,
                    "arm": args.arm,
                },
                "cpu",
            )
            out = train_config(system_idx, system, window, variant, args.arm)
            for i, value in enumerate(out["curve"]):
                tracker.log((i + 1) * 100, mean_loss=float(value))
            tracker.finish(to_json({k: v for k, v in out.items() if k != "curve"}))
            print(f"{run}: best val {out['val_mse'].min():.4f}", flush=True)


if __name__ == "__main__":
    main()
