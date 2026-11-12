from pathlib import Path

import numpy as np
import torch

from anchor.config import (
    BURN_IN,
    K_MAX,
    TRAIN_DATA_SEED,
    VAL_DATA_SEED,
    ObservabilityMode,
)
from anchor.envs.henon import generate_windows, observation_dim
from anchor.losses import (
    anchor_loss_bootstrapped,
    anchor_loss_composed,
    anchor_loss_rollout,
    anchor_loss_unit_rollout,
    grounding_loss,
    multistep_loss,
    one_step_loss,
    rollout_error,
)
from anchor.metrics import (
    direct_median_error,
    direct_mse,
    rollout_median_error,
    rollout_mse,
)
from anchor.models import HorizonMLP, UnitStep
from anchor.tracking import RunLogger

HIDDEN_SIZE = 256
N_TRAJ = 1000
LEN_TRAJ = 36
LEARNING_RATE = 1e-3
N_STEPS = 40000
BATCH_WINDOWS = 512
# rollouts backpropagate through up to K_MAX chaotic steps; applied to every
# variant so the optimization is the same for all of them
GRAD_CLIP = 1.0

# family, auxiliary loss and the path used to select the checkpoint and lambda.
# Every variant trains the same model F(z, k); the "onestep" family only uses
# it with k = 1. "direct" and "onestep" are the lambda = 0 baselines.
VARIANTS = {
    "direct": ("direct", None, "direct"),
    "A": ("direct", anchor_loss_composed, "direct"),
    "B": ("direct", anchor_loss_bootstrapped, "direct"),
    "E": ("direct", anchor_loss_unit_rollout, "rollout"),
    "onestep": ("onestep", None, "rollout"),
    "C": ("onestep", multistep_loss, "rollout"),
    "D": ("onestep", anchor_loss_rollout, "rollout"),
}

PATHS = ("direct", "rollout")


def run_name(
    variant: str,
    mode: ObservabilityMode,
    lam: float,
    seed: int,
    max_horizon: int | None = None,
) -> str:
    horizon = f"_ka{max_horizon}" if max_horizon else ""
    return f"{variant}_{mode}{horizon}_lam{lam}_seed{seed}"


def grad_norm(loss: torch.Tensor, params: list[torch.Tensor]) -> float:
    """Norm of the gradient of one loss term, without touching .grad."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)

    return torch.sqrt(sum((g**2).sum() for g in grads if g is not None)).item()


def train_variant(
    variant: str,
    mode: ObservabilityMode,
    lam: float,
    seed: int,
    device: str,
    models_dir: Path,
    results_dir: Path,
    n_steps: int = N_STEPS,
    max_horizon: int | None = None,
    select_metric: str = "mse",
    val_horizon: int | None = None,
    log_grads: bool = False,
    use_wandb: bool = False,
) -> float:
    """
    Trains one variant and saves two checkpoints: the one with the lowest
    validation error of the direct prediction F(z, k) and the one with the
    lowest validation error of the unit-step rollout F(., 1)^k, both averaged
    over every horizon. Each path is later evaluated with its own checkpoint,
    so no variant is judged on a path its checkpoint was not selected for.
    The variant's selection path decides which value is returned for the
    choice of lambda.

    Every variant trains the same model F(z, k) with the same number of
    steps, minibatches of windows, learning rate and gradient clipping.

    Args:
        variant (str): one of VARIANTS
        mode (ObservabilityMode): insufficient or sufficient observability
        lam (float): weight of the auxiliary loss
        seed (int): seed for weight initialization and decompositions
        device (str): device used for training
        models_dir (Path): directory for checkpoints
        results_dir (Path): directory for the run summary
        n_steps (int): number of optimization steps
        max_horizon (int | None): largest horizon of the auxiliary loss,
            K_MAX if None. Supported by variants C, D and E.
        select_metric (str): "mse" (e01) or "median", the validation
            statistic used to select checkpoints
        val_horizon (int | None): with "mse", evaluate validation over
            horizons 1..val_horizon only; all horizons if None (e01)
        log_grads (bool): print the gradient norms of the base and the
            weighted auxiliary term every 100 steps
        use_wandb (bool): mirror the training curve to Weights & Biases

    Returns:
        float: best validation error of the variant's selection path
    """

    family, aux_loss, select = VARIANTS[variant]

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train = generate_windows(
        N_TRAJ, LEN_TRAJ, BURN_IN, K_MAX, mode, np.random.default_rng(TRAIN_DATA_SEED)
    ).to(device)
    val = generate_windows(
        N_TRAJ // 5,
        LEN_TRAJ,
        BURN_IN,
        K_MAX,
        mode,
        np.random.default_rng(VAL_DATA_SEED),
    ).to(device)

    model = HorizonMLP(K_MAX, observation_dim(mode), HIDDEN_SIZE).to(device)
    unit = UnitStep(model, K_MAX)
    if family == "direct":
        base_loss, aux_model = grounding_loss, model
    else:
        base_loss, aux_model = one_step_loss, unit
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    aux_kwargs = {"max_horizon": max_horizon} if max_horizon else {}
    if select_metric == "mse" and val_horizon:
        val_metrics = {
            "direct": lambda m, w: direct_mse(m, w, val_horizon),
            "rollout": lambda m, w: rollout_mse(m, w, val_horizon),
        }
    elif select_metric == "mse":
        val_metrics = {"direct": grounding_loss, "rollout": rollout_error}
    else:
        val_metrics = {"direct": direct_median_error, "rollout": rollout_median_error}
    name = run_name(variant, mode, lam, seed, max_horizon)
    tracker = RunLogger(
        name,
        results_dir,
        {
            "variant": variant,
            "mode": mode,
            "lambda": lam,
            "seed": seed,
            "steps": n_steps,
            "selected_by": select,
            "max_horizon": max_horizon,
            "select_metric": select_metric,
            "val_horizon": val_horizon,
        },
        device,
        use_wandb,
        group=Path(results_dir).parent.name,
    )

    best = {path: {"val": float("inf"), "step": 0, "state": None} for path in PATHS}

    for step in range(n_steps):
        model.train()
        idx = torch.randint(0, len(train), (BATCH_WINDOWS,), generator=generator)
        batch = train[idx.to(device)]
        loss = base_loss(aux_model, batch)
        if aux_loss is not None:
            aux = lam * aux_loss(aux_model, batch, generator, **aux_kwargs)
            if log_grads and (step + 1) % 100 == 0:
                params = list(model.parameters())
                base_norm, aux_norm = grad_norm(loss, params), grad_norm(aux, params)
                print(
                    f"{name} | Step: {step + 1:05d} | Grad base: {base_norm:.3g} | "
                    f"Grad aux: {aux_norm:.3g} | Ratio: {aux_norm / base_norm:.3g}"
                )
            loss = loss + aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_errors = {
                    "direct": val_metrics["direct"](model, val).item(),
                    "rollout": val_metrics["rollout"](unit, val).item(),
                }
            for path in PATHS:
                if val_errors[path] < best[path]["val"]:
                    best[path] = {
                        "val": val_errors[path],
                        "step": step + 1,
                        "state": {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                    }
            tracker.log(
                step + 1,
                loss=loss.item(),
                val_direct=val_errors["direct"],
                val_rollout=val_errors["rollout"],
            )
            print(
                f"{name} | Step: {step + 1:05d} | "
                f"Loss: {loss.item():.4g} | Val direct: {val_errors['direct']:.4g} | "
                f"Val rollout: {val_errors['rollout']:.4g}"
            )

    models_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in PATHS:
        torch.save(best[path]["state"], models_dir / f"{name}_{path}.pt")
        results[f"val_{path}"] = best[path]["val"]
        results[f"best_step_{path}"] = best[path]["step"]
    tracker.finish(results)

    return best[select]["val"]
