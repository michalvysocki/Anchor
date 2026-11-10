import argparse
import json

import numpy as np
import torch

from anchor_loss import (
    UnitStep,
    anchor_loss_bootstrapped,
    anchor_loss_composed,
    anchor_loss_rollout,
    anchor_loss_unit_rollout,
    grounding_loss,
    multistep_loss,
    one_step_loss,
    rollout_error,
)
from config import (
    BURN_IN,
    DP_HIDDEN_SIZE,
    K_MAX,
    MODELS_DIR,
    RESULTS_DIR,
    TRAIN_DATA_SEED,
    TRAINING_SEEDS,
    VAL_DATA_SEED,
    ObservabilityMode,
)
from direct_predictor import DirectPredictor
from toystateworld import generate_windows, observation_dim

LAMBDAS = (0.1, 0.3, 1.0, 3.0)
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

VARIANT_MODELS_DIR = MODELS_DIR / "variants"
VARIANT_RESULTS_DIR = RESULTS_DIR / "variants"


def run_name(variant: str, mode: ObservabilityMode, lam: float, seed: int) -> str:
    return f"{variant}_{mode}_lam{lam}_seed{seed}"


def train_variant(
    variant: str,
    mode: ObservabilityMode,
    lam: float,
    seed: int,
    device: str,
    n_steps: int = N_STEPS,
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
        n_steps (int): number of optimization steps

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

    model = DirectPredictor(K_MAX, observation_dim(mode), DP_HIDDEN_SIZE).to(device)
    unit = UnitStep(model, K_MAX)
    if family == "direct":
        base_loss, aux_model = grounding_loss, model
    else:
        base_loss, aux_model = one_step_loss, unit
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best = {path: {"val": float("inf"), "step": 0, "state": None} for path in PATHS}

    for step in range(n_steps):
        model.train()
        idx = torch.randint(0, len(train), (BATCH_WINDOWS,), generator=generator)
        batch = train[idx.to(device)]
        loss = base_loss(aux_model, batch)
        if aux_loss is not None:
            loss = loss + lam * aux_loss(aux_model, batch, generator)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_errors = {
                    "direct": grounding_loss(model, val).item(),
                    "rollout": rollout_error(unit, val).item(),
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
            print(
                f"{run_name(variant, mode, lam, seed)} | Step: {step + 1:05d} | "
                f"Loss: {loss.item():.4g} | Val direct: {val_errors['direct']:.4g} | "
                f"Val rollout: {val_errors['rollout']:.4g}"
            )

    name = run_name(variant, mode, lam, seed)
    VARIANT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VARIANT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "variant": variant,
        "mode": mode,
        "lambda": lam,
        "seed": seed,
        "steps": n_steps,
        "selected_by": select,
    }
    for path in PATHS:
        torch.save(best[path]["state"], VARIANT_MODELS_DIR / f"{name}_{path}.pt")
        summary[f"val_{path}"] = best[path]["val"]
        summary[f"best_step_{path}"] = best[path]["step"]
    with open(VARIANT_RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(summary, f)

    return best[select]["val"]


def best_lambda(variant: str, mode: ObservabilityMode, seed: int) -> float:
    """Lambda with the lowest validation loss in the sweep of one seed."""
    losses = {}
    for lam in LAMBDAS:
        with open(
            VARIANT_RESULTS_DIR / f"{run_name(variant, mode, lam, seed)}.json"
        ) as f:
            losses[lam] = json.load(f)[f"val_{VARIANTS[variant][2]}"]

    return min(losses, key=losses.get)


def main():
    parser = argparse.ArgumentParser(description="Train Anchor variants.")
    parser.add_argument("stage", choices=["sweep", "final", "baseline"])
    parser.add_argument("--family", choices=["direct", "onestep"], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    modes = ("insufficient", "sufficient")
    variants = [
        v for v, (family, aux, _) in VARIANTS.items() if family == args.family and aux
    ]
    sweep_seed = TRAINING_SEEDS[0]

    if args.stage == "sweep":
        for mode in modes:
            for variant in variants:
                for lam in LAMBDAS:
                    train_variant(variant, mode, lam, sweep_seed, args.device)

    elif args.stage == "final":
        for mode in modes:
            for variant in variants:
                lam = best_lambda(variant, mode, sweep_seed)
                print(f"{variant} {mode}: selected lambda {lam}")
                for seed in TRAINING_SEEDS[1:]:
                    train_variant(variant, mode, lam, seed, args.device)

    elif args.stage == "baseline":
        for mode in modes:
            for seed in TRAINING_SEEDS:
                train_variant(args.family, mode, 0.0, seed, args.device)


if __name__ == "__main__":
    main()
