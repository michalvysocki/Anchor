"""Training stages of the shelved limited-horizon experiment, see README.md in this directory."""

import argparse
import json
from pathlib import Path

from anchor.config import TRAINING_SEEDS
from anchor.training import N_STEPS, run_name, train_variant

EXPERIMENT_DIR = Path(__file__).resolve().parent
LH_MODELS_DIR = EXPERIMENT_DIR / "models"
LH_RESULTS_DIR = EXPERIMENT_DIR / "results"
CHECK_DIR = "gradcheck"
HORIZONS = (4, 8)
LAMBDAS = (0.001, 0.01, 0.1, 1.0)
FAMILIES = {"onestep": ("C", "D"), "direct": ("E",)}
SELECT_METRIC = "median"
# hyperparameters are chosen in this mode only and reused in the other one
SELECTION_MODE = "insufficient"
CHECK_STEPS = 2000


def train(
    variant: str,
    mode: str,
    lam: float,
    seed: int,
    device: str,
    max_horizon: int | None = None,
    check: bool = False,
) -> float:
    """Runs train_variant with the limited-horizon protocol."""
    models_dir, results_dir = LH_MODELS_DIR, LH_RESULTS_DIR
    if check:
        models_dir, results_dir = models_dir / CHECK_DIR, results_dir / CHECK_DIR

    return train_variant(
        variant,
        mode,
        lam,
        seed,
        device,
        models_dir,
        results_dir,
        n_steps=CHECK_STEPS if check else N_STEPS,
        max_horizon=max_horizon,
        select_metric=SELECT_METRIC,
        log_grads=check,
    )


def best_setting(variant: str) -> tuple[int, float]:
    """(K_A, lambda) with the lowest validation rollout error of the sweep."""
    vals = {}
    for max_horizon in HORIZONS:
        for lam in LAMBDAS:
            name = run_name(
                variant, SELECTION_MODE, lam, TRAINING_SEEDS[0], max_horizon
            )
            with open(LH_RESULTS_DIR / f"{name}.json") as f:
                vals[(max_horizon, lam)] = json.load(f)["val_rollout"]

    return min(vals, key=vals.get)


def main():
    parser = argparse.ArgumentParser(description="Train the limited-horizon variants.")
    parser.add_argument("stage", choices=["check", "baseline", "sweep", "final"])
    parser.add_argument("--family", choices=list(FAMILIES), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    variants = FAMILIES[args.family]
    sweep_seed = TRAINING_SEEDS[0]

    if args.stage == "check":
        for variant in variants:
            for max_horizon in HORIZONS:
                train(
                    variant,
                    SELECTION_MODE,
                    1.0,
                    sweep_seed,
                    args.device,
                    max_horizon,
                    check=True,
                )

    elif args.stage == "baseline":
        if args.family != "onestep":
            parser.error("the only baseline here is onestep")
        for mode in ("insufficient", "sufficient"):
            for seed in TRAINING_SEEDS:
                train("onestep", mode, 0.0, seed, args.device)

    elif args.stage == "sweep":
        for variant in variants:
            for max_horizon in HORIZONS:
                for lam in LAMBDAS:
                    train(
                        variant,
                        SELECTION_MODE,
                        lam,
                        sweep_seed,
                        args.device,
                        max_horizon,
                    )

    elif args.stage == "final":
        for variant in variants:
            max_horizon, lam = best_setting(variant)
            print(f"{variant}: selected K_A {max_horizon}, lambda {lam}")
            for seed in TRAINING_SEEDS[1:]:
                train(variant, SELECTION_MODE, lam, seed, args.device, max_horizon)
            for seed in TRAINING_SEEDS:
                train(variant, "sufficient", lam, seed, args.device, max_horizon)


if __name__ == "__main__":
    main()
