"""Training stages of e01, see README.md in this directory."""

import argparse
import json
from pathlib import Path

from anchor.config import TRAINING_SEEDS, ObservabilityMode
from anchor.training import VARIANTS, run_name, train_variant

EXPERIMENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LAMBDAS = (0.1, 0.3, 1.0, 3.0)


def train(variant: str, mode: ObservabilityMode, lam: float, seed: int, device: str):
    """Runs train_variant with the e01 protocol."""
    train_variant(variant, mode, lam, seed, device, MODELS_DIR, RESULTS_DIR)


def best_lambda(variant: str, mode: ObservabilityMode, seed: int) -> float:
    """Lambda with the lowest validation loss in the sweep of one seed."""
    losses = {}
    for lam in LAMBDAS:
        with open(RESULTS_DIR / f"{run_name(variant, mode, lam, seed)}.json") as f:
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
                    train(variant, mode, lam, sweep_seed, args.device)

    elif args.stage == "final":
        for mode in modes:
            for variant in variants:
                lam = best_lambda(variant, mode, sweep_seed)
                print(f"{variant} {mode}: selected lambda {lam}")
                for seed in TRAINING_SEEDS[1:]:
                    train(variant, mode, lam, seed, args.device)

    elif args.stage == "baseline":
        for mode in modes:
            for seed in TRAINING_SEEDS:
                train(args.family, mode, 0.0, seed, args.device)


if __name__ == "__main__":
    main()
