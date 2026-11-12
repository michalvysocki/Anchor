"""Training stages of e01b, see README.md in this directory."""

import argparse
import json
from pathlib import Path

import numpy as np

from anchor.config import ObservabilityMode
from anchor.training import N_STEPS, VARIANTS, run_name, train_variant

EXPERIMENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
CHECK_DIR = "gradcheck"
MAX_HORIZON = 4
VAL_HORIZON = 10
LAMBDAS = (0.001, 0.01, 0.1, 1.0)
MODES = ("insufficient", "sufficient")
# lambda is chosen on these seeds and evaluated on disjoint ones, so the
# reported result does not reuse the data that chose it
SELECTION_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = (3, 4, 5, 6, 7)
# baseline of each evaluation path on the test set
BASELINES = {"direct": "direct", "rollout": "onestep"}
# a lambda is admissible when, averaged over the selection seeds, its
# validation error in the sufficient mode is at most this factor times that
# of the same family at lambda = 0 with the same seed
SUFFICIENT_TOLERANCE = 1.10
CHECK_STEPS = 2000


def family_variants(family: str) -> list[str]:
    """Anchor variants of one family, without its baseline."""
    return [v for v, (f, aux, _) in VARIANTS.items() if f == family and aux]


def train(
    variant: str,
    mode: ObservabilityMode,
    lam: float,
    seed: int,
    device: str,
    check: bool = False,
) -> float:
    """Runs train_variant with the e01b protocol."""
    models_dir, results_dir = MODELS_DIR, RESULTS_DIR
    if check:
        models_dir, results_dir = models_dir / CHECK_DIR, results_dir / CHECK_DIR
    is_baseline = VARIANTS[variant][1] is None

    return train_variant(
        variant,
        mode,
        lam,
        seed,
        device,
        models_dir,
        results_dir,
        n_steps=CHECK_STEPS if check else N_STEPS,
        max_horizon=None if is_baseline else MAX_HORIZON,
        val_horizon=VAL_HORIZON,
        log_grads=check,
    )


def run_file(variant: str, mode: ObservabilityMode, lam: float, seed: int) -> str:
    """Name of the run summary and checkpoints of one run."""
    is_baseline = VARIANTS[variant][1] is None

    return run_name(variant, mode, lam, seed, None if is_baseline else MAX_HORIZON)


def validation_error(
    variant: str, mode: ObservabilityMode, lam: float, seed: int, path: str
) -> float:
    """Validation error of one path of one run."""
    with open(RESULTS_DIR / f"{run_file(variant, mode, lam, seed)}.json") as f:
        return json.load(f)[f"val_{path}"]


def relative_validation_error(
    variant: str, mode: ObservabilityMode, lam: float
) -> float:
    """
    Validation error of the variant's selection path relative to the same
    family at lambda = 0 with the same seed, averaged over the selection
    seeds. Comparing with the same family isolates the effect of the
    auxiliary term; pairing by seed removes the shared effect of the
    initialization.
    """

    family, _, path = VARIANTS[variant]

    return float(
        np.mean(
            [
                validation_error(variant, mode, lam, seed, path)
                / validation_error(family, mode, 0.0, seed, path)
                for seed in SELECTION_SEEDS
            ]
        )
    )


def best_lambda(variant: str) -> float:
    """
    Lambda for the final runs: among the lambdas whose relative validation
    error in the sufficient mode is at most SUFFICIENT_TOLERANCE, the one with
    the lowest relative validation error in the insufficient mode. If no
    lambda is admissible, the smallest one.
    """

    admissible = [
        lam
        for lam in LAMBDAS
        if relative_validation_error(variant, "sufficient", lam) <= SUFFICIENT_TOLERANCE
    ]
    if not admissible:
        return min(LAMBDAS)

    return min(
        admissible,
        key=lambda lam: relative_validation_error(variant, "insufficient", lam),
    )


def main():
    parser = argparse.ArgumentParser(description="Train the e01b variants.")
    parser.add_argument("stage", choices=["check", "baseline", "sweep", "final"])
    parser.add_argument("--family", choices=["direct", "onestep"], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    variants = family_variants(args.family)

    if args.stage == "check":
        for variant in variants:
            train(variant, "insufficient", 1.0, SELECTION_SEEDS[0], args.device, True)

    elif args.stage == "baseline":
        for mode in MODES:
            for seed in SELECTION_SEEDS + EVALUATION_SEEDS:
                train(args.family, mode, 0.0, seed, args.device)

    elif args.stage == "sweep":
        for mode in MODES:
            for variant in variants:
                for lam in LAMBDAS:
                    for seed in SELECTION_SEEDS:
                        train(variant, mode, lam, seed, args.device)

    elif args.stage == "final":
        for variant in variants:
            lam = best_lambda(variant)
            print(f"{variant}: selected lambda {lam}")
            for mode in MODES:
                for seed in EVALUATION_SEEDS:
                    train(variant, mode, lam, seed, args.device)


if __name__ == "__main__":
    main()
