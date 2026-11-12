"""Runs every numerical check of the theory (README.md lists what each one verifies)."""

import sys

import harmlessness
import improvement
import markov_gap
import residual_lemma
import rollout_bound
import target_variance

MODULES = [
    target_variance,
    markov_gap,
    residual_lemma,
    harmlessness,
    improvement,
    rollout_bound,
]


def main():
    failures = sum(module.main() for module in MODULES)
    print(f"{failures} failed checks in total")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
