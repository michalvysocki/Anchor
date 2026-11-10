# Anchor

Grounded predictive path consistency for latent world models. The same future state can be reached by different predictive paths: one direct prediction over the whole horizon, or a composition of shorter predictions. Anchor trains these paths to agree while keeping predictions grounded in observed future states. The goal is to reduce model-induced compounding error in long rollouts.

This is ongoing research. The Anchor objective itself is not implemented yet.

## Status

Stage 0: controlled experiments on the Hénon map (deterministic, chaotic) with two observation modes:

- `sufficient`: delay embedding (x_{t-1}, x_t), which is the full state,
- `insufficient`: only x_t, so the one-step model is Markov-misspecified.

Current result (5 training seeds, held-out test set, `results/direct_vs_rollout.png`):

- `insufficient`: a direct multi-horizon predictor beats one-step rollout at every horizon k = 1..20.
- `sufficient`: rollout is better up to k ≈ 14. Beyond that both errors saturate; direct converges to the error of predicting the attractor mean and rollout to the error of a random attractor point.

Next: the Anchor loss (composition with a stop-gradient direct target, and a bootstrapped variant anchored at later observed states), compared against multi-horizon and direct baselines.

## Project structure

```
Anchor/
├── src/
│   ├── config.py            # constants, seeds, model paths
│   ├── toystateworld.py     # Hénon map, observation modes, dataset generation
│   ├── model.py             # one-step predictor
│   ├── direct_predictor.py  # direct predictor conditioned on one-hot horizon k
│   ├── train.py             # trains the one-step predictor for every mode and seed
│   ├── train_direct.py      # trains the direct predictor for every mode and seed
│   ├── evaluate.py          # rollout drift and diverged-rollout rate
│   ├── evaluate_direct.py   # direct prediction error per horizon
│   ├── compare.py           # paired test errors and their correlations, per seed
│   ├── aggregate_seeds.py   # mean and 95% CI over seeds, comparison plot
│   └── experiments/         # hyperparameter searches
├── tests/                   # invariants of data generation and evaluation
├── models/                  # checkpoints (gitignored)
├── results/                 # paired errors and plots
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
uv sync
```

## Usage

Scripts import each other as top-level modules, so run them from `src/`:

```bash
cd src
uv run python train.py
uv run python train_direct.py
uv run python compare.py
uv run python aggregate_seeds.py
```

`train_direct.py` uses Apple MPS when available. Run the tests from the repository root:

```bash
uv run pytest
```

## Reproducibility

Training, validation and test data come from separate generators (`TRAIN_DATA_SEED`, `VAL_DATA_SEED`, `TEST_DATA_SEED` in `config.py`). `TRAINING_SEEDS` set only the weight initialization, so all seeds see the same data. Each run saves the checkpoint with the lowest validation loss.
