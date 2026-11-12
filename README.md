# Anchor

Grounded predictive path consistency for latent world models. The same future state can be reached by different predictive paths: one direct prediction over the whole horizon, or a composition of shorter predictions. Anchor trains these paths to agree while keeping predictions grounded in observed future states. The goal is to reduce model-induced compounding error in long rollouts.

This is ongoing research.

## Status

- **e01** (`experiments/e01_v0`, done): Hénon map with full (`sufficient`) and partial (`insufficient`) observation, raw observations, Anchor variants A–E against one-step and direct baselines. Direct prediction beats one-step rollout under partial observation. No Anchor variant improved the rollout without harming the well-specified case; the auxiliary horizons reached beyond the predictable range and dominated the gradients (see the README's addendum).
- **Shelved** (`experiments/shelved/limited_horizon`): the same comparison with horizon-limited auxiliary losses, not run, because raw one-dimensional observations leave no latent for Anchor to shape.
- **Next** (e02): a partially observed linear-Gaussian system with a latent model, comparing bootstrapped (Anchor) and Monte Carlo (TD-MPC-style) multi-step targets under observation noise.

Every experiment has a README written before it runs, with its hypotheses and decision criteria, followed by the result and conclusion.

## Project structure

```
Anchor/
├── src/anchor/              # library, installed as the `anchor` package
│   ├── config.py            # data seeds, training seeds, K_MAX, burn-in
│   ├── envs/henon.py        # Hénon map, observation modes, trajectory windows
│   ├── models.py            # horizon-conditioned MLP F(z, k), unit-step wrapper
│   ├── losses.py            # grounding, one-step, multi-step and Anchor losses
│   ├── metrics.py           # path errors, median errors, sharpness, off-attractor, CIs
│   ├── training.py          # variant catalogue and the shared training loop
│   └── plotting.py          # fixed variant colors
├── experiments/
│   ├── e01_v0/              # README, train.py, evaluate.py, bayes_floor.py, results/, models/
│   ├── shelved/             # experiments designed but not run
│   └── archive/pre_v0/      # frozen pipeline from before e01
├── tests/                   # losses, metrics and data invariants
├── pyproject.toml
└── uv.lock
```

Checkpoints (`*.pt`) are not in git; they are reproducible from the seeds.

## Setup

```bash
uv sync
```

## Usage

Experiment scripts can be run from any directory:

```bash
uv run python -u experiments/e01b_corrected/run.py baseline --family onestep | tee logs/e01b_onestep.log
uv run python experiments/e01b_corrected/evaluate.py
make test
```

Use `python -u` for long runs piped to `tee`, otherwise the output is buffered and the log stays empty until the end. Each experiment's README lists the full sequence of commands as they were run.

Common tasks are in the `Makefile`: `make test`, `make lint`, `make format`, `make theory` (numerical checks of the derivations), `make paper` (build `paper/main.pdf` with Tectonic).

## Run tracking

Every training run writes `results/{run}.json` (config, results and metadata: git commit and whether the tree was dirty, device, host, versions, start time, duration) and `results/{run}.curve.jsonl` (training curve). Local files are the source of truth; Weights & Biases can mirror them: install the optional extra with `uv sync --extra tracking`, pass `use_wandb=True` to `train_variant`; runs are offline unless `WANDB_MODE` says otherwise and land in the experiment's `wandb/` directory (ignored by git), to be uploaded with `wandb login` and `wandb sync <experiment>/wandb/offline-run-*`. e01b predates run tracking; its curves were recovered from the training logs by `experiments/e01b_corrected/backfill_curves.py`.

## Reproducibility

Training, validation and test data come from separate generators (`TRAIN_DATA_SEED`, `VAL_DATA_SEED`, `TEST_DATA_SEED` in `src/anchor/config.py`). `TRAINING_SEEDS` set weight initialization and sampling, so all seeds see the same data. Evaluation on CPU is deterministic: re-running it reproduces the saved results bit for bit.
