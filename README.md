# Anchor

Multi-step consistency targets for latent world models. A world model with a one-step dynamics model is rolled out recursively; when its latent does not summarize the history (a Markov gap), the rollout drifts. Anchor is an auxiliary loss that pulls the $k$-step rollout toward targets anchored in real observations: the encoded true future (the Monte Carlo target of TD-MPC and SPR) or the model's own rollout from a later encoded observation (a bootstrapped target), mixed over anchor points with weights chosen from estimated bias and variance.

The theory (paper draft in `paper/`) shows that these targets leave the one-step optimum unchanged when the latent is Markov and the target is under stop-gradient, strictly improve the rollout under a Markov gap, and that the conditional mean of the bootstrap residual (the critic) measures the rollout bias without the target variance.

This is ongoing research. Every experiment has a README written before it runs, with its hypotheses, decision criteria and design review, followed by the result and the conclusion.

## Status

| experiment | system | outcome |
|---|---|---|
| e01 (`experiments/e01_v0`) | Hénon map, raw observations, variants A–E | **superseded by e01b.** Direct beats rollout under partial observation (stands); no variant helped without harm, because of design errors (horizons beyond predictability, unbalanced gradients) |
| archived (`experiments/archive/limited_horizon`) | e01 with limited horizons | designed, not run: raw observations leave no latent to shape |
| e01b (`experiments/e01b_corrected`) | e01 with its design errors corrected | R1 formally, R3 in practice: short horizons and small λ do no harm, but no fixed λ helps in one regime without harming the other |
| e02 (`experiments/e02_lgssm`) | linear-Gaussian systems, history-window latent, C, D and the mixture M\* | R3 formally, R1 after diagnosis: population theory predicted every trained gain to the third decimal; M\* matches the better of C and D but beats C nowhere |
| next: e03 | Hénon with observation noise and a learned latent | does M\* gain where bootstrapped targets have the advantage? |

Every statement of the theory has a numerical check in `theory/`; its README is the map of the theory (assumptions, results, checks, experiments that test them).

## Project structure

```
Anchor/
├── src/anchor/              # library, installed as the `anchor` package
│   ├── config.py            # data seeds, training seeds, K_MAX, burn-in (Hénon experiments)
│   ├── envs/henon.py        # Hénon map, observation modes, trajectory windows
│   ├── envs/lgssm.py        # AR(1)/AR(2) observed with noise: exact covariances, sampling, Kalman optimum
│   ├── linear.py            # linear window model: population losses, stop-gradient fixed points,
│   │                        #   anchor mixtures, bias–variance weights, critic
│   ├── models.py            # horizon-conditioned MLP F(z, k), unit-step wrapper
│   ├── losses.py            # grounding, one-step, multi-step and Anchor losses (Hénon)
│   ├── metrics.py           # path errors, median errors, sharpness, off-attractor, CIs
│   ├── training.py          # Hénon variant catalogue and the shared training loop
│   ├── stats.py             # t-intervals for any number of seeds, paired log ratios, verdicts with a margin
│   ├── tracking.py          # run summary with metadata, JSONL training curves, optional W&B
│   └── plotting.py          # fixed variant colors
├── experiments/
│   ├── e01_v0/              # each experiment: README, scripts, results/ (models/ not in git)
│   ├── e01b_corrected/
│   ├── e02_lgssm/           # design.py, predictions.py (before training), run.py, evaluate.py
│   └── archive/             # pre_v0 (frozen pipeline before e01), limited_horizon (designed, not run)
├── theory/                  # one numerical check per statement of the theory; README.md is the map
├── paper/                   # LaTeX draft (Tectonic), synced to Overleaf later
├── tests/                   # losses, metrics, invariants, LG-SSM and linear theory, tracking
├── Makefile
├── pyproject.toml
└── uv.lock
```

Checkpoints (`*.pt`), `logs/` and offline W&B runs are not in git; checkpoints are reproducible from the seeds.

**What goes where.** The library holds what more than one experiment uses or what is part of the method or the theory: environments, models, losses, metrics, statistics, run tracking. The protocol of an experiment (its grid, seeds, stages, batched training as in e02, verdicts) stays in that experiment's scripts. One model family is one place in the library: `models.py`, `losses.py`, `training.py` and `metrics.py` hold the horizon-conditioned predictor F(z, k) of e01 and e01b; `linear.py` holds the linear window model of e02; e03 adds `latent.py`. Results of finished experiments were produced with the code at their commit: e01 and e01b keep their own interval code (`metrics.mean_ci`), e02 its own in `evaluate.py`; new experiments use `stats.py`.

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

Common tasks are in the `Makefile`: `make test`, `make lint`, `make format`, `make theory` (numerical checks of the derivations), `make paper` (build `paper/main.pdf` with Tectonic), `make e02-predictions` and `make e02-evaluate`.

## Run tracking

Every training run writes `results/{run}.json` (config, results and metadata: git commit and whether its code differed from it (`*.py`, `pyproject.toml`, `uv.lock`; results and notes may change while experiments run), device, host, versions, start time, duration) and `results/{run}.curve.jsonl` (training curve). Local files are the source of truth; Weights & Biases can mirror them: install the optional extra with `uv sync --extra tracking`, pass `use_wandb=True` to `train_variant`; runs are offline unless `WANDB_MODE` says otherwise and land in the experiment's `wandb/` directory (ignored by git), to be uploaded with `wandb login` and `wandb sync <experiment>/wandb/offline-run-*`. e01b predates run tracking; its curves were recovered from the training logs by `experiments/e01b_corrected/backfill_curves.py`.

## Reproducibility

Hénon experiments (e01, e01b): training, validation and test data come from separate generators (`TRAIN_DATA_SEED`, `VAL_DATA_SEED`, `TEST_DATA_SEED` in `src/anchor/config.py`); `TRAINING_SEEDS` set weight initialization and sampling, so all seeds see the same data. e02: every seed draws its own training set (shared by all variants of that seed, so comparisons are paired), and models are evaluated exactly from the known covariance. Evaluation on CPU is deterministic: re-running it reproduces the saved results bit for bit.
