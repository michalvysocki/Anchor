# Archive — pre-v0 pipeline

Frozen. The code here is kept for the record and does not run in this layout: it imports `config`, `toystateworld`, `model` and `direct_predictor` as top-level modules from the old `src/`.

To run it, check out commit `adfffcd` (the last commit before the repository restructuring) and follow the README there.

## Contents

- `src/`: the separate one-step (`model.py`, `train.py`, `evaluate.py`) and direct (`train_direct.py`, `evaluate_direct.py`) pipelines, the paired comparison (`compare.py`, `aggregate_seeds.py`) and the hyperparameter searches in `src/experiments/`.
- `results/`: paired test errors per seed and `direct_vs_rollout.png`, from the first version of Q1 (2026-11-10).
- `models/` (not in git): checkpoints of those runs, and `archive_pre_fix/` with checkpoints from before the bug fixes of 2026-11-09.

## Why it was superseded

The direct model had about 400 times more parameters than the one-step model, so the comparison was not fair. e01 (`experiments/e01_v0`) repeats it with one horizon-conditioned model for every variant and supersedes these results. The project log has the details.
