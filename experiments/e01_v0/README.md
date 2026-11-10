# e01 — v0: direct vs rollout, and Anchor variants

Status: running (started 2026-11-10). Decision criteria below were written before any test-set result was seen.

Code: `src/train_variants.py`, `src/anchor_loss.py`, `src/evaluate_variants.py`, at the commit that sets `N_STEPS = 40000` (on top of `01d753d`).

## Questions

**Q1 (motivation).** With equal model capacity and training, does direct prediction F(z, k) beat the unit-step rollout F(·, 1)^k?

**Q2 (main).** Does an Anchor objective reduce rollout error compared to the same model trained without it, and compared to a plain multi-step loss?

## Setup

- Environment: Hénon map (a = 1.4, b = 0.3), burn-in 10, trajectories of 36 steps, K_MAX = 20.
- Observation modes: `sufficient` (x_{t-1}, x_t), which is the full state; `insufficient` (x_t only), Markov-misspecified.
- Data: train, validation and test windows from separate generators (`TRAIN_DATA_SEED`, `VAL_DATA_SEED`, `TEST_DATA_SEED`), every horizon equally represented.
- One model for every variant: F(z, k), MLP 2×256 tanh, one-hot k. The one-step family uses it only with k = 1.
- Same protocol for every variant: Adam, lr 1e-3, 40 000 steps, batches of 512 windows, gradient clipping at norm 1.0.
- Each run keeps two checkpoints, best validation error of the direct path and of the rollout path (both averaged over k). Each path is evaluated with its own checkpoint.
- λ ∈ {0.1, 0.3, 1, 3}, chosen on seed 0 by the validation error of the variant's target path; seeds 1–4 use the chosen λ.
- 5 training seeds (initialization and sampling); data identical across seeds.

## Variants

| variant | loss | target path |
|---|---|---|
| direct | grounding: F(z_t, k) vs true z_{t+k}, all k | direct |
| A | direct + λ‖F(F(z_t, p), q) − sg F(z_t, k)‖² | direct |
| B | direct + λ‖F(z_t, k) − sg F(z_{t+p}^true, q)‖² | direct |
| E | direct + λ‖F(·,1)^k(z_t) − sg F(z_t, k)‖² | rollout |
| onestep | F(z_t, 1) vs true z_{t+1} | rollout |
| C | onestep + λ‖F(·,1)^k(z_t) − z_{t+k}^true‖² (multi-step loss, control) | rollout |
| D | onestep + λ‖F(·,1)^k(z_t) − sg F(·,1)^q(z_{t+p}^true)‖² (plug-in Anchor) | rollout |

k ~ U{2..20}, p ~ U{1..k−1}, q = k − p. C uses the same distribution of k as D, so C and D differ only in the target.

## Metrics (test set)

- Primary: median Euclidean error per k, averaged over k = 1..20, per seed.
- Comparison: paired difference to the baseline of the same seed, mean and 95% t-interval over 5 seeds. A difference is called significant when its interval excludes 0.
- Guards: fraction of diverged rollouts (counted as failures, error ∞), and median distance of predictions to the attractor (detects predictions that win by predicting the mean).

## Hypotheses

- H1: in `insufficient`, the direct path of the direct baseline has lower error than the rollout of the onestep baseline. In `sufficient`, the rollout is better up to the horizon where errors saturate.
- H2: in `insufficient`, D has lower rollout error than both onestep and C.
- H3: in `sufficient`, no Anchor variant is significantly worse than its baseline.

## Decision criteria

The primary comparison is the rollout path in `insufficient`. A variant only counts as an improvement if, in addition, it is not significantly worse in `sufficient` and its off-attractor distance at k = 20 is not significantly larger than its baseline's.

| scenario | result | next research step |
|---|---|---|
| S1 | D or E significantly better than onestep **and** than C | build the `AnchorLoss` module, Stage 1 (learned encoder), linear-Gaussian theory |
| S2 | D ≈ C, both better than onestep | Anchor adds nothing over a multi-step loss on raw observations; run Stage 1 as the decisive test; if still D ≈ C, frame the paper as an analysis of when path consistency helps |
| S3 | E or A better, D not | the useful signal is consistency with a direct head; method is E with a training-time head |
| S4 | improvement only with larger off-attractor distance, or harm in `sufficient` | point-wise MSE rewards the mean in chaos; needs a distributional variant or metric |
| S5 | no variant better | first check λ range and training length; if still nothing, negative result and a theory-driven paper |

The target venue is ICML in every scenario; the scenario decides what the paper claims, not where it is submitted.

The result of this experiment is recorded as it comes out and is not reinterpreted later. Any new idea that follows from it is a new hypothesis, tested in a new experiment with its own README written before it runs.

## Known limitations

- λ is chosen on seed 0, whose test result is then reported with the other seeds. Selection uses validation, not test data.
- Test windows from the same trajectory overlap, so they are not independent; confidence intervals are over training seeds, not over windows.
- One environment, raw observations, no actions.

## History

A first version of Q1 (2026-11-10, `compare.py`, `aggregate_seeds.py`) found that direct beats rollout in `insufficient` at every horizon. Its models had different capacity (direct ~72k parameters, one-step ~190), so it is superseded by this experiment. See the project log for details.

## Result

Not available yet.

## Conclusion

Not available yet.
