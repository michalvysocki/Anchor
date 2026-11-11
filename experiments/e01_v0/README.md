# e01 — v0: direct vs rollout, and Anchor variants

Status: done 2026-11-11 (started 2026-11-10). Decision criteria below were written before any test-set result was seen.

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

Evaluated 2026-11-11 (`src/evaluate_variants.py`, output in `logs/evaluate_variants.log`, `results/variants/test_results.json`, `results/variants/variants_comparison.png`). All 100 runs finished with 40 000 steps; no rollout diverged.

Selected λ (seed 0, validation): A 0.1 / 0.1, B 0.1 / 0.1, E 3.0 / 3.0, C 3.0 / 0.1, D 3.0 / 3.0 (insufficient / sufficient). Five of ten choices sit at the upper edge of the grid.

Mean median test error over k = 1..20, paired difference to the baseline, 95% t-interval over 5 seeds:

| mode | path | variant | error | diff vs baseline |
|---|---|---|---|---|
| insufficient | direct | direct | 0.442 ± 0.004 | — |
| insufficient | direct | A | 0.458 ± 0.002 | +0.016 ± 0.003 |
| insufficient | direct | B | 0.442 ± 0.003 | +0.000 ± 0.003 |
| insufficient | direct | E | 0.488 ± 0.009 | +0.046 ± 0.008 |
| insufficient | rollout | onestep | 0.540 ± 0.019 | — |
| insufficient | rollout | C | 0.527 ± 0.003 | −0.013 ± 0.021 |
| insufficient | rollout | D | 0.523 ± 0.009 | −0.017 ± 0.027 |
| insufficient | rollout | E | 0.550 ± 0.003 | +0.010 ± 0.018 |
| insufficient | rollout | A | 0.485 ± 0.005 | −0.055 ± 0.022 |
| insufficient | rollout | B | 0.548 ± 0.024 | +0.008 ± 0.026 |
| sufficient | direct | direct | 0.500 ± 0.004 | — |
| sufficient | direct | A | 0.564 ± 0.003 | +0.064 ± 0.006 |
| sufficient | direct | B | 0.504 ± 0.005 | +0.003 ± 0.007 |
| sufficient | direct | E | 0.697 ± 0.022 | +0.196 ± 0.022 |
| sufficient | rollout | onestep | 0.180 ± 0.031 | — |
| sufficient | rollout | C | 0.694 ± 0.025 | +0.514 ± 0.027 |
| sufficient | rollout | D | 0.700 ± 0.027 | +0.520 ± 0.032 |
| sufficient | rollout | E | 0.773 ± 0.016 | +0.593 ± 0.043 |
| sufficient | rollout | A | 0.780 ± 0.022 | +0.600 ± 0.043 |

Per horizon:

- `insufficient`: the direct path of the direct baseline is below the onestep rollout at k = 2..5 (k = 3: 0.208 vs 0.438) and they meet near k ≈ 8–10, where both approach saturation. C, D and A are worse than onestep at k = 1..3 and better from k ≈ 8 on. The onestep baseline reaches its best validation rollout at step 500–2500 in every seed.
- `sufficient`: the onestep rollout error is 3·10⁻⁴ at k = 1 and stays below 0.03 up to k = 10. Every multi-step or Anchor variant raises the rollout error at k = 1 by two orders of magnitude (C 0.067, D 0.099, E 0.305) and saturates by k ≈ 8.
- Off-attractor distance at k = 20, `sufficient` rollout: onestep 0.001, C 0.225, D 0.378, E 0.288, A 0.070. In `insufficient` it is 0.0001 for every variant.

Hypotheses:

- H1: confirmed in both modes.
- H2: not confirmed. D is not significantly better than onestep or C.
- H3: refuted. A and E are significantly worse on the direct path, and every Anchor variant is significantly worse on the rollout path in `sufficient`.

Problems with the protocol found when reading the result:

- The off-attractor guard cannot detect blurring in `insufficient`. The observation is one-dimensional, so the attractor's projection is an interval that contains the mean prediction.
- The λ grid was too narrow. Half of the choices are at its upper edge. In `sufficient`, even λ = 0.1 already harms C.
- Checkpoints and λ are selected by mean squared error averaged over k, while the primary metric is the median Euclidean error. At long horizons in chaos, MSE favours predicting the mean.

## Conclusion

S4 by the pre-registered criteria. On the Hénon map with raw observations, no Anchor variant in its current form improves the rollout without harming the well-specified case; the only significant improvement (A's rollout in `insufficient`) comes with significant harm in `sufficient`.

The result does not distinguish two causes of the harm:

- (a) the objective: point-wise MSE at long horizons in chaos pulls predictions towards the mean (certain for E, whose target is the blurred direct prediction);
- (b) the optimization: for C and D the exact map has zero loss, so the harm can come from gradients through up to 19 chaotic steps rather than from the objective.

Next experiments, each with its own README written before it runs: a test separating (a) from (b), and the linear-Gaussian system.
