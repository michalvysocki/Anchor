# Shelved — Anchor within the predictable horizon (formerly e02)

Status: shelved 2026-11-11, not run (only a 500-step smoke test of the pipeline). Decision criteria below were written before any training run of this experiment.

Reasons for shelving, found in a design review after the code was written:

1. The primary metric averages the median error over k = 1..20. In `insufficient` the rollout saturates by k ≈ 3–5, so most of the average comes from horizons where a prediction of the mean has a lower point error than a sharp prediction. Hyperparameter selection uses the same metric, so it would favour blurred settings; the sharpness guard only acts afterwards. In the smoke test the variant with the lowest error had sharpness 0.005.
2. With raw one-dimensional observations the model is a map f: ℝ → ℝ with no room for hidden state, so no objective can make f^k(x_t) approximate E[x_{t+k} | x_t] for every k. Path consistency can only move error between horizons. The thesis is about shaping a latent towards a Markov state, which needs a model with a latent.

The code (horizon limit, median selection, sharpness, gradient logging) is kept for later experiments.

Code: `run_limited_horizon.py` (training stages, settings, selection) and `evaluate_limited_horizon.py` in this directory, on top of `src/anchor_loss.py` (`max_horizon`, median validation errors), `src/train_variants.py` (`max_horizon`, `select_metric`, output directories, gradient logging) and `src/evaluate_variants.py` (`sharpness`). The defaults of the changed functions reproduce e01. Outputs go to `models/shelved_limited_horizon/` and `results/shelved_limited_horizon/`.

Run from the repository root, one terminal per family:

```
uv run python experiments/shelved/limited_horizon/run_limited_horizon.py check --family onestep
uv run python experiments/shelved/limited_horizon/run_limited_horizon.py baseline --family onestep
uv run python experiments/shelved/limited_horizon/run_limited_horizon.py sweep --family onestep
uv run python experiments/shelved/limited_horizon/run_limited_horizon.py final --family onestep

uv run python experiments/shelved/limited_horizon/run_limited_horizon.py check --family direct --device mps
uv run python experiments/shelved/limited_horizon/run_limited_horizon.py sweep --family direct --device mps
uv run python experiments/shelved/limited_horizon/run_limited_horizon.py final --family direct --device mps

uv run python experiments/shelved/limited_horizon/evaluate_limited_horizon.py
```

## Motivation

The e01 addendum found that e01 did not test path consistency as such. Its auxiliary losses used horizons up to 20, beyond the horizon where even a near-exact one-step model is saturated (~10–13 steps on the training windows). Their gradients were 10³–10⁶ times larger than the one-step gradient, so they dominated training for every λ in the grid. This experiment repeats the rollout-path comparison with both problems removed.

## Question

**Q.** When the auxiliary loss only uses horizons inside the predictable range and λ covers the range where the two gradients are of comparable size, does an Anchor objective reduce rollout error under Markov misspecification without harming the well-specified case?

## Setup

Same as e01 unless listed here: Hénon map, burn-in 10, trajectories of 36 steps, K_MAX = 20 for evaluation, the same train / validation / test generators, model F(z, k) MLP 2×256 tanh with one-hot k, Adam lr 1e-3, 40 000 steps, batches of 512 windows, gradient clipping at 1.0, 5 training seeds.

## Variants

All are evaluated on the rollout path F(·, 1)^k.

| variant | loss |
|---|---|
| onestep | F(z_t, 1) vs true z_{t+1} (baseline) |
| C | onestep + λ‖F(·,1)^k(z_t) − z_{t+k}^true‖², k ≤ K_A (control) |
| D | onestep + λ‖F(·,1)^k(z_t) − sg F(·,1)^q(z_{t+p}^true)‖², k ≤ K_A (plug-in Anchor) |
| E | direct grounding + λ‖F(·,1)^k(z_t) − sg F(z_t, k)‖², k ≤ K_A |

k ~ U{2..K_A}, p ~ U{1..k−1}, q = k − p. A and B are dropped: they target the direct path, and e01 showed nothing on the rollout path that is not covered here.

E is kept because it is the only variant whose target carries information the one-step model lacks: in `insufficient`, the direct path beats the rollout (e01, Q1).

## Gradient scales measured before training

Ratio of the auxiliary gradient norm to the base gradient norm (one-step loss for C and D, grounding loss for E), batch of 512 training windows. "init" is a fresh model (seed 0), "trained" is the e01 seed-0 checkpoint of the baseline of the same family.

| mode | K_A | state | C | D | E |
|---|---|---|---|---|---|
| insufficient | 4 | init | 0.88 | 0.016 | 0.28 |
| insufficient | 4 | trained | 4.4 | 3.1 | 64 |
| insufficient | 8 | init | 0.88 | 0.009 | 0.24 |
| insufficient | 8 | trained | 30 | 16 | 1.2·10³ |
| sufficient | 4 | init | 0.72 | 0.005 | 0.17 |
| sufficient | 4 | trained | 74 | 95 | 5.0·10³ |
| sufficient | 8 | init | 0.62 | 0.003 | 0.17 |
| sufficient | 8 | trained | 1.4·10³ | 1.0·10³ | 3.2·10⁶ |
| (e01) | 20 | trained | 3.1·10³ – 3.6·10⁶ | 1.6·10³ – 2.8·10⁶ | 3.2·10⁸ – 3.1·10¹⁷ |

The ratio grows by orders of magnitude during training, so no fixed λ keeps the terms balanced throughout. The λ grid below spans the balanced range for K_A ≤ 8 in `insufficient`. Dynamic normalization is left for a later experiment, if this one shows a signal.

## Hyperparameters

- K_A ∈ {4, 8}: both below the horizon where the near-exact model saturates on the training windows.
- λ ∈ {0.001, 0.01, 0.1, 1}.
- (K_A, λ) is chosen per variant on seed 0 in `insufficient` only, by the validation metric below.
- **The same (K_A, λ) is then used in `sufficient`.** A plug-in method cannot know whether the observation is Markov, so it must not be tuned separately per mode.

## Changes to the e01 protocol

1. Auxiliary horizon limited to K_A by a parameter, not by cutting windows (E reads the one-hot size from the window length).
2. Checkpoints and hyperparameters are selected by the validation median Euclidean rollout error averaged over k = 1..20, the same statistic as the primary test metric, instead of MSE.
3. A blurring guard that works in one dimension: sharpness ratio s_k = std of predictions / std of true points at horizon k, over test windows and coordinates. A prediction of the mean has s_k → 0. The off-attractor distance is still reported for `sufficient`.
4. The onestep baseline is retrained with the new selection rule (5 seeds × 2 modes).
5. Before the long run, one short run per variant (2 000 steps) logs the gradient norms of both terms, to confirm the measurement above on the training path.

Runs: sweep 3 variants × 8 settings (insufficient, seed 0), final 3 × 4 seeds (insufficient), 3 × 5 seeds (sufficient), baseline 10. 61 runs in total.

## Metrics (test set)

- Primary: median Euclidean rollout error per k, averaged over k = 1..20, per seed.
- Comparison: paired difference to onestep of the same seed, mean and 95% t-interval over 5 seeds. Significant when the interval excludes 0.
- Guards: sharpness ratio at k = 20 (both modes), off-attractor distance at k = 20 (`sufficient`), fraction of diverged rollouts.

## Hypotheses

- H1: in `insufficient`, at least one of C, D, E has significantly lower rollout error than onestep.
- H2: in `insufficient`, D has significantly lower rollout error than C.
- H3: in `sufficient`, with the hyperparameters chosen in `insufficient`, no variant is significantly worse than onestep.

## Decision criteria

A variant counts as an improvement only if it is significantly better than onestep in `insufficient`, is not significantly worse in `sufficient`, and its sharpness ratio at k = 20 is not significantly lower than onestep's in either mode.

| scenario | result | next research step |
|---|---|---|
| S1 | D or E is an improvement and significantly better than C | path consistency works inside the predictable horizon; build the `AnchorLoss` module, then linear-Gaussian and a learned latent; method paper |
| S2 | C is an improvement, D and E are not significantly better than C | a horizon-limited multi-step loss is what helps; an Anchor target adds nothing on raw observations; the learned latent is the decisive test |
| S3 | some variant is better in `insufficient` but fails the `sufficient` or sharpness condition | path consistency needs a mechanism that turns itself off when the model is well specified; analysis-paper direction |
| S4 | no variant is better in `insufficient` | limiting the horizon does not rescue path consistency on raw Hénon observations; analysis paper, the linear-Gaussian experiment decides |

The result is recorded as it comes out and is not reinterpreted later. A new idea that follows from it is a new hypothesis with its own README.

## Known limitations

- The test set is the same as in e01, whose per-horizon test results were seen before this design. K_A is justified by training-window measurements, but the choice of {4, 8} was made knowing the e01 curves.
- (K_A, λ) is chosen on seed 0, which is also reported.
- A fixed λ cannot keep the gradients balanced through training (see the table above).
- One environment, raw observations, no actions.

## Result

Not available yet.

## Conclusion

Not available yet.
