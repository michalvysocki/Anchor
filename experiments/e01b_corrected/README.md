# e01b — e01 with the design errors corrected

Status: pre-registered 2026-11-11, not run (only 500-step smoke tests of the pipeline, before the statistical corrections). Decision criteria below were written before any training run of this experiment.

e01 is not modified; its result stands as recorded. This experiment repeats its comparison with the design errors found after it (e01 addendum, post-hoc analysis and the shelved limited-horizon review) corrected. It supersedes `experiments/shelved/limited_horizon`.

Code: `run.py` and `evaluate.py` in this directory, on the `anchor` package. Outputs go to `results/` and `models/` here.

## Design errors of e01 and their corrections

| e01 error | evidence | correction here |
|---|---|---|
| auxiliary horizons up to 20, beyond the predictable range (~10–13 steps at near-exact precision) | addendum: a near-exact model has rollout MSE 0.8 at k = 20 | auxiliary horizons k ≤ K_A = 4 |
| auxiliary gradients 10³–10⁶ × the base gradient, λ grid 0.1–3 never reached balance | addendum, gradient table | λ ∈ {0.001, 0.01, 0.1, 1}, spanning gradient ratios from below to above 1 at K_A = 4 (measured, see below) |
| primary metric (median error) does not match MSE training when the future is multimodal | post-hoc floor analysis: conditional median vs mean differ 5× at k = 1 | primary metric: test MSE, the statistic MSE training optimizes |
| metric averaged over saturated horizons | shelved review | primary metric averaged over k = 1..10; k = 1..20 reported as secondary |
| checkpoints and λ selected by a different statistic than the primary metric | e01 limitations | selection by validation MSE over k = 1..10, the primary statistic |
| off-attractor guard blind in 1D | e01 result (0.0001 for every variant in `insufficient`) | `insufficient`: MSE and sharpness compared with the conditional mean (see below); `sufficient`: sharpness and off-attractor guards |
| λ tuned separately per mode, which a plug-in method cannot do | shelved review | one λ per variant for both modes, chosen by a rule that uses both modes (see Protocol) |
| λ chosen on seed 0, which was then also reported | e01 limitations | λ chosen on selection seeds 0–2, results reported on disjoint evaluation seeds 3–7 |

## Design review (done before the training run)

- **Gradient normalization was considered and rejected.** Scaling the auxiliary gradient to a fixed fraction of the base gradient would make the auxiliary term vanish as the base loss converges. In `sufficient`, where the base gradient of a near-exact model is ~5·10⁻⁴, the variants would be harmless by construction and the main question would be answered by the protocol, not by the data.
- **The `insufficient` mode cannot show a real improvement of the rollout as a simulator.** With raw one-dimensional observations the model is a map f: ℝ → ℝ with no hidden state, so no objective makes f^k(x_t) approximate E[x_{t+k} | x_t] for all k. A lower MSE there can only come from moving the rollout toward the conditional mean. It is kept because e01 had it, and its result is read against the conditional mean (below).
- **K_A = 4 is fixed, not tuned.** At K_A = 4 a near-exact one-step model has rollout MSE ~2·10⁻⁵ (training windows), and in `insufficient` the direct path is far from saturation. Tuning K_A as well would double the sweep for little information.
- **Found in the smoke test: the sharpness guard contradicts MSE in `insufficient`.** Under MSE the best prediction is the conditional mean, which for a multimodal future has sharpness below 1. A variant moving toward it would be rejected as blurred. Therefore in `insufficient` sharpness is not a guard; MSE and sharpness are reported next to those of the conditional mean, estimated by nearest neighbours (200 neighbours, 1 156 932 windows from `REFERENCE_DATA_SEED`, as in the e01 floor analysis). In `sufficient` the future is deterministic, the conditional mean is the true point, and the sharpness guard stays.
- **Found in the smoke test: choosing λ in `insufficient` only would decide the `sufficient` result.** MSE in `insufficient` rewards the strongest pull toward the mean, so the selection picked λ = 1 for D and E, which then harms `sufficient` almost by construction. Therefore the sweep runs in both modes and λ is chosen by a rule that uses both (Protocol), and the whole λ trade-off is reported.
- **Found in the second smoke test: the admissibility rule compared E with the wrong baseline.** E belongs to the direct family but is selected on the rollout path; comparing it with onestep made it inadmissible for every λ, because the rollout of the direct family at λ = 0 is already much worse than onestep's in `sufficient`. That measured the family, not the auxiliary term. The rule now compares each variant with its own family at λ = 0.
- **Found in a statistical review: "not significantly worse" would not show harmlessness.** The onestep baseline in `sufficient` varies by about ±35% across seeds (e01 smoke and e01 results), so with 5 seeds almost no harm would reach significance and H1 would pass regardless of the data. Harmlessness is therefore an equivalence test: the upper end of the 95% interval of the paired relative MSE difference must be below +10%; a wider interval is "inconclusive", not "harmless".
- **Found in a statistical review: a single selection seed with a 5% threshold is below the noise.** In the smoke test C at λ = 0.1 appeared 45% better than its family in `sufficient`, which is noise around a near-exact model. λ is now chosen on the mean over 3 selection seeds of errors paired by seed with the same family at λ = 0 (the same seed gives the same initialization), with a 10% threshold.
- **Found in a statistical review: the selection seed was also reported.** Evaluation now uses seeds 3–7, disjoint from the selection seeds 0–2.
- **Found in the third smoke test: the "harmless" verdict compared E with onestep.** The rollout of the direct family at λ = 0 is about four times worse than onestep's in `sufficient`, so E would always be "worse" whatever its auxiliary term does. Harmlessness is now measured against the same family at λ = 0, like the admissibility rule; "better" in `insufficient` stays against onestep, the best alternative. The sharpness condition in `sufficient` is measured against the same family too, and is part of the verdict ("blurred").
- **A and B are kept as secondary.** They target the direct path; all decision criteria use C, D and E on the rollout path, so adding A and B does not add comparisons to the main conclusions. They are reported because B is the bootstrapped target of TD Networks for the direct prediction and A is composition consistency without grounding in a later observation, both useful comparisons for later work.
- **Remaining limitation:** the test windows are the same as in e01, and this design was made knowing the e01 test curves.

## Questions

**Q1 (main).** Is there a main variant (C, D, E) whose chosen λ reduces rollout MSE in `insufficient` while being harmless (equivalence test) in the well-specified case (`sufficient`)?

**Q2.** At its chosen λ, is D (whose loss is zero for the exact map) harmless in `sufficient`?

**Q3 (replication).** Does Q1 of e01 (direct beats rollout in `insufficient`, rollout beats direct in `sufficient`) hold under the MSE metric over k = 1..10?

## Setup

As e01 unless listed: Hénon map, burn-in 10, trajectories of 36 steps, K_MAX = 20 for evaluation, same train / validation / test generators, model F(z, k) (MLP 2×256 tanh, one-hot k) for every variant, Adam lr 1e-3, 40 000 steps, batches of 512 windows, gradient clipping 1.0. Selection seeds 0–2, evaluation seeds 3–7 (training seeds set initialization and sampling; data are identical across seeds).

Variants as in e01 (direct, A, B, E on the direct family; onestep, C, D on the one-step family), all auxiliary losses with k ~ U{2..4}. Main variants: C, D, E. Secondary: A, B.

## Gradient scales measured before training

Ratio of auxiliary to base gradient norm at K_A = 4 (batch of 512 training windows):

| mode | state | C | D | E |
|---|---|---|---|---|
| insufficient | init | 0.88 | 0.016 | 0.28 |
| insufficient | trained | 4.4 | 3.1 | 64 |
| sufficient | init | 0.72 | 0.005 | 0.17 |
| sufficient | trained | 74 | 95 | 5.0·10³ |

With λ ∈ {0.001, 0.01, 0.1, 1}, the weighted ratio for C and D spans roughly 10⁻³ to 10². The smoke test (200 steps, λ = 1, `insufficient`) gave ratios 2.4 (C), 6.1 (D), 0.20 (A), 1.2 (B), 1.5 (E).

## Protocol

1. Gradient check: 2 000 steps per variant at λ = 1 in `insufficient` with gradient logging.
2. Baselines (direct, onestep): 8 seeds (selection and evaluation) × 2 modes.
3. Sweep: each Anchor variant × 4 λ × 2 modes × 3 selection seeds.
4. λ for the final runs, per variant, from validation MSE over k = 1..10 of the variant's selection path (direct for A and B, rollout for C, D and E). For each λ and mode, the relative error is the ratio to the same family at λ = 0 with the same seed (the direct baseline for A, B and E, the onestep baseline for C and D), averaged over the selection seeds. A λ is admissible if its relative error in `sufficient` is at most 1.10; among admissible λ, the one with the lowest relative error in `insufficient`; if none is admissible, the smallest λ. Comparing with the same family isolates the effect of the auxiliary term; the test-set comparison stays against direct (direct path) and onestep (rollout path), as in e01.
5. Final: the chosen λ, evaluation seeds 3–7, both modes.

Runs: 32 baselines + 120 sweep + 50 final = 202 (plus 5 short gradient checks): 84 in the one-step family (CPU) and 118 in the direct family (MPS). Each family's final stage needs only its own baseline and sweep.

```
uv run python experiments/e01b_corrected/run.py check --family onestep
uv run python experiments/e01b_corrected/run.py baseline --family onestep
uv run python experiments/e01b_corrected/run.py sweep --family onestep

uv run python experiments/e01b_corrected/run.py check --family direct --device mps
uv run python experiments/e01b_corrected/run.py baseline --family direct --device mps
uv run python experiments/e01b_corrected/run.py sweep --family direct --device mps

uv run python experiments/e01b_corrected/run.py final --family onestep
uv run python experiments/e01b_corrected/run.py final --family direct --device mps

uv run python experiments/e01b_corrected/evaluate.py
```

## Metrics (test set)

- Primary: MSE per k averaged over k = 1..10, per seed and path.
- Comparison: paired absolute and relative difference to the baseline of the same seed and path (direct for the direct path, onestep for the rollout path), mean and 95% t-interval over the 5 evaluation seeds.
- "Better" (`insufficient`): the interval of the absolute difference lies below 0.
- Verdict in `sufficient`, measured against the same family at λ = 0 on the rollout path (onestep for C and D, the rollout of the direct baseline for E), which isolates the auxiliary term: "blurred" if the interval of the paired sharpness difference at k = 10 lies below 0; otherwise "harmless" if the upper end of the interval of the relative MSE difference is below +0.10; otherwise "worse" if the interval of the absolute MSE difference lies above 0; otherwise "inconclusive".
- `insufficient` reference: MSE and sharpness of the conditional mean (nearest-neighbour estimate).
- `sufficient` guards: sharpness at k = 10 (part of the verdict, see below), off-attractor distance at k = 10.
- Both modes: fraction of diverged rollouts.
- Secondary: MSE over k = 1..20, the e01 median metric, the λ trade-off table (validation errors of every λ in both modes relative to the same family at λ = 0, averaged over the selection seeds), and the results of A and B.

## Hypotheses

- H1: for C and D at the chosen λ, the rollout is harmless in `sufficient` (equivalence test).
- H2: in `insufficient`, any variant with significantly lower rollout MSE than onestep has a rollout sharpness at k = 10 closer to the conditional mean's than onestep has (it improves by moving toward the conditional mean, as the structural argument says).
- H3: Q3 replicates.

## Decision criteria

| scenario | result | consequence for the method |
|---|---|---|
| R1 | some main variant at its chosen λ is "better" in `insufficient` and "harmless" in `sufficient` (which includes not being "blurred") | path consistency with short horizons and a suitable λ can be safe in the well-specified case; horizon limits and the λ range carry over to e02 |
| R2 | no λ is admissible for any main variant (every λ harms `sufficient` on validation) | path losses harm a good model even when short and weak; a mechanism that switches off (adaptive λ) is necessary, not optional |
| R3 | admissible λ exist, but at them no main variant is both "better" and "harmless" | the gain and the harm come together, or the evidence cannot separate them; λ cannot be fixed across specifications; same consequence as R2 for adaptive λ |
| R4 | a variant lowers `insufficient` MSE below the conditional-mean floor, or improves while moving away from it | contradicts the structural argument; check the code and data before interpreting |

If the `sufficient` verdicts are mostly "inconclusive", the result is reported as such: the experiment then shows that 5 seeds cannot resolve harm of this size, which is itself a reason to measure harmlessness in e02, where the baseline varies less.

The result is recorded as it comes out and is not reinterpreted later.

## Known limitations

- Same test windows as e01; the design was made knowing the e01 test curves and after two 500-step smoke tests of this pipeline.
- The admissibility threshold (1.10) and the equivalence margin (+10%) are judgements, fixed here before training.
- 3 selection seeds still leave noise in the choice of λ; the λ trade-off table shows how close the alternatives were.
- A fixed λ cannot keep the gradients balanced through all of training.
- One environment, raw observations, no actions, no latent.

## Predictions recorded before results (2026-11-11, training running)

Written while the training runs, before any result. They come from reasoning and from the direction of the 500-step smoke tests, which are not results.

- Q3 replicates (≈90%).
- C and D: "harmless" in `sufficient` at λ 0.001–0.01 (≈65%), harm at λ 0.1–1, stronger for D; "not better" in `insufficient` at admissible λ.
- E: the most uncertain variant; it may reach R1 (in the smoke test at λ = 0.01 it improved its own family's rollout by ~15% in both modes).
- Scenarios: R1 ≈25% (mostly through E), R2 ≈15%, R3 ≈50%, R4 ≈5%, mostly "inconclusive" ≈5%.

How R1 through E is to be read, fixed before the result:

- "Harmless" for E is relative to the rollout of the direct family, which in `sufficient` is about four times worse than onestep's. R1 through E would show that the E term improves and does not harm the rollout of a model trained for direct prediction, not that it gives a rollout as good as the best one-step model in every case.
- A gain in `insufficient` counts as the intended effect only if E's rollout sharpness stays near the conditional mean's (H2); a gain with sharpness far below it is a pull toward an averaged forecast (R4 territory), not a better simulator.
- E needs a direct head trained alongside, so it is not plug-in. A positive E result is a signal to test a direct-head target for the latent in e02, not evidence that Anchor works.

## Result

Evaluated 2026-11-12 (`evaluate.py`; output in `results/test_results.json`, `results/e01b_comparison.png`, printed summary in the local `logs/e01b_evaluate.log`). All 202 runs are present (checked against the protocol in `run.py`), every run reached 40 000 steps, no validation value is non-finite and no rollout diverged at k = 10. The only warnings are 12 overflows in `envs/henon.py` during trajectory generation, where unstable trajectories are discarded.

Selected λ (selection seeds 0–2, validation): A 0.001, B 0.01, C 0.01, D 0.01, E 0.1.

Rollout MSE over k = 1..10, relative paired difference to onestep (95% t-interval over evaluation seeds 3–7) and verdict:

| variant | λ | insufficient | verdict | sufficient | verdict |
|---|---|---|---|---|---|
| C | 0.01 | +0.5% ± 1.0 | not better | −25% ± 24 | blurred |
| D | 0.01 | −2.3% ± 2.0 | better | −12% ± 31 | inconclusive |
| E | 0.1 | −14.8% ± 2.0 | better | (vs direct family) MSE −0.528 ± 0.115 relative, sharpness +0.045 ± 0.052 | harmless |

Absolute rollout MSE (k ≤ 10): onestep 0.547 / 0.0004, C 0.549 / 0.0003, D 0.534 / 0.0003, E 0.466 / 0.192, rollout of the direct model 0.532 / 0.406 (insufficient / sufficient). The conditional-mean floor in insufficient is 0.2325; no variant is below it (best direct path 0.277, best rollout 0.466).

Direct path, relative to direct: A and B within ±1% in both modes; E +6.5% ± 2.9 (insufficient) and +73% ± 19 (sufficient).

Rollout sharpness at k = 10 in insufficient: floor 0.575, onestep 0.727, D 0.758, C 0.806, E 0.496 ± 0.228. E's sharpness is flat at about 0.5 from k = 2 to 20, while the conditional mean's falls from 0.96 to 0.29; at k = 1–2 E's rollout MSE is above onestep's.

λ trade-off on validation (relative to the same family at λ = 0, insufficient / sufficient): C at λ = 1: 0.91 / 10.6; D at λ = 0.1: 0.91 / 7.0, at λ = 1: 0.82 / 75.4; E at λ = 1: 0.88 / 0.40.

Against the pre-registered criteria and predictions:

- **Q3 / H3 replicates** (direct beats rollout in insufficient, rollout beats direct in sufficient). Predicted ≈90%.
- **Q1: formally R1 through E**, as predicted ("R1 ≈25%, mostly through E", with the caveat that E is harmless only relative to the direct family). Assessment: **R1 formally, R3 in practice.** E is about 500× worse than onestep in sufficient, its rollout is uniformly damped rather than closer to the conditional mean, and it worsens its own direct path in both modes.
- **Q2 / H1 not confirmed:** C is "blurred", D "inconclusive". The "blurred" verdict is an artefact of the pre-registered rule: sharpness 1.001 vs 1.001 (difference −0.001 ± 0.000) is significant only because the interval around a near-exact model is extremely narrow; the rule had no practical threshold. The equivalence test with a +10% relative margin cannot pass when the baseline MSE is ~4·10⁻⁴ (relative intervals ±24–31%). Reported as pre-registered; both are protocol lessons, not evidence of harm.
- **Prediction for C and D** ("not better" in insufficient at admissible λ): holds for C; D is formally "better" with an interval that almost touches zero.
- **H2:** E's sharpness at k = 10 is closer to the floor than onestep's, so H2 passes at the pre-registered horizon, but over the whole curve E does not follow the conditional mean (see above). D improves while its sharpness moves away from the floor (the R4 clause), but the sharpness difference is not significant (+0.031 ± 0.054).
- **Theory-based prediction from AR(1)** (order by bias E ≥ C > D, recorded in the research note before the evaluation): partly wrong. E is best, but D beats C at every λ on validation. The variance caveat recorded with the prediction fits (on chaotic dynamics the variance of the Monte Carlo target grows roughly as e^{2·0.42k}), but this is not tested here.
- **Compared with e01:** at short horizons and λ ≤ 0.01, C and D leave the near-exact one-step model intact in sufficient (MSE 0.0003 vs 0.0004), while in e01 every variant raised the error at k = 1 from 0.0003 to 0.07–0.3. This supports the e01 diagnosis (horizons beyond predictability and unbalanced gradients), not a failure of path consistency itself.

## Conclusion

Written 2026-11-12.

1. Path consistency with short auxiliary horizons and a small λ is not inherently destructive: C and D leave the near-exact one-step model intact in the well-specified case, which confirms that the failure of e01 came from its configuration. A gain in the misspecified case exists, but only at λ values that harm the well-specified case; at admissible λ it is negligible (C +0.5%, D −2.3%). D beating C at every λ is consistent with a bias–variance trade-off of Monte Carlo targets on chaotic dynamics; this is a prediction for e02, not a result of e01b. E's gain, together with the worse direct path and the uniformly damped rollout, argues against sharing one model between the direct head and the dynamics.
2. No fixed λ serves both observability regimes, which argues for an adaptive Anchor whose strength depends on the regime (λ(Δ)).
3. Carried to e02: short horizons and a gradient-scale check; the prediction C ≥ D on a stable linear system; equivalence tested on log-MSE or with an absolute margin; a practical threshold for sharpness verdicts; a separate head or no head instead of a shared model.

## Exploratory analysis (after the conclusion, not pre-registered)

Question: does the path disagreement of the baseline models (λ = 0, evaluation seeds) separate the two observability modes, and does it track how much the rollout loses to the direct prediction? Script `explore_disagreement.py`, output `results/explore_disagreement.json`, run 2026-11-12. All measures are on the test windows at k = 2..4, averaged over the splits k = p + q and divided by the variance of the true point at horizon k (mean ± 95% t-interval over 5 seeds).

| measure | k | insufficient | sufficient | ratio |
|---|---|---|---|---|
| bootstrap residual of the one-step model, $\|g^k(z_t) - g^q(z_{t+p})\|^2$ (D loss) | 2 / 4 | 0.31 / 0.98 | 1.4·10⁻⁶ / 1.4·10⁻⁵ | 2·10⁵ / 7·10⁴ |
| direct vs rollout, $\|F(z_t,k) - g^k(z_t)\|^2$ | 2 / 4 | 0.12 / 0.73 | 0.005 / 0.009 | 25 / 80 |
| rollout gap, MSE of $g^k$ minus MSE of $F$ | 2 / 4 | +0.11 / +0.65 | −0.005 / −0.009 | — |
| composition of the direct model, $\|F(F(z_t,p),q) - F(z_t,k)\|^2$ (A loss) | 2 / 4 | 0.12 / 0.50 | 0.13 / 0.15 | 1.0 / 3.4 |

Observations:

- The bootstrap residual, which needs only the trained one-step model and the data, separates the modes by 4–5 orders of magnitude. Caveat: it is the D loss, i.e. the variance of the target plus the squared Markov-gap bias. Hénon has no process noise, so in the sufficient mode the variance part is zero; with process noise the residual stays positive for a Markov latent, and the bias part (the disagreement Δ of the theory) has to be separated from it. This separation is the open part of T4.
- The direct-vs-rollout disagreement has almost the magnitude of the rollout gap (0.12 vs 0.11, 0.55 vs 0.50, 0.73 vs 0.65), but not its sign: in the sufficient mode the rollout is the better path. This is largely mechanical: when one predictor is near-optimal, the squared disagreement approximates the excess error of the other.
- The composition inconsistency of the direct model does not separate the modes at k = 2 (0.12 vs 0.13). Self-consistency without grounding in a later observation does not measure the Markov gap; it reflects the direct model's own error. This supports A as the ablation of grounding.
- Limitation: the two modes are extreme (an exact Markov state without noise vs a strongly non-Markov observation). Whether the disagreement tracks the size of the gap in between is a question for e02, where κ can be varied continuously.

**Correction (2026-11-12, from Stage 1 theory T4).** For a linear one-step model the mean of the bootstrap residual given $z_t$ is $g^q\big(g^p(z_t) - \mathbb E[z_{t+p} \mid z_t]\big)$, the rollout bias at horizon $p$ carried forward by $g^q$. For $p = 1$ and a one-step model at its optimum it is zero. In the linear analogue, the residual at k = 2 (only $p = 1$) is therefore pure target variance, not the Markov gap: the separation of the modes above comes from the sufficient mode being deterministic, not from detecting the gap. The gap can appear only for $p \ge 2$ (k ≥ 3). Hénon is nonlinear, so this is a strong indication, not a proof. The observation that the direct-vs-rollout disagreement tracks the rollout gap fits the same lemma: with $p = k$ (the Monte Carlo target) the conditional mean of the residual is exactly the rollout bias at k.
