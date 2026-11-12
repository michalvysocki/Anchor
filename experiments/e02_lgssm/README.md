# e02 — Anchor on linear-Gaussian systems with a history window

Pre-registered 2026-11-12, before any training. The population predictions below were computed before the training code existed (`predictions.py`, output in `results/predictions.txt` and `results/predictions.json`).

## Purpose

Test the Stage 1 theory where the optimum is known exactly (Kalman filter) and every theory quantity can be computed in closed form:

- **T1:** for a Markov latent the anchor term does not move the model.
- **T2:** under a Markov gap a small anchor term strictly improves the rollout.
- **Residual lemma and critic:** the critic measures the rollout bias without the target variance.

It also tests the method chosen on 2026-11-12: **Anchor as a mixture of targets over anchor points with bias–variance weights (M\*)**.

## Setup

**Systems.** Scalar states observed with noise, $y_t = x_t + v_t$, all stationary and zero-mean:
- AR(1) with $a \in \{0.5, 0.9, 0.99\}$;
- AR(2) with complex roots of radius 0.9 and angle π/6 (a damped oscillation).

The signal fraction $\kappa = s/(s+r) \in \{1, 0.8, 0.5\}$ sets the observation noise ($\kappa = 1$: none). That gives 12 systems; $a = 0.99$ is the near-unstable arm.

**Latent.** The window of the last $L \in \{1, 2, 4, 8\}$ observations, a fixed linear history encoder. It is Markov iff $\kappa = 1$ and $L$ covers the AR order (15 of the 48 configurations). Otherwise the Markov gap shrinks as $L$ grows.

**Model.** A linear one-step predictor $y_{t+1} = \theta \cdot z_t$; its rollout shifts the window (companion matrix). There is no encoder to learn, so collapse and latent impoverishment are not tested here (left for e03).

**Variants.** All share the one-step loss; the auxiliary horizons are $k \in \{3, 4\}$, each sampled uniformly; anchor points $p \ge 2$.

| variant | auxiliary target for the rollout to $k$ |
|---|---|
| onestep | none (λ = 0) |
| C | true $y_{t+k}$ (anchor $p = k$, Monte Carlo; the TD-MPC/SPR loss) |
| D | $\mathrm{sg}[g^{k-p}(z_{t+p})]$, $p$ uniform on $2..k-1$ (bootstrap) |
| M | $\mathrm{sg}[\sum_{p=2}^{k} w_p T_p]$ with equal weights (mixture) |
| M\* | the same with weights minimizing $w^\top(B + V/N)w$ on the simplex: $B$ from the difference of two critics (regression of $T_p$ and of $y_{t+k}$ on $z_t$), $V$ the covariance of their residuals, $N$ the number of training windows (overlapping windows overstate the effective sample size, which shifts the weights toward C, the cautious direction); recomputed every 100 steps on the training data, under stop-gradient |

References (not trained):
- direct: the best linear prediction of $y_{t+k}$ from the window;
- Kalman: the best prediction from the whole past.

**Data.** Exact stationary trajectories of 64 observations.
- **Large arm:** 2000 trajectories, about 100k windows. All 48 configurations.
- **Small arm:** 20 trajectories, about 1000 windows. Only $a \in \{0.9, 0.99\}$ and AR(2), with $\kappa \in \{1, 0.5\}$ and $L \in \{1, 2\}$ (12 configurations).

**Every seed draws its own training set** (finding 1 of the design review): with a linear model and a fixed training set, all seeds would converge to almost the same θ, and the intervals would miss the variability that matters, the draw of the data. The validation set is shared and used only to select λ. There is no test set (see Metrics).

**Training.** Data in units of the true $\sqrt{\mathrm{Var}(y)}$; $\theta$ initialized at 0, as in practice; Adam, learning rate $10^{-2}$ with cosine decay to 0 over 3000 steps of 256 windows (the decay was added after the smoke test: with a constant step the trained models jittered up to 1.5% above the predicted fixed point; with the decay they are within 0.2%); λ ∈ {0.01, 0.1, 1, 10}. All λ and seeds of one configuration and variant are trained together as a batch of independent models. The checkpoint is the last one: linear least-squares-type objectives have no overfitting regime that would need selection. The smoke test checks that 3000 steps reach the population fixed point within 1% on the large arm. Because θ = 0 lies near the degenerate fixed point found in T2 ($b = 0$), training may converge to a different branch than the one predicted (tracked from the one-step optimum): the distance of every trained θ to the predicted fixed point is reported, and convergence to another branch is a separate finding, not a failure of the theory (finding 2).

**Seeds.** Selection seeds 0–2, evaluation seeds 3–32 (30 seeds: runs are cheap and the small arm varies with the data draw; finding 6). Disjoint, as in e01b.

**Secondary ablations** (added 2026-11-12 before the full runs; outside the decision criteria H1–H5, like A and B in e01b). Each is compared with the variant it modifies:

| ablation | modifies | what it isolates |
|---|---|---|
| M\*_oracle | M\* | weights from the true moments instead of estimates from the training data: separates "the rule is wrong" from "the weights are badly estimated" |
| M\*_p1 | M\* | anchor points from $p = 1$: tests the Remark that $p = 1$ carries no signal at the one-step optimum |
| D_nosg | D | no stop-gradient on the bootstrap target: T1 and the lemma assume it |

A and B are not applicable here: for a one-step model, composition consistency (A) holds by definition ($g^q(g^p(z)) = g^k(z)$), and there is no direct head (B). They return in e03.

The no-stop-gradient ablation was first planned on M\*. The population predictions showed that without stop-gradient the coupled fixed point of the model and of M\*'s weights often does not exist (48 of 96 configuration–arm pairs), and a smoke run showed that the outcome then depends on where the weights settle, not on the gradient. It was moved to D, where the effect of the stop-gradient is isolated and every fixed point exists.

## Questions

**Q1 (T2, main).** In configurations with a gap (population onestep → direct improvement ≥ 2%, the "gap configurations"), does each variant at its selected λ lower the rollout MSE relative to onestep, and does the size agree with the population prediction?

**Q2 (T1).** In the 15 Markov configurations, is every variant harmless at every λ?

**Q3 (T1 boundary, the e01b harm).** With little data (an inexact model), does a fixed λ harm in the Markov configurations, and more so near instability ($a = 0.99$)?

**Q4 (method).** Does M\* match the better of C and D in both data arms, i.e. recover C where bias dominates and gain where the target variance matters?

**Q5 (lemma, diagnostic).** Does the critic, estimated from training data at the trained onestep model, track the true rollout bias across configurations, while the raw residual does not?

## Metrics

- **Rollout MSE** at $k = 1..10$ in units of $\mathrm{Var}(y)$, computed **exactly** from the known covariance for every trained θ (`linear.rollout_mse`), so no test sample adds noise and all spread comes from the training data and the optimization (finding 3). The primary statistic is its mean over $k$.
- **Comparison:** the paired difference of the log of the primary statistic to onestep of the same seed (a ratio, robust near zero, the e01b lesson). Mean and 95% t-interval over 30 evaluation seeds (t = 2.045).
- **Fraction of the gap closed:** (onestep − variant) / (onestep − direct), all population or all empirical; only in gap configurations, where the denominator is not near zero (finding 8).
- **Distance to the predicted fixed point:** $\|\hat\theta - \theta^\star\|$ for every trained model.
- **Critic and residual of D's targets** at the trained onestep model, from regressions on the training windows.
- **Gradient scales:** the norm of the auxiliary gradient relative to the one-step gradient at the end of training; reported, not used in a decision. (At θ = 0 the auxiliary gradient is exactly zero, because the rollout rows of $k \ge 2$ and their derivatives vanish there, so the ratio at initialization carries no information. Smoke test at $a = 0.99$, κ = 0.8, $L = 2$: the ratio at the end is about λ, no imbalance as in e01.)

**λ selection.** For each configuration and variant, the λ with the lowest mean validation rollout MSE (k = 1..10) over the selection seeds, on a validation sample (selecting on the population would be an oracle).

## Hypotheses and decision criteria

| | prediction (population, `results/predictions.txt`) | criterion on the test set |
|---|---|---|
| H1 (Q1) | in the gap configurations C, M, M\* close 0–90% of the gap (almost nothing in AR(2) with κ = 0.5, L = 1); D closes less | assessed only where the predicted improvement at the selected λ is at least 1% (finding 4; the other gap configurations are reported descriptively): **agreement with theory** (large arm), the predicted log ratio lies within the CI or within 0.01 of it, and **"better"**, the upper end of the CI of the log ratio < 0. H1 holds if both hold in at least 80% of these configurations for each of C, M, M\* |
| H2 (Q2) | exactly zero change at every λ for every variant | verdicts per configuration, variant and λ with the margin δ = log 1.02 (see "Verdicts with a margin" below). H2 holds if no test is "material harm"; a single "inconclusive" or "small increase" is not harm (finding 9) |
| H3 (Q3) | the population says no harm; with little data a small harm is possible, larger at $a = 0.99$ | the same verdicts in the small arm (finding 5). "Better" is possible: with finite data the lemma holds only approximately and extra multi-step targets use more of the data; it would be a new effect, not a confirmation of T1 |
| H4 (Q4) | large arm: C ≥ M\* ≥ M ≥ D, with M\* within 1% of C; small arm: open (T3) | "M\* matches the better": in every non-Markov configuration of an arm, the upper end of the CI of the paired log(M\* / better of C and D, each at its selected λ) is < log 1.02; H4 is assessed per arm |
| H5 (Q5) | critic ≈ 0 in Markov configurations and ordered with the gap; the residual stays at 0.03–0.6 regardless | Spearman correlation over the 33 non-Markov large-arm configurations between the estimated critic and the population onestep − direct gap (absolute, in units of Var(y), the scale of the critic) ≥ 0.8; in every Markov configuration critic < 10⁻³ while residual > 10⁻² |

**Verdicts with a margin** (smoke-test finding, 2026-11-12, see Design review). For the log ratio to onestep with margin δ = log 1.02:

| verdict | condition on the 95% CI |
|---|---|
| better | upper end < 0 |
| material harm | lower end > δ |
| small increase | lower end > 0, but not material harm (also when the upper end < δ) |
| harmless | upper end < δ, lower end ≤ 0, not better |
| inconclusive | otherwise |

Both "material harm" and "small increase" counts are reported.

**Outcomes:**

| | outcome | meaning |
|---|---|---|
| R1 | H1, H2 and H4 hold | the theory predicts the linear case and the mixture is a safe default: proceed to e03 with M\* as the method |
| R2 | H1 and H2 hold, H4 fails in the small arm (M\* worse than the better of C, D) | the bias–variance rule is wrong or badly estimated; fix the weight rule before e03 |
| R3 | H2 fails at large data | contradicts T1: check the code and the estimation before interpreting |
| R4 | H1 fails (no improvement where the population predicts one) | the population prediction does not transfer to training: check optimization (convergence, λ range) before interpreting |

## Design review (before any training code)

Nine findings, all adopted (2026-11-12):

1. **Seeds did not resample the data.** With a linear model all seeds would converge to almost the same θ, so the intervals would be nearly zero and miss the data-draw variability; now every seed draws its own training set.
2. **Initialization at 0 is near the degenerate fixed point** of T2; the branch reached by training is checked and reported.
3. **Exact evaluation** from the known covariance instead of a test sample.
4. **Agreement with theory is trivial for tiny effects;** H1 is assessed only where the predicted improvement is at least 1%.
5. **H3 allows "better"** with little data.
6. **30 evaluation seeds** instead of 10.
7. **The effective sample size of M\* is overstated** by overlapping windows; it errs toward C.
8. **The fraction of the gap closed** is computed only in gap configurations.
9. **Multiple tests in H2:** a single "inconclusive" is not harm.

**Final check before training** (2026-11-12): the training losses of `run.py` equal the library losses of `linear.py` evaluated on the empirical second moment of the same windows (C, D, M, M\*, and the D gradient under stop-gradient, to 12 digits); every seed's observations are shared across window lengths; the counts of Markov configurations were corrected (15, not 13: AR(1) with κ = 1 is Markov at every $L$); H4 now uses the paired per-seed CI that the criterion states, not the difference of means. Second check (after adding the ablations): H5 now uses the absolute gap (onestep − direct in units of Var(y)) as the criterion states, not the relative one; H2 and H3 count only the main variants, so the ablations cannot enter the decision criteria.

The questions, systems, variants, thresholds and population predictions are unchanged.

**Smoke-test findings** (three configurations, 2026-11-12):

10. **Constant Adam step left a jitter** of up to 1.5% around the fixed point; cosine decay brought trained models within 0.2% of the predicted population fixed points.
11. **"Harm" had no practical threshold.** In the small arm ($a = 0.99$, κ = 1, $L = 1$) differences of about +0.05% in MSE were significant because the intervals are very narrow, so the rule "lower end > 0" called them harm (the same artefact as "blurred" in e01b). The harm verdict now uses the margin δ = log 1.02 registered before the smoke test (standard non-inferiority): "material harm" needs the lower end above δ; significant increases within the margin are reported separately as "small increase". No new threshold, no change to predictions. Disclosed: this rule was adjusted after seeing one smoke-test configuration.

## Known limitations

- **The encoder is fixed,** so the main argument for bootstrap targets (their effect on a learned encoder, candidate 1) is not tested. In the population, C optimizes the evaluated rollout error itself, so with a fixed encoder and large data C is expected to win. e02 tests the bias side of the trade-off; e03 tests the variance side and the encoder.
- **Scalar observations and linear models:** no Jensen term, no nonlinearity.
- **$N$ in M\*'s rule is the number of training windows,** a heuristic for the effective sample size of SGD. It is fixed here, not tuned.
- **The thresholds** (2% gap, log 1.02 margin, ρ ≥ 0.8, 1% for "within") are judgements, fixed before training.
- **The λ grid ends at 10,** and in persistent systems ($a \ge 0.9$) the population optimum of C and M\* is at the largest λ (T2: the optimum tends to λ → ∞ there). A selected λ at the edge is expected; agreement with theory is assessed at the selected λ, so it is not affected.

## Predictions recorded before results (2026-11-12)

From the population fixed points (`results/predictions.txt`):
- every fixed point is stable;
- zero change in all Markov configurations;
- largest gains at $a = 0.99$, $\kappa = 0.8$, $L = 1$: C −29.7%, M\* −29.7%, M −27.1%, D −23.7%, direct −33.1%;
- gains vanish for $L \ge 4$–8 as the gap closes;
- in the population M\* (large) equals C in every configuration and M\* (small) is within 0.2% of it;
- **H5 in the population:** the Spearman correlation over the 33 non-Markov configurations between the critic at the one-step optimum and the absolute gap is 0.97, against 0.52 for the raw residual.

Secondary ablations (`results/predictions.txt`, recomputed with the ablations; the original predictions are unchanged to the last digit):
- **M\*_oracle** equals M\* in the population, so any difference measures the estimation of the weights.
- **M\*_p1** is within 0.03% of M\* in every configuration (the rule puts almost no weight on $p = 1$).
- **D_nosg** breaks T1:
  - in Markov configurations it **raises** the error by 0.8–28% at λ = 1 and 2.6–45% at λ = 10 (log ratio), where D changes nothing;
  - in 12 of the 15 configurations with a gap, its best λ is worse than D's.

  Smoke runs at $a = 0.9$, κ = 1, $L = 1$ and $a = 0.99$, κ = 0.8, $L = 2$ reproduce the predicted log ratios at λ = 1 and 10 to the third digit (e.g. +0.1299 ± 0.0021 trained vs +0.1305 predicted).

Claude's subjective probabilities, before any training run:

| | probability |
|---|---|
| H2 holds at large data | ≈90% |
| H1 holds with agreement in most gap configurations | ≈75% (risk: optimization not reaching the fixed point at λ = 10) |
| H3: no harm with little data | ≈60% |
| H4 in the large arm | ≈80% |
| H4 in the small arm | ≈50% (the variance advantage of bootstrap may be too small in stable linear systems) |
| H5 | ≈70% |
| R1 | ≈40% |
| R2 | ≈35% |
| R3 | ≈5% |
| R4 | ≈10% |
| mixed or inconclusive | ≈10% |

## Result

Evaluated 2026-11-12 (`evaluate.py`; output in `results/test_results.json`, `results/e02_summary.png`, printed summary in the local `logs/e02_evaluate.log`).

**Completeness.** All 480 runs are present:
- large arm: 48 configurations × 8 variants;
- small arm: 12 configurations × 8 variants.

Every run comes from commit 04335d4 and every value is finite. The runs record `git_dirty: true` because the tree also contained the committed smoke-test result files, which the runs overwrote, and edits to `paper/`. No `.py` file differed from the commit.

**Verdicts against the pre-registered criteria:**

| | result | prediction |
|---|---|---|
| H1 (agreement with theory and "better", gap configurations with predicted gain ≥ 1%) | C 12/12, M 12/12, M\* 12/12: **holds** | ≈75% |
| H2 (Markov, large arm, 240 tests) | 171 harmless, 21 better, 47 small increase, **1 material harm**: **fails formally** | ≈90% |
| H3 (Markov, small arm, 80 tests) | 75 harmless, 5 small increase, 0 material harm | ≈60% no harm |
| H4 large (M\* vs the better of C and D) | 33/33; largest upper end of the CI of the log ratio $8 \cdot 10^{-5}$: **holds** | ≈80% |
| H4 small | 7/7; largest upper end 0.0019: **holds** | ≈50% |
| H5 (critic tracks the gap) | Spearman 0.957 for the critic vs 0.523 for the raw residual (predicted 0.97 / 0.52); every Markov configuration has critic < $10^{-3}$: **holds** | ≈70% |

**Agreement with the population theory** (large arm, non-Markov configurations, selected λ):
- trained and predicted log ratios differ by at most 0.0015 (median $4 \cdot 10^{-5}$);
- trained models are at a median distance of 0.006 from the predicted fixed points (at most 0.031).

Examples:
- AR(2) with κ = 1, $L = 1$: C −0.2465 ± 0.0015 trained vs −0.2465 predicted; D −0.2107 vs −0.2111;
- $a = 0.99$, κ = 0.8, $L = 2$: C −0.1240 vs −0.1241.

The largest gain is at $a = 0.99$, κ = 0.8, $L = 1$: C −0.354 (30% lower MSE).

Fraction of the onestep → direct gap closed (15 gap configurations):

| C | M\* | D |
|---|---|---|
| 0.01–0.90 | 0.01–0.90 | 0.00–0.74 |

**Selected λ.** λ = 10 (the edge of the grid) was selected in 23 of 33 configurations for C, 22 for M\* and 24 for D, as expected from T2 for persistent systems.

**The material harm in H2.** AR(2), κ = 1, $L = 8$, D at λ = 10: +0.241 log ratio (+27% MSE), where 0 is predicted. The trained θ is 1.65 from the predicted fixed point, against a median of 0.006 elsewhere. The 47 small increases (median +0.1%) are also at λ = 10 in long windows or near instability, with distances of 0.2–0.6. The cause is examined in the exploratory analysis below.

**Small arm.** M\* matches C but never beats it:

| configuration | C | M\* | M\*_oracle | D |
|---|---|---|---|---|
| $a = 0.99$, κ = 0.5, $L = 1$ | −0.1167 | −0.1153 | −0.1154 | −0.0693 |
| $a = 0.99$, κ = 0.5, $L = 2$ | −0.1101 | −0.1085 | −0.1086 | −0.0372 |

D is the weakest variant in every configuration of both arms. No configuration of e02 shows an advantage of bootstrapped targets.

**Secondary ablations:**
- **D_nosg** (no stop-gradient): material harm in 26 of 60 Markov tests in the large arm and 10 of 20 in the small arm; worse than D in 27 of 33 large-arm gap configurations and 5 of 7 small-arm ones. This is as predicted: the trained models match the population fixed points of D_nosg within 0.01 in 190 of 192 large-arm cases. In the remaining two ($a = 0.99$, κ = 1, $L = 1$ and $L = 2$, λ = 10) training reaches a different and much worse point (log ratio about +2.2, i.e. roughly nine times the MSE, against +0.04 and +0.07 predicted); found while writing the paper section, 2026-11-12.
- **M\*_oracle:** within $10^{-4}$ of M\*; better in 3 of 33 and 4 of 7 configurations by tiny amounts. The weights are well estimated.
- **M\*_p1:** within $10^{-6}$ of M\* on average. The anchor point $p = 1$ adds nothing.

**Outcome.** Formally **R3** (H2 fails by one test). Per R3 the code and the estimation were checked before interpreting (below). Assessment agreed with the user: **R3 formally, R1 after diagnosis.**

## Exploratory analysis (after the evaluation, not pre-registered)

**Convergence check** (`convergence_check.py`, output `results/convergence_check.json`, log `logs/e02_convergence.log`, run 2026-11-12). All four main variants were retrained at λ = 10 in the 15 Markov configurations of the large arm, with 30 000 steps instead of 3000 (same data, seeds and cosine schedule). Log ratios are relative to the original 3000-step onestep runs:

| verdict | 3000 steps | 30 000 steps |
|---|---|---|
| material harm | 1 | 0 |
| small increase | 23 | 1 |
| harmless | 33 | 11 |
| better | 3 | 48 |

- **The material harm disappears.** AR(2), $L = 8$, D goes from +0.2407 to −0.0001, and its distance from 1.645 to 0.062.
- **Distances fall** from 0.2–0.6 to 0.01–0.06.
- **One small increase remains:** $a = 0.5$, $L = 8$, D, +0.0023, distance 0.098, still not fully converged.

**The "better" verdicts after convergence are an artefact of the baseline.** The 3000-step onestep models are themselves not fully converged. At $a = 0.99$, $L = 2$ their MSE is 0.27% above their own least-squares solution, exactly the apparent gain; at $a = 0.99$, $L = 8$ it is 0.04%, and in the AR(2) and $a = 0.9$ configurations about 0.01%. Once both sides converge, the anchored models sit at the one-step optimum, as T1 states.

**Consequences:**
- The failure of H2 is an optimization budget, not a violation of T1. A large λ in an ill-conditioned problem (long windows, $a$ near 1) slows convergence, and with a fixed number of steps it looks like harm.
- The under-converged baseline shifts the log ratios by at most about 0.3%, against gains of 1–30% in H1 and a criterion of agreement at the third digit, so H1 and H4 are unaffected.

**Lessons for e03:**
1. The training budget must follow a convergence criterion (for example a plateau of the training loss), applied to the baselines as well.
2. The harm seen in e01b (D at λ = 1, 75× in sufficient) now has two competing explanations: approximation error amplified by chaos, or optimization that does not converge through chaotic rollouts. e03 must tell them apart, for instance by checking whether the harm shrinks with longer training.

**Exploratory ensemble check** (`explore_ensemble.py`, output `results/explore_ensemble.json`, run 2026-11-12). Question: does the disagreement of an ensemble of one-step models see the Markov gap? Ensembles of 5 least-squares one-step models on bootstrap resamples of the same windows (as in PETS), 10 seeds, four configurations, 315 to 126 000 training windows; exact population moments, units of Var(y). The prediction was written in the script before the first run: the ensemble disagreement falls with N, while the excess rollout error and the critic stay at their population values.

| configuration | windows | disagreement, k = 10 | excess rollout MSE, k = 10 | critic |
|---|---|---|---|---|
| AR(1) a = 0.9, κ = 1, L = 1 (Markov) | 315 | 6.0e-3 | 0.0140 | 0.0035 |
| | 126 000 | 1.3e-5 | 0.0000 | 0.00001 |
| AR(1) a = 0.9, κ = 0.8, L = 1 | 315 | 4.0e-4 | 0.0553 | 0.0131 |
| | 126 000 | 7.4e-7 | 0.0582 | 0.0111 |
| AR(1) a = 0.9, κ = 0.5, L = 1 | 315 | 6.9e-7 | 0.0301 | 0.0072 |
| | 126 000 | 2.7e-10 | 0.0303 | 0.0083 |
| AR(2), κ = 0.8, L = 2 | 310 | 1.5e-4 | 0.0392 | 0.0304 |
| | 124 000 | 1.8e-7 | 0.0397 | 0.0301 |

Population values (excess, critic): 0 and 0 for the Markov configuration; 0.0583 and 0.0112, 0.0303 and 0.0084, 0.0397 and 0.0300 for the three with a gap.

- **The prediction holds.** With a gap, the excess rollout error and the critic stay at their population values at every N, while the ensemble disagreement falls by a factor of 500 to 2 600 over the 400-fold increase in data.
- **The ensemble is blind to the gap even with little data.** At 315 windows, its disagreement is 140 to 44 000 times smaller than the excess error it should warn about, because every member converges to the same biased one-step solution. Only for the Markov latent are both quantities of the same order, since all remaining error is estimation error.
- **Limits.** The members are linear and differ only by resampling. Neural ensembles add diversity from initialization, but the bias of the population one-step optimum is shared by every member regardless; whether this holds for learned latents is a question for e03.

## Conclusion

Written by the user, 2026-11-12.

1. e02 confirmed the theory quantitatively in the linear-Gaussian setting with a fixed window encoder:
   - the population fixed points predicted, before training, the gains of every variant to the third decimal (T2);
   - for Markov latents nothing changes once optimization converges (T1);
   - the critic measures the rollout bias and the raw residual does not (lemma, H5);
   - the stop-gradient is necessary (D_nosg);
   - the anchor point $p = 1$ carries no signal.

   Nonlinear dynamics, learned encoders and the matrix versions of the theorems were not tested.
2. M\* is safe: it never falls behind the better of C and D by more than 0.2%. But it does not beat C anywhere in e02. In a stable linear system the variance of the Monte Carlo target does not matter even with about 1000 windows, so the bias–variance rule rightly picks C. Whether M\* gains where bootstrapped targets have an advantage (chaotic dynamics with noise and a learned latent, where D beat C in e01b) is the main question of e03, which decides whether Anchor improves on the standard Monte Carlo loss.
3. H2 fails formally by one test. The convergence check shows an optimization budget rather than a violation of T1: with ten times more steps the harm disappears and the models reach the one-step optimum. The results support the theory and the safety of the method; its advantage is still open.
4. Carried to e03:
   - a training budget set by a convergence criterion, applied to the baselines as well;
   - a check that separates the two explanations of the e01b harm (approximation error amplified by chaos vs non-convergence through chaotic rollouts);
   - the protocol of e02: each seed draws its data, verdicts use a margin, and predictions are recorded before training where theory allows.
