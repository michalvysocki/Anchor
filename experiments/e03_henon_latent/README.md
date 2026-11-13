# e03 — Anchor on the Hénon map with noise and a learned latent

Pre-registration drafted 2026-11-14, before any code. Design review, smoke tests and predictions are added below before the full runs.

## Purpose

e02 confirmed the theory in the linear-Gaussian case, where every quantity is exact. e03 tests what carries over to **chaotic nonlinear dynamics with a learned latent**, organized by the two sources of rollout bias found after e02 (exploratory analyses in `experiments/e02_lgssm/README.md`):

- **Model bias.** A dynamics model with one transition treats an encoded latent and its own prediction alike, so under a Markov gap no multi-step loss can make the rollout optimal at every horizon (the ceiling). A one-bit **flag** (encoded vs predicted input) removes it in the linear case: there C+F closed the whole gap between the one-step model and the best direct prediction in 32 of 32 configurations, against at most 0.90 without the flag.
- **Target bias.** A bootstrapped target computed from the window at $t+p$ lacks what the window at $t$ knew, so it is biased even under a perfect model (D+F: median 0.74 of the gap). A critic $\delta(z_t)$ adds the missing part back (**Debiased Bootstrap, DB**); in the linear case DB equals C exactly.

The questions are whether the flag still helps when the mean of the state is not propagated exactly (Jensen), whether DB adds anything over C when the dynamics are nonlinear, whether M\* chooses a safe target in every regime, and where the harm seen in e01b came from.

## Setup

**System.** Hénon map, $a = 1.4$, $b = 0.3$. Only $x_t$ is observed, with Gaussian noise: $y_t = x_t + v_t$, $\mathrm{Var}(v) = r$, signal fraction $\kappa = \mathrm{Var}(x) / (\mathrm{Var}(x) + r) \in \{1, 0.8, 0.5\}$, $\mathrm{Var}(x)$ measured on the attractor. Trajectories start uniformly in $[-1, 1]^2$, burn-in 100 steps, unstable ones rejected; 64 observations each; data in units of $\sqrt{\mathrm{Var}(y)}$.

**Latent.** An encoder $\varphi$ of the last $L \in \{1, 2, 4\}$ observations: MLP, 2 hidden layers of 128 (tanh), latent dimension 16. Nine configurations. With $\kappa = 1$ and $L \ge 2$ the window determines the state (its second coordinate is $b\,x_{t-1}$), so the latent can be Markov: the two **Markov controls**. The other seven have a Markov gap.

**Model.** Encoder $\varphi$, dynamics $g$ in the latent (MLP, 2 hidden layers of 128, tanh, residual), decoder $r$ to $y$ (MLP, 1 hidden layer of 64). All trained jointly.
- **Without flag:** $u_1 = \varphi(z_t) + \mathrm{MLP}(\varphi(z_t))$, $u_{j+1} = u_j + \mathrm{MLP}(u_j)$, prediction $\hat y_{t+j} = r(u_j)$.
- **With flag (F):** the flag enters the MLP as a one-hot input $f$ (encoded or predicted), $u' = u + \mathrm{MLP}([u, f])$, with $f = \mathrm{enc}$ at the first step and $f = \mathrm{pred}$ after it. The extra capacity is one input column of the first layer (finding 2 of the design review: an embedding added to $u$ would shift the state itself, not only the transition).

**Losses.** Every variant has the one-step loss $(r(u_1) - y_{t+1})^2 + \|u_1 - \mathrm{sg}[\varphi(z_{t+1})]\|^2$ (mean over latent dimensions): the prediction of the next observation and, as in TD-MPC and SPR, the consistency of the predicted latent with the encoding of the next window, which makes the predicted latent one the dynamics can be applied to again (added after the gap stage, finding 18). The multi-step targets are in observation space and under stop-gradient (the latent-target version of the multi-step term, as in TD-MPC2, is left for e05); the observation term keeps the latent informative, so the encoder cannot collapse. the auxiliary term is $\lambda$ times the mean over $k$ of $(r(u_k) - \mathrm{sg}[\text{target}_k])^2$, training horizons $k \in \{2, 3, 4\}$ (with $k = 1$, the first $2n$ horizons for a state of dimension $n = 2$; the identifiability argument of the flag check), $k$ sampled uniformly.

**Variants (core).** Each without and with the flag, so 8 in all.

| variant | target for the rollout to $k$ |
|---|---|
| onestep | none (λ = 0) |
| C | the observed $y_{t+k}$ (Monte Carlo; the target of TD-MPC and SPR, in observation space) |
| DB | $T_{k,p} + \delta_{k,p}(z_t)$, $p$ uniform on $2..k-1$ ($k = 2$: the observed $y_{t+2}$). $T_{k,p}$ is the decoded rollout of the same model from the window at $t+p$, its first step flagged as encoded. $\delta$ is a critic head on $\mathrm{sg}[\varphi(z_t)]$ (MLP, 1 hidden layer of 64, one output per pair), trained on $\mathrm{sg}[y_{t+k} - T_{k,p}]$ with 3× the model's learning rate |
| M\* | mixture over anchor points $p = 2..k$ ($p = k$ is C, $p < k$ is $T_{k,p}$) with weights minimizing $w^\top(B + V/N)w$ on the simplex, $B$ from the difference of two critic heads (regressions of $T_{k,p}$ and of $y_{t+k}$ on $\mathrm{sg}[\varphi(z_t)]$), $V$ the covariance of their residuals, $N$ the number of training windows; the weights stay on C for the first 1000 steps, while the critic heads train, then are recomputed every 100 steps on a training batch, under stop-gradient (as in e02) |

**Ablations.** On two gap configurations, $\kappa = 0.5, L = 1$ and $\kappa = 0.8, L = 2$, main data arm. Each is compared with the variant it modifies:

| ablation | modifies | what it isolates |
|---|---|---|
| D, D+F | DB, DB+F | $\delta = 0$: the plain bootstrapped target and its information bias |
| TDB, TDB+F | DB, DB+F | telescoped correction: one $\delta_p$ per anchor point added to $u_{p+1}$ and propagated by $g$ |
| DB oracle, DB+F oracle | DB, DB+F | $\delta$ fitted on a large reference sample: separates the rule from the estimation of $\delta$ |
| F at λ = 0 | onestep+F | the flag without a multi-step term (prediction: no gain; $e_{\mathrm{pred}}$ is not trained) |
| C+F on {3, 4} | C+F | training horizons without $k = 2$ (identifiability) |
| C, matched capacity | C | without flag, dynamics widened to the parameter count of the two-network flag |
| C+F, two networks | C+F | separate networks $g_0$, $g_1$ instead of one network with an embedding |

**GRU arm** (encoder with memory, the "filter" row of the bias table). The window encoder is replaced by a GRU over the last 16 observations (hidden size 16, the latent). The window length no longer applies, so the arm has 3 configurations, $\kappa \in \{1, 0.8, 0.5\}$ ($\kappa = 1$ is a Markov control), with the variants onestep, C, C+F, DB, DB+F, M\* and M\*+F. Prediction: both biases shrink, so the flag and $\delta$ matter less than with a window.

**Variance-regime arm.** On 4 configurations, $\kappa \in \{1, 0.5\} \times L \in \{1, 2\}$: (i) small data, (ii) long training horizons $k \in \{2..8\}$, closer to the limit of predictability. Variants C, D, DB, M\*, each without and with the flag; onestep and direct are trained in the small-data arm and taken from the main arm for the long horizons. This is where target variance may matter, so where bootstrapped targets and M\* can gain.

**Not applicable.** A (composition consistency) holds by definition for a one-step model; B and E need a direct head, which is not a variant here. M (uniform mixture) and D without stop-gradient were settled in e02.

**Reference (not a variant).** A direct network $F(z_t, k)$: the same encoder as the world model (window MLP 2×128 to the latent of 16), then a head on [latent, one-hot $k$] (1 hidden layer of 128), $k = 1..10$, trained on the same data; in the GRU arm the same head on a GRU encoder (finding 1). The gap of a configuration is the test MSE of onestep minus that of direct, with the same encoder.

**Data.** Separate generators per split (tags below). Main arm: 2000 training trajectories (about $10^5$ windows). Small-data arm: 20 trajectories (about $10^3$ windows). Validation: 200 trajectories, shared, used for λ selection and the stopping rule. Test: 500 trajectories, fixed. Every seed draws its own training set (lesson of e02). Windows hold $L$ (or 16 for the GRU) past observations and 10 future ones.

**Training.** Adam, learning rate $10^{-3}$, batches of 512 windows, gradient clipping 1.0. **Budget by convergence, identical for every variant, baseline and reference:** the validation loss of the variant is evaluated every 1000 steps; the learning rate is multiplied by 0.3 after 6 evaluations without an improvement of both 0.01% relative and $10^{-6}$ absolute (first drafted as 3 evaluations and 0.1%, changed by the correctness review and the gap stage below); training stops when it falls below $10^{-5}$ or at 100 000 steps (runs stopped by the cap are reported). Each variant stops on its own loss, the objective it minimizes, so each reaches its own optimum; the validation rollout MSE is logged along the way, to show whether the primary metric was still moving at the stop (finding 4). λ ∈ {0.1, 1, 10}. Models of one configuration and variant are trained together as a batch of independent models where the code allows it.

**Seeds.** Selection seeds 0–2, evaluation seeds 3–32 (30 seeds, as in e02; first drafted as 10 and raised before any Anchor run, because external compute became available and onestep varies strongly between seeds, see the correctness review), disjoint.

**Implementation details.** Every anchor point $p = 2..k-1$ of a horizon is used in every step and the targets are averaged, instead of sampling one $p$: the expected loss is the same and its noise lower. One rollout from the window at $t+p$ gives $T_{k,p}$ for every $k > p$. All variants of a configuration see the same sequence of training batches (the reference batches of the oracle have their own generator). The trained parameters of every run are saved next to its results.

## Questions

**Q1 (model bias, main).** In the gap configurations, does the flag lower the rollout error of C (C+F vs C), and by how much of the gap?

**Q2 (T1 and the e01b harm).** In the Markov controls, is every variant harmless at every λ once training has converged? If a harm remains, the e01b harm is approximation error amplified by chaos, not non-convergence.

**Q3 (target bias).** Does DB improve on D, as the information bias predicts, or does it lose, which would mean that D's advantage in e01b came from its bias acting as a regularizer?

**Q4 (debiased bootstrap vs Monte Carlo).** Does DB+F differ from C+F, in the main arm and in the variance-regime arm?

**Q5 (method).** Does M\* match the best fixed target in every configuration and arm, while every fixed target loses somewhere?

**Q6 (critic).** Does the critic of the trained onestep model track the gap?

## Metrics

- **Primary:** test rollout MSE against $y_{t+k}$, mean over $k = 1..4$ (decided on the gap stage), in units of $\mathrm{Var}(y)$; the mean over $k = 1..10$ is secondary.
- **Comparison:** paired log ratio of the primary metric between two variants on the same seed (the same training data), mean and 95% t-interval over 30 seeds ($t = 2.045$), verdicts with the margin δ = log 1.02 as in e02 (better, material harm, small increase, harmless, inconclusive).
- **Fraction of the gap closed:** (onestep − variant) / (onestep − direct), in the gap configurations.
- Secondary: MSE against the clean $x_{t+k}$; MSE per $k$; fraction of diverged rollouts ($|\hat y| > 10$); steps to convergence; weights of M\*; validation errors of every λ (sensitivity of the flagged variants to λ).
- **Critic:** at the trained onestep model, the mean squared output of a critic head trained on $\mathrm{sg}[\text{rollout} - T_{k,p}]$ (the residual mean, Corollary), and the raw squared residual.

**λ selection.** Per configuration and variant, the λ with the lowest mean validation rollout MSE over the primary horizons ($k = 1..4$) over the selection seeds; a λ whose training diverged is never selected.

## Hypotheses and decision criteria

| | prediction | criterion (test set, evaluation seeds) |
|---|---|---|
| H1 (Q1) | the flag helps, by less than in the linear case: only the model part of the bias is removed, the spread of the belief remains | C+F vs C "better" in at least 5 of the 7 gap configurations of the main arm; reported separately for the gap without noise ($\kappa = 1, L = 1$, missing memory only) and the gaps with noise (finding 6), and with the number of configurations whose improvement exceeds the flag's effect in the Markov controls (control for H1, Gap measured before the Anchor runs) |
| H2 (Q2) | no harm after convergence | in the 2 Markov controls, no "material harm" for any core variant at any λ, against onestep without flag |
| H3 (Q3) | two-sided; no direction is predicted | verdict of DB vs D and of DB+F vs D+F in the ablation configurations and the variance-regime arm (both flag settings in both); interpretation fixed now: DB "better" or "harmless" ⇒ D's advantage was the form of its target; DB "material harm" ⇒ it was its bias |
| H4 (Q4) | main arm: DB+F within the margin of C+F; any gain of DB+F only in the variance-regime arm | DB+F vs C+F verdict per configuration; H4 holds if "harmless" or "better" in every main-arm gap configuration. **H4b** (secondary, reported, not part of the outcomes): DB vs C without the flag in the same configurations, i.e. the debiased bootstrap as a pure plug-in in a model with one transition |
| H-M\* (Q5) | M\* is never materially worse than the best fixed target; every fixed target is materially worse than the best somewhere | for every configuration and arm, the upper end of the CI of log(M\* / best fixed target with the same flag setting, each at its selected λ) is < log 1.02, the best taken over {C, DB} in the main grid and the GRU arm and over {C, D, DB} where D is trained (ablation configurations, variance-regime arm; finding 3), for both flag settings; and for each fixed target there is a configuration where its lower end against the best is > 0 |
| H5 (Q6) | the critic tracks the gap | Spearman correlation between the critic and the gap over the 9 window configurations and the 3 GRU configurations ≥ 0.8; critic in the Markov controls below that of every gap configuration |

**Outcomes.**

| | outcome | consequence |
|---|---|---|
| R1 | H1 and H2 hold | the flag carries to e05 as part of the method; H3–H5 and H-M\* set the default target (C, DB or M\*) |
| R2 | H1 fails, H2 holds | the flag is a linear-case result and a diagnostic (the decomposition of the bias), not a method component for chaotic dynamics; Anchor stays the plug-in loss |
| R3 | H2 fails | harm after convergence: the e01b harm is approximation error amplified by chaos; a mechanism that switches off (adaptive λ) becomes necessary, not optional |
| R4 | the stopping rule fails for most runs (the cap of 100 000 steps is reached), or most rollouts of a variant diverge | the training setup does not support the comparison; check the code and the budget before interpreting |

The paper's claim about the method follows the decision rule recorded on 2026-11-13: the dynamics-side method (flag plus target) stays if it beats C; the encoder-side arm is parked and not part of e03.

## Design review (before any training code)

Seven findings, all adopted (2026-11-14):

1. **Reference for the GRU arm.** The direct network had only a window encoder, so the gap of a GRU configuration would be measured against a different encoder. The GRU arm gets its own direct reference with a GRU encoder.
2. **How the flag enters the dynamics.** Adding an embedding to $u$ in a residual dynamics model shifts the state itself, not only the transition. The flag is a one-hot input of the MLP, $u' = u + \mathrm{MLP}([u, f])$.
3. **H-M\* referred to D where D is not trained.** The best fixed target is taken over {C, DB} in the main grid and over {C, D, DB} where D exists.
4. **The stopping rule uses each variant's own loss**, which contains the λ term, so variants stop on different objectives. This is intended (each reaches its own optimum); the validation rollout MSE is logged to show whether the primary metric still moved at the stop.
5. **Unknown compute.** About 2500 models with a budget of up to 100 000 steps may cost far more than e02. The smoke test measures time per model and steps to convergence; if the full run exceeds about two days, the selection seeds or the λ grid are reduced before the full run, recorded as a smoke-test finding.
6. **H1 mixes two kinds of gap.** $\kappa = 1, L = 1$ has a gap from missing memory only, the others from noise and memory. The criterion stays at 5 of 7; the two kinds are reported separately.
7. **No closed-form predictions.** Before the full run, the gap (onestep − direct) of every configuration is measured from a short training of the two baselines on one seed and recorded as the size at stake. Configurations whose gap is too small to resolve (below 1% of the onestep MSE, as the threshold of H1 in e02) are reported descriptively and excluded from H1 and H4, decided before any Anchor variant is trained.

## Correctness review of the code (2026-11-14, before any Anchor run)

1. **The stopping rule was too lax: the onestep baseline had not converged.** With 0.1% and patience 3, onestep ($\kappa = 0.5$, $L = 1$, seed 0) stopped at 14 000 steps with test rollout MSE 1.200; stricter rules gave 1.152 (0.01%, patience 3), 1.128 (0.001%, patience 3) and 1.122 (0.01%, patience 6, stop at 32 000 steps), while C+F stayed at 0.978 under every rule. The lax rule would have favoured every Anchor variant against onestep, the error e02 warned about. Changed to 0.01% and patience 6 (stops at 27 000–40 000 steps), about 2.3 times the cost.
2. **Trained parameters were not saved**, so H5 (a critic fitted on the trained onestep model) and any later analysis would need retraining. They are saved now.
3. **M\* computed its first weights from untrained critic heads.** The weights now stay on C for the first 1000 steps.
4. **The oracle drew its reference batches from the shared batch generator**, so its training batches differed from the other variants'. It has its own generator now.
5. **Anchor points are averaged, not sampled** (Implementation details above).
6. **The onestep rollout error varies strongly between seeds** (test MSE 1.05 to 1.18 at $\kappa = 0.5$, $L = 1$), so comparisons with onestep will have wide intervals.
7. **The evaluation script and the critic of H5 are not written yet**; they are needed before the final stage, not before the selection stage.

Second review (the same day, after raising the evaluation seeds to 30):

8. **Initialization streams collided between modules**: the random stream of the encoder's first bias was the stream of the dynamics' first weights (correlation 1.0). Every tensor now has its own key (model, module, layer, kind), hashed by SeedSequence; the encoder of a model is still initialized identically across variants, which keeps the comparisons paired.
9. **The direct reference was trained only in the gap stage**, on one seed, so the fraction of the gap closed would have compared 30 seeds with one. It is now trained in every configuration where onestep is (not in the long arm, whose data and reference are those of the main arm).

Third review (final check before the runs):

10. **H2 judges harm at every λ, but the final stage trained only the selected λ.** In the two Markov controls of the main arm, every Anchor variant is now trained at all three λ on the evaluation seeds.
11. **R4 called a rollout model better than the direct reference a contradiction.** The direct network is an estimate of the reachable level, not the optimum (Known limitations), so a rollout model may beat it. R4 now covers only a failed stopping rule or diverging rollouts; a variant that beats the direct reference is reported, and its fraction of the gap closed exceeds 1.
12. The description of the direct reference matched the code only roughly; it now states the architecture.

Checked without change: disjoint generators for training (per arm and seed), validation, test and reference data; selection seeds 0–2 and evaluation seeds 3–32 draw different training sets; the window at $t+p$ and the targets $y_{t+k}$ are indexed correctly for the window and the GRU encoders; every target is under stop-gradient; critics are trained only by their own losses; the hypotheses H1, H3, H4, H-M\* and H5 use variants that the plan trains in the stated configurations.

Fourth review:

13. **Pairing depended on the size of a run.** Initialization keys and the batch generator depended on a model's position in its batch of models, so the models of one seed started from different parameters and saw different batches at different λ, and in the Markov controls (90 models per run) differently from onestep (30). Both are now keyed by the seed alone: the models of one seed share their initial encoder and their sequence of batches across λ, variants and run sizes.
14. **λ selection could pick a diverged run**: `argmin` returns a NaN validation error as the minimum. Non-finite validation errors are now treated as infinite.

Fifth review:

15. **The onestep baseline of the long arm duplicated the main arm's**: the same large data, seeds and batches, and onestep does not use the training horizons. The long arm now uses the main arm's onestep and direct runs (165 → 161 jobs per stage).
16. **An interrupted stage restarted from the first job.** A job whose parameters are saved is now skipped (`--force` reruns it), so a stage can be resumed on a shared machine.
17. Memory: the largest validation passes (90 models; GRU with 30 models; the long arm) peak at about 2 GB per process, so 16 processes fit in 128 GB with room to spare.

Code layout (after the fifth review): the world models, targets and critics are in the library (`src/anchor/latent.py`, with the batched networks in `src/anchor/ensemble.py` and the noise model in `src/anchor/envs/henon.py`), as the repository's rule asks for the method; this directory keeps the settings (`design.py`), the data assembly (`data.py`), the stages (`run.py`) and the launcher. The move was mechanical (sizes passed as parameters); the smoke run of C at κ = 0.5, L = 1 reproduces its test MSE before the move (0.9902), and `tests/test_latent.py` now checks automatically what the reviews checked by hand: the flag separates the first step, every target is under stop-gradient, the model loss does not train the critics and the critic loss does not train the model, the cached bootstraps equal separate rollouts, M\* weights lie on the simplex, initialization is paired across variants, and the window indices.

An end-to-end test of the stages gap, selection and final (a few steps per job) and of the launcher with three processes passed.

The losses after the change that computes one rollout per anchor point equal the previous ones bit for bit (28 cases, window and GRU encoders, short and long horizons).

## Finding of the first gap stage (2026-11-14, before any Anchor run)

18. **The onestep baseline had no meaningful rollout.** In the Markov control $\kappa = 1$, $L = 2$, the first gap run (one seed) predicted one step almost exactly (one-step loss $3 \cdot 10^{-7}$, test MSE 0.000 at $k = 1$) but reached 2.50 at $k = 2$, more than the variance of the data, and 2.2 on average over $k = 1..10$. The one-step loss asked only that the predicted latent $u_1$ decode to $y_{t+1}$; nothing made it a latent to which the dynamics can be applied again, so the second step acted on an object outside the space the dynamics was trained on. (In e02 the latent was the window and the rollout shifted it, so the question did not arise.) Every Anchor variant would have beaten such a baseline trivially, which would have voided H1, H2 and the gaps. The one-step loss of every variant now also asks $u_1$ to match $\mathrm{sg}[\varphi(z_{t+1})]$, as the latent consistency of TD-MPC and SPR (Losses). Under a Markov gap the predicted latent then approaches the conditional mean of the next encoding, the situation the flag addresses. The gap stage was stopped and restarted with this loss; its first results are kept in `results/runs_invalid_gap_v1/`.

19. **Smoke test of the corrected baseline** (one seed, trained to convergence): at $\kappa = 1$, $L = 2$ onestep now reaches 0.009 at $k = 2$ (2.50 before) and grows exponentially after $k \approx 4$ (3.50 at $k = 10$), while direct stays at 0.23 at $k = 10$; at $\kappa = 0.5$, $L = 1$ onestep and direct differ only at $k = 2..3$ (0.997 against 0.966) and both are near 0.98, unpredictable, from $k = 4$. Three consequences, settled on the full gap stage before any Anchor run: (a) the mean over $k = 1..10$ dilutes the effect into horizons where nothing is predictable, so the range of the primary metric is chosen from the gap stage; (b) onestep − direct mixes the Markov gap with approximation error amplified by chaos (1.39 in a Markov control), so the flag's effect in the Markov controls is used as a control for H1; (c) without noise the one-step loss keeps improving by more than 0.01% relative and training ran to the step cap, so an improvement must now also exceed $10^{-6}$ in absolute terms.

## Every target with and without the flag (2026-11-14)

Decided after the gap stage and while the selection stage ran, before any test result and before any selection result was read (user). Every target is now trained without and with the flag in every arm: the variance-regime arm adds D, DB and M\*; the GRU arm adds DB and M\*; the ablations add TDB and DB oracle. C+F on {3, 4}, C wide and C+F with two networks are ablations of the flag itself and have no unflagged counterpart. H3 and H-M\* are judged in both flag settings, and H4b compares DB with C without the flag: whether the debiased bootstrap helps as a pure plug-in, without changing the dynamics. 195 jobs per stage (161 before). The selection stage already running covers the 161; the added jobs run when the stage is started again (finished jobs are skipped).

## Gap measured before the Anchor runs

Gap stage, 2026-11-14, on the MacBook (CPU), one selection seed, onestep and direct trained to the stopping rule, with the one-step latent consistency (finding 18). `gap_report.py` prints the test MSE at every $k$ (`results/gap_report.json`).

| configuration | relative gap over $k = 1..4$ | $k$ where direct < 0.95 Var(y) | note |
|---|---|---|---|
| κ = 1, L = 1 (gap without noise) | +38.6% | 1–10 | |
| κ = 0.8, L = 1, 2, 4 | +6.5%, +4.9%, +3.5% | 1–5, 1–5, 1–6 | |
| κ = 0.8, GRU | +13.8% | 1–9 | |
| κ = 0.5, L = 1, 2, 4 | +2.0%, +3.7%, +1.0% | 1, 1–3, 1–3 | |
| κ = 0.5, GRU | +1.7% | 1–3 | |
| κ = 1, L = 2, 4, GRU (Markov) | +92%, −185%, +96% | 1–10 | onestep nearly exact up to $k \approx 3$, then chaos amplifies its error |

All runs at κ = 1 reached the cap of 100 000 steps (4 of the 12 configurations); the others stopped at 29 000–84 000 steps.

**Decisions, before any Anchor run (accepted 2026-11-14):**
1. **Primary metric over $k = 1..4$** (the training horizons; with noise the map is predictable for 3 to 6 steps, and the mean over $k = 1..10$ diluted the gap into unpredictable horizons). λ is selected on the same range. The mean over $k = 1..10$ is reported as secondary.
2. **H1 and H4 use all 7 window configurations with a gap**: each has a relative gap of at least 1% over $k = 1..4$ (κ = 0.5, L = 4 at 1.03%).
3. **Control for H1.** In the Markov controls (κ = 1, L = 2 and 4) the flag has no model bias to remove, so its effect there measures what it does for other reasons, such as damping the growth of error under chaos. The effect in a gap configuration is attributed to the Markov gap only if its improvement, −(mean log ratio of C+F to C), exceeds the largest improvement in the Markov controls; the count of attributed configurations is reported with H1.

## Known limitations

- The optimal predictor of a noisy Hénon map is not available in closed form; the direct network is an estimate of the reachable level, not the optimum.
- Neural training adds approximation and optimization error, which the population theory does not cover; the convergence rule controls only the second.
- The design was made after the exploratory e02 analyses (flag, DB, TDB), which motivate it; they are reported as exploratory and not counted as confirmation.
- Generator tags: training 30, validation 31, test 32, reference 33 (new tags, so that no draw is shared with e01 or e01b).

## How to run

All stages run on CPU. The jobs are independent, so a machine with many cores runs them as parallel processes (`launch.sh`, every process takes every N-th job; results are written per job, so shards and machines can be merged by copying `results/runs/`). Trained parameters (`.pt`) are ignored by git and must be copied back with the JSON files.

On a Linux machine (e.g. a Hetzner Cloud CCX server, Ubuntu 22.04 or 24.04, 48 dedicated vCPU):

```
# on the laptop: copy the repository (no GitHub access needed)
rsync -az --exclude .venv --exclude logs --exclude '*/results/runs' ./ root@<ip>:anchor/

# on the server
apt update && apt install -y tmux
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
cd anchor && uv sync
tmux new -s e03
bash experiments/e03_henon_latent/launch.sh gap 12 2
bash experiments/e03_henon_latent/launch.sh select 24 2
bash experiments/e03_henon_latent/launch.sh final 24 2

# on the laptop: copy the results and logs back, then delete the server
rsync -az root@<ip>:anchor/experiments/e03_henon_latent/results/ experiments/e03_henon_latent/results/
rsync -az root@<ip>:anchor/logs/ logs/
```

The gap stage is recorded in this README (finding 7) before the selection stage starts; the final stage needs every selection run.
