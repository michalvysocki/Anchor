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

**Losses.** Targets are in observation space and under stop-gradient, so collapse cannot occur (the latent-target version, as in TD-MPC2, is left for e05). Every variant has the one-step loss $(r(u_1) - y_{t+1})^2$; the auxiliary term is $\lambda$ times the mean over $k$ of $(r(u_k) - \mathrm{sg}[\text{target}_k])^2$, training horizons $k \in \{2, 3, 4\}$ (with $k = 1$, the first $2n$ horizons for a state of dimension $n = 2$; the identifiability argument of the flag check), $k$ sampled uniformly.

**Variants (core).** Each without and with the flag, so 8 in all.

| variant | target for the rollout to $k$ |
|---|---|
| onestep | none (λ = 0) |
| C | the observed $y_{t+k}$ (Monte Carlo; the target of TD-MPC and SPR, in observation space) |
| DB | $T_{k,p} + \delta_{k,p}(z_t)$, $p$ uniform on $2..k-1$ ($k = 2$: the observed $y_{t+2}$). $T_{k,p}$ is the decoded rollout of the same model from the window at $t+p$, its first step flagged as encoded. $\delta$ is a critic head on $\mathrm{sg}[\varphi(z_t)]$ (MLP, 1 hidden layer of 64, one output per pair), trained on $\mathrm{sg}[y_{t+k} - T_{k,p}]$ with 3× the model's learning rate |
| M\* | mixture over anchor points $p = 2..k$ ($p = k$ is C, $p < k$ is $T_{k,p}$) with weights minimizing $w^\top(B + V/N)w$ on the simplex, $B$ from the difference of two critic heads (regressions of $T_{k,p}$ and of $y_{t+k}$ on $\mathrm{sg}[\varphi(z_t)]$), $V$ the covariance of their residuals, $N$ the number of training windows; recomputed every 100 steps on a training batch, under stop-gradient (as in e02) |

**Ablations.** On two gap configurations, $\kappa = 0.5, L = 1$ and $\kappa = 0.8, L = 2$, main data arm. Each is compared with the variant it modifies:

| ablation | modifies | what it isolates |
|---|---|---|
| D, D+F | DB, DB+F | $\delta = 0$: the plain bootstrapped target and its information bias |
| TDB+F | DB+F | telescoped correction: one $\delta_p$ per anchor point added to $u_{p+1}$ and propagated by $g$ |
| DB+F oracle | DB+F | $\delta$ fitted on a large reference sample: separates the rule from the estimation of $\delta$ |
| F at λ = 0 | onestep+F | the flag without a multi-step term (prediction: no gain; $e_{\mathrm{pred}}$ is not trained) |
| C+F on {3, 4} | C+F | training horizons without $k = 2$ (identifiability) |
| C, matched capacity | C | without flag, dynamics widened to the parameter count of the two-network flag |
| C+F, two networks | C+F | separate networks $g_0$, $g_1$ instead of one network with an embedding |

**GRU arm** (encoder with memory, the "filter" row of the bias table). The window encoder is replaced by a GRU over the last 16 observations (hidden size 16, the latent). The window length no longer applies, so the arm has 3 configurations, $\kappa \in \{1, 0.8, 0.5\}$ ($\kappa = 1$ is a Markov control), with the variants onestep, C, C+F, DB+F and M\*+F. Prediction: both biases shrink, so the flag and $\delta$ matter less than with a window.

**Variance-regime arm.** On 4 configurations, $\kappa \in \{1, 0.5\} \times L \in \{1, 2\}$: (i) small data, (ii) long training horizons $k \in \{2..8\}$, closer to the limit of predictability. Variants C, C+F, D+F, DB+F, M\*+F. This is where target variance may matter, so where bootstrapped targets and M\* can gain.

**Not applicable.** A (composition consistency) holds by definition for a one-step model; B and E need a direct head, which is not a variant here. M (uniform mixture) and D without stop-gradient were settled in e02.

**Reference (not a variant).** A direct network $F(z_t, k)$ (window MLP 2×128, one-hot $k$, $k = 1..10$), trained on the same data; in the GRU arm the same head on a GRU encoder (finding 1). The gap of a configuration is the test MSE of onestep minus that of direct, with the same encoder.

**Data.** Separate generators per split (tags below). Main arm: 2000 training trajectories (about $10^5$ windows). Small-data arm: 20 trajectories (about $10^3$ windows). Validation: 200 trajectories, shared, used for λ selection and the stopping rule. Test: 500 trajectories, fixed. Every seed draws its own training set (lesson of e02). Windows hold $L$ (or 16 for the GRU) past observations and 10 future ones.

**Training.** Adam, learning rate $10^{-3}$, batches of 512 windows, gradient clipping 1.0. **Budget by convergence, identical for every variant, baseline and reference:** the validation loss of the variant is evaluated every 1000 steps; the learning rate is multiplied by 0.3 after 3 evaluations without a relative improvement of 0.1%; training stops when it falls below $10^{-5}$ or at 100 000 steps (runs stopped by the cap are reported). Each variant stops on its own loss, the objective it minimizes, so each reaches its own optimum; the validation rollout MSE is logged along the way, to show whether the primary metric was still moving at the stop (finding 4). λ ∈ {0.1, 1, 10}. Models of one configuration and variant are trained together as a batch of independent models where the code allows it.

**Seeds.** Selection seeds 0–2, evaluation seeds 3–12 (10 seeds), disjoint.

## Questions

**Q1 (model bias, main).** In the gap configurations, does the flag lower the rollout error of C (C+F vs C), and by how much of the gap?

**Q2 (T1 and the e01b harm).** In the Markov controls, is every variant harmless at every λ once training has converged? If a harm remains, the e01b harm is approximation error amplified by chaos, not non-convergence.

**Q3 (target bias).** Does DB improve on D, as the information bias predicts, or does it lose, which would mean that D's advantage in e01b came from its bias acting as a regularizer?

**Q4 (debiased bootstrap vs Monte Carlo).** Does DB+F differ from C+F, in the main arm and in the variance-regime arm?

**Q5 (method).** Does M\* match the best fixed target in every configuration and arm, while every fixed target loses somewhere?

**Q6 (critic).** Does the critic of the trained onestep model track the gap?

## Metrics

- **Primary:** test rollout MSE against $y_{t+k}$, mean over $k = 1..10$, in units of $\mathrm{Var}(y)$.
- **Comparison:** paired log ratio of the primary metric between two variants on the same seed (the same training data), mean and 95% t-interval over 10 seeds ($t = 2.262$), verdicts with the margin δ = log 1.02 as in e02 (better, material harm, small increase, harmless, inconclusive).
- **Fraction of the gap closed:** (onestep − variant) / (onestep − direct), in the gap configurations.
- Secondary: MSE against the clean $x_{t+k}$; MSE per $k$; fraction of diverged rollouts ($|\hat y| > 10$); steps to convergence; weights of M\*; validation errors of every λ (sensitivity of the flagged variants to λ).
- **Critic:** at the trained onestep model, the mean squared output of a critic head trained on $\mathrm{sg}[\text{rollout} - T_{k,p}]$ (the residual mean, Corollary), and the raw squared residual.

**λ selection.** Per configuration and variant, the λ with the lowest mean validation rollout MSE (k = 1..10) over the selection seeds.

## Hypotheses and decision criteria

| | prediction | criterion (test set, evaluation seeds) |
|---|---|---|
| H1 (Q1) | the flag helps, by less than in the linear case: only the model part of the bias is removed, the spread of the belief remains | C+F vs C "better" in at least 5 of the 7 gap configurations of the main arm; reported separately for the gap without noise ($\kappa = 1, L = 1$, missing memory only) and the gaps with noise (finding 6) |
| H2 (Q2) | no harm after convergence | in the 2 Markov controls, no "material harm" for any core variant at any λ, against onestep without flag |
| H3 (Q3) | two-sided; no direction is predicted | verdict of DB vs D and of DB+F vs D+F in the ablation configurations and the variance-regime arm; interpretation fixed now: DB "better" or "harmless" ⇒ D's advantage was the form of its target; DB "material harm" ⇒ it was its bias |
| H4 (Q4) | main arm: DB+F within the margin of C+F; any gain of DB+F only in the variance-regime arm | DB+F vs C+F verdict per configuration; H4 holds if "harmless" or "better" in every main-arm gap configuration |
| H-M\* (Q5) | M\* is never materially worse than the best fixed target; every fixed target is materially worse than the best somewhere | for every configuration and arm, the upper end of the CI of log(M\* / best fixed target with the same flag setting, each at its selected λ) is < log 1.02, the best taken over {C, DB} in the main grid and over {C, D, DB} where D is trained (ablation configurations, variance-regime arm; finding 3); and for each fixed target there is a configuration where its lower end against the best is > 0 |
| H5 (Q6) | the critic tracks the gap | Spearman correlation between the critic and the gap over the 9 window configurations and the 3 GRU configurations ≥ 0.8; critic in the Markov controls below that of every gap configuration |

**Outcomes.**

| | outcome | consequence |
|---|---|---|
| R1 | H1 and H2 hold | the flag carries to e05 as part of the method; H3–H5 and H-M\* set the default target (C, DB or M\*) |
| R2 | H1 fails, H2 holds | the flag is a linear-case result and a diagnostic (the decomposition of the bias), not a method component for chaotic dynamics; Anchor stays the plug-in loss |
| R3 | H2 fails | harm after convergence: the e01b harm is approximation error amplified by chaos; a mechanism that switches off (adaptive λ) becomes necessary, not optional |
| R4 | a variant beats the direct reference on the primary metric by more than the margin, or the stopping rule fails for most runs | contradicts the setup; check the code and data before interpreting |

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

## Gap measured before the Anchor runs

To do (finding 7).

## Known limitations

- The optimal predictor of a noisy Hénon map is not available in closed form; the direct network is an estimate of the reachable level, not the optimum.
- Neural training adds approximation and optimization error, which the population theory does not cover; the convergence rule controls only the second.
- 10 evaluation seeds give wider intervals than e02 (30).
- The design was made after the exploratory e02 analyses (flag, DB, TDB), which motivate it; they are reported as exploratory and not counted as confirmation.
- Generator tags: training 30, validation 31, test 32, reference 33 (new tags, so that no draw is shared with e01 or e01b).
