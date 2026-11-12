# Theory: statements and their numerical checks

Every statement of the theory has one script that checks it numerically: closed forms against Monte Carlo estimates from 400k stationary AR(1) chains (a check passes within 4 standard errors), and properties of the linear window model against its exact population losses. Derivations are in the research note ("Derivations on AR(1) with observation noise"); statements and proofs in `paper/sections/theory.tex` and the appendix.

```bash
make theory                              # every check
uv run python theory/residual_lemma.py   # one statement
```

## Map of the theory

The chain of results, from the setting to the method:

| statement (paper label) | assumes | shows | check | tested in experiments |
|---|---|---|---|---|
| Markov gap of a noisy observation (setting of T2, T4) | AR(1), observation noise, $\|a\| < 1$ | $\mathbb E[y_{t+j} \mid y_t] = a^j\kappa y_t$; each bootstrap point shrinks the target by κ; the rollout of the best one-step model reaches κ^{k−1} of the optimum | `markov_gap.py` | e01 (direct beats rollout under partial observation), e02 (gap configurations) |
| Target variance (T3, scalar part) | Markov state, ideal model | the bootstrapped target removes the noise of the last $q$ steps | `target_variance.py` | e01b (D > C on chaotic dynamics); e02 (no effect on stable linear systems) |
| Residual lemma (`lem:residual`) | any process, affine one-step model | $\mathbb E[R \mid z_t] = A^q(g^p(z_t) - \mathbb E[z_{t+p} \mid z_t])$: the rollout bias at $p$ carried forward by $q$ steps; the raw residual adds the target variance | `residual_lemma.py` | e01b exploratory analysis (corrected), e02 H5 |
| Critic (`cor:critic`) | the lemma; a head trained by MSE on the target | the head minus the rollout measures the bias without the target variance; with $p = k$ it is the direct predictor | `residual_lemma.py` | e02 H5 (Spearman 0.96 vs 0.52 for the raw residual) |
| No signal at $p = 1$ (`rem:p1`) | one-step model at its optimum | anchor points $p = 1$ add only variance | `residual_lemma.py` | e02 ablation M\*_p1 |
| Harmlessness T1 (`prop:t1`) | latent Markov in the mean, exact affine model, stop-gradient | the one-step optimum is a fixed point for every λ and weights; without stop-gradient it moves | `harmlessness.py` | e02 H2 (after the convergence check), ablation D_nosg |
| Small Anchor helps, T2 (`prop:t2-ar1`) | AR(1), $k = 3$, $p = 2$, stop-gradient | $b'(0) = 3a^5\kappa^4(1-\kappa)$; first-order gain at every $k \ge 2$, second-order cost at $k = 1$; λ → ∞ gives exactness at $k = p$ and a degenerate branch toward 0 | `improvement.py` | e02 H1 (gains predicted to the third decimal) |
| Rollout bound T4 (`prop:t4-ar1`) | AR(1), $\|a\| < 1$, ideal predictors | rollout bias at $k$ ≤ $(k-1)\|a\|^{k-2}$ × disagreement at $k = 2$ | `rollout_bound.py` | not yet tested directly |
| Bias–variance weights (method M\*) | the critic estimates the bias, the residual the variance | weights minimizing $w^\top(B + V/N)w$; C as $N \to \infty$ | `tests/test_lgssm.py` | e02 H4 (M\* matches the better of C and D) |

`common.py` holds the shared simulation, regression and report. The population theory of the linear window model (fixed points, critic, weights) is in the library, `src/anchor/linear.py`, because e02 also trains with it.

## Open

- Matrix versions of T2 and T4 on the linear-Gaussian state-space model (telescoping the rollout error).
- T1 and the lemma for nonlinear models: the Jensen term $\mathbb E[g^q(Y)] - g^q(\mathbb E[Y])$.
- T3 in full: the bias–variance trade-off that sets the weights, and when bootstrapped targets win (chaos, learned encoders).
- T5: no collapse of a learned encoder.
- Stability of the fixed points and the optimal λ (numerically it depends on the dynamics more than on the gap).

## Result (2026-11-12)

All 166 checks pass:

| script | checks |
|---|---|
| `target_variance.py` | 18 |
| `markov_gap.py` | 27 |
| `residual_lemma.py` | 60 |
| `harmlessness.py` | 31 |
| `improvement.py` | 20 |
| `rollout_bound.py` | 10 |

The AR(1) checks are the 115 of the former `theory/stage1_ar1/verify.py`, regrouped by statement; each configuration now has its own generator, so the estimates differ in the last digits. The T1 and T2 checks are new.
