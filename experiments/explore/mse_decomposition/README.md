# Exploratory analysis: which part of the rollout error grows beyond the trained horizon?

Not pre-registered and not part of e03. No training: it reads the saved measures of the calibration pilot (`../calibration/results/pilot.json`) and does not change them. Decided on 2026-11-23 as the first step of the diagnosis of the error beyond $K$ (the kinematics of the trajectory).

The sharpness and the correlation of these arms were read before the predictions below were written (calibration pilot, 2026-11-16), so they are not blind and cannot confirm anything; a confirmation belongs to e04.

**Decomposition.** Per seed and $k$, with $b$ the difference of the means of the predictions and of $y_{t+k}$, $s$ the sharpness (standard deviation of the predictions over that of $y_{t+k}$), $\rho$ their correlation and $V$ the variance of $y_{t+k}$:

$$
\mathrm{MSE} = \underbrace{b^2}_{\text{shift}} + \underbrace{V(s-\rho)^2}_{\text{miscalibration}} + \underbrace{V(1-\rho^2)}_{\text{lost information}}
$$

The first two terms can be removed by a linear rescaling of the predictions; the third cannot. Each term is reported as a share of the MSE.

**Arms.** henon_C6 and henon_CL6 (Hénon, $\kappa = 1$, $L = 1$, $K = 6$, 10 seeds, $k \le 20$). The AR(2) arms of the pilot are not covered by the predictions.

**Predictions** (written 2026-11-23, before the decomposition was computed):

| | Range | Arm | Prediction |
|---|---|---|---|
| P1 | $k = 7..8$ | C | lost information is the largest term; miscalibration is at least 25% of the MSE |
| P2 | $k = 7..8$ | CL | lost information is the largest term; the share of miscalibration is at most 0.5 × that of C |
| P3 | $k = 15..20$ | C | miscalibration is the largest term; its share grows by at least 15 percentage points from $k = 7..8$ |
| P4 | $k = 15..20$ | CL | lost information is the largest term |
| P5 | both ranges | C, CL | $b^2$ is below 10% of the MSE |

**Evaluation.**
- Shares are computed per seed and $k$ and averaged over the $k$ of the range. Thresholds apply to the median over the 10 seeds: for P2, the median share of CL against the median share of C; for P3, the median of the per-seed growth.
- "Largest term" holds per seed on the averaged shares: confirmed in at least 8 of 10 seeds, inconclusive in 6 or 7, refuted in at most 5.
- A prediction holds if both of its parts hold. It is refuted if either part is refuted, and inconclusive otherwise.
- P5, per arm and range on the median share: confirmed below 10%; not confirmed from 10% to 40%; refuted above 40%, or at 10% or more while exceeding the median share of miscalibration.
- Correctness check: the three terms sum to the MSE within 1% for every seed and $k$. If not, the code is checked and nothing is interpreted.

## Results (2026-11-23)

Reproduce with `python experiments/explore/mse_decomposition/analyze.py`; per-seed shares are in `results/decomposition.json`. Correctness check passed: the three terms sum to the MSE within 0.3% for every seed and $k$.

| | Verdict | Numbers |
|---|---|---|
| P1 | refuted | information largest in 10/10, but miscalibration is 20.4% of the MSE (threshold 25%) |
| P2 | holds | information largest in 10/10; miscalibration 0.013 against a limit of 0.102 |
| P3 | refuted | miscalibration largest in 2/10; its share changes by −4.7 percentage points |
| P4 | holds | information largest in 10/10 |
| P5 | refuted | holds for CL; for C at $k = 15..20$, $b^2$ is 17.5% of the MSE and exceeds miscalibration (12.8%) |

Median shares of the MSE over 10 seeds:

| Arm | Range | $b^2$ | Miscalibration | Lost information | Largest in seeds ($b^2$ / miscalibration / information) |
|---|---|---|---|---|---|
| C | $k = 7..8$ | 0.025 | 0.204 | 0.774 | 0 / 0 / 10 |
| C | $k = 15..20$ | 0.175 | 0.128 | 0.387 | 4 / 2 / 4 |
| CL | $k = 7..8$ | 0.022 | 0.013 | 0.952 | 0 / 0 / 10 |
| CL | $k = 15..20$ | 0.017 | 0.175 | 0.785 | 0 / 0 / 10 |

Read after the fact, not predicted (10 seeds, one system, one $K$):

- **The edge of C has two steps.** From $k = 6$ to 7 the median MSE grows by 0.528, mostly lost information (+0.317, largest in 9/10 seeds), with miscalibration +0.164 and $b^2$ +0.025. From $k = 7$ to 8 it grows by 0.137, mostly miscalibration (+0.094, largest in 7/10). In CL the step from 6 to 7 adds 0.095, with no growth of miscalibration.
- **CL keeps more information at the edge.** With $V \approx 1$, $\rho \approx 0.27$ (C) and 0.48 (CL) at $k = 7$, and 0.15 and 0.34 at $k = 8$.
- **The far tail of C has no single regime.** At $k = 15..20$, $\rho \approx 0$ in every seed. Seeds 0, 5, 6, 8 are dominated by a shift of the mean (MSE 3.2–10.1), seeds 2, 7 by amplitude ($s \ge 1.09$, MSE 2.3–3.0), and seeds 1, 3, 4, 9 by lost information with a contracted amplitude ($s \le 0.42$, MSE 1.1–1.4). Six of the 10 seeds exceed $2V$ throughout this range. The grouping is post hoc.
- **CL rarely drifts.** The median $b^2$ is about 2% of the MSE; seeds 8 and 9 reach 22% and 18% at $k = 15..20$. No seed of CL exceeds $2V$ in this range.
- **The MSE in the tail favours contraction.** With $\rho = 0$, a constant at the mean has MSE $V$, while a faithful simulation of the system has $2V$. The advantage of CL at $k = 15..20$ is therefore not evidence of more information; a shift of the mean is a defect under both readings.
