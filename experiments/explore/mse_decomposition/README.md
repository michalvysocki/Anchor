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
