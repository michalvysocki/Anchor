# Exploratory pilot: does the edge move with the trained horizon, and is what breaks beyond it calibration?

Not pre-registered and not part of e03; it imports e03's code and the code of the earlier pilots without changing it. Decided on 2026-11-16 after the c_vs_cl pilot (`../c_vs_cl`), which found, on Hénon at κ = 1, L = 1, that observation targets (C) keep information up to the trained horizon $K$ and lose it in the next step ($k^* = 6$ at $K = 4$, $k^* = 9$ at $K = 8$, both 10/10 seeds), and that co-adaptation of the decoder does not explain it.

Read after the fact in c_vs_cl, not predicted: C8 was calibrated up to $k = 8$ and kept the amplitude of its predictions without the information beyond it; CL8 lowered its amplitude with its information. This pilot tests both findings on arms that have not been run.

**Measures.**
- Information horizon $k^*$: the first $k$ at which the correlation of the predictions with $y_{t+k}$ falls below 0.3, per seed (as in c_vs_cl; first crossing).
- Calibration gap: sharpness minus correlation, per seed and $k$, where sharpness is the standard deviation of the predictions over that of $y_{t+k}$. For the prediction that minimizes MSE given any information the two are equal (the covariance of $\mathbb{E}[y \mid x]$ with $y$ is its variance), and the best rescaling of any prediction makes them equal, so the gap needs no floor and applies to Hénon. A positive gap means more amplitude than information.
- For AR(2), the floor from $y_t$ (the best prediction from what the encoder sees) as an additional reference, computed empirically: the MSE of $\rho_k y_t$ on the same validation windows. On these 8 800 windows (200 trajectories) that MSE differs from the analytic floor by up to 0.044 at a single $k$ (1.026 against 0.982 at $k = 3$), the size of the margin in P3, so the analytic floor would mostly measure the sample. Both are stored.

**Arms** (training of the horizon pilot: 30 000 steps, Adam $10^{-3}$ reduced tenfold for the last 20%, λ = 1, generator seed 17; seeds 300–309; κ = 1, L = 1; the model class of c_vs_cl):
- **henon_C6, henon_CL6:** C and CL with horizons 2..6, on the data of c_vs_cl. Together with $K = 4$ (off-curve pilot) and $K = 8$ (c_vs_cl) this gives three trained horizons, the new one between the other two.
- **ar2_C8, ar2_CL8:** C and CL with horizons 2..8, on the AR(2) data of the off-curve pilot (radius 0.9, angle π/6, the same draws). $K = 4$ there is the off-curve pilot's ar2_C and ar2_CL.

The auxiliary terms are averaged over horizons, as in e03 and c_vs_cl, so the comparison with the existing arms holds; each short term then weighs $1/(K-1)$.

**Predictions** (written before the first run, with tolerances). Only the four new arms count; the existing arms are reported with the same numbers but cannot confirm anything, since the thresholds below were chosen after seeing C8 and CL8 on Hénon (C8: $|\text{gap}| \le 0.02$ inside the window, about 0.6 beyond it; CL8 0.10–0.18 beyond it).
- **P1 (the edge at $K + 1$, Hénon only):** $k^*(\mathrm{C6}) = 7$ in at least 8 of the 10 seeds, and $k^*(\mathrm{CL6}) \ge k^*(\mathrm{C6})$ in at least 8 of the 10 seeds (paired by seed).
- **P2 (calibration).** Per seed, the inside gap is the median of |gap| over $k = 1..K$ and the outside gap the median of the gap over $k = K+2..K+6$.
  - For each C arm (henon_C6, ar2_C8): inside gap ≤ 0.1 and outside gap ≥ 0.3 in the same seed, in at least 8 of the 10 seeds (the claim is about a single model: calibrated inside the window and not beyond it).
  - For each CL arm (henon_CL6, ar2_CL8): |outside gap| < 0.3 in at least 8 of the 10 seeds. Two-sided, because the property in question is calibration beyond $K$: a CL arm that shrinks its predictions far below its information is a different defect, not an example of it.
  - P2 holds if all four parts hold; a C arm whose seeds meet only one of the two conditions is read as partial.
  - Decided before the run, after the checks: first written as two separate counts for C and a one-sided condition for CL (outside gap < 0.3); made joint and two-sided before any new arm was trained.
- **P3 (AR(2), against the empirical floor):** the median over seeds of the mean of MSE − floor over $k = 1..8$ for ar2_C8 is at most 0.02, and the mean of MSE − floor over $k = 9..14$ is larger for ar2_C8 than for ar2_CL8 in at least 8 of the 10 seeds.

**Caveats.**
- In AR(2) the correlation oscillates and changes sign (the autocorrelation crosses zero), so $k^*$ is not used there (P1 is Hénon only). The calibration gap is still defined: a negative correlation with positive sharpness gives a large gap.
- $k^*$ is the first crossing, so a single dip counts (seed 304 of CL8 in c_vs_cl).
- With $K = 6$ the short terms weigh 1/5 instead of 1/3 or 1/7, so changes in the MSE over $k = 1..4$ between the three Hénon horizons mix the horizon with the weights.

**Checks before the run** (300 steps where training is involved; all passed):
- The AR(2) data are the off-curve pilot's draws (identical tensors; training windows of width 11, so targets up to $k = 10$ fit).
- With horizons 2..4 on the AR(2) data, the model class here gives weights identical to the off-curve pilot's for C and CL (maximum difference 0.0).
- The loss of C6 and CL8 equals the one-step term plus the mean of the $K - 1$ multi-step terms (difference ≤ 1.2e-7).
- The inside and outside gaps take the intended ranges ($k = 1..K$ and $k = K+2..K+6$), checked on synthetic values.
- The gaps recomputed on the existing arms match the values quoted above (Hénon C8: 0.02 inside, 0.63 outside).
- The calibration gap of the best prediction from $y_t$ on the validation windows is between −0.055 and +0.021 at $k = 1, 3, 5, 9, 10, 12, 14$ (zero up to sampling noise); a prediction with the same information and full amplitude ($\mathrm{sign}(\rho_k)\, y_t$) gives +0.50 at $k = 5$ and +0.86 at $k = 10$.
- The empirical floor reproduces the MSE of $\rho_k y_t$ computed separately (0.272, 1.026, 0.780, 1.011 at $k = 1, 3, 5, 10$). Against it, the off-curve pilot's ar2_C (K = 4) lies 0.004 above the floor over $k = 1..4$ (0.033 against the analytic floor).
- A step of ar2_C8 takes about 0.20 s while e03 runs on the same machine.
- The smoke run of all four arms runs end to end and writes the AR(2) floors.

The checks also printed the gaps of the existing arms against the thresholds, after the predictions above were written; the predictions were not changed. Outside gap ≥ 0.3 (one-sided counts, printed before P2 was made joint and two-sided): Hénon C4 8/10, CL4 7/10, C8 9/10, CL8 0/10; AR(2) C4 5/10, CL4 6/10. So at $K = 4$ the CL arms do not meet the CL condition of P2 either, and AR(2) C4 does not meet the C condition. With the final conditions (printed after the change, by the smoke run): C joint count Hénon C4 8/10, C8 9/10, AR(2) C4 5/10; CL two-sided count Hénon CL4 3/10, CL8 10/10, AR(2) CL4 4/10.

Run: `caffeinate -i uv run python experiments/explore/calibration/pilot.py "" 2` (results in `results/pilot.json`, finished arms are skipped).

## Results (2026-11-16)

All four arms finished (10 seeds each, no NaN). The existing arms are listed with them for comparison and do not count toward the predictions.

**Outcome of the predictions.**
- **P1 fails.** $k^*(\mathrm{C6}) = 7$ in 7 of 10 seeds and 8 in the other three. $k^*(\mathrm{CL6}) \ge k^*(\mathrm{C6})$ in 10/10 pairs.
- **P2 fails.** henon_C6 10/10, henon_CL6 3/10, ar2_C8 4/10, ar2_CL8 9/10.
- **P3 partly holds.** First part: ar2_C8 is 0.008 above the floor over $k = 1..8$ (threshold 0.02). Second part fails: C8 has the larger excess over $k = 9..14$ in 6 of 10 seeds (threshold 8/10; medians 0.106 against 0.082).

Information horizon $k^*$ per seed on Hénon (seeds 300–309; K = 4 from the off-curve pilot, K = 8 from c_vs_cl):

| $K$ | C | CL |
|---|---|---|
| 4 | 6 6 6 6 6 6 6 6 6 6 | 8 9 9 9 9 8 8 8 9 9 |
| 6 | 7 8 8 8 7 7 7 7 7 7 | 9 9 9 9 9 8 9 9 8 9 |
| 8 | 9 9 9 9 9 9 9 9 9 9 | 10 9 10 10 2 9 10 10 10 10 |

Calibration gaps (medians over seeds; outside with [min, max]), the P2 count under the final conditions, and the MSE over $k = 1..4$. For AR(2), also the MSE above the empirical floor over $k = 1..K$ and $k = K+1..K+6$. New arms in bold.

| arm | inside gap | outside gap | P2 count | MSE $k = 1..4$ | above floor $1..K$ | above floor $K+1..K+6$ |
|---|---|---|---|---|---|---|
| henon_C4 | 0.01 | 0.55 [0.23, 0.82] | 8/10 | 0.274 | | |
| henon_CL4 | 0.03 | 0.32 [0.07, 0.48] | 3/10 | 0.299 | | |
| **henon_C6** | 0.01 | 0.64 [0.35, 1.20] | 10/10 | 0.278 | | |
| **henon_CL6** | 0.04 | 0.38 [0.17, 0.51] | 3/10 | 0.339 | | |
| henon_C8 | 0.02 | 0.63 [0.21, 1.84] | 9/10 | 0.285 | | |
| henon_CL8 | 0.08 | 0.18 [−0.01, 0.28] | 10/10 | 0.364 | | |
| ar2_C4 | 0.04 | 0.33 [0.20, 0.96] | 5/10 | 0.760 | 0.004 | 0.250 |
| ar2_CL4 | 0.17 | 0.33 [0.24, 0.51] | 4/10 | 0.826 | 0.070 | 0.285 |
| **ar2_C8** | 0.05 | 0.18 [−0.03, 0.69] | 4/10 | 0.763 | 0.008 | 0.106 |
| **ar2_CL8** | 0.23 | −0.00 [−0.16, 0.32] | 9/10 | 1.024 | 0.218 | 0.082 |

The variance of $y$ is 0.99–1.00 on Hénon and about 1.04 on AR(2).

**Reading.**
- **Hénon, the edge.** The edge of C moves monotonically with the trained horizon in every seed, between $K + 1$ and $K + 2$ (6 at $K = 4$, 7–8 at $K = 6$, 9 at $K = 8$). The point prediction of P1 misses by one seed. The $k^*$ of CL hardly depends on $K$ (8–9 at $K = 4$ and 6, 9–10 at $K = 8$) and is never below C's.
- **Hénon, short horizons.** The MSE of CL over $k = 1..4$ grows with $K$ (0.299, 0.339, 0.364), while C's stays near 0.28 (0.274, 0.278, 0.285). The growth is confounded with the weights $1/(K-1)$ of the short terms.
- **Hénon, calibration of C.** C is calibrated inside the window and loses calibration beyond it at all three $K$ (henon_C6 10/10). Beyond the window the median MSE of C6 grows to 2.8 at $k = 20$ and exceeds 2 in 6 of 10 seeds. The largest value is 12.5, in one seed at $k = 18$.
- **Hénon, calibration of CL.** At $K = 4$ and 6 CL also loses calibration beyond the window, but less than C: its outside gap is smaller than C6's in 8 of 10 pairs, and no CL seed exceeds an MSE of 1.63. CL is calibrated beyond the window only at $K = 8$ (median 0.18, 10/10).
- **AR(2), C8.** C8 is calibrated inside the window (0.03–0.09 in every seed) and at the floor there (0.008). Its outside gap varies by seed (median 0.18, −0.03 to 0.69), so the C condition of P2 fails.
- **AR(2), C at K = 4 and 8.** Over the same steps relative to the end of the window ($K+1..K+6$), C's excess above the floor falls from 0.25 at $K = 4$ to 0.106 at $K = 8$, about 2.4-fold. Decay of the signal does not explain this: the autocorrelation of AR(2) oscillates with period 12 and is still sizeable at the end of the window and beyond it (−0.28 at $k = 8$, 0.24 and 0.28 at $k = 11$ and 12). Two candidate causes, not separated by this design:
  - $K = 8$ covers two thirds of the period, which constrains the rotation in the transition more;
  - the weights $1/(K-1)$ of the short terms change with $K$.
- **AR(2), CL8.** CL8 meets the CL condition beyond the window at the cost of a large gap inside it (median 0.23; 0.218 above the floor over $k = 1..8$). This is excess amplitude, not shrinkage:
  - at $k = 1$ it is at the floor (0.273 against 0.272);
  - at $k = 2$ and 3 the sharpness is 0.92 and 0.74, against correlations of 0.52 and 0.10;
  - at $k = 4$ the correlation is negative (−0.19);
  - the MSE at $k = 3$ and 4 (1.47 and 1.26) is above the variance of $y$.
- **Limits.** The comparisons across $K$ are after the fact and confounded with the term weights. AR(2) has two trained horizons and Hénon three. The two causes of the AR(2) difference between $K = 4$ and 8 are hypotheses.
