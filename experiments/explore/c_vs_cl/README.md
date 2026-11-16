# Exploratory pilot: what keeps information in the rollout under latent targets

Not pre-registered and not part of e03; it imports e03's code and the measurements of the off-curve pilot without changing them. Decided on 2026-11-15 after the off-curve pilot (`../offcurve`), which refuted drift off the curve of encodings as the cause of the horizon reversal. At $k = 5$ in Hénon, C and CL lay equally far from the curve but differed in what the off-curve component carried (memory gain 0.16 against 0.35), so the difference lies in the content of the rollout state.

**Measure.** The information horizon $k^*$ is the first $k$ at which the correlation of the predictions with $y_{t+k}$ falls below 0.3, per seed. In the off-curve pilot (Hénon, κ = 1, L = 1, horizons 2..4) it was 6 for C in 10/10 seeds and 8–9 for CL. It is the first crossing, so a correlation that oscillates around 0.3 counts at its first dip.

**Two candidates** (not mutually exclusive):
- (a) **Specialization to the trained horizon.** C trains the compositions $g^k$ to hit observations up to $k = 4$, and nothing requires the state after $k$ steps to be a good start for further steps. Under CL the target at every $k$ is of the same kind (a state like an encoding).
- (b) **Co-adaptation of the decoder.** In C the terms at $k = 2..4$ also train the decoder, on rollout states, so the decoder and the dynamics can agree on states that only this decoder reads, and only up to $k = 4$. Under CL the decoder is trained on $u_1$ alone. In the off-curve pilot C's decoder shift grew with $k$ much faster than CL's (0.86 against 0.14 at $k = 20$).

**Arms.** Hénon at κ = 1, L = 1, seeds 300–309, training of the horizon pilot (30 000 steps, Adam $10^{-3}$ reduced tenfold for the last 20%, λ = 1, generator seed 17):
- **C8:** C with horizons 2..8, testing (a).
- **CL8:** CL with horizons 2..8, the reference for C8.
- **C_sgdec:** C with horizons 2..4 whose terms at $k \ge 2$ reach the encoder and the dynamics but not the decoder, which learns from $k = 1$ only, as in CL. This tests (b).

C and CL with horizons 2..4 are not rerun: their results are in the off-curve pilot, and the model class here reproduces them exactly (checked below). Measurements are those of the off-curve pilot (`measure`), with $k^*$ added.

**Predictions** (written before the first run, with tolerances):
- Q1 (specialization): if (a) holds, $k^*(\mathrm{C8}) \ge 9$ in at least 8 of the 10 seeds; if not, $k^*(\mathrm{C8}) \le 7$ in at least 8 of the 10 seeds. A value in between is read as partial.
- Q2 (co-adaptation of the decoder): if (b) holds, $k^*(\mathrm{C\_sgdec}) \ge 8$ in at least 8 of the 10 seeds, with an MSE over $k = 1..4$ within +0.02 of C's (0.274).
- Q3 (descriptive): $k^*(\mathrm{CL8})$ against $k^*(\mathrm{CL})$.

**Caveat.** The attainable floor for Hénon from $y_t$ alone is unknown, so $k^* \approx 9$ may be the limit of predictability from one observation rather than an effect of the target. If C8 and CL8 both stop at 9, it may be the system's ceiling. CL8 is in the design so that a value beyond 9 would rule that out.

A second caveat: the auxiliary terms are averaged over horizons, as in e03, so in C8 and CL8 each of the terms at $k = 2..4$ weighs 1/7 instead of 1/3. A change of the MSE over $k = 1..4$ in these arms may come from this dilution rather than from the longer horizons.

**Checks before the run** (300 steps):
- All three arms start from the same initial weights as the horizon pilot's models (the horizons change no module).
- The loss of C8 equals the one-step term plus the mean of the seven terms at $k = 2..8$; the targets up to $k = 8$ fit in the training windows (width 11).
- With horizons 2..4 and the decoder trained, the model class here gives weights identical to the horizon pilot's for C and CL (maximum difference 0.0).
- The cut decoder gives the same values as the decoder and the same gradient with respect to its input, and no gradient to its own parameters.
- In C_sgdec the decoder's gradient equals that of the one-step term alone, while the gradients of the encoder and the dynamics equal plain C's.
- $k^*$ recomputed on the off-curve results gives 6 for C in every seed and 8–9 for CL.
- A step of C8 takes about 0.12 s.
- The smoke run of all three arms (2 min) runs end to end.

Run: `caffeinate -i uv run python experiments/explore/c_vs_cl/pilot.py "" 2` (results in `results/pilot.json`, finished runs are skipped).

## Results (2026-11-16)

All three arms finished. C and CL are the off-curve pilot's runs (same seeds, horizons 2..4). The variance of $y$ is 0.99–1.00 at every $k$.

Information horizon $k^*$ per seed (seeds 300–309):

| arm | $k^*$ |
|---|---|
| C | 6 6 6 6 6 6 6 6 6 6 |
| CL | 8 9 9 9 9 8 8 8 9 9 |
| C8 | 9 9 9 9 9 9 9 9 9 9 |
| CL8 | 10 9 10 10 2 9 10 10 10 10 |
| C_sgdec | 6 6 6 7 6 6 6 6 6 6 |

Rollout MSE, median over seeds [min, max]:

| arm | $k = 1..4$ | $k = 5..8$ | $k = 9..14$ |
|---|---|---|---|
| C | 0.274 [0.273, 0.275] | 1.123 [0.984, 1.521] | 1.392 [1.068, 2.414] |
| CL | 0.299 [0.287, 0.369] | 0.819 [0.749, 0.873] | 1.295 [1.235, 1.367] |
| C8 | 0.285 [0.279, 0.288] | 0.595 [0.568, 0.605] | 1.775 [1.204, 4.198] |
| CL8 | 0.364 [0.303, 0.718] | 0.722 [0.656, 0.899] | 1.006 [0.991, 1.199] |
| C_sgdec | 0.279 [0.277, 0.281] | 1.252 [1.006, 1.528] | 1.635 [1.119, 2.019] |

Number of seeds (of 10) with the lower MSE:

| | $k = 1..4$ | $k = 5..8$ | $k = 9..14$ |
|---|---|---|---|
| C8 vs C | 10 | 10 | 2 |
| C8 vs CL8 | 10 | 10 | 0 |
| CL8 vs CL | 1 | 8 | 10 |
| C_sgdec vs C | 0 | 4 | 5 |

Medians over seeds at $k = 2, 4, 8, 10, 12$ (sharpness is the standard deviation of the predictions over that of $y$; for the prediction that minimizes MSE it equals the correlation):

| | C8 | CL8 |
|---|---|---|
| correlation | .85, .72, .62, .07, −.01 | .78, .67, .50, .20, .11 |
| sharpness | .86, .72, .60, .66, .65 | .65, .57, .40, .30, .29 |
| distance to the curve | .37, 1.21, 3.45, 4.75, 6.02 | .26, .29, .39, .42, .42 |
| decoder shift | .44, 1.02, .73, 1.64, 1.70 | .24, .27, .30, .43, .44 |

C8's bias grows beyond $k = 8$ (median −0.13 at $k = 10$, −0.28 at $k = 14$).

- **Q1 holds: (a).** $k^*(\mathrm{C8}) = 9$ in 10/10 seeds. The correlation falls from .62 at $k = 8$ to .18 at $k = 9$; CL8 still has .32 at $k = 9$, so this is not the system's ceiling. Beyond $k = 8$ C8 is worse than C (lower in 2/10 seeds over $k = 9..14$, median 1.78, above the variance of $y$).
- **Q2 fails.** $k^*(\mathrm{C\_sgdec})$ is 6 in 9 seeds and 7 in one. The MSE condition holds (+0.002 to +0.007 over C at $k = 1..4$); the information horizon is what fails. The correlation profile of C_sgdec matches C's, so cutting the decoder off the multi-step terms changes nothing.
- **Q3.** Against CL, $k^*(\mathrm{CL8})$ is larger in 8 seeds, equal in 1 (301) and smaller in 1 (304). In seed 304 the value 2 comes from the first-crossing definition: the correlation dips to −0.15 at $k = 2$ and returns to 0.30–0.63 over $k = 3..8$ (counted from $k = 3$, $k^* = 9$). The seed is nonetheless poorly fitted at short horizons (MSE 1.33 at $k = 2$, 0.718 over $k = 1..4$).

**Reading.** The horizon reversal moves with the trained horizon: with horizons 2..8, C8 is better than CL8 up to $k = 8$ and worse beyond (10/10 both ways), as C and CL were around $k = 4$. Observation targets keep information up to the trained horizon and lose it in the next step; co-adaptation of the decoder is excluded. Read after the fact, not predicted: C8 is calibrated up to $k = 8$ (sharpness equals correlation) and beyond it keeps the amplitude of its predictions without the information; CL8 is under-sharp inside the window (.65 against .78 at $k = 2$) and lowers its amplitude with its information beyond it. C8's state leaves the curve far already inside the window (1.21 at $k = 4$) while staying informative, and the decoder reads that displacement (decoder shift about 1), so drift in directions the decoder ignores does not describe it. Limits: one system, two trained horizons, a calibration reading made after the results, and CL8's short-horizon cost confounded with the 1/7 weights.
