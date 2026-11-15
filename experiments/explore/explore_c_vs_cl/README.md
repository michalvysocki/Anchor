# Exploratory pilot: what keeps information in the rollout under latent targets

Not pre-registered and not part of e03; it imports e03's code and the measurements of the off-curve pilot without changing them. Decided on 2026-11-15 after the off-curve pilot (`../explore_offcurve`), which refuted drift off the curve of encodings as the cause of the horizon reversal. At $k = 5$ in Hénon, C and CL lay equally far from the curve but differed in what the off-curve component carried (memory gain 0.16 against 0.35), so the difference lies in the content of the rollout state.

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

Run: `caffeinate -i uv run python experiments/explore/explore_c_vs_cl/pilot.py "" 2` (results in `results/pilot.json`, finished runs are skipped).
