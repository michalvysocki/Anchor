# Exploratory pilot: what observation targets do to the latent beyond the trained horizon

Not pre-registered and not part of e03; it imports e03's code and the horizon pilot's model without changing them. Decided on 2026-11-15 after the horizon pilot (`../horizon`). There, at κ = 1, L = 1:
- observation targets (C) and latent targets (CL) carried the same information about $y_{t+k}$ on the trained horizons (correlation 0.72 against 0.70 at $k = 4$);
- beyond them C lost it within two or three steps, while CL kept it (0.09 against 0.37 at $k = 7$);
- the mean latent norm did not explain the difference.

The aim is to explain the mechanism before any method is built on it, and to check the reversal on a second system.

**Working hypothesis.** Observation targets constrain only what the decoder reads, so the dynamics may move the state in directions of the latent the decoder does not see, and these build up under composition. Latent targets keep the whole state close to the image of the encoder, so the composition keeps working.

**Why the measurement has to separate memory from noise.** With a window of $L = 1$ the encoder sees one observation, so every encoding lies on a one-dimensional curve $E(y)$ in the 16-dimensional latent. Leaving the curve is not harmful in itself. When the latent is not Markov (Hénon or AR(2) observed through one value), a rollout that predicts well must carry memory off the curve, which CL penalizes up to $k = 4$ and C allows. The hypothesis therefore predicts off-curve displacement *without* information, not off-curve displacement as such.

**Setup.** Variants C and CL and their training exactly as in the horizon pilot: seeds 300–309, 30 000 steps, Adam $10^{-3}$ reduced tenfold for the last 20%, λ = 1, horizons {2, 3, 4}, generator seed 17. All systems at κ = 1 (no observation noise), L = 1:
- **henon:** the horizon pilot's configuration, retrained identically. Its MSE must match the stored one, which checks reproducibility.
- **ar2:** AR(2) with roots of radius 0.9 and angle π/6, as in e02. Not Markov at L = 1; the linear counterpart of Hénon.
- **ar1:** AR(1) with a = 0.99. Markov at L = 1, which makes it the control.

The AR data are drawn exactly from the stationary distribution (`anchor.envs.lgssm`, generator tag 50, no draw shared with e02 or e03), with 2000 trajectories of 64 observations per seed for training and 200 for validation, in units of $\sqrt{\mathrm{Var}(y)}$. These are the same sizes as e03.

**Measured** on validation windows extended to $k = 20$, per seed and per $k$:
- rollout MSE, and the exact correlation, bias and sharpness of the predictions;
- **distance to the curve** $d_k$: the mean distance of $u_k$ to the nearest point of $E(y)$ on a grid of 2001 values of $y$, in units of the spread of the encodings (the encodings' own distance is recorded as the grid resolution);
- **decoder shift:** $\mathbb E[(D(u_k) - D(u_k^{\mathrm{on}}))^2]$, with $u_k^{\mathrm{on}}$ the nearest curve point, i.e. how much the decoder sees of the off-curve component;
- **memory gain:** the out-of-sample increase in $R^2$ when the off-curve component $u_k - u_k^{\mathrm{on}}$ is added to a cubic in the curve position $y^*_k$. It is computed for two targets:
  - $y_{t+k-1}$, the second coordinate of the Markov state (the measure chosen in the design; at κ = 1 the state is observed exactly);
  - $y_{t+k+1}$, the next observation.

  Fits use one half of the validation trajectories and are evaluated on the other half.
- for the AR systems, two exact floors of the MSE: the best prediction from $y_t$ alone (the attainable floor for this latent) and the Kalman prediction from the whole past.

**Predictions** (written before the first run):
- P1: in henon and ar2, beyond $k = 4$, C's distance to the curve grows and its memory gain falls, while CL's distance stays small.
- P2: per seed of C, the growth of $d_k$ without memory gain precedes the loss of correlation. Operationally, the first $k > 4$ at which $d_k$ exceeds twice its value at $k = 4$ comes no later than the first $k$ at which the correlation drops below half its value at $k = 4$, in at least 8 of the 10 seeds.
- P3: in ar1 both variants stay close to the curve and there is no reversal: CL is not lower than C over $k = 5..10$ in at least 8 of the 10 seeds.

**Caveats.**
- With $L = 1$ the rollout latent is a function of $y_t$ alone. In a linear system the curve position already determines $y_t$ whenever the map $y_t \mapsto y^*_k$ is invertible. For ar2 the memory gain can therefore be non-zero only where the autocorrelation at that lag is near zero (the oscillation crosses zero), and small elsewhere even for a perfect model.
- The thresholds in P2 (a factor of 2 and a half) were chosen without data and can be revised before the run.
- The memory gain is a linear readout, so information stored nonlinearly off the curve is undercounted.

**Smoke run** (300 steps, all six runs, 3 min 20 s): the pipeline runs end to end. The reproducibility check is meaningful only at the full 30 000 steps.

Run: `caffeinate -i uv run python experiments/explore/offcurve/pilot.py "" 2` (results in `results/pilot.json`, finished runs are skipped). Use two threads, as in the horizon pilot, so the henon runs reproduce it exactly.

## Results (2026-11-15)

All six runs finished. The henon runs reproduce the horizon pilot exactly: the maximum difference of the per-seed, per-$k$ MSE is 0.0 for both variants.

Rollout MSE, mean over seeds, with the number of seeds in which C is lower than CL:

| | $k = 1..4$ | $k = 5..10$ | $k = 11..20$ |
|---|---|---|---|
| henon C / CL | 0.274 / 0.309 (10/10) | 1.268 / 0.948 (0/10) | 1.609 / 1.361 (4/10) |
| ar2 C / CL | 0.761 / 0.833 (10/10) | 1.210 / 1.162 (7/10) | 2.289 / 1.219 (1/10) |
| ar1 C / CL | 0.0485 / 0.0484 (3/10) | 0.1347 / 0.1345 (1/10) | 0.2516 / 0.2509 (0/10) |

The tables below give medians over seeds at $k = 1, 4, 5, 7, 10, 15, 20$. The encodings themselves lie about 0.001 from the curve.

| henon | C | CL |
|---|---|---|
| correlation | .96, .72, .45, .11, .00, −.01, .00 | .96, .70, .69, .42, .10, −.02, −.01 |
| distance to the curve | .15, .38, .40, .49, .66, .96, 1.47 | .16, .24, .24, .26, .31, .36, .42 |
| memory gain ($y_{t+k-1}$) | .70, .36, .16, .08, .03, .01, .00 | .80, .39, .35, .11, .03, .00, .00 |
| decoder shift | .01, .31, .18, .37, .48, .57, .86 | .02, .21, .14, .14, .19, .21, .14 |

| ar2 | C | CL |
|---|---|---|
| MSE | .27, .98, .79, .84, 1.75, 1.82, 1.78 | .28, 1.00, 1.17, 1.06, 1.23, 1.18, 1.19 |
| correlation | .86, .25, .50, .48, −.14, −.02, .02 | .86, .24, −.39, .08, .00, .00, .00 |
| distance to the curve | .06, .23, .28, .41, .63, 1.08, 1.58 | .03, .04, .04, .04, .05, .06, .07 |
| memory gain ($y_{t+k-1}$) | ≈ 0 (largest .05 at $k = 7$) | ≈ 0 (largest .20 at $k = 7$) |

The ar2 floors at the same $k$ are .26, .95, .79, .79, .99, 1.00, .99 for the best prediction from $y_t$ and .09, .68, .72, .74, .91, .96, .98 for Kalman. In ar1 both variants lie on the floor at every $k$ (.020 at $k = 1$, .31 at $k = 20$; the two floors coincide), with distances to the curve of at most 0.07 and memory gains and decoder shifts of about 0.

- **P1 holds in part.**
  - In henon, C's distance grows (0.38 → 1.47) and its memory gain falls (0.36 → 0), but CL's distance is not small: at $k = 4, 5$ it equals C's at $k = 1..4$, and it grows to 0.42. CL's memory gain falls to 0 as well, only more slowly.
  - In ar2 the memory part cannot be tested, since the gain is about 0 for both variants at every $k$ (as the caveat anticipated); the distance part holds.
  - Over $k = 5..7$ in ar2, C lies 0.28–0.41 from the curve and still on the floor, so leaving the curve does no harm there.
- **P2 fails.**
  - In henon it is satisfied in 0/10 seeds. The correlation halves first ($k = 5..6$) and the distance doubles later ($k = 8..14$). Between $k = 4$ and $k = 5$ C's distance barely moves (0.38 → 0.40) while its correlation falls from 0.72 to 0.45; over the same step CL's distance (0.24) and correlation (0.70 → 0.69) stay put.
  - In ar2 it is satisfied in 7/10 seeds, but the criterion does not fit there: the correlation oscillates and at $k = 4$ is near zero (the autocorrelation crosses zero), so half its value is a poor baseline.
  - In ar1 the 10/10 is vacuous, since the correlation never drops.
- **P3 holds in substance and fails formally.** Both variants stay near the curve, on the floor, with the same MSE. CL is lower than C over $k = 5..10$ in 9/10 seeds, though by 0.0002, so the criterion fails. This is the same fragility as H2 of the horizon pilot: counting seeds on a tie picks up noise. Later predictions should state a tolerance.

**Reading.**
- The working hypothesis, that drift off the curve causes the loss of information, is refuted:
  - in henon the information is lost before the drift;
  - in ar2 C is off the curve and optimal at the same time.
- At $k = 5$ in henon the two variants are equally far from the curve but differ in what the off-curve component carries (memory gain 0.16 against 0.35). The difference therefore lies in the content of the rollout state, not in its position.
- The reversal itself does not generalize in its henon form. In ar2, C is lower over $k = 5..10$ in 7/10 seeds and lies on the floor there. CL is lower only beyond $k = 10$, where C's MSE rises well above the reference (1.75 at $k = 10$) while CL stays near it.
- At $k = 5$ in ar2, CL's predictions have the opposite sign to the optimal one (correlation −0.39), giving an MSE of 1.17 against a floor of 0.79.
