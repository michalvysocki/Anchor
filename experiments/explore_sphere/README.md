# Exploratory pilot: spherical encodings

Not pre-registered and not part of e03; it imports e03's code without changing it. Design fixed on 2026-11-14 after a check (four corrections: P1 only for variants whose later latents are supervised, a latent-target variant CL added, P2 against a binned estimate of the variance of the next encoding, a check that encodings lie near the sphere).

**Idea.** A loss $\mu\,(\|\varphi(z_t)\|^2 - 1)^2$ keeps encodings near the unit sphere and leaves predictions free. With latent consistency a predicted latent approaches the mean of the future encodings, and for unit vectors $\|\mathbb E[\varphi]\|^2 = 1 - \operatorname{tr}\operatorname{Cov}(\varphi)$: the norm of a prediction carries the uncertainty of the future, acts as an implicit flag that vanishes for a Markov latent without noise, and lets the rollout blur with the horizon.

**Setup.** Hénon as in e03, configurations κ = 0.8, L = 1 and κ = 1, L = 1 (gaps) and κ = 1, L = 2 (Markov control); variants onestep, C (observation targets) and CL (latent targets, as TD-MPC2), each without and with the sphere; 3 seeds, 30 000 steps, Adam 1e-3 reduced tenfold for the last 20%, λ = μ = 1.

**Predictions** (written before the first run, also in `pilot.py`):
- P1: for CL+S the mean norm of $u_k$ falls with $k$ in the gap configurations and stays near 1 for the first steps in the Markov control.
- P2: for the sphere variants, $1 - \|u_1\|^2$ tracks the variance of the next encoding given $y_t$ (20 quantile bins, window 1): positive rank correlation and, for CL+S, values of similar size.
- P3: the sphere does not raise the rollout MSE over $k = 1..4$ and brings the rollout's sharpness closer to the direct reference's.
- Check: encodings of the sphere variants have norm close to 1.

**Measurements on the validation set**, not the test set, which is reserved for e03's final stage; the sharpness of the direct reference comes from e03's gap-stage direct model (seed 0). Both added by a second check before the full run.

**Smoke run** (300 steps, κ = 0.8, L = 1): encodings of the sphere variants at norm 1.001–1.006 ± 0.01, the pipeline runs, about 30 ms per step with the selection stage of e03 running alongside.

Run: `uv run python experiments/explore_sphere/pilot.py` (results in `results/pilot.json`, finished configurations are skipped).

## Pilot 2: scale control and a centred sphere

Decided after pilot 1 (2026-11-14). Pilot 1 confirmed the identity for CL+S (the deficit $1 - \|u_1\|^2$ equal to the binned variance of the next encoding, 0.134 vs 0.136 and 0.079 vs 0.081), showed that the norm falls only a little and plateaus with $k$, that the mechanism needs latent multi-step targets (C+S drifts outside the sphere), and that without the sphere the latent scale drifts (encoding norms 2–71), a confound for any effect of the sphere on the error. `pilot2.py` trains only CL, with four penalties: none (CL), the scale alone (CL+N: the mean squared norm over the batch is 1), the sphere (CL+S) and a centred sphere (CL+SC: plus the squared norm of the batch mean of the encodings, so that an unpredictable future can bring the predicted mean close to zero and the norm can measure a wider range of uncertainty). Same configurations, seeds, training and validation measurements as pilot 1.

**Predictions** (written before the first run):
- Q1: if CL+N matches CL+S in rollout MSE over $k = 1..4$, the sphere's effect on the error in pilot 1 came from normalizing the scale, not from the belief mechanism.
- Q2: the deficit matches the binned variance only for the sphere variants (CL+S, CL+SC), not for CL+N.
- Q3: centring makes the norm track the growing uncertainty along the rollout more faithfully: for $k = 1..6$ (window 1), the mean absolute difference between the deficit $1 - \|u_k\|^2$ and the binned variance of $\varphi(z_{t+k})$ given $y_t$ is smaller for CL+SC than for CL+S. (First drafted as "the norm falls more clearly"; redefined by a check before the full run, because centring changes the scale of the norms and a steeper fall alone is not evidence that the norm measures the uncertainty.)

Run: `uv run python experiments/explore_sphere/pilot2.py` (results in `results/pilot2.json`).

## Results of pilot 1 (3 seeds, validation)

| | κ = 0.8, L = 1 | κ = 1, L = 1 | κ = 1, L = 2 (Markov) |
|---|---|---|---|
| encoding norm without / with the sphere | 1.1–2.5 / 1.00 | 1.9–71 / 1.00 | 2.2–12.5 / 1.00 |
| CL+S: $\|u_k\|$, k = 1, 2, 4, 8 | 0.93, 0.92, 0.92, 0.92 | 0.96, 0.96, 0.93, 0.90 | 1.00, 1.00, 0.99, 0.99 |
| CL+S: deficit vs binned variance at k = 1 | 0.134 vs 0.136 (ρ +0.72) | 0.079 vs 0.081 (ρ +0.80) | — |
| C+S: $\|u_k\|$ at k = 8 | 0.90 | 1.69 | 3.80 |
| MSE k = 1..4, CL vs CL+S | 0.863 vs 0.867 | 0.297 vs 0.389 | 0.129 vs 0.013 |
| MSE k = 1..10, C vs C+S | 0.929 vs 0.930 | 0.889 vs 0.891 | 0.002 vs 1.92 |

Against the predictions: P2 holds for CL+S (the identity, to the second decimal) and only weakly or with the wrong sign for onestep+S and C+S; P1 holds only in part (the norm falls a little and plateaus); P3 is mixed, with large effects in both directions.

## Results of pilot 2 (3 seeds, validation)

| | κ = 0.8, L = 1 | κ = 1, L = 1 | κ = 1, L = 2 (Markov) |
|---|---|---|---|
| norm of the mean encoding, CL+S / CL+SC | 0.92 / 0.01 | 0.89 / 0.01 | 0.85 / 0.01 |
| $\|u_k\|$, k = 1, 2, 4, 8, CL+SC | 0.42, 0.24, 0.16, 0.05 | 0.81, 0.70, 0.66, 0.55 | 1.00, 1.00, 0.99, 0.99 |
| mean \|deficit − variance\|, k = 1..6: CL+N / CL+S / CL+SC | 0.007 / 0.005 / 0.007 | 0.010 / 0.008 / 0.129 | — |
| CL+SC deficit vs variance at k = 1 and 6 | 0.78 vs 0.79; 0.99 vs 0.98 | 0.32 vs 0.36; 0.63 vs 0.87 | — |
| MSE k = 1..4: CL / CL+N / CL+S / CL+SC | 0.863 / 0.869 / 0.867 / 0.840 | 0.297 / 0.363 / 0.380 / 0.338 | 0.129 / 0.004 / 0.143 / 0.002 |

Against the predictions: Q1 holds (CL+N matches CL+S in MSE, so the sphere's effect on the error came from normalizing the scale); Q2 fails, because the scale penalty alone already gave encodings of nearly constant norm (1.00 ± 0.08–0.10), so the identity held for CL+N too; Q3 fails on the measure fixed before the run (no better at κ = 0.8, much worse at κ = 1, L = 1), although centring widened the range of the deficit from about 0.15 to 0.3–0.99.

**Reproducibility.** CL agreed between the two pilots to four digits; CL+S agreed at κ = 0.8 and κ = 1, L = 1 but not in the Markov control (MSE 0.013 in pilot 1, 0.143 in pilot 2), where the two scripts compute the sphere penalty in a different order of floating-point sums. Per-seed check (`check_stability.py`, results in `results/check_stability.json`), validation MSE over $k = 1..4$ of the three seeds:

| code path | 2 threads | 4 threads |
|---|---|---|
| pilot 1 | 0.0009, 0.0349, 0.0022 | 0.0009, 0.0349, 0.0022 |
| pilot 2 | 0.0059, 0.1713, 0.2518 | 0.0059, 0.1713, 0.2518 |

The number of threads changes nothing (CPU training is deterministic here); the code path changes every seed, two of them by a factor of 5 to 100. Within one code path the seeds already span a factor of 40. In the noise-free Markov configuration the outcome of training is decided by perturbations at the level of floating-point rounding, so it behaves as random: the variance between seeds is large and a mean over three seeds says little.

## Properties learned (exploratory; to be confirmed in a pre-registered experiment)

1. **The norm of a prediction measures the uncertainty of the future, if encodings have nearly constant norm.** With latent multi-step consistency, $1 - \|u_k\|^2$ equals the variance of the future encodings given $z_t$ to about 0.002 at $k = 1$ and 0.005–0.010 over $k = 1..6$. The sphere is one way to get constant norm; a penalty on the mean scale gave nearly the same (norm 1.00 ± 0.08–0.10).
2. **The mechanism needs latent multi-step targets.** With observation targets (C+S) the later latents are unsupervised, leave the ball (norm up to 3.8) and the rollout of the Markov control breaks beyond $k = 4$; with one-step training (onestep+S) the later latents are unsupervised as well.
3. **Without a norm constraint the latent scale drifts** (encoding norms 1–71), which changes the balance between the latent and the observation terms. The effects of the sphere on the error came mostly from fixing this scale.
4. **Uncentred encodings cluster on a small part of the sphere** (norm of the mean 0.85–0.92), so the norm of a prediction has a narrow range (deficit about 0.15). **Centring spreads them** (norm of the mean about 0.01) and widens the range to 0.3–0.99, so the norm becomes a high-range signal of uncertainty.
5. **The rollout under-propagates growing uncertainty where it comes from missing memory and chaos.** With centring, the deficit tracks the variance along the horizon at κ = 0.8 (to about 0.01) but lags it at κ = 1, L = 1 (0.63 against 0.87 at $k = 6$): there the model is overconfident at long horizons, and there CL+SC has a higher error than CL.
6. **The mechanism switches itself off when the future is certain:** in the Markov control the norm of predictions stays at 0.99–1.00 in every constrained variant.
7. **CL+SC had the lowest error in two of three configurations** (0.840 against 0.863; 0.002 against 0.129) and a higher one where property 5 bites (0.338 against 0.297). Three seeds and no intervals: a signal, not a result.
8. **Training in the noise-free Markov configuration is decided by rounding-level perturbations**: the same variant differed tenfold between two code paths, every seed changed with the order of floating-point sums (not with the number of threads), and seeds of one code path spanned a factor of 40. Comparisons there need many seeds and robust summaries.

**Open questions for a better mechanism:** how to make the rollout propagate the growth of uncertainty at long horizons (property 5); whether centring should be kept with a different measure of tracking (relative error); whether the decoder should read the norm explicitly; how the mechanism behaves with actions (e04) and inside TD-MPC2, whose SimNorm normalizes predictions as well.
