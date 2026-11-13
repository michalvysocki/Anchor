# Exploratory pilot: spherical encodings

Not pre-registered and not part of e03; it imports e03's code without changing it. Design accepted by the user on 2026-11-14 after a check (four corrections: P1 only for variants whose later latents are supervised, a latent-target variant CL added, P2 against a binned estimate of the variance of the next encoding, a check that encodings lie near the sphere).

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

Decided by the user after pilot 1 (2026-11-14). Pilot 1 confirmed the identity for CL+S (the deficit $1 - \|u_1\|^2$ equal to the binned variance of the next encoding, 0.134 vs 0.136 and 0.079 vs 0.081), showed that the norm falls only a little and plateaus with $k$, that the mechanism needs latent multi-step targets (C+S drifts outside the sphere), and that without the sphere the latent scale drifts (encoding norms 2–71), a confound for any effect of the sphere on the error. `pilot2.py` trains only CL, with four penalties: none (CL), the scale alone (CL+N: the mean squared norm over the batch is 1), the sphere (CL+S) and a centred sphere (CL+SC: plus the squared norm of the batch mean of the encodings, so that an unpredictable future can bring the predicted mean close to zero and the norm can measure a wider range of uncertainty). Same configurations, seeds, training and validation measurements as pilot 1.

**Predictions** (written before the first run):
- Q1: if CL+N matches CL+S in rollout MSE over $k = 1..4$, the sphere's effect on the error in pilot 1 came from normalizing the scale, not from the belief mechanism.
- Q2: the deficit matches the binned variance only for the sphere variants (CL+S, CL+SC), not for CL+N.
- Q3: centring makes the norm track the growing uncertainty along the rollout more faithfully: for $k = 1..6$ (window 1), the mean absolute difference between the deficit $1 - \|u_k\|^2$ and the binned variance of $\varphi(z_{t+k})$ given $y_t$ is smaller for CL+SC than for CL+S. (First drafted as "the norm falls more clearly"; redefined by a check before the full run, because centring changes the scale of the norms and a steeper fall alone is not evidence that the norm measures the uncertainty.)

Run: `uv run python experiments/explore_sphere/pilot2.py` (results in `results/pilot2.json`).
