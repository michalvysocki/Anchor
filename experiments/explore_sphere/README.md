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
