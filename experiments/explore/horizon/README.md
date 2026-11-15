# Exploratory pilot: the horizon reversal between observation and latent targets

Not pre-registered and not part of e03; it imports e03's code without changing it. Decided on 2026-11-14, as the first step of checking whether the reversal seen in the first sphere pilot is real before any explanation or method is built on it. There (3 seeds), observation targets (C) had a lower rollout MSE than latent targets (CL) over the trained horizons $k = 1..4$ in every configuration, while at κ = 1, L = 1 CL was about 20% lower over $k = 1..10$ (0.704 against 0.889).

**Setup.** Hénon as in e03, window $L = 1$, κ ∈ {1, 0.8}; variants C and CL defined as in the first sphere pilot (one-step loss of e03 with latent consistency at $k = 1$, plus λ · observation targets or λ · latent consistency at $k = 2, 3, 4$; no flag, no sphere); ten seeds (300–309, no draw shared with e03 or the other pilots); 30 000 steps, Adam $10^{-3}$ reduced tenfold for the last 20%; λ = 1. Measured on e03's validation trajectories, with windows extended to $k = 20$ (8800 windows; e03's test set is not touched), per seed and per $k$: rollout MSE, sharpness (standard deviation of the predictions over that of $y$) and the mean norm of the predicted latent. The variance of $y$ at each $k$ is recorded as the reference (the MSE of predicting the mean; about 1). For reading H1: if CL is lower beyond $k = 4$ only because it stays at that reference while C exceeds it, the reversal reflects the absence of drift rather than better generalization; CL's MSE against the reference and its sharpness separate the two.

**Predictions** (written before the first run):
- H1: at κ = 1, C has a lower MSE than CL over $k = 1..4$ and CL a lower MSE than C over $k = 5..10$, each in at least 8 of the 10 seeds (paired by seed).
- H2: at κ = 0.8, no reversal: C is at most CL over both $k = 1..4$ and $k = 5..10$ in at least 8 of the 10 seeds.
- H3 (descriptive, for the explanation): beyond $k = 4$ the norm of C's predicted latent drifts away from the range of the encodings, while CL's stays within it.

Training at κ = 1 is noise-free, where single seeds can differ widely (property 8 of the sphere pilots); hence the counts over seeds and the per-seed values instead of means alone.

**Smoke run** (300 steps, κ = 1): the pipeline runs, the long validation windows start with the same values as e03's, about 0.1 s per step for ten models.

Run: `uv run python experiments/explore/horizon/pilot.py "" 2` (results in `results/pilot.json`, finished runs are skipped).

## Results (2026-11-15)

All four runs finished (10 seeds each, 8800 validation windows; the variance of $y$ is about 1 at every $k$). Rollout MSE, mean over seeds, with the number of seeds in which C is lower than CL:

| | κ = 1, C | κ = 1, CL | C < CL | κ = 0.8, C | κ = 0.8, CL | C < CL |
|---|---|---|---|---|---|---|
| $k = 1..4$ | 0.274 | 0.309 | 10/10 | 0.817 | 0.859 | 10/10 |
| $k = 5..10$ | 1.268 | 0.948 | 0/10 | 0.997 | 0.995 | 3/10 |
| $k = 11..20$ | 1.609 | 1.361 | 4/10 | 1.006 | 1.003 | 0/10 |

- **H1 holds.** At κ = 1, C is lower over $k = 1..4$ and CL over $k = 5..10$, both in 10/10 seeds. Beyond $k = 10$ both are above the reference and CL is lower in only 6/10 seeds.
- **H2 fails formally.** At κ = 0.8, C is at most CL over $k = 5..10$ in only 3/10 seeds. The difference is about 0.002 and both variants sit at the reference (CL at 0.995, equal to the variance of $y$). Beyond $k = 4$ nothing is predictable at κ = 0.8 (sharpness at $k = 10$: C 0.047, CL 0.005). CL is lower only because it converges to the mean faster, so there is no reversal in substance.
- **H3 fails over $k = 5..10$ and holds only beyond $k = 10$.** At κ = 1:
  - Over $k = 5..10$, C's mean predicted latent norm stays within the 5–95% range of its encodings in 9/10 seeds. The exception, seed 306, is not among the seeds with the largest MSE (seeds 300 and 301, at 1.54 and 1.74, stay within the range).
  - Over $k = 11..20$, C leaves the range in 6/10 seeds, up to 2.7 times the median encoding norm.
  - CL stays within the range throughout, in every seed.
  - The norm therefore does not explain the reversal over $k = 5..10$. A mean norm is also a coarse measure: predictions can leave the region the decoder was trained on while keeping a typical norm.

**Correlation with the target.** The correlation between predictions and $y_{t+k}$ follows from the MSE, the sharpness $s$ and the variance $v$ of $y$ as $(v + s^2 v - \mathrm{MSE}) / (2 s v)$. Omitting the bias makes this a lower bound. Medians over seeds:

| $k$ | 1 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| κ = 1, C | 0.96 | 0.72 | 0.44 | 0.20 | 0.09 | −0.01 | −0.03 |
| κ = 1, CL | 0.96 | 0.70 | 0.69 | 0.49 | 0.37 | 0.30 | 0.09 |
| κ = 0.8, C | 0.61 | 0.24 | 0.19 | 0.09 | 0.02 | 0.00 | −0.07 |
| κ = 0.8, CL | 0.61 | 0.23 | 0.17 | 0.10 | 0.03 | −0.01 | −0.07 |

The reading note above anticipated a reversal produced only by CL staying at the reference. The correlations rule that out: CL keeps information about $y_{t+k}$ for several steps beyond the trained horizon, while C loses it within two or three steps and keeps a large variance of predictions (sharpness 0.64 at $k = 5$), which puts its MSE above the reference. On the trained horizons the two variants carry the same information (0.72 and 0.70 at $k = 4$). C's lower MSE there must therefore come from calibration (scale and bias of the predictions), not from more information. The pilot did not store predictions, so the split between scale and bias is approximate.

**Open.**
- Why latent targets carry information past the trained horizon and observation targets do not.
- Whether this is general or a property of noise-free chaotic Hénon.
- How much is predictable at all over $k = 5..10$: there is no floor for the attainable MSE.
- In the language of the method requirements, C satisfies requirement 1 and violates requirement 2, and CL the reverse.
