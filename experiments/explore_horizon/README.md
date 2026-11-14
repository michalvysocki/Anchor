# Exploratory pilot: the horizon reversal between observation and latent targets

Not pre-registered and not part of e03; it imports e03's code without changing it. Decided on 2026-11-14, as the first step of checking whether the reversal seen in the first sphere pilot is real before any explanation or method is built on it. There (3 seeds), observation targets (C) had a lower rollout MSE than latent targets (CL) over the trained horizons $k = 1..4$ in every configuration, while at κ = 1, L = 1 CL was about 20% lower over $k = 1..10$ (0.704 against 0.889).

**Setup.** Hénon as in e03, window $L = 1$, κ ∈ {1, 0.8}; variants C and CL defined as in the first sphere pilot (one-step loss of e03 with latent consistency at $k = 1$, plus λ · observation targets or λ · latent consistency at $k = 2, 3, 4$; no flag, no sphere); ten seeds (300–309, no draw shared with e03 or the other pilots); 30 000 steps, Adam $10^{-3}$ reduced tenfold for the last 20%; λ = 1. Measured on e03's validation trajectories, with windows extended to $k = 20$ (8800 windows; e03's test set is not touched), per seed and per $k$: rollout MSE, sharpness (standard deviation of the predictions over that of $y$) and the mean norm of the predicted latent. The variance of $y$ at each $k$ is recorded as the reference (the MSE of predicting the mean; about 1). For reading H1: if CL is lower beyond $k = 4$ only because it stays at that reference while C exceeds it, the reversal reflects the absence of drift rather than better generalization; CL's MSE against the reference and its sharpness separate the two.

**Predictions** (written before the first run):
- H1: at κ = 1, C has a lower MSE than CL over $k = 1..4$ and CL a lower MSE than C over $k = 5..10$, each in at least 8 of the 10 seeds (paired by seed).
- H2: at κ = 0.8, no reversal: C is at most CL over both $k = 1..4$ and $k = 5..10$ in at least 8 of the 10 seeds.
- H3 (descriptive, for the explanation): beyond $k = 4$ the norm of C's predicted latent drifts away from the range of the encodings, while CL's stays within it.

Training at κ = 1 is noise-free, where single seeds can differ widely (property 8 of the sphere pilots); hence the counts over seeds and the per-seed values instead of means alone.

**Smoke run** (300 steps, κ = 1): the pipeline runs, the long validation windows start with the same values as e03's, about 0.1 s per step for ten models.

Run: `uv run python experiments/explore_horizon/pilot.py "" 2` (results in `results/pilot.json`, finished runs are skipped).
