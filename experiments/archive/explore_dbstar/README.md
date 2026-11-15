# Exploratory pilot: DB*, a multi-anchor control-variate target

Not pre-registered and not part of e03; it imports e03's code without changing it. Decided on 2026-11-14 after a review of an external analysis of `latent.py`, which proposed combining DB and M*. The review kept the idea and rejected two claims: that the optimal weights always lie strictly inside the simplex with a variance strictly below both C and DB (if a bootstrapped anchor equals the conditional mean of $y_{t+k}$ given the later window, DB alone is optimal, on the boundary), and that the reduction of variance per sample must improve the estimator (with a linear critic fitted on the same data DB equals C exactly, e02). Related work to cite: control variates in multi-step TD targets (De Asis and Sutton, 2018).

**Target.** For a horizon $k$, each bootstrapped anchor is debiased by the critics, $T_{k,p} + h_y(z_t) - h_p(z_t)$, with $h_p$ the regression of $T_{k,p}$ and $h_y$ the regression of $y_{t+k}$ on $\mathrm{sg}[\varphi(z_t)]$ (M*'s critic heads). With exact critics every candidate, the debiased anchors and $y_{t+k}$, has conditional mean $\mathbb E[y_{t+k} \mid z_t]$, so the weights minimize the variance alone, $w = \arg\min_{w \in \Delta} w^\top V w$, $V$ the covariance of the candidates around $h_y$; no bias term and no division by the number of windows.

**Setup.** Hénon as in e03, window $L = 1$, κ ∈ {0.8, 0.5, 1}; variants C+F, D+F, DB+F (as in e03) and DB*+F, all with the flag so that the comparison concerns the target alone; five seeds (100–104, no draw shared with e03); 30 000 steps, Adam $10^{-3}$ (critics $3 \cdot 10^{-3}$) reduced tenfold for the last 20%; λ = 1; horizons {2, 3, 4}; DB*'s weights stay on C for the first 1000 steps, then are recomputed every 100 steps. Measured on the validation set: rollout MSE over $k = 1..4$ per seed; for DB*+F at $k = 3, 4$, the variance of each target around $h_y$ and the weights.

**Predictions** (written before the first run):
- R1: the variance of the DB* target around $h_y$ is at most that of DB and below that of C.
- R2: DB*'s weights lie near the boundary of the simplex, with little weight on $y_{t+k}$, where the observation noise is large (κ = 0.5).
- R3: no prediction on the rollout MSE; it is reported per seed.

**Caveat for R1.** The variance of a target around $h_y$ contains the critic's misfit as well as the conditional variance; a target that is nearly constant (an undertrained rollout) also has a small variance. R1 is read together with the MSE. A second caveat, found by a check before the run: when an anchor is nearly constant given $z_t$ (as at κ = 0.5, where the future is barely predictable), its debiased form $T_{k,p} - h_p(z_t) + h_y(z_t)$ is close to $h_y(z_t)$ itself, so weight on it means distilling the direct critic of $y_{t+k}$ into the rollout (like E in e01b). The DB* target is then as good as $h_y$; low variance there does not mean a better target unless the MSE says so.

**Smoke run** (1300 steps, κ = 0.5): the pipeline runs past the warm-up, the weights move (at $k = 4$: 0.82, 0.18, 0.00 on the anchors $p = 2, 3$ and on $y_{t+k}$), about 14 ms per step with four threads.

Run: `uv run python experiments/explore_dbstar/pilot.py "" 2` (results in `results/pilot.json`, finished runs are skipped).
