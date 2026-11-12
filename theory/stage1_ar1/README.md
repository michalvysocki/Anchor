# Stage 1 theory: AR(1) with observation noise

Derivations are in the research note ("Stage 1 Theory: AR(1) with Observation Noise", 2026-11-11). `verify.py` compares every closed form with a Monte Carlo estimate from 400k independent stationary chains (pass = within 4 standard errors) and checks the step 5 inequality on a grid.

```bash
uv run python theory/stage1_ar1/verify.py
```

Result (2026-11-11): all 55 checks pass for κ = 0.5, 0.8 and 1 (Markov observation). In the Markov case the rollout bias and the k = 2 disagreement are both zero up to sampling noise, so their ratio there is noise.
