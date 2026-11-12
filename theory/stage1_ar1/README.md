# Stage 1 theory: AR(1) with observation noise

Derivations are in the research note ("Derivations on AR(1) with observation noise"). `verify.py` compares every closed form (steps 2–6) with a Monte Carlo estimate from 400k independent stationary chains (pass = within 4 standard errors) and checks the step 5 inequality on a grid.

```bash
uv run python theory/stage1_ar1/verify.py
```

Result (2026-11-11): all 55 checks of steps 2–5 pass for κ = 0.5, 0.8 and 1 (Markov observation). Step 6 added 2026-11-12 (residual of a linear one-step model, the lemma for the optimal and a misfit model, the critic): all 115 checks pass. The critic check uses the same regression as the residual check, so it confirms the sign convention, not independent evidence. In the Markov case the rollout bias and the k = 2 disagreement are both zero up to sampling noise, so their ratio there is noise.
