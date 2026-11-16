# Exploratory pilots

Pilots outside the pre-registered experiments: not part of any decision rule, each with its predictions written in its README before the first run. They import the code of e03 without changing it. Findings from here enter a paper only after confirmation in a pre-registered experiment, or as clearly labelled exploratory analyses in an appendix.

| folder | question | outcome |
|---|---|---|
| `sphere/` | do spherical encodings make the norm of a prediction carry the uncertainty of the future? (2026-11-14) | eight properties recorded; the effect on the error came mostly from fixing the latent scale |
| `us/` | does a latent split into u (compared with encodings) and s (free memory) close the gap? (2026-11-14) | S1 and S2 refuted |
| `horizon/` | does the horizon reversal between observation (C) and latent (CL) targets hold up with 10 seeds? (2026-11-14/15) | holds at κ = 1: C better on the trained horizons, CL keeps information beyond them |
| `offcurve/` | does drift off the curve of encodings cause C's loss of information? Second system: linear-Gaussian (2026-11-15) | refuted; the difference lies in the content of the state; the reversal does not generalize to AR(2) in its Hénon form |
| `c_vs_cl/` | specialization to the trained horizon or co-adaptation of the decoder? (2026-11-15/16) | specialization: with horizons 2..8 the reversal moves to $k = 8$; co-adaptation of the decoder excluded |

Pilots load each other by their folder names, so the folders stay side by side. Run commands are given from the repository root.
