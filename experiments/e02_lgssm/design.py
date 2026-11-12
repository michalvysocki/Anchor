"""Settings of e02 shared by the predictions, training and evaluation, see README.md."""

import numpy as np

from anchor.envs.lgssm import ar1, ar2

HORIZON = 10
WINDOWS = (1, 2, 4, 8)
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
KAPPAS = (1.0, 0.8, 0.5)
SYSTEMS = [ar1(a, k) for a in (0.5, 0.9, 0.99) for k in KAPPAS] + [
    ar2(0.9, np.pi / 6, k) for k in KAPPAS
]
VARIANTS = ("C", "D", "M", "M*")
# secondary ablations of M*, outside the decision criteria (README)
ABLATIONS = ("M*_oracle", "M*_p1", "D_nosg")
# each ablation is compared with the variant it modifies
ABLATION_OF = {"M*_oracle": "M*", "M*_p1": "M*", "D_nosg": "D"}

# training data: trajectories of LEN_TRAJ observations; the small-data arm
# probes the regime where target variance matters
LEN_TRAJ = 64
N_TRAJ = {"large": 2000, "small": 20}

SELECTION_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = tuple(range(3, 33))


def is_markov(system, window: int) -> bool:
    """The window is a Markov latent iff there is no noise and it covers the AR order."""
    return system.kappa == 1.0 and window >= len(system.phi)


def n_windows(data: str, window: int) -> int:
    """Number of training windows, the effective sample size of M*'s weights."""
    return N_TRAJ[data] * (LEN_TRAJ - window - HORIZON + 1)
