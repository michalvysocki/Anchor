"""Settings of e03 shared by data, training and evaluation, see README.md."""

import numpy as np

from anchor.envs.henon import x_moments
from anchor.latent import Sizes

KAPPAS = (1.0, 0.8, 0.5)
WINDOWS = (1, 2, 4)
GRU_HISTORY = 16
LATENT, HIDDEN, DECODER_HIDDEN, CRITIC_HIDDEN = 16, 128, 64, 64
EVAL_HORIZON = 10
SHORT_HORIZONS = (2, 3, 4)
LONG_HORIZONS = tuple(range(2, 9))
LAMBDAS = (0.1, 1.0, 10.0)
# horizons (inclusive) of the primary metric and of lambda selection, fixed
# from the gap stage before the selection stage (README, Gap measured before
# the Anchor runs): the training horizons, within the range where the noisy
# Hénon map is still predictable; k = 1..10 is reported as secondary
PRIMARY_K = (1, 4)
LEN_TRAJ = 64
N_TRAJ = {"large": 2000, "small": 20}
N_VAL_TRAJ, N_TEST_TRAJ, N_REF_TRAJ = 200, 500, 20000

SELECTION_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = tuple(range(3, 33))

# generator tags: new, so that no draw is shared with e01 or e01b
TRAIN_TAG, VAL_TAG, TEST_TAG, REF_TAG = 30, 31, 32, 33

# training: Adam, plateau schedule, convergence stop (README, Training)
LR, CRITIC_LR_FACTOR, BATCH, CLIP = 1e-3, 3.0, 512, 1.0
# stricter than first drafted: with 0.1% and patience 3 the onestep baseline
# stopped before its rollout error had settled (correctness review, README)
EVAL_EVERY, PATIENCE, MIN_REL_IMPROVEMENT = 1000, 6, 1e-4
# an improvement must also exceed this absolute amount (units of Var(y)):
# without noise the one-step loss falls towards 1e-7 and keeps improving by
# more than 0.01% relative, so the relative rule alone never stops
MIN_ABS_IMPROVEMENT = 1e-6
LR_FACTOR, LR_STOP, MAX_STEPS = 0.3, 1e-5, 100_000
MSTAR_EVERY = 100
# M* keeps the C weights until its critic heads have trained for this long
MSTAR_WARMUP = 1000

CORE = ("onestep", "C", "DB", "M*")
ABLATION_CONFIGS = ((0.5, 1), (0.8, 2))
ABLATIONS = ("D", "D+F", "TDB+F", "DB+F_oracle", "C+F_h34", "C_wide", "C+F_two")
VARIANCE_CONFIGS = ((1.0, 1), (1.0, 2), (0.5, 1), (0.5, 2))
VARIANCE_VARIANTS = ("onestep", "C", "C+F", "D+F", "DB+F", "M*+F")
GRU_VARIANTS = ("onestep", "C", "C+F", "DB+F", "M*+F")

SIZES = Sizes(LATENT, HIDDEN, DECODER_HIDDEN, CRITIC_HIDDEN, EVAL_HORIZON, GRU_HISTORY)

# moments of x on the attractor, for centring and the noise level
X_MEAN, X_VAR = x_moments(200_000)


def is_markov(kappa: float, window: int | None) -> bool:
    """A noiseless window of two observations determines the state; a GRU on 16 too."""
    return kappa == 1.0 and (window is None or window >= 2)


def main_variants() -> list[str]:
    return [v + f for v in CORE for f in ("", "+F")]


def config_name(kappa: float, window: int | None) -> str:
    enc = "gru" if window is None else f"L{window}"
    return f"k{kappa}_{enc}"


def rng(tag: int, *key: int) -> np.random.Generator:
    return np.random.default_rng([tag, *key])
