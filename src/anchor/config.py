"""
Settings of the Hénon experiments (e01, e01b; e03 will reuse the data
seeds): separate data generators per split, the training seeds, the window
length K_MAX and the burn-in of the trajectories.
"""

from typing import Literal

ObservabilityMode = Literal["insufficient", "sufficient"]

# separate generators for each data split, shared by every experiment
TRAIN_DATA_SEED = 0
TEST_DATA_SEED = 1
VAL_DATA_SEED = 2
# large sample for nearest-neighbour estimates of conditional expectations
REFERENCE_DATA_SEED = 3
TRAINING_SEEDS = (0, 1, 2, 3, 4)

K_MAX = 20
BURN_IN = 10
