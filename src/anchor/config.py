from typing import Literal

ObservabilityMode = Literal["insufficient", "sufficient"]

# separate generators for each data split, shared by every experiment
TRAIN_DATA_SEED = 0
TEST_DATA_SEED = 1
VAL_DATA_SEED = 2
TRAINING_SEEDS = (0, 1, 2, 3, 4)

K_MAX = 20
BURN_IN = 10
