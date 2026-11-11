"""Invariants of data generation that earlier bugs violated."""

import numpy as np
import pytest

from anchor.config import BURN_IN, K_MAX, TEST_DATA_SEED, TRAIN_DATA_SEED, VAL_DATA_SEED
from anchor.envs.henon import generate_windows

MODES = ("insufficient", "sufficient")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", MODES)
def test_every_window_reaches_k_max(mode):
    windows = generate_windows(
        50, 36, BURN_IN, K_MAX, mode, np.random.default_rng(TRAIN_DATA_SEED)
    )

    assert len(windows) > 0
    assert windows.shape[1] == K_MAX + 1
    assert np.isfinite(windows.numpy()).all()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", MODES)
def test_too_short_trajectory_raises(mode):
    rng = np.random.default_rng(TRAIN_DATA_SEED)

    with pytest.raises(ValueError):
        generate_windows(10, BURN_IN + K_MAX - 1, BURN_IN, K_MAX, mode, rng)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_train_val_test_are_disjoint():
    def starts(seed):
        windows = generate_windows(
            200, 36, BURN_IN, K_MAX, "sufficient", np.random.default_rng(seed)
        )
        return {tuple(row) for row in windows[:, 0].tolist()}

    train = starts(TRAIN_DATA_SEED)
    val = starts(VAL_DATA_SEED)
    test = starts(TEST_DATA_SEED)

    assert not train & val
    assert not train & test
    assert not val & test
