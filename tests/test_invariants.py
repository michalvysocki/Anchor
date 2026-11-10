"""Invariants of data generation and evaluation that earlier bugs violated."""

import numpy as np
import pytest
import torch
from torch import nn

from config import BURN_IN, K_MAX, TEST_DATA_SEED, TRAIN_DATA_SEED, VAL_DATA_SEED
from evaluate import average_drift
from toystateworld import (
    generate_dataset,
    generate_horizon_dataset,
    generate_windows,
    horizon_input,
)

MODES = ("insufficient", "sufficient")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", MODES)
def test_every_horizon_has_equal_count(mode):
    rng = np.random.default_rng(TRAIN_DATA_SEED)
    X, _ = generate_horizon_dataset(50, 36, BURN_IN, K_MAX, mode, rng)

    counts = X[:, -K_MAX:].sum(dim=0)

    assert counts.min() > 0
    assert torch.all(counts == counts[0])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", MODES)
def test_too_short_trajectory_raises(mode):
    rng = np.random.default_rng(TRAIN_DATA_SEED)

    with pytest.raises(ValueError):
        generate_horizon_dataset(10, BURN_IN + K_MAX - 1, BURN_IN, K_MAX, mode, rng)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_train_val_test_are_disjoint():
    def inputs(seed):
        X, _ = generate_dataset(
            200, 36, BURN_IN, "sufficient", np.random.default_rng(seed)
        )
        return {tuple(row) for row in X.tolist()}

    train = inputs(TRAIN_DATA_SEED)
    val = inputs(VAL_DATA_SEED)
    test = inputs(TEST_DATA_SEED)

    assert not train & val
    assert not train & test
    assert not val & test


class Exploding(nn.Module):
    def forward(self, x):
        return 3.0 * x


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_diverged_rollouts_are_counted_not_dropped():
    with torch.inference_mode():
        medians, diverged_rate, _ = average_drift(
            Exploding(), 50, 20, BURN_IN, "insufficient", np.random.default_rng(0)
        )

    assert diverged_rate[-1] == 1.0
    assert np.isinf(medians[-1])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", MODES)
def test_windows_match_horizon_dataset(mode):
    X, y = generate_horizon_dataset(
        30, 36, BURN_IN, K_MAX, mode, np.random.default_rng(TRAIN_DATA_SEED)
    )
    windows = generate_windows(
        30, 36, BURN_IN, K_MAX, mode, np.random.default_rng(TRAIN_DATA_SEED)
    )

    n = windows.shape[0]
    k = torch.arange(1, K_MAX + 1).repeat(n)
    starts = windows[:, 0].repeat_interleave(K_MAX, dim=0)
    targets = windows[:, 1:].reshape(n * K_MAX, -1)

    assert torch.equal(horizon_input(starts, k, K_MAX), X)
    assert torch.equal(targets, y)
