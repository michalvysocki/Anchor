"""Checks of the test-set metrics used to compare Anchor variants."""

import numpy as np
import torch
from torch import nn

from anchor.config import K_MAX
from anchor.losses import grounding_loss, rollout_error
from anchor.metrics import (
    direct_mse,
    mse_per_horizon,
    neighbour_futures,
    off_attractor,
    path_errors,
    rollout_mse,
    sharpness,
)
from anchor.models import HorizonMLP, UnitStep


class IgnoresHorizon(nn.Module):
    """Returns the input point for every horizon."""

    def forward(self, x):
        return x[:, :1]


class Exploding(nn.Module):
    def forward(self, x):
        return 3.0 * x[:, :1] + 1.0


def test_constant_model_errors_are_distances_to_start():
    windows = torch.randn(16, K_MAX + 1, 1)
    expected = (windows[:, 1:] - windows[:, :1]).abs().squeeze(-1).numpy()

    for path in ("direct", "rollout"):
        errors, _ = path_errors(IgnoresHorizon(), windows, path)
        assert np.allclose(errors, expected, atol=1e-6)


def test_diverged_rollout_errors_are_infinite_from_divergence_on():
    windows = torch.full((4, K_MAX + 1, 1), 0.5)

    errors, _ = path_errors(Exploding(), windows, "rollout")

    assert np.isinf(errors[:, -1]).all()
    first = np.argmax(np.isinf(errors), axis=1)
    assert all(np.isinf(errors[i, first[i] :]).all() for i in range(4))


def test_points_on_reference_are_on_attractor():
    reference = np.random.default_rng(0).normal(size=(100, 2))
    preds = reference[:20].reshape(4, 5, 2)

    assert np.allclose(off_attractor(preds, reference), 0, atol=1e-5)


def test_sharpness_is_one_for_truth_and_zero_for_constant():
    windows = torch.randn(200, K_MAX + 1, 2)
    truth = windows[:, 1:].numpy()

    assert np.allclose(sharpness(truth, windows), 1.0)
    assert np.allclose(sharpness(np.zeros_like(truth), windows), 0.0)


def test_path_mse_without_horizon_matches_training_losses():
    torch.manual_seed(0)
    windows = torch.randn(32, K_MAX + 1, 2)
    model = HorizonMLP(K_MAX, 2, 16)
    unit = UnitStep(model, K_MAX)

    assert torch.allclose(direct_mse(model, windows), grounding_loss(model, windows))
    assert torch.allclose(rollout_mse(unit, windows), rollout_error(unit, windows))


def test_path_mse_with_horizon_uses_only_first_horizons():
    windows = torch.randn(16, K_MAX + 1, 1)
    # constant model: the error at horizon k is the distance to the start point
    expected = ((windows[:, 1:4] - windows[:, :1]) ** 2).mean()

    assert torch.allclose(direct_mse(IgnoresHorizon(), windows, 3), expected)
    assert torch.allclose(rollout_mse(IgnoresHorizon(), windows, 3), expected)


def test_mse_per_horizon_from_euclidean_errors():
    errors = np.array([[1.0, np.inf], [3.0, 2.0]])

    mse = mse_per_horizon(errors, point_dim=2)

    assert mse[0] == (1 + 9) / 2 / 2
    assert np.isinf(mse[1])


def test_neighbour_futures_are_futures_of_the_closest_starts():
    # futures are a deterministic function of the start, so the nearest
    # neighbours of a start that is in the reference return its own future
    starts = np.linspace(-1, 1, 101)
    reference = np.stack([starts, 2 * starts, 3 * starts], axis=1)
    rng = np.random.default_rng(0)
    reference = reference[rng.permutation(len(reference))]

    futures = neighbour_futures(reference, np.array([0.5, -0.2]), n_neighbours=1)

    assert futures.shape == (2, 1, 2)
    assert np.allclose(futures[:, 0], [[1.0, 1.5], [-0.4, -0.6]])
