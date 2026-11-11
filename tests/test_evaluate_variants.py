"""Checks of the test-set metrics used to compare Anchor variants."""

import numpy as np
import torch
from torch import nn

from config import K_MAX
from evaluate_variants import off_attractor, path_errors, sharpness


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
