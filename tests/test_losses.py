"""Checks of the Anchor objectives on small synthetic batches."""

import numpy as np
import pytest
import torch
from torch import nn

from anchor.config import BURN_IN, K_MAX, TRAIN_DATA_SEED
from anchor.envs.henon import generate_windows
from anchor.losses import (
    anchor_loss_bootstrapped,
    anchor_loss_composed,
    anchor_loss_rollout,
    anchor_loss_unit_rollout,
    grounding_loss,
    multistep_loss,
    rollout_error,
    rollout_to,
    sample_decomposition,
)
from anchor.metrics import median_error
from anchor.models import HorizonMLP, UnitStep, predict


@pytest.mark.parametrize("p_one", [False, True])
def test_decomposition_is_valid(p_one):
    g = torch.Generator().manual_seed(0)
    k, p, q = sample_decomposition(5000, K_MAX, p_one, g, torch.device("cpu"))

    assert k.min() >= 2 and k.max() <= K_MAX
    assert p.min() >= 1 and q.min() >= 1
    assert torch.equal(p + q, k)
    if p_one:
        assert torch.all(p == 1)
    else:
        assert p.max() == K_MAX - 1


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_grounding_loss_averages_every_horizon():
    torch.manual_seed(0)
    model = HorizonMLP(K_MAX, 1, 16)
    windows = generate_windows(
        20, 36, BURN_IN, K_MAX, "insufficient", np.random.default_rng(TRAIN_DATA_SEED)
    )
    n = len(windows)
    per_horizon = [
        nn.functional.mse_loss(
            predict(model, windows[:, 0], torch.full((n,), k), K_MAX), windows[:, k]
        )
        for k in range(1, K_MAX + 1)
    ]

    assert torch.allclose(
        grounding_loss(model, windows), torch.stack(per_horizon).mean()
    )


class IgnoresHorizon(nn.Module):
    """Returns the input point for every horizon, so every path agrees."""

    def forward(self, x):
        return x[:, :1]


def test_composed_loss_is_zero_for_consistent_model():
    windows = torch.randn(64, K_MAX + 1, 1)
    g = torch.Generator().manual_seed(0)

    assert anchor_loss_composed(IgnoresHorizon(), windows, g) == 0


def test_stop_gradient_blocks_target_path():
    torch.manual_seed(0)
    windows = torch.randn(64, K_MAX + 1, 1)

    for loss_fn in (anchor_loss_composed, anchor_loss_bootstrapped):
        grads = {}
        for symmetric in (False, True):
            torch.manual_seed(1)
            model = HorizonMLP(K_MAX, 1, 16)
            loss = loss_fn(
                model, windows, torch.Generator().manual_seed(0), symmetric=symmetric
            )
            loss.backward()
            grads[symmetric] = model.output_layer.weight.grad.clone()

        assert not torch.allclose(grads[False], grads[True])


class TrueHenon(nn.Module):
    """Exact one-step dynamics of the Henon map in the sufficient mode."""

    def forward(self, z):
        x_prev, x = z[:, 0], z[:, 1]
        return torch.stack([x, 1 - 1.4 * x**2 + 0.3 * x_prev], dim=1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_true_dynamics_have_zero_rollout_losses():
    windows = generate_windows(
        20, 36, BURN_IN, K_MAX, "sufficient", np.random.default_rng(TRAIN_DATA_SEED)
    )
    g = torch.Generator().manual_seed(0)
    # float32 rounding grows through 20 chaotic steps, so the losses are tiny, not 0
    tol = 1e-5

    assert rollout_error(TrueHenon(), windows) < tol
    assert multistep_loss(TrueHenon(), windows, g) < tol
    assert anchor_loss_rollout(TrueHenon(), windows, g) < tol


def test_rollout_to_picks_each_sample_horizon():
    class AddOne(nn.Module):
        def forward(self, z):
            return z + 1

    z = torch.zeros(4, 1)
    steps = torch.tensor([1, 3, 2, 5])

    assert torch.equal(rollout_to(AddOne(), z, steps).squeeze(1), steps.float())


def test_rollout_stop_gradient_blocks_target_path():
    torch.manual_seed(0)
    windows = torch.randn(64, K_MAX + 1, 1)
    grads = {}
    for symmetric in (False, True):
        torch.manual_seed(1)
        model = HorizonMLP(K_MAX, 1, 16)
        loss = anchor_loss_rollout(
            UnitStep(model, K_MAX),
            windows,
            torch.Generator().manual_seed(0),
            symmetric=symmetric,
        )
        loss.backward()
        grads[symmetric] = model.output_layer.weight.grad.clone()

    assert not torch.allclose(grads[False], grads[True])


def test_unit_step_is_the_horizon_one_prediction():
    torch.manual_seed(0)
    model = HorizonMLP(K_MAX, 2, 16)
    z = torch.randn(8, 2)
    expected = model(
        torch.cat(
            [
                z,
                nn.functional.one_hot(torch.zeros(8, dtype=torch.int64), K_MAX).float(),
            ],
            dim=1,
        )
    )

    assert torch.allclose(UnitStep(model, K_MAX)(z), expected)


def test_unit_rollout_loss_is_zero_for_consistent_model():
    windows = torch.randn(64, K_MAX + 1, 1)
    g = torch.Generator().manual_seed(0)

    assert anchor_loss_unit_rollout(IgnoresHorizon(), windows, g) == 0


@pytest.mark.parametrize(
    "loss_fn",
    [
        anchor_loss_composed,
        anchor_loss_bootstrapped,
        multistep_loss,
        anchor_loss_rollout,
        anchor_loss_unit_rollout,
    ],
)
def test_default_horizon_is_k_max(loss_fn):
    torch.manual_seed(0)
    windows = torch.randn(64, K_MAX + 1, 1)
    model = HorizonMLP(K_MAX, 1, 16)
    one_step_family = loss_fn in (multistep_loss, anchor_loss_rollout)
    aux_model = UnitStep(model, K_MAX) if one_step_family else model

    default = loss_fn(aux_model, windows, torch.Generator().manual_seed(0))
    explicit = loss_fn(
        aux_model, windows, torch.Generator().manual_seed(0), max_horizon=K_MAX
    )

    assert torch.equal(default, explicit)


def test_max_horizon_two_gives_two_step_losses():
    torch.manual_seed(0)
    windows = torch.randn(64, K_MAX + 1, 1)
    model = HorizonMLP(K_MAX, 1, 16)
    unit = UnitStep(model, K_MAX)
    g = torch.Generator().manual_seed(0)
    two = torch.full((64,), 2, dtype=torch.int64)
    rolled = unit(unit(windows[:, 0]))
    mse = nn.functional.mse_loss

    assert torch.allclose(
        multistep_loss(unit, windows, g, max_horizon=2), mse(rolled, windows[:, 2])
    )
    assert torch.allclose(
        anchor_loss_rollout(unit, windows, g, max_horizon=2),
        mse(rolled, unit(windows[:, 1])),
    )
    assert torch.allclose(
        anchor_loss_unit_rollout(model, windows, g, max_horizon=2),
        mse(rolled, predict(model, windows[:, 0], two, K_MAX)),
    )
    # k = 2 forces p = q = 1
    one = torch.ones(64, dtype=torch.int64)
    direct = predict(model, windows[:, 0], two, K_MAX)
    assert torch.allclose(
        anchor_loss_composed(model, windows, g, max_horizon=2),
        mse(
            predict(model, predict(model, windows[:, 0], one, K_MAX), one, K_MAX),
            direct,
        ),
    )
    assert torch.allclose(
        anchor_loss_bootstrapped(model, windows, g, max_horizon=2),
        mse(direct, predict(model, windows[:, 1], one, K_MAX)),
    )


def test_median_error_counts_nan_as_infinite():
    targets = torch.zeros(3, 2, 1)
    preds = torch.tensor(
        [[[1.0], [1.0]], [[2.0], [float("nan")]], [[3.0], [float("nan")]]]
    )

    # horizon 1: median of 1, 2, 3; horizon 2: median of 1, inf, inf
    assert median_error(preds, targets) == float("inf")
    assert median_error(preds[:, :1], targets[:, :1]) == 2.0
