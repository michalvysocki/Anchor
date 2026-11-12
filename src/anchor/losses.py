"""
Losses of the F(z, k) family (e01, e01b): grounding of the direct path,
the one-step loss, the multi-step loss C, and the Anchor variants A, B, D
and E with random decompositions k = p + q. The functions of the same name
in linear.py are the population losses of the linear window model (e02),
not these.
"""

import torch
from torch import nn

from anchor.models import UnitStep, predict


def grounding_loss(model: nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """
    MSE of direct predictions against true future points for every
    horizon of every window. With no Anchor term this is the same objective
    as the direct predictor.

    Args:
        model (nn.Module): horizon-conditioned model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)

    Returns:
        torch.Tensor: scalar loss
    """

    n, length, _ = windows.shape
    k_max = length - 1
    k = torch.arange(1, k_max + 1, device=windows.device).repeat(n)
    starts = windows[:, 0].repeat_interleave(k_max, dim=0)
    targets = windows[:, 1:].reshape(n * k_max, -1)

    return nn.functional.mse_loss(predict(model, starts, k, k_max), targets)


def sample_decomposition(
    n: int, k_max: int, p_one: bool, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples a horizon k >= 2 and its decomposition k = p + q for each window.

    Args:
        n (int): number of windows
        k_max (int): maximum step
        p_one (bool): if True p is always 1, otherwise p ~ U{1, ..., k - 1}
        generator (torch.Generator): generator for sampling, on CPU
        device (torch.device): device of the returned tensors

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: k, p and q of size (n,)
    """

    k = torch.randint(2, k_max + 1, (n,), generator=generator)
    if p_one:
        p = torch.ones(n, dtype=torch.int64)
    else:
        p = (torch.rand(n, generator=generator) * (k - 1)).long() + 1

    return k.to(device), p.to(device), (k - p).to(device)


def anchor_loss_composed(
    model: nn.Module,
    windows: torch.Tensor,
    generator: torch.Generator,
    p_one: bool = False,
    symmetric: bool = False,
    max_horizon: int | None = None,
) -> torch.Tensor:
    """
    Variant A: the composed path F(F(z, p), q) is pulled toward the direct
    prediction F(z, k), which is grounded by grounding_loss.

    Args:
        model (nn.Module): horizon-conditioned model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)
        generator (torch.Generator): generator for sampling decompositions
        p_one (bool): if True p is always 1
        symmetric (bool): if True gradients flow through both paths
        max_horizon (int | None): largest sampled k, k_max if None. The
            one-hot size still comes from the window length.

    Returns:
        torch.Tensor: scalar loss
    """

    n, length, _ = windows.shape
    k_max = length - 1
    k, p, q = sample_decomposition(
        n, max_horizon or k_max, p_one, generator, windows.device
    )
    z = windows[:, 0]

    direct = predict(model, z, k, k_max)
    if not symmetric:
        direct = direct.detach()
    composed = predict(model, predict(model, z, p, k_max), q, k_max)

    return nn.functional.mse_loss(composed, direct)


def anchor_loss_bootstrapped(
    model: nn.Module,
    windows: torch.Tensor,
    generator: torch.Generator,
    p_one: bool = False,
    symmetric: bool = False,
    max_horizon: int | None = None,
) -> torch.Tensor:
    """
    Variant B: the prediction F(z_t, k) is pulled toward the shorter
    prediction F(z_{t+p}, q) started from the true observation at t + p.

    Args:
        model (nn.Module): horizon-conditioned model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)
        generator (torch.Generator): generator for sampling decompositions
        p_one (bool): if True p is always 1
        symmetric (bool): if True gradients flow through both paths
        max_horizon (int | None): largest sampled k, k_max if None. The
            one-hot size still comes from the window length.

    Returns:
        torch.Tensor: scalar loss
    """

    n, length, _ = windows.shape
    k_max = length - 1
    k, p, q = sample_decomposition(
        n, max_horizon or k_max, p_one, generator, windows.device
    )
    rows = torch.arange(n, device=windows.device)

    anchor = predict(model, windows[rows, p], q, k_max)
    if not symmetric:
        anchor = anchor.detach()
    long_path = predict(model, windows[:, 0], k, k_max)

    return nn.functional.mse_loss(long_path, anchor)


def rollout_to(model: nn.Module, z: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """
    Rolls a one-step model forward and returns, for each sample, the state
    after its own number of steps.

    Args:
        model (nn.Module): one-step model
        z (torch.Tensor): starting points of size (batch, point_dim)
        steps (torch.Tensor): number of steps, at least 1, of size (batch,)

    Returns:
        torch.Tensor: states of size (batch, point_dim)
    """

    states = []
    for _ in range(int(steps.max())):
        z = model(z)
        states.append(z)
    states = torch.stack(states, dim=1)
    rows = torch.arange(len(steps), device=steps.device)

    return states[rows, steps - 1]


def one_step_loss(model: nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """
    MSE of one-step predictions for every consecutive pair in the windows.

    Args:
        model (nn.Module): one-step model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)

    Returns:
        torch.Tensor: scalar loss
    """

    point_dim = windows.shape[-1]
    inputs = windows[:, :-1].reshape(-1, point_dim)
    targets = windows[:, 1:].reshape(-1, point_dim)

    return nn.functional.mse_loss(model(inputs), targets)


def rollout_error(model: nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """
    MSE of free-running rollouts against true future points, averaged over
    every horizon from 1 to k_max. Used to select one-step models.

    Args:
        model (nn.Module): one-step model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)

    Returns:
        torch.Tensor: scalar loss
    """

    z = windows[:, 0]
    states = []
    for _ in range(windows.shape[1] - 1):
        z = model(z)
        states.append(z)

    return nn.functional.mse_loss(torch.stack(states, dim=1), windows[:, 1:])


def multistep_loss(
    model: nn.Module,
    windows: torch.Tensor,
    generator: torch.Generator,
    max_horizon: int | None = None,
) -> torch.Tensor:
    """
    Variant C: a k-step rollout from z_t against the true z_{t+k}, with k
    sampled like in variant D. This is variant D with p = k.

    Args:
        model (nn.Module): one-step model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)
        generator (torch.Generator): generator for sampling horizons
        max_horizon (int | None): largest sampled k, k_max if None

    Returns:
        torch.Tensor: scalar loss
    """

    n, length, _ = windows.shape
    k, _, _ = sample_decomposition(
        n, max_horizon or length - 1, False, generator, windows.device
    )
    rows = torch.arange(n, device=windows.device)

    return nn.functional.mse_loss(rollout_to(model, windows[:, 0], k), windows[rows, k])


def anchor_loss_rollout(
    model: nn.Module,
    windows: torch.Tensor,
    generator: torch.Generator,
    p_one: bool = False,
    symmetric: bool = False,
    max_horizon: int | None = None,
) -> torch.Tensor:
    """
    Variant D: a k-step rollout from z_t is pulled toward a q-step rollout
    started from the true observation z_{t+p}. Works with any one-step model.

    Args:
        model (nn.Module): one-step model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)
        generator (torch.Generator): generator for sampling decompositions
        p_one (bool): if True p is always 1
        symmetric (bool): if True gradients flow through both paths
        max_horizon (int | None): largest sampled k, k_max if None

    Returns:
        torch.Tensor: scalar loss
    """

    n, length, _ = windows.shape
    k, p, q = sample_decomposition(
        n, max_horizon or length - 1, p_one, generator, windows.device
    )
    rows = torch.arange(n, device=windows.device)

    anchor = rollout_to(model, windows[rows, p], q)
    if not symmetric:
        anchor = anchor.detach()
    long_path = rollout_to(model, windows[:, 0], k)

    return nn.functional.mse_loss(long_path, anchor)


def anchor_loss_unit_rollout(
    model: nn.Module,
    windows: torch.Tensor,
    generator: torch.Generator,
    p_one: bool = False,
    symmetric: bool = False,
    max_horizon: int | None = None,
) -> torch.Tensor:
    """
    Variant E: the unit-step rollout F(., 1)^k from z_t is pulled toward
    the direct prediction F(z_t, k) of the same model, which is grounded by
    grounding_loss.

    Args:
        model (nn.Module): horizon-conditioned model
        windows (torch.Tensor): windows of size (n, k_max + 1, point_dim)
        generator (torch.Generator): generator for sampling horizons
        p_one (bool): unused, kept for the same signature as other variants
        symmetric (bool): if True gradients flow through both paths
        max_horizon (int | None): largest sampled k, k_max if None. The
            one-hot size still comes from the window length.

    Returns:
        torch.Tensor: scalar loss
    """

    n, length, _ = windows.shape
    k_max = length - 1
    k, _, _ = sample_decomposition(
        n, max_horizon or k_max, False, generator, windows.device
    )
    z = windows[:, 0]

    direct = predict(model, z, k, k_max)
    if not symmetric:
        direct = direct.detach()
    rolled = rollout_to(UnitStep(model, k_max), z, k)

    return nn.functional.mse_loss(rolled, direct)
