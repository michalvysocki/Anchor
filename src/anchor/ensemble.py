"""
Ensembles of independent small networks trained in one batch (e03): every
parameter has a leading model dimension M, every activation has shape
(M, batch, ...), so M models with their own data, initialization, learning
rate and stopping time train in one forward and backward pass. Used for the
learned-latent world models of e03; the networks of e01 and e01b are in
models.py.
"""

import itertools
import math

import numpy as np
import torch
from torch import nn

Key = tuple[int, ...]


def init_uniform(shape: tuple[int, ...], fan_in: int, keys: list[Key]) -> torch.Tensor:
    """
    PyTorch's default uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)), one
    generator per model. Each tensor has its own key (model, module, layer,
    kind), hashed by SeedSequence, so no two tensors share a random stream.
    """

    bound = 1 / math.sqrt(fan_in)
    out = []
    for key in keys:
        seed = int(np.random.SeedSequence(list(key)).generate_state(1)[0])
        gen = torch.Generator().manual_seed(seed)
        out.append((torch.rand(shape, generator=gen) * 2 - 1) * bound)

    return torch.stack(out)


class EnsembleLinear(nn.Module):
    """M independent linear layers: (M, B, n_in) -> (M, B, n_out)."""

    def __init__(self, n_in: int, n_out: int, keys: list[Key]):
        super().__init__()
        self.weight = nn.Parameter(
            init_uniform((n_in, n_out), n_in, [(*k, 0) for k in keys])
        )
        self.bias = nn.Parameter(init_uniform((n_out,), n_in, [(*k, 1) for k in keys]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mbi,mio->mbo", x, self.weight) + self.bias[:, None]


class EnsembleMLP(nn.Module):
    """M independent MLPs with tanh hidden layers and a linear output."""

    def __init__(self, sizes: list[int], keys: list[Key]):
        super().__init__()
        self.layers = nn.ModuleList(
            EnsembleLinear(a, b, [(*k, i) for k in keys])
            for i, (a, b) in enumerate(itertools.pairwise(sizes))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class EnsembleGRU(nn.Module):
    """M independent single-layer GRUs; returns the last hidden state, (M, B, hidden)."""

    def __init__(self, n_in: int, hidden: int, keys: list[Key]):
        super().__init__()
        self.hidden = hidden
        self.w_in = nn.Parameter(
            init_uniform((n_in, 3 * hidden), hidden, [(*k, 0) for k in keys])
        )
        self.w_h = nn.Parameter(
            init_uniform((hidden, 3 * hidden), hidden, [(*k, 1) for k in keys])
        )
        self.b = nn.Parameter(
            init_uniform((3 * hidden,), hidden, [(*k, 2) for k in keys])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x of shape (M, B, T, n_in)."""
        n_models, batch, steps, _ = x.shape
        h = x.new_zeros(n_models, batch, self.hidden)
        gates_in = torch.einsum("mbti,mio->mbto", x, self.w_in) + self.b[:, None, None]
        for t in range(steps):
            gh = torch.einsum("mbh,mho->mbo", h, self.w_h)
            ri, zi, ni = gates_in[:, :, t].chunk(3, dim=-1)
            rh, zh, nh = gh.chunk(3, dim=-1)
            r = torch.sigmoid(ri + rh)
            z = torch.sigmoid(zi + zh)
            n = torch.tanh(ni + r * nh)
            h = (1 - z) * n + z * h
        return h


class EnsembleAdam:
    """
    Adam over parameters with a leading model dimension, with a learning
    rate per model (a model with rate 0 is frozen) and gradient clipping of
    each model's total gradient norm.
    """

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: torch.Tensor,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        clip: float | None = 1.0,
    ):
        self.params = [p for p in params]
        self.lr = lr.clone()
        self.betas, self.eps, self.clip = betas, eps, clip
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = torch.zeros(len(lr), device=lr.device)

    def model_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    @torch.no_grad()
    def step(self):
        grads = [
            p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params
        ]
        active = (self.lr > 0).float()
        if self.clip is not None:
            sq = sum(self.model_view(g).pow(2).sum(1) for g in grads)
            scale = torch.clamp(self.clip / (sq.sqrt() + 1e-12), max=1.0)
        else:
            scale = torch.ones(len(self.lr), device=self.lr.device)
        self.t += active
        b1, b2 = self.betas
        bc1 = 1 - b1 ** self.t.clamp(min=1)
        bc2 = 1 - b2 ** self.t.clamp(min=1)
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            shape = (-1,) + (1,) * (p.dim() - 1)
            g = g * scale.view(shape)
            a = active.view(shape)
            m.mul_(1 - a * (1 - b1)).add_(a * (1 - b1) * g)
            v.mul_(1 - a * (1 - b2)).add_(a * (1 - b2) * g * g)
            step = (
                (self.lr / bc1).view(shape)
                * m
                / ((v / bc2.view(shape)).sqrt() + self.eps)
            )
            p.sub_(a * step)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
