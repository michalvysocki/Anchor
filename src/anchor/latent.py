"""
Latent world models with Anchor targets, as ensembles of M independent
models trained in one batch (e03; to be reused by e04 and e05): an encoder
of the observation window (MLP) or history (GRU), residual latent dynamics
with an optional flag (encoded vs predicted input), a decoder to the
observation, and the critic heads of the targets. The one-step loss
predicts y_{t+1} and keeps the predicted latent consistent with the encoding
of the next window (as TD-MPC and SPR); the multi-step targets live in
observation space and are under stop-gradient:

- C: the observed future y_{t+k} (Monte Carlo)
- D: the decoded rollout of the same model from the window at t + p
- DB: D plus a critic delta(z_t) of the part D misses (Debiased Bootstrap)
- TDB: one critic shift per anchor point, added to the latent and propagated
- M*: a mixture over anchor points with bias-variance weights from critics

Variant names: a target, "+F" for the flag, and an optional suffix for an
ablation ("_two": two dynamics networks instead of a flag input; "_wide":
unflagged dynamics with the parameter count of "_two"; others are handled
by the experiment). The F(z, k) networks of e01 and e01b are in models.py.
"""

from dataclasses import dataclass

import numpy as np
import torch

from anchor.ensemble import EnsembleGRU, EnsembleMLP
from anchor.linear import simplex_minimizer


@dataclass(frozen=True)
class Sizes:
    latent: int = 16
    hidden: int = 128
    decoder_hidden: int = 64
    critic_hidden: int = 64
    eval_horizon: int = 10
    gru_history: int = 16

    def past(self, window: int | None) -> int:
        """Observations the encoder sees: the window, or the GRU's history."""
        return self.gru_history if window is None else window


DEFAULT_SIZES = Sizes()


def flagged(variant: str) -> bool:
    return "+F" in variant


def base(variant: str) -> str:
    """Target of a variant name: onestep, C, D, DB, TDB, M* (or direct)."""
    return variant.split("+")[0].split("_")[0]


def anchor_pairs(horizons: tuple[int, ...]) -> list[tuple[int, int]]:
    """(k, p) with anchor points p = 2..k-1 for every horizon k >= 3."""
    return [(k, p) for k in horizons for p in range(2, k)]


class WorldModels(torch.nn.Module):
    """
    M latent world models: encoder phi (window MLP or GRU), residual
    dynamics g with an optional one-hot flag (encoded / predicted input),
    decoder r to y, and the critic heads a variant needs. The heads take
    sg[phi(z_t)] and are trained only by their own losses.
    """

    def __init__(
        self,
        variant: str,
        window: int | None,
        horizons: tuple[int, ...],
        seeds: list[int],
        sizes: Sizes = DEFAULT_SIZES,
    ):
        super().__init__()
        self.s = sizes
        self.variant, self.target = variant, base(variant)
        self.flag = flagged(variant)
        self.two_nets = variant.endswith("_two")
        self.window, self.past = window, sizes.past(window)
        self.horizons = horizons
        self.pairs = anchor_pairs(horizons)

        # initialization keys (model seed, module); see ensemble.init_uniform
        def keys(module):
            return [(m, module) for m in seeds]

        if window is None:
            self.enc = EnsembleGRU(1, self.s.latent, keys(1))
        else:
            self.enc = EnsembleMLP(
                [window, self.s.hidden, self.s.hidden, self.s.latent], keys(1)
            )
        width = (
            wide_hidden(sizes.latent, sizes.hidden)
            if variant.endswith("_wide")
            else sizes.hidden
        )
        n_in = self.s.latent + (2 if self.flag and not self.two_nets else 0)
        self.dyn = EnsembleMLP([n_in, width, width, self.s.latent], keys(2))
        if self.two_nets:
            self.dyn_pred = EnsembleMLP(
                [self.s.latent, self.s.hidden, self.s.hidden, self.s.latent], keys(3)
            )
        self.dec = EnsembleMLP([self.s.latent, self.s.decoder_hidden, 1], keys(4))
        n_out = {
            "DB": len(self.pairs),
            "TDB": self.s.latent * len(self.points()),
            "M*": len(self.pairs) + len(horizons),
        }.get(self.target)
        self.critic = (
            EnsembleMLP([self.s.latent, self.s.critic_hidden, n_out], keys(5))
            if n_out
            else None
        )
        # M* weights per horizon: (M, number of anchor points p = 2..k), C at start
        self.weights = {
            k: torch.eye(k - 1)[-1].expand(len(seeds), -1).clone() for k in horizons
        }

    def points(self) -> list[int]:
        return sorted({p for _, p in self.pairs})

    def model_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("critic")]

    def critic_parameters(self):
        return [] if self.critic is None else list(self.critic.parameters())

    # -- forward pieces

    def encode(self, w: torch.Tensor, start: int) -> torch.Tensor:
        """Latent of the window ending at y_{t+start}; w of shape (M, B, width)."""
        seq = w[..., start : start + self.past]
        return self.enc(seq[..., None]) if self.window is None else self.enc(seq)

    def step(self, u: torch.Tensor, first: bool) -> torch.Tensor:
        if self.two_nets:
            return u + (self.dyn(u) if first else self.dyn_pred(u))
        if self.flag:
            f = u.new_zeros(*u.shape[:-1], 2)
            f[..., 0 if first else 1] = 1.0
            return u + self.dyn(torch.cat([u, f], -1))
        return u + self.dyn(u)

    def rollout(self, z: torch.Tensor, steps: int, shift=None) -> list[torch.Tensor]:
        """Latents u_1..u_steps from an encoded z; shift is added to u_1 (TDB)."""
        u = self.step(z, True)
        if shift is not None:
            u = u + shift
        out = [u]
        for _ in range(steps - 1):
            u = self.step(u, False)
            out.append(u)
        return out

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return self.dec(u)[..., 0]

    def y(self, w: torch.Tensor, k: int) -> torch.Tensor:
        return w[..., self.past + k - 1]

    # -- targets

    def critic_out(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.critic(z_t.detach())

    def tdb_shift(self, out: torch.Tensor, p: int) -> torch.Tensor:
        i = self.points().index(p)
        return out[..., i * self.s.latent : (i + 1) * self.s.latent]

    @torch.no_grad()
    def bootstraps(self, w: torch.Tensor, out=None) -> dict:
        """
        T_{k,p} for every pair: one rollout from the window at t + p serves
        every k > p (TDB adds its shift to u_{p+1}), under stop-gradient.
        """

        tb, top = {}, max(self.horizons)
        for p in self.points():
            shift = self.tdb_shift(out, p) if self.target == "TDB" else None
            us = self.rollout(self.encode(w, p), top - p, shift)
            for k in self.horizons:
                if k > p:
                    tb[k, p] = self.decode(us[k - p - 1])
        return tb

    @torch.no_grad()
    def targets(self, w, z_t, boot) -> dict[int, torch.Tensor]:
        """Target of the rollout to every training horizon k >= 2, under stop-gradient."""
        out = self.critic_out(z_t) if self.target == "DB" else None
        tg = {}
        for k in self.horizons:
            y = self.y(w, k)
            if self.target == "C" or k == 2:
                tg[k] = y
                continue
            rows = []
            for p in range(2, k):
                t = boot[k, p]
                if self.target == "DB":
                    t = t + out[..., self.pairs.index((k, p))]
                rows.append(t)
            if self.target == "M*":
                wts = self.weights[k].to(w.dtype)
                tg[k] = (
                    sum(wts[:, i, None] * r for i, r in enumerate(rows))
                    + wts[:, -1, None] * y
                )
            else:
                tg[k] = torch.stack(rows).mean(0)
        return tg

    def boot_for(self, w, z_t):
        out = self.critic_out(z_t).detach() if self.target == "TDB" else None
        return self.bootstraps(w, out)

    # -- losses, each of shape (M,)

    def losses(
        self, w: torch.Tensor, lam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Model loss and critic loss. The one-step loss has two terms, as in
        TD-MPC and SPR: the prediction of y_{t+1} and the consistency of the
        predicted latent with sg[phi(z_{t+1})]. Without the second, nothing
        makes u_1 a latent that the dynamics can be applied to again, and a
        model trained on one step has no meaningful rollout. The auxiliary
        multi-step term is weighted by lam.
        """

        z_t = self.encode(w, 0)
        us = self.rollout(z_t, max(self.horizons) if self.target != "onestep" else 1)
        preds = [self.decode(u) for u in us]
        with torch.no_grad():
            z_next = self.encode(w, 1)
        consistency = (us[0] - z_next).pow(2).mean((-2, -1))
        one = (preds[0] - self.y(w, 1)).pow(2).mean(-1) + consistency
        if self.target == "onestep":
            return one, one.new_zeros(one.shape)
        boot = self.boot_for(w, z_t) if self.target != "C" else {}
        tg = self.targets(w, z_t, boot)
        aux = torch.stack(
            [(preds[k - 1] - tg[k]).pow(2).mean(-1) for k in self.horizons]
        ).mean(0)
        return one + lam * aux, self.critic_loss(w, z_t, boot)

    def critic_loss(
        self, w: torch.Tensor, z_t: torch.Tensor, boot=None
    ) -> torch.Tensor:
        if self.critic is None:
            return z_t.new_zeros(z_t.shape[0])
        if boot is None and self.target in ("DB", "M*"):
            boot = self.bootstraps(w)
        out = self.critic_out(z_t)
        terms = []
        if self.target == "DB":
            for i, (k, p) in enumerate(self.pairs):
                resid = self.y(w, k) - boot[k, p]
                terms.append((out[..., i] - resid).pow(2).mean(-1))
        elif self.target == "TDB":
            # one shift per anchor point, fitted on the true futures k = p+1..self.s.eval_horizon
            # through the model's own dynamics; the model is not updated by this loss
            for p in self.points():
                us = self.rollout(
                    self.encode(w, p).detach(),
                    self.s.eval_horizon - p,
                    self.tdb_shift(out, p),
                )
                for j, u in enumerate(us):
                    terms.append(
                        (self.decode(u) - self.y(w, p + j + 1)).pow(2).mean(-1)
                    )
        else:  # M*: regressions of every T_{k,p} and of y_{t+k} on z_t
            for i, (k, p) in enumerate(self.pairs):
                terms.append((out[..., i] - boot[k, p]).pow(2).mean(-1))
            for j, k in enumerate(self.horizons):
                terms.append(
                    (out[..., len(self.pairs) + j] - self.y(w, k)).pow(2).mean(-1)
                )
        return torch.stack(terms).mean(0)

    @torch.no_grad()
    def update_weights(self, w: torch.Tensor, n_windows: int):
        """M*: weights minimizing w'(B + V/N)w on the simplex, from the critic heads."""
        z_t = self.encode(w, 0)
        out = self.critic_out(z_t)
        boot = self.bootstraps(w)
        for k in self.horizons:
            if k == 2:
                continue
            ps = list(range(2, k))
            h = torch.stack(
                [out[..., self.pairs.index((k, p))] for p in ps]
                + [out[..., len(self.pairs) + self.horizons.index(k)]]
            )  # (P, M, B)
            t = torch.stack([boot[k, p] for p in ps] + [self.y(w, k)])
            bias = h - h[-1:]
            resid = t - h
            resid = resid - resid.mean(-1, keepdim=True)
            b = torch.einsum("pmb,qmb->mpq", bias, bias) / w.shape[1]
            v = torch.einsum("pmb,qmb->mpq", resid, resid) / (w.shape[1] - 1)
            q = (b + v / n_windows).cpu().double().numpy()
            self.weights[k] = torch.tensor(
                np.stack([simplex_minimizer(qm) for qm in q]),
                dtype=w.dtype,
                device=w.device,
            )

    @torch.no_grad()
    def rollout_mse(self, w: torch.Tensor, clean: torch.Tensor | None = None) -> dict:
        """Per model: MSE of the rollout against y (and clean x) at k = 1..self.s.eval_horizon."""
        us = self.rollout(self.encode(w, 0), self.s.eval_horizon)
        preds = torch.stack([self.decode(u) for u in us])  # (K, M, B)
        ys = torch.stack([self.y(w, k) for k in range(1, self.s.eval_horizon + 1)])
        out = {
            "mse": (preds - ys).pow(2).mean(-1).T,
            "diverged": (preds.abs() > 10).any(0).float().mean(-1),
        }
        if clean is not None:
            xs = torch.stack(
                [
                    clean[..., self.past + k - 1]
                    for k in range(1, self.s.eval_horizon + 1)
                ]
            )
            out["mse_clean"] = (preds - xs).pow(2).mean(-1).T
        return out


def wide_hidden(latent: int, hidden: int) -> int:
    """Width of the unflagged dynamics with the parameter count of the two-network flag."""

    def count(h):
        return (latent + 1) * h + (h + 1) * h + (h + 1) * latent

    target = 2 * count(hidden)
    return min(range(hidden, 4 * hidden), key=lambda h: abs(count(h) - target))


class DirectModels(torch.nn.Module):
    """Reference F(z_t, k): the same encoder, a head on [phi(z_t), one-hot k], k = 1..10."""

    def __init__(
        self, window: int | None, seeds: list[int], sizes: Sizes = DEFAULT_SIZES
    ):
        super().__init__()
        self.s = sizes
        self.window, self.past = window, sizes.past(window)
        keys = [(m, 1) for m in seeds]
        if window is None:
            self.enc = EnsembleGRU(1, self.s.latent, keys)
        else:
            self.enc = EnsembleMLP(
                [window, self.s.hidden, self.s.hidden, self.s.latent], keys
            )
        self.head = EnsembleMLP(
            [self.s.latent + self.s.eval_horizon, self.s.hidden, 1],
            [(m, 6) for m in seeds],
        )

    def model_parameters(self):
        return list(self.parameters())

    def critic_parameters(self):
        return []

    def predict_all(self, w: torch.Tensor) -> torch.Tensor:
        seq = w[..., : self.past]
        z = self.enc(seq[..., None]) if self.window is None else self.enc(seq)
        preds = []
        for k in range(self.s.eval_horizon):
            onehot = z.new_zeros(*z.shape[:-1], self.s.eval_horizon)
            onehot[..., k] = 1.0
            preds.append(self.head(torch.cat([z, onehot], -1))[..., 0])
        return torch.stack(preds)  # (K, M, B)

    def losses(self, w, lam):
        ys = torch.stack([w[..., self.past + k] for k in range(self.s.eval_horizon)])
        loss = (self.predict_all(w) - ys).pow(2).mean((0, -1))
        return loss, loss.new_zeros(loss.shape)

    @torch.no_grad()
    def rollout_mse(self, w, clean=None):
        preds = self.predict_all(w)
        ys = torch.stack([w[..., self.past + k] for k in range(self.s.eval_horizon)])
        out = {
            "mse": (preds - ys).pow(2).mean(-1).T,
            "diverged": (preds.abs() > 10).any(0).float().mean(-1),
        }
        if clean is not None:
            xs = torch.stack(
                [clean[..., self.past + k] for k in range(self.s.eval_horizon)]
            )
            out["mse_clean"] = (preds - xs).pow(2).mean(-1).T
        return out
