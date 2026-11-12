"""
Proposition T1 (paper: prop:t1): for a latent that is Markov in the mean and
an exact affine one-step model, the gradient of every anchor term vanishes
at the one-step optimum, so it is a fixed point for every lambda. Checked on
the linear window model of e02 (population, exact): the update field is zero
at the one-step optimum and the stop-gradient fixed point does not move, for
C, D, M and M*. Without the stop-gradient (D_nosg) it moves, so the
stop-gradient is a condition of T1; with a Markov gap every variant moves.
"""

import sys

import numpy as np
import torch
from common import Report

from anchor import linear
from anchor.envs.lgssm import ar1, ar2, observation_covariance

HORIZON = 10
MARKOV = [(ar1(0.9, 1.0), 1), (ar1(0.99, 1.0), 4), (ar2(0.9, np.pi / 6, 1.0), 2)]
GAP = [(ar1(0.9, 0.5), 1), (ar2(0.9, np.pi / 6, 0.8), 2)]
LAMBDAS = (0.1, 10.0)
N_EFF = 1000


def setup(system, window):
    cov = observation_covariance(system, linear.max_horizon_size(window, HORIZON))
    return cov, linear.one_step_solution(cov, window)


def main() -> int:
    report = Report("harmlessness for Markov latents (T1)")
    for system, window in MARKOV:
        cov, optimum = setup(system, window)
        cov_t = torch.as_tensor(cov / cov[0, 0])
        theta = torch.as_tensor(optimum)
        for variant in ("C", "D", "M", "M*"):
            for lam in LAMBDAS:
                weights = None
                if variant == "M":
                    weights = linear.uniform_weights()
                if variant == "M*":
                    weights = linear.bias_variance_weights(optimum, cov, N_EFF)
                field = linear.update_field(variant, lam, cov_t, weights)(theta)
                point = linear.fixed_point(variant, lam, cov, window, n_eff=N_EFF)
                shift = float(np.abs(point["theta"] - optimum).max())
                report.record(
                    f"{system.name} L{window} {variant} λ={lam}",
                    field.norm().item() < 1e-10 and shift < 1e-8,
                    f"|field| {field.norm().item():.1e}, fixed point moved {shift:.1e}",
                )
        point = linear.fixed_point("D_nosg", 1.0, cov, window)
        shift = float(np.abs(point["theta"] - optimum).max())
        report.record(
            f"{system.name} L{window} D_nosg λ=1 moves",
            shift > 1e-3,
            f"moved {shift:.3f}",
        )
    for system, window in GAP:
        cov, optimum = setup(system, window)
        for variant in ("C", "D"):
            point = linear.fixed_point(variant, 1.0, cov, window)
            shift = float(np.abs(point["theta"] - optimum).max())
            report.record(
                f"gap: {system.name} L{window} {variant} λ=1 moves",
                shift > 1e-3,
                f"moved {shift:.3f}",
            )

    return report.summary()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
