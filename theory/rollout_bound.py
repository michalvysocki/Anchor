"""
Proposition T4 for AR(1) (paper: prop:t4-ar1; research note: step 5): the
bias of the k-step rollout of the best one-step model is bounded by the
path disagreement at k = 2,

    |a^k kappa (1 - kappa^{k-1})| <= (k-1) |a|^{k-2} |a^2 kappa (1 - kappa)|,

from 1 - kappa^{k-1} <= (k-1)(1 - kappa), also checked on a grid.
"""

import sys

import numpy as np
from common import CONFIGS, N_SE, Report, chains, slope


def main() -> int:
    report = Report("rollout bias bounded by short disagreement (T4, AR(1))")
    for index in range(len(CONFIGS)):
        a, _, kappa, _, y = chains(index)
        direct_2, _ = slope(y[:, 2], y[:, 0])
        disagreement = abs(direct_2 - (a * kappa) ** 2)
        for k in (3, 5, 8):
            direct_k, se = slope(y[:, k], y[:, 0])
            bias = abs(direct_k - (a * kappa) ** k)
            bound = (k - 1) * abs(a) ** (k - 2) * disagreement
            ratio = bias / bound if bound else float("nan")
            report.record(
                f"bias <= bound, k={k}",
                bias <= bound + N_SE * se,
                f"bias {bias:.5f}, bound {bound:.5f}, ratio {ratio:.3f}",
            )

    kappas = np.linspace(0.0, 1.0, 1001)
    worst = min(
        float(((k - 1) * (1 - kappas) - (1 - kappas ** (k - 1))).min())
        for k in range(2, 51)
    )
    report.record("1 - kappa^(k-1) <= (k-1)(1-kappa), grid", worst >= -1e-12)

    return report.summary()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
