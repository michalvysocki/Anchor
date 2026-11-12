"""
The Markov gap of a noisy observation in AR(1) (research note: steps 3-4;
paper: setting of Propositions T2 and T4). With kappa = s / (s + r):

    E[y_{t+j} | y_t] = a^j kappa y_t,
    E[D | y_t] = a^k kappa^2 y_t      for the ideal D = a^q kappa y_{t+p},
    E[D_rollout | y_t] = a^k kappa^{q+1} y_t   for D = (a kappa)^q y_{t+p},

so each bootstrap point shrinks the target by kappa and the rollout of the
best one-step model aims at kappa^{k-1} of the optimum.
"""

import sys

from common import CONFIGS, Report, chains, slope


def main() -> int:
    report = Report("Markov gap of a noisy observation")
    for index in range(len(CONFIGS)):
        a, _, kappa, _, y = chains(index)
        for j in (1, 2, 5):
            report.check(
                f"E[y_t+{j} | y_t] slope", slope(y[:, j], y[:, 0]), a**j * kappa
            )
        for k, p in ((2, 1), (5, 2), (8, 3)):
            q = k - p
            report.check(
                f"E[D | y_t] slope, k={k}, p={p}",
                slope(a**q * kappa * y[:, p], y[:, 0]),
                a**k * kappa**2,
            )
            report.check(
                f"E[D_rollout | y_t] slope, k={k}, p={p}",
                slope((a * kappa) ** q * y[:, p], y[:, 0]),
                a**k * kappa ** (q + 1),
            )

    return report.summary()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
