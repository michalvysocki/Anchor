"""
Variance of the multi-step targets given a Markov state (paper: T3, scalar
part; research note: step 2). With the state observed and the ideal model,
the Monte Carlo target C = x_{t+k} and the bootstrapped target
D = a^q x_{t+p} have

    Var[C | x_t] = sigma2 sum_{j<k} a^{2j},   Var[D | x_t] = sigma2 sum_{q<=j<k} a^{2j},

so D removes the process noise of the last q steps.
"""

import sys

from common import CONFIGS, Report, chains, variance


def main() -> int:
    report = Report("target variance (T3, scalar)")
    for index in range(len(CONFIGS)):
        a, sigma2, _, x, _ = chains(index)
        for k, p in ((2, 1), (5, 2), (8, 3)):
            q = k - p
            mean = a**k * x[:, 0]
            report.check(
                f"Var[C | x_t], k={k}",
                variance(x[:, k] - mean),
                sigma2 * sum(a ** (2 * j) for j in range(k)),
            )
            report.check(
                f"Var[D | x_t], k={k}, p={p}",
                variance(a**q * x[:, p] - mean),
                sigma2 * sum(a ** (2 * j) for j in range(q, k)),
            )

    return report.summary()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
