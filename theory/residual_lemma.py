"""
Residual lemma, critic and the remark on p = 1 (paper: lem:residual,
cor:critic, rem:p1; research note: step 6). For a linear one-step model
g(y) = b y the bootstrap residual R = g^k(y_t) - g^q(y_{t+p}) has

    E[R | y_t] = b^q (b^p - a^p kappa) y_t,

the rollout bias at horizon p carried forward by g^q. It vanishes for p = 1
at the one-step optimum b = a kappa, and a critic regressed on the target
g^q(y_{t+p}) recovers it without the target variance. In the Markov case
with the ideal model the residual has mean 0 but positive variance.
"""

import sys

from common import CONFIGS, Report, chains, slope, variance


def main() -> int:
    report = Report("residual lemma, critic, p = 1")
    for index in range(len(CONFIGS)):
        a, sigma2, kappa, x, y = chains(index)
        for k, p in ((2, 1), (5, 2)):
            q = k - p
            resid = a**k * x[:, 0] - a**q * x[:, p]
            report.check(
                f"Markov: E[R | x_t] slope, k={k}, p={p}", slope(resid, x[:, 0]), 0.0
            )
            report.check(
                f"Markov: Var[R | x_t], k={k}, p={p}",
                variance(resid),
                sigma2 * a ** (2 * q) * sum(a ** (2 * j) for j in range(p)),
            )
        # b = a kappa is the best one-step model from y; the other b is misfit
        for b, label in ((a * kappa, "optimal"), (0.6 * a, "misfit")):
            for k, p in ((2, 1), (4, 2), (8, 3), (8, 8)):
                q = k - p
                lemma = b**q * (b**p - a**p * kappa)
                report.check(
                    f"{label}: E[R | y_t], k={k}, p={p}",
                    slope(b**k * y[:, 0] - b**q * y[:, p], y[:, 0]),
                    lemma,
                )
                critic, se = slope(b**q * y[:, p], y[:, 0])
                report.check(
                    f"{label}: critic h - g^k, k={k}, p={p}",
                    (critic - b**k, se),
                    -lemma,
                )

    return report.summary()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
