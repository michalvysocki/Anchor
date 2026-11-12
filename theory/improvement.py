"""
Proposition T2 for AR(1) (paper: prop:t2-ar1; research note: step 8). For
g(y) = b y with the anchor term at k = 3, p = 2 under stop-gradient, the
fixed point solves

    F(b, lambda) = (b - a kappa) + 3 lambda b^3 (b^2 - a^2 kappa) = 0,

with b'(0) = 3 a^5 kappa^4 (1 - kappa). The excess errors E_k = (b^k - a^k
kappa)^2 have zero derivative at k = 1 and a negative derivative along
b'(0) at every k >= 2, so a small lambda strictly lowers sum_k E_k. As
lambda grows the fixed point tends to a sqrt(kappa), exact at k = p = 2, and
a degenerate branch tends to 0.
"""

import math
import sys

from common import Report

CASES = [(0.9, 0.5), (0.5, 0.8), (-0.7, 0.6), (0.99, 0.9)]
K = 10


def field(b: float, lam: float, a: float, kappa: float) -> float:
    return (b - a * kappa) + 3 * lam * b**3 * (b**2 - a**2 * kappa)


def root(lam: float, a: float, kappa: float, lo: float, hi: float) -> float:
    """Bisection for F = 0 between lo and hi, where F changes sign."""
    f_lo = field(lo, lam, a, kappa)
    if f_lo == 0:
        return lo
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = field(mid, lam, a, kappa)
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def branch(lam: float, a: float, kappa: float) -> float:
    """Fixed point connected to lambda = 0: between a kappa and a sqrt(kappa)."""
    return root(lam, a, kappa, a * kappa, a * math.sqrt(kappa))


def excess(b: float, a: float, kappa: float) -> list[float]:
    return [(b**k - a**k * kappa) ** 2 for k in range(1, K + 1)]


def main() -> int:
    report = Report("small Anchor always helps (T2, AR(1))")
    h = 1e-6
    for a, kappa in CASES:
        name = f"a={a}, kappa={kappa}"
        slope_fd = (branch(h, a, kappa) - branch(0.0, a, kappa)) / h
        slope_th = 3 * a**5 * kappa**4 * (1 - kappa)
        report.record(
            f"{name}: b'(0)",
            abs(slope_fd - slope_th) < 1e-4 * max(1.0, abs(slope_th)),
            f"finite difference {slope_fd:+.6f}, formula {slope_th:+.6f}",
        )
        e0, e1 = (
            excess(branch(0.0, a, kappa), a, kappa),
            excess(branch(h, a, kappa), a, kappa),
        )
        d_total = (sum(e1) - sum(e0)) / h
        d_first = (e1[0] - e0[0]) / h
        report.record(
            f"{name}: d sum E_k / d lambda < 0", d_total < 0, f"{d_total:+.3e}"
        )
        report.record(
            f"{name}: k = 1 is second order",
            abs(d_first) < 1e-3 * abs(d_total),
            f"dE_1/dλ {d_first:+.1e}",
        )
        limit = branch(1e8, a, kappa)
        report.record(
            f"{name}: λ → ∞ gives a sqrt(kappa)",
            abs(limit - a * math.sqrt(kappa)) < 1e-3,
            f"b(1e8) {limit:+.5f}, a sqrt(kappa) {a * math.sqrt(kappa):+.5f}",
        )
        # degenerate branch: sign opposite to a, size about (3 lambda a)^(-1/3)
        degenerate = root(1e8, a, kappa, -math.copysign(0.05, a), 0.0)
        report.record(
            f"{name}: degenerate branch → 0",
            abs(field(degenerate, 1e8, a, kappa)) < 1e-6 and abs(degenerate) < 0.01,
            f"b {degenerate:+.5f}",
        )

    return report.summary()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
