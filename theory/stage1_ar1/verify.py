"""
Numerical check of the Stage 1 derivations on AR(1) with observation noise,
see "Stage 1 Theory" in the research note.

    x_{t+1} = a x_t + w_t,  w_t ~ N(0, sigma2)
    y_t     = x_t + v_t,    v_t ~ N(0, r)

Every closed form is compared with a Monte Carlo estimate from independent
stationary chains; a check passes when the difference is within N_SE
standard errors. The bound of step 5 is also checked on a grid.
"""

import sys

import numpy as np

N_CHAINS = 400_000
K_MAX = 8
N_SE = 4.0
SEED = 0
# (a, sigma2, r as a fraction of the stationary variance s)
CONFIGS = (
    (0.9, 1.0, 1.0),  # kappa = 0.5
    (0.5, 1.0, 0.25),  # kappa = 0.8
    (-0.7, 1.0, 0.0),  # kappa = 1, Markov observation
)


def simulate(a: float, sigma2: float, r: float, rng: np.random.Generator):
    """Stationary chains of states and observations, each of size (N_CHAINS, K_MAX + 1)."""
    s = sigma2 / (1 - a**2)
    x = np.empty((N_CHAINS, K_MAX + 1))
    x[:, 0] = rng.normal(0.0, np.sqrt(s), N_CHAINS)
    for t in range(K_MAX):
        x[:, t + 1] = a * x[:, t] + rng.normal(0.0, np.sqrt(sigma2), N_CHAINS)
    y = x + rng.normal(0.0, np.sqrt(r), x.shape)

    return x, y


def slope(target: np.ndarray, regressor: np.ndarray) -> tuple[float, float]:
    """Least-squares slope through the origin and its standard error."""
    b = float(target @ regressor / (regressor @ regressor))
    resid = target - b * regressor
    se = float(np.sqrt(resid.var() / (regressor @ regressor)))

    return b, se


def variance(values: np.ndarray) -> tuple[float, float]:
    """Sample variance and its standard error."""
    v = float(values.var())

    return v, v * np.sqrt(2.0 / (len(values) - 1))


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, name: str, estimate: tuple[float, float], theory: float):
        value, se = estimate
        ok = abs(value - theory) <= N_SE * se + 1e-12
        self.failures += not ok
        print(
            f"  {'ok  ' if ok else 'FAIL'} {name:<34} "
            f"sim {value:+.5f} ± {se:.5f}   theory {theory:+.5f}"
        )


def check_config(a: float, sigma2: float, r_frac: float, report: Report, rng):
    s = sigma2 / (1 - a**2)
    r = r_frac * s
    kappa = s / (s + r)
    print(f"\na = {a}, sigma2 = {sigma2}, r = {r:.4f}, kappa = {kappa:.4f}")
    x, y = simulate(a, sigma2, r, rng)

    print(" step 2: target variance given the state")
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

    print(" step 3: regression on the observation")
    for j in (1, 2, 5):
        report.check(f"E[y_t+{j} | y_t] slope", slope(y[:, j], y[:, 0]), a**j * kappa)
    for k, p in ((2, 1), (5, 2), (8, 3)):
        q = k - p
        d = a**q * kappa * y[:, p]
        report.check(
            f"E[D | y_t] slope, k={k}, p={p}", slope(d, y[:, 0]), a**k * kappa**2
        )

    print(" step 4: rollout family")
    for k, p in ((2, 1), (5, 2), (8, 3)):
        q = k - p
        d = (a * kappa) ** q * y[:, p]
        report.check(
            f"E[D_rollout | y_t] slope, k={k}, p={p}",
            slope(d, y[:, 0]),
            a**k * kappa ** (q + 1),
        )

    print(" step 5: rollout bias vs disagreement at k = 2 (empirical slopes)")
    direct_2, _ = slope(y[:, 2], y[:, 0])
    disagreement = abs(direct_2 - (a * kappa) ** 2)
    for k in (3, 5, 8):
        direct_k, se = slope(y[:, k], y[:, 0])
        bias = abs(direct_k - (a * kappa) ** k)
        bound = (k - 1) * abs(a) ** (k - 2) * disagreement
        ok = bias <= bound + N_SE * se
        report.failures += not ok
        print(
            f"  {'ok  ' if ok else 'FAIL'} k={k}: bias {bias:.5f} <= "
            f"bound {bound:.5f} (ratio {bias / bound if bound else float('nan'):.3f})"
        )


def check_bound_grid(report: Report):
    """1 - kappa^(k-1) <= (k-1)(1 - kappa) on a grid."""
    kappas = np.linspace(0.0, 1.0, 1001)
    worst = 0.0
    for k in range(2, 51):
        gap = (k - 1) * (1 - kappas) - (1 - kappas ** (k - 1))
        worst = min(worst, float(gap.min()))
    ok = worst >= -1e-12
    report.failures += not ok
    print(f"\nstep 5 inequality on a grid (k = 2..50): {'ok' if ok else 'FAIL'}")


def main():
    rng = np.random.default_rng(SEED)
    report = Report()
    for config in CONFIGS:
        check_config(*config, report, rng)
    check_bound_grid(report)
    print(f"\n{report.failures} failed checks")
    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
