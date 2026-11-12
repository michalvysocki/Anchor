"""
Shared tools of the numerical checks of the theory: stationary AR(1) chains
observed with noise, regression slopes with standard errors, and a report
that counts checks and failures. See README.md for the map of statements.

    x_{t+1} = a x_t + w_t,  w_t ~ N(0, sigma2)
    y_t     = x_t + v_t,    v_t ~ N(0, r)

A Monte Carlo check passes when the estimate is within N_SE standard errors
of the closed form.
"""

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


def parameters(a: float, sigma2: float, r_frac: float) -> tuple[float, float]:
    """Observation-noise variance r and signal fraction kappa of a configuration."""
    s = sigma2 / (1 - a**2)
    r = r_frac * s

    return r, s / (s + r)


def simulate(a: float, sigma2: float, r: float, rng: np.random.Generator):
    """Stationary chains of states and observations, each of size (N_CHAINS, K_MAX + 1)."""
    s = sigma2 / (1 - a**2)
    x = np.empty((N_CHAINS, K_MAX + 1))
    x[:, 0] = rng.normal(0.0, np.sqrt(s), N_CHAINS)
    for t in range(K_MAX):
        x[:, t + 1] = a * x[:, t] + rng.normal(0.0, np.sqrt(sigma2), N_CHAINS)
    y = x + rng.normal(0.0, np.sqrt(r), x.shape)

    return x, y


def chains(config_index: int):
    """Configuration parameters and its chains, from a generator fixed per configuration."""
    a, sigma2, r_frac = CONFIGS[config_index]
    r, kappa = parameters(a, sigma2, r_frac)
    x, y = simulate(a, sigma2, r, np.random.default_rng([SEED, config_index]))
    print(f"\na = {a}, sigma2 = {sigma2}, r = {r:.4f}, kappa = {kappa:.4f}")

    return a, sigma2, kappa, x, y


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
    """Counts checks and failures and prints one line per check."""

    def __init__(self, title: str):
        self.checks = 0
        self.failures = 0
        print(f"== {title}")

    def check(self, name: str, estimate: tuple[float, float], theory: float):
        """Monte Carlo estimate (value, standard error) against a closed form."""
        value, se = estimate
        self.record(
            name,
            abs(value - theory) <= N_SE * se + 1e-12,
            f"sim {value:+.5f} ± {se:.5f}   theory {theory:+.5f}",
        )

    def record(self, name: str, ok: bool, detail: str = ""):
        """Any other check."""
        self.checks += 1
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<38} {detail}")

    def summary(self) -> int:
        print(f"{self.checks - self.failures}/{self.checks} checks passed\n")
        return self.failures
