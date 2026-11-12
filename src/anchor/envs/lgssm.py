"""
Scalar linear-Gaussian processes observed with noise: AR(1) and AR(2)
states x_t, observations y_t = x_t + v_t. Everything is stationary and
zero-mean, so the whole joint distribution of observations is given by
their autocovariance, which is computed exactly.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ARSystem:
    """
    AR(p) state with p <= 2, observed with noise.

    Attributes:
        name (str): short identifier used in run names
        phi (tuple[float, ...]): AR coefficients (phi_1,) or (phi_1, phi_2)
        sigma2 (float): variance of the process noise
        kappa (float): signal fraction s / (s + r), with s the stationary
            state variance and r the observation-noise variance; 1 means
            the state is observed exactly
    """

    name: str
    phi: tuple[float, ...]
    sigma2: float
    kappa: float

    @property
    def state_variance(self) -> float:
        """Stationary variance s of the state."""
        return float(state_autocovariance(self, 1)[0])

    @property
    def noise_variance(self) -> float:
        """Observation-noise variance r = s (1 - kappa) / kappa."""
        return self.state_variance * (1 - self.kappa) / self.kappa


def ar1(a: float, kappa: float, sigma2: float = 1.0) -> ARSystem:
    return ARSystem(f"ar1_a{a}_k{kappa}", (a,), sigma2, kappa)


def ar2(radius: float, angle: float, kappa: float, sigma2: float = 1.0) -> ARSystem:
    """AR(2) with complex roots radius * exp(+-i angle): a damped oscillation."""
    phi = (2 * radius * np.cos(angle), -(radius**2))
    return ARSystem(f"ar2_r{radius}_k{kappa}", phi, sigma2, kappa)


def state_autocovariance(system: ARSystem, n_lags: int) -> np.ndarray:
    """gamma_x(0..n_lags-1) of the stationary state."""
    phi, sigma2 = system.phi, system.sigma2
    gamma = np.zeros(max(n_lags, 2))
    if len(phi) == 1:
        (a,) = phi
        assert abs(a) < 1, "AR(1) must be stationary"
        gamma[0] = sigma2 / (1 - a**2)
        gamma[1] = a * gamma[0]
    else:
        p1, p2 = phi
        roots = np.roots([1, -p1, -p2])
        assert np.all(np.abs(roots) < 1), "AR(2) must be stationary"
        gamma[0] = sigma2 * (1 - p2) / ((1 + p2) * ((1 - p2) ** 2 - p1**2))
        gamma[1] = p1 * gamma[0] / (1 - p2)
    for j in range(2, len(gamma)):
        gamma[j] = sum(c * gamma[j - 1 - i] for i, c in enumerate(phi))

    return gamma[:n_lags]


def observation_autocovariance(system: ARSystem, n_lags: int) -> np.ndarray:
    """gamma_y(0..n_lags-1): the state's plus noise at lag 0."""
    gamma = state_autocovariance(system, n_lags).copy()
    gamma[0] += system.noise_variance

    return gamma


def observation_covariance(system: ARSystem, length: int) -> np.ndarray:
    """Covariance matrix of (y_1, ..., y_length), a symmetric Toeplitz matrix."""
    gamma = observation_autocovariance(system, length)
    lags = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))

    return gamma[lags]


def sample_observations(
    system: ARSystem, n_traj: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Stationary observation trajectories of size (n_traj, length), drawn
    exactly from the joint Gaussian (no burn-in needed).
    """

    chol = np.linalg.cholesky(observation_covariance(system, length))
    # matmul through Accelerate on macOS can raise spurious floating-point
    # warnings on finite inputs; the result is checked instead
    with np.errstate(all="ignore"):
        out = rng.standard_normal((n_traj, length)) @ chol.T
    assert np.isfinite(out).all()

    return out


def windows(observations: np.ndarray, window: int, horizon: int) -> np.ndarray:
    """
    All windows of window + horizon consecutive observations, size
    (n, window + horizon): the first `window` entries are the latent z_t,
    the rest are y_{t+1..t+horizon}.
    """

    length = observations.shape[1]
    span = window + horizon
    idx = np.arange(length - span + 1)[:, None] + np.arange(span)

    return observations[:, idx].reshape(-1, span)


def optimal_mse(system: ARSystem, horizon: int, history: int = 400) -> np.ndarray:
    """
    MSE of the best prediction of y_{t+k}, k = 1..horizon, from a long
    history of observations (the steady-state Kalman predictor), in units
    of Var(y).
    """

    cov = observation_covariance(system, history + horizon)
    past = cov[:history, :history]
    out = []
    for k in range(1, horizon + 1):
        cross = cov[:history, history + k - 1]
        out.append(cov[0, 0] - cross @ np.linalg.solve(past, cross))

    return np.array(out) / cov[0, 0]
