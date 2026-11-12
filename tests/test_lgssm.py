import numpy as np
import pytest
import torch

from anchor import linear
from anchor.envs.lgssm import (
    ar1,
    ar2,
    observation_autocovariance,
    observation_covariance,
    optimal_mse,
    sample_observations,
    windows,
)

HORIZON = 10
SYSTEMS = [ar1(0.9, 0.5), ar1(0.5, 1.0), ar2(0.9, np.pi / 6, 0.8)]


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_autocovariance_matches_simulation(system):
    y = sample_observations(system, 200_000, 5, np.random.default_rng(0))
    assert np.isfinite(y).all()
    empirical = np.array([np.mean(y[:, 0] * y[:, j]) for j in range(5)])
    exact = observation_autocovariance(system, 5)
    assert np.allclose(empirical, exact, atol=0.03 * exact[0])


def test_windows_layout():
    obs = np.arange(12.0).reshape(2, 6)
    out = windows(obs, window=2, horizon=1)
    assert out.shape == (8, 3)
    assert np.array_equal(out[0], [0, 1, 2]) and np.array_equal(out[4], [6, 7, 8])


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_population_errors_are_ordered(system):
    window = 2
    cov = observation_covariance(system, window + HORIZON)
    theta = linear.one_step_solution(cov, window)
    rollout = linear.rollout_mse(theta, cov, HORIZON)
    direct = linear.direct_mse(cov, window, HORIZON)
    best = optimal_mse(system, HORIZON)
    assert np.isclose(rollout[0], direct[0])
    assert np.all(best <= direct + 1e-9) and np.all(direct <= rollout + 1e-9)


@pytest.mark.parametrize("variant", ["C", "D"])
@pytest.mark.parametrize("window", [1, 4])
def test_markov_latent_fixed_point_is_one_step_optimum(variant, window):
    """T1: for a Markov latent the auxiliary term does not move the model."""
    system = ar1(0.9, 1.0)
    cov = observation_covariance(system, window + HORIZON)
    point = linear.fixed_point(variant, 10.0, cov, window)
    assert point["residual"] < 1e-10
    assert np.allclose(point["theta"], linear.one_step_solution(cov, window))


def test_bootstrap_improves_rollout_under_gap():
    """T2: a small D term lowers the summed rollout error when kappa < 1."""
    system = ar1(0.9, 0.5)
    cov = observation_covariance(system, 1 + HORIZON)
    base = linear.rollout_mse(linear.one_step_solution(cov, 1), cov, HORIZON)
    point = linear.fixed_point("D", 0.1, cov, 1)
    anchored = linear.rollout_mse(point["theta"], cov, HORIZON)
    assert anchored.sum() < base.sum()
    assert anchored[0] > base[0]


def test_critic_vanishes_without_gap_but_residual_does_not():
    """Lemma: the critic is zero for a Markov latent, the raw residual is not."""
    cov = observation_covariance(ar1(0.9, 1.0), 1 + HORIZON)
    critic, residual = linear.critic_and_residual(linear.one_step_solution(cov, 1), cov)
    assert critic < 1e-12 and residual > 0.1

    cov = observation_covariance(ar1(0.9, 0.5), 1 + HORIZON)
    critic, _ = linear.critic_and_residual(linear.one_step_solution(cov, 1), cov)
    assert critic > 1e-4


def test_population_loss_matches_sampled_loss():
    system = ar1(0.5, 0.8)
    window = 2
    cov = observation_covariance(system, window + HORIZON)
    theta = torch.tensor([0.1, 0.3], dtype=torch.float64)
    cov_t = torch.as_tensor(cov / cov[0, 0])
    obs = sample_observations(system, 20_000, 40, np.random.default_rng(1))
    data = torch.as_tensor(windows(obs, window, HORIZON) / np.sqrt(cov[0, 0]))
    pred = data[:, :window] @ theta
    sampled = ((pred - data[:, window]) ** 2).mean()
    assert torch.isclose(linear.one_step_loss(theta, cov_t), sampled, rtol=0.02)


def test_mixture_with_weight_on_true_future_is_multistep_loss():
    cov = observation_covariance(ar1(0.9, 0.5), 2 + HORIZON)
    cov_t = torch.as_tensor(cov / cov[0, 0])
    theta = torch.tensor([0.1, 0.4], dtype=torch.float64)
    one_hot = {k: np.eye(k - 1)[-1] for k in linear.AUX_HORIZONS}
    assert torch.isclose(
        linear.mixture_loss(theta, cov_t, one_hot), linear.multistep_loss(theta, cov_t)
    )


def test_simplex_minimizer():
    w = linear.simplex_minimizer(np.diag([1.0, 3.0]))
    assert np.allclose(w, [0.75, 0.25])
    # a dominated coordinate is dropped
    w = linear.simplex_minimizer(np.array([[1.0, 1.0], [1.0, 4.0]]))
    assert np.allclose(w, [1.0, 0.0])


def test_target_moments_bias_vanishes_for_markov_latent():
    """Lemma: at the one-step optimum of a Markov latent every target is unbiased."""
    cov = observation_covariance(ar1(0.9, 1.0), 1 + HORIZON)
    theta = linear.one_step_solution(cov, 1)
    for k in linear.AUX_HORIZONS:
        b, v = linear.target_moments(theta, cov, k)
        assert np.abs(b).max() < 1e-12
        # variance shrinks as the anchor moves toward t (fewer random steps)
        assert np.all(np.diff(np.diag(v)) > 0)


def test_bias_variance_weights_recover_monte_carlo_with_infinite_data():
    cov = observation_covariance(ar1(0.9, 0.5), 1 + HORIZON)
    point = linear.fixed_point("M*", 1.0, cov, 1, n_eff=1e12)
    for w in point["weights"].values():
        assert np.isclose(w[-1], 1.0)
    c_point = linear.fixed_point("C", 1.0, cov, 1)
    assert np.allclose(point["theta"], c_point["theta"], atol=1e-8)


def test_bias_variance_weights_move_to_bootstrap_with_little_data():
    cov = observation_covariance(ar1(0.9, 0.5), 1 + HORIZON)
    weights = linear.bias_variance_weights(linear.one_step_solution(cov, 1), cov, 10)
    assert all(w[-1] < 1.0 for w in weights.values())


def test_ablation_with_first_anchor_point_keeps_p1():
    cov = observation_covariance(ar1(0.9, 0.5), 1 + HORIZON)
    theta = linear.one_step_solution(cov, 1)
    weights = linear.bias_variance_weights(theta, cov, 10, first=1)
    for k, w in weights.items():
        assert len(w) == k and linear.first_point(k, w) == 1


def test_without_stop_gradient_markov_fixed_point_moves():
    """T1 needs the stop-gradient: without it the one-step optimum is not a fixed point."""
    cov = observation_covariance(ar1(0.5, 1.0), 1 + HORIZON)
    theta = linear.one_step_solution(cov, 1)
    with_sg = linear.fixed_point("D", 1.0, cov, 1)
    without = linear.fixed_point("D_nosg", 1.0, cov, 1)
    assert np.allclose(with_sg["theta"], theta)
    assert abs(without["theta"][0] - theta[0]) > 0.1
