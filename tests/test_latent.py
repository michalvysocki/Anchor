import pytest
import torch

from anchor.latent import Sizes, WorldModels, base, flagged

SIZES = Sizes(latent=4, hidden=8, decoder_hidden=8, critic_hidden=8)
HORIZONS = (2, 3, 4)


def batch(n_models=2, n=16, past=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n_models, n, past + SIZES.eval_horizon, generator=gen)


def model(variant, window=2, seeds=(0, 1)):
    return WorldModels(variant, window, HORIZONS, list(seeds), SIZES)


def test_variant_names():
    assert base("DB+F_oracle") == "DB" and flagged("DB+F_oracle")
    assert base("C_wide") == "C" and not flagged("C_wide")
    assert base("M*+F") == "M*"


def test_flag_separates_the_first_step():
    u = torch.randn(2, 5, SIZES.latent)
    m = model("C+F")
    assert not torch.allclose(m.step(u, True), m.step(u, False))
    plain = model("C")
    assert torch.equal(plain.step(u, True), plain.step(u, False))


@pytest.mark.parametrize("variant", ["D+F", "DB+F", "TDB+F", "M*+F"])
def test_targets_are_under_stop_gradient(variant):
    m = model(variant)
    w = batch()
    z = m.encode(w, 0)
    for t in m.targets(w, z, m.boot_for(w, z)).values():
        assert not t.requires_grad


@pytest.mark.parametrize("variant", ["DB+F", "TDB+F", "M*+F"])
def test_model_loss_does_not_train_the_critic(variant):
    m = model(variant)
    loss, _ = m.losses(batch(), torch.ones(2))
    grads = torch.autograd.grad(loss.sum(), m.critic_parameters(), allow_unused=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)


@pytest.mark.parametrize("variant", ["DB+F", "M*+F"])
def test_critic_loss_does_not_train_the_model(variant):
    m = model(variant)
    _, closs = m.losses(batch(), torch.ones(2))
    grads = torch.autograd.grad(closs.sum(), m.model_parameters(), allow_unused=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)


def test_bootstraps_equal_separate_rollouts():
    m = model("D+F")
    w = batch()
    boot = m.bootstraps(w)
    for (k, p), t in boot.items():
        with torch.no_grad():
            direct = m.decode(m.rollout(m.encode(w, p), k - p)[-1])
        assert torch.allclose(t, direct)


def test_mstar_weights_lie_on_the_simplex():
    m = model("M*+F")
    m.update_weights(batch(n=64), n_windows=1000)
    for w in m.weights.values():
        assert torch.all(w >= 0)
        assert torch.allclose(w.sum(-1), torch.ones(w.shape[0]), atol=1e-6)


def test_initialization_is_paired_across_variants():
    a, b = model("C"), model("DB+F")
    assert torch.equal(a.enc.layers[0].weight, b.enc.layers[0].weight)
    assert torch.equal(a.dec.layers[0].weight, b.dec.layers[0].weight)


def test_gru_encoder_and_windows():
    m = WorldModels("C+F", None, HORIZONS, [0], SIZES)
    w = batch(n_models=1, past=SIZES.gru_history)
    loss, _ = m.losses(w, torch.ones(1))
    assert torch.isfinite(loss).all()


def test_target_of_y_is_the_observation_k_steps_ahead():
    m = model("C")
    w = torch.arange(2 + SIZES.eval_horizon, dtype=torch.float32)[None, None]
    # window of 2 past values (indices 0, 1); y_{t+k} is at index 1 + k
    assert [m.y(w, k).item() for k in (1, 3)] == [2.0, 4.0]
    assert m.past == 2


def test_one_step_loss_includes_latent_consistency():
    m = model("onestep")
    w = batch()
    loss, _ = m.losses(w, torch.zeros(2))
    with torch.no_grad():
        u1 = m.step(m.encode(w, 0), True)
        pred = (m.decode(u1) - m.y(w, 1)).pow(2).mean(-1)
        cons = (u1 - m.encode(w, 1)).pow(2).mean((-2, -1))
    assert torch.allclose(loss, pred + cons)
    # the target encoding is under stop-gradient
    grads = torch.autograd.grad(loss.sum(), m.enc.parameters(), allow_unused=True)
    assert any(g is not None for g in grads)
