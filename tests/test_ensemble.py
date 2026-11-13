import torch

from anchor.ensemble import EnsembleAdam, EnsembleGRU, EnsembleLinear, EnsembleMLP


def test_linear_models_are_independent():
    layer = EnsembleLinear(3, 2, keys=[(0,), (1,)])
    x = torch.randn(2, 5, 3)
    out = layer(x)
    for m in range(2):
        expected = x[m] @ layer.weight[m] + layer.bias[m]
        assert torch.allclose(out[m], expected, atol=1e-6)


def test_initialization_depends_only_on_the_seed():
    a = EnsembleMLP([3, 8, 1], keys=[(5,), (7,)])
    b = EnsembleMLP([3, 8, 1], keys=[(7,)])
    assert torch.equal(a.layers[0].weight[1], b.layers[0].weight[0])


def test_no_two_tensors_share_a_random_stream():
    mlp = EnsembleMLP([4, 8, 8], keys=[(0, 1), (0, 2)])
    tensors = [t[m].flatten()[:8] for t in mlp.parameters() for m in range(2)]
    for a in range(len(tensors)):
        for b in range(a + 1, len(tensors)):
            ra, rb = (
                tensors[a] / tensors[a].abs().max(),
                tensors[b] / tensors[b].abs().max(),
            )
            assert not torch.allclose(ra, rb)


def test_gru_models_are_independent():
    gru = EnsembleGRU(1, 4, keys=[(0,), (1,)])
    x = torch.randn(2, 3, 6, 1)
    h = gru(x)
    x2 = x.clone()
    x2[1] += 1.0
    h2 = gru(x2)
    assert h.shape == (2, 3, 4)
    assert torch.allclose(h[0], h2[0]) and not torch.allclose(h[1], h2[1])


def test_adam_with_zero_rate_freezes_a_model():
    layer = EnsembleLinear(2, 1, keys=[(0,), (1,)])
    before = layer.weight.detach().clone()
    opt = EnsembleAdam(list(layer.parameters()), torch.tensor([1e-2, 0.0]))
    for _ in range(3):
        loss = layer(torch.randn(2, 4, 2)).pow(2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert not torch.equal(layer.weight[0], before[0])
    assert torch.equal(layer.weight[1], before[1])


def test_adam_clips_each_model_separately():
    p = torch.nn.Parameter(torch.zeros(2, 3))
    opt = EnsembleAdam([p], torch.tensor([1.0, 1.0]), clip=1.0)
    p.grad = torch.tensor([[100.0, 0, 0], [0.1, 0, 0]])
    opt.step()
    # the first step of Adam moves by about lr * sign(g) regardless of scale
    assert torch.allclose(p[:, 0].abs(), torch.ones(2), atol=1e-3)
