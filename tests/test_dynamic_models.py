"""Small full dynamic-model graphs and their input gradients on CPU/Metal."""
import copy

import pytest
import torch

from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
from actionmesh.model.temporal_denoiser import ActionMeshDenoiser
from triposg.models.autoencoders import TripoSGVAEModel
from triposg.models.transformers import TripoSGDiTModel

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True, params=[False, True], ids=["native", "chunked"])
def attention_chunks(request, monkeypatch):
    if request.param:
        from genia.core.ops.attention import scaled_dot_product_attention
        from actionmesh.model.utils import attention_processor as action_attention
        from triposg.models import attention_processor as tripo_attention

        def bounded(query, key, value, **kwargs):
            if query.device.type == "mps":
                kwargs["max_score_elements"] = 96
            return scaled_dot_product_attention(query, key, value, **kwargs)

        monkeypatch.setattr(action_attention, "scaled_dot_product_attention", bounded)
        monkeypatch.setattr(tripo_attention, "scaled_dot_product_attention", bounded)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.manual_seed(903)
    yield
    torch.set_num_threads(previous)


def compare(model, inputs, run):
    model.eval().requires_grad_(False)
    metal = copy.deepcopy(model).to('mps')
    cpu = [v.detach().requires_grad_(v.is_floating_point()) for v in inputs]
    mps = [v.detach().to('mps').requires_grad_(v.is_floating_point()) for v in inputs]
    expected = run(model, *cpu)
    actual = run(metal, *mps)
    torch.testing.assert_close(actual.cpu(), expected, atol=8e-6, rtol=2e-4)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (actual * weight.to('mps')).sum().backward()
    for c, m in zip(cpu, mps):
        if c.grad is not None:
            assert m.grad is not None and m.grad.isfinite().all()
            torch.testing.assert_close(m.grad.cpu(), c.grad, atol=3e-5, rtol=2e-3)
    return metal, [v.detach() for v in mps]


def test_temporal_denoiser_inflated_attention_and_frequency_cache():
    model = ActionMeshDenoiser(in_channels=8, width=32, num_layers=3,
        num_attention_heads=4, cross_attention_dim=16, inflated_layers=(0, 2))
    inputs = [torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 16),
              torch.tensor([[0., 1., 2.], [1., 3., 6.]]), torch.tensor([.3, .8])]
    def run(m, x, context, framestep, time):
        mask = x.new_tensor([[1, 0, 0], [0, 1, 0]])
        return m(x, context, framestep, time, mask=mask)[0]
    metal, values = compare(model, inputs, run)
    with torch.no_grad():
        first, cache = metal(*values)
        second, _ = metal(*values, freqs_rot=cache)
        torch.testing.assert_close(first, second)
        with torch.autocast('mps', dtype=torch.bfloat16):
            assert metal(*values)[0].isfinite().all()


def test_temporal_autoencoder_query_and_latent_gradients():
    model = ActionMeshAutoencoder(latent_channels=8, width=32, num_layers=2,
        num_attention_heads=4, verbose=False)
    inputs = [torch.randn(1, 3, 5, 8), torch.tensor([[0., 1., 3.]]), torch.tensor([0.]),
              torch.tensor([[.25, .75]]), torch.rand(1, 13, 6)]
    compare(model, inputs, lambda m, *args: m(*args))


def test_triposg_denoiser():
    model = TripoSGDiTModel(in_channels=8, width=32, num_layers=3,
                          num_attention_heads=4, cross_attention_dim=16)
    inputs = [torch.randn(2, 5, 8), torch.tensor([300., 800.]), torch.randn(2, 7, 16)]
    compare(model, inputs, lambda m, *args: m(*args).sample)


def test_triposg_vae_encode_decode_and_query_chunks():
    model = TripoSGVAEModel(latent_channels=8, width_encoder=32, width_decoder=32,
        num_layers_encoder=1, num_layers_decoder=1, num_attention_heads=4)
    inputs = [torch.randn(2, 43, 6), torch.rand(2, 17, 3)]
    def run(m, surface, queries):
        posterior = m.encode(surface, num_tokens=6, seed=30).latent_dist
        return m.decode(posterior.mode(), queries, num_chunks=5).sample
    metal, values = compare(model, inputs, run)
    with torch.no_grad():
        z = metal.encode(values[0], num_tokens=6, seed=30).latent_dist.mode()
        chunked = metal.decode(z, values[1], num_chunks=5).sample
        torch.testing.assert_close(chunked, metal.decode(z, values[1], num_chunks=100).sample)


@pytest.mark.parametrize('kind', ['action_denoiser', 'action_autoencoder', 'tripo_denoiser', 'tripo_vae'])
def test_production_precision_matches_fp32_model(kind):
    """ActionMesh uses bf16 autocast; its TripoSG stage stores fp16 weights."""
    if kind == 'action_denoiser':
        model = ActionMeshDenoiser(in_channels=8, width=32, num_layers=3,
            num_attention_heads=4, cross_attention_dim=16, inflated_layers=(0, 2))
        values = [torch.randn(1, 3, 5, 8), torch.randn(1, 3, 7, 16),
                  torch.tensor([[0., 1., 3.]]), torch.tensor([.4])]
        run = lambda m, *x: m(*x)[0]
    elif kind == 'action_autoencoder':
        model = ActionMeshAutoencoder(latent_channels=8, width=32, num_layers=2,
            num_attention_heads=4, verbose=False)
        values = [torch.randn(1, 3, 5, 8), torch.tensor([[0., 1., 3.]]),
                  torch.tensor([0.]), torch.tensor([[.25, .75]]), torch.rand(1, 13, 6)]
        run = lambda m, *x: m(*x)
    elif kind == 'tripo_denoiser':
        model = TripoSGDiTModel(in_channels=8, width=32, num_layers=3,
            num_attention_heads=4, cross_attention_dim=16)
        values = [torch.randn(1, 5, 8), torch.tensor([300.]), torch.randn(1, 7, 16)]
        run = lambda m, *x: m(*x).sample
    else:
        model = TripoSGVAEModel(latent_channels=8, width_encoder=32, width_decoder=32,
            num_layers_encoder=1, num_layers_decoder=1, num_attention_heads=4)
        values = [torch.randn(1, 43, 6), torch.rand(1, 17, 3)]
        def run(m, surface, queries):
            z = m.encode(surface, num_tokens=6, seed=30).latent_dist.mode()
            return m.decode(z, queries, num_chunks=5).sample

    model.eval().requires_grad_(False)
    metal = copy.deepcopy(model).to('mps')
    use_autocast = kind.startswith('action_')
    metal_values = [v.to('mps') for v in values]
    if not use_autocast:
        metal.half()
        metal_values = [v.half() for v in metal_values]
    with torch.no_grad():
        reference = run(model, *values)
        with torch.autocast('mps', dtype=torch.bfloat16, enabled=use_autocast):
            actual = run(metal, *metal_values).float().cpu()
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, reference, atol=.02, rtol=.05)
    assert (actual - reference).norm() / reference.norm().clamp_min(1e-6) < .025
