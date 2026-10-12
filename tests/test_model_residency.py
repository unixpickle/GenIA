from types import SimpleNamespace

import pytest
import torch

from genia.core.utils.model_residency import offload_sampling_models


def backbone(device):
    conditioner = torch.nn.Linear(4, 4).to(device)
    return SimpleNamespace(
        models=torch.nn.ModuleDict({
            "ss_generator": torch.nn.Linear(4, 4).to(device),
            "slat_generator": torch.nn.Linear(4, 4).to(device),
            "slat_decoder_gs": torch.nn.Linear(4, 4).to(device),
            "slat_decoder_mesh": torch.nn.Linear(4, 4).to(device),
        }),
        condition_embedders={"ss_condition_embedder": conditioner,
                              "slat_condition_embedder": conditioner},
    )


def test_cpu_residency_unchanged():
    pipeline = backbone("cpu")
    weight = pipeline.models["ss_generator"].weight
    with offload_sampling_models(pipeline, "cpu"):
        assert pipeline.models["ss_generator"].weight is weight
    assert pipeline.models["ss_generator"].weight is weight


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("fail", [False, True])
def test_metal_decoders_stay_resident_and_generators_restore(fail):
    pipeline = backbone("mps")
    sampling = [pipeline.models["ss_generator"], pipeline.models["slat_generator"],
                pipeline.condition_embedders["ss_condition_embedder"]]
    originals = [{key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                 for module in sampling]
    seen = False
    try:
        with offload_sampling_models(pipeline, "mps"):
            seen = True
            assert all(next(module.parameters()).device.type == "cpu" for module in sampling)
            assert pipeline.models["slat_decoder_gs"].weight.device.type == "mps"
            assert pipeline.models["slat_decoder_mesh"].weight.device.type == "mps"
            # Exercise decoder differentiation while the sampling networks are away.
            latent = torch.randn(3, 4, device="mps", requires_grad=True)
            pipeline.models["slat_decoder_gs"](latent).square().sum().backward()
            assert torch.isfinite(latent.grad).all()
            if fail:
                raise RuntimeError("simulated fitting failure")
    except RuntimeError as exc:
        assert fail and str(exc) == "simulated fitting failure"
    assert seen
    for module, original in zip(sampling, originals):
        assert next(module.parameters()).device.type == "mps"
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value.cpu(), original[key], atol=0, rtol=0)
