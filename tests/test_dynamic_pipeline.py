"""Offline checkpoint loading, denoising integration and adaptive extraction."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model as ReferenceDINO

from genia.core.models.hf_dinov2 import Dinov2Model
from actionmesh.model.utils.pointcloud_sampling import sample_pc, sample_from_indices
from triposg.inference_utils import flash_extract_geometry, hierarchical_extract_geometry
from triposg.models.autoencoders import TripoSGVAEModel
from triposg.models.transformers import TripoSGDiTModel
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_checkpoint_local_encoder_and_denoising(tmp_path, monkeypatch, device):
    import triposg.pipelines.pipeline_triposg as pipeline_module
    torch.manual_seed(653)
    reference = TripoSGPipeline(
        TripoSGVAEModel(latent_channels=8, width_encoder=32, width_decoder=32,
                       num_layers_encoder=1, num_layers_decoder=1, num_attention_heads=4),
        TripoSGDiTModel(in_channels=8, width=32, num_layers=3, num_attention_heads=4, cross_attention_dim=32),
        RectifiedFlowScheduler(),
        ReferenceDINO(Dinov2Config(hidden_size=32, num_attention_heads=4, num_hidden_layers=1,
                                  image_size=28, patch_size=7)),
        BitImageProcessor(size={'shortest_edge': 28}, crop_size={'height': 28, 'width': 28}),
    )
    reference.save_pretrained(tmp_path)
    actual = TripoSGPipeline.from_pretrained(tmp_path, local_files_only=True).to(device)
    assert type(actual.image_encoder_dinov2) is Dinov2Model
    outputs = []
    def extract(geometric_func, target, **kwargs):
        queries = torch.linspace(-1, 1, 39, device=target).reshape(1, 13, 3)
        outputs.append(geometric_func(queries).cpu())
        return [(np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]]), np.array([[0, 1, 2]]))]
    monkeypatch.setattr(pipeline_module, 'hierarchical_extract_geometry', extract)
    image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (39, 57, 3), dtype=np.uint8))
    latents = torch.randn(1, 6, 8)
    trajectories = []
    for pipe in (reference, actual):
        pipe.vae.eval(); pipe.transformer.eval(); pipe.image_encoder_dinov2.eval()
        pipe.set_progress_bar_config(disable=True)
        path = []
        def callback(_, i, t, values):
            path.append(values['latents'].cpu().clone())
            return values
        result = pipe(image, num_inference_steps=3, num_tokens=6, latents=latents.clone(),
                      use_flash_decoder=False, callback_on_step_end=callback)
        assert len(result.meshes) == 1
        trajectories.append(path)
    for e, a in zip(*trajectories):
        torch.testing.assert_close(a, e, atol=4e-6, rtol=3e-5)
    torch.testing.assert_close(outputs[1], outputs[0], atol=4e-6, rtol=3e-5)


def test_actionmesh_cache_paths_packaged_configs_and_lazy_device(tmp_path, monkeypatch):
    import actionmesh.pipeline as module
    from genia.core.actionmesh_video import build_pipeline
    calls = []
    def download(repo):
        calls.append(repo)
        return str(tmp_path/repo)
    monkeypatch.setattr(module, 'download_if_missing', download)
    monkeypatch.setenv('GENIA_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu')
    pipeline = build_pipeline(fast=True, low_ram=True, dtype='bfloat16')
    assert pipeline.device.type == ('mps' if torch.backends.mps.is_available() else 'cpu')
    assert pipeline._lazy_loading and pipeline.image_to_3d_pipe is None
    assert pipeline.cfg.model.image_encoder.pretrained_dino_model == str(tmp_path/'facebook/dinov2-large')
    assert pipeline._triposg_weights_dir == str(tmp_path/'VAST-AI/TripoSG')
    assert calls == ['VAST-AI/TripoSG', 'facebook/dinov2-large', 'briaai/RMBG-1.4', 'facebook/ActionMesh']


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('cap', [None, 13, 6, 1000])
def test_chunked_fps_indices_refer_to_original_cloud(device, cap):
    torch.manual_seed(56)
    points = torch.randn(2, 23, 6, device=device).requires_grad_()
    sampled, indices = sample_pc(points, 6, 'fps', fps_max_points=cap,
                                 fps_random=False, fps_chunks=3)
    torch.testing.assert_close(sample_from_indices(points, indices), sampled)
    assert sampled.shape == (2, 6, 6)
    sampled.sum().backward()
    expected = torch.zeros_like(points)
    expected.scatter_add_(1, indices[..., None].expand(-1, -1, 6), torch.ones_like(sampled))
    torch.testing.assert_close(points.grad, expected)


class AnalyticVAE:
    decoder = SimpleNamespace(set_topk=lambda _: None)

    def decode(self, latents, queries):
        queries = queries.float()
        xyz = queries*queries.new_tensor([1., .8, 1.1])+queries.new_tensor([.031, .019, -.05])
        return SimpleNamespace(sample=(.6-xyz.norm(dim=-1, keepdim=True))*10)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('flash', [False, True])
def test_adaptive_geometry_extraction(device, flash):
    model = AnalyticVAE()
    def run(target):
        if flash:
            return flash_extract_geometry(torch.zeros(1, 2, 3, device=target), model,
                octree_depth=5, min_resolution=7, mini_grid_num=2, num_chunks=1000)[0]
        return hierarchical_extract_geometry(lambda q: model.decode(None, q).sample,
                torch.device(target), dense_octree_depth=3, hierarchical_octree_depth=4)[0]
    expected_v, expected_f = run('cpu')
    vertices, faces = run(device)
    assert len(vertices) > 100 and len(faces) > 100
    assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
    np.testing.assert_allclose(vertices, expected_v, atol=3e-5, rtol=2e-4)
    np.testing.assert_array_equal(faces, expected_f)
