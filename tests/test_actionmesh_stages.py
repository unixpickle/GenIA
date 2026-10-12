"""Temporal pipeline stages across two autoregressive windows on CPU/Metal."""
import copy

import numpy as np
import pytest
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image

from actionmesh.io.video_input import ActionMeshInput
from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
from actionmesh.model.temporal_denoiser import ActionMeshDenoiser
from actionmesh.model.utils.storage import LatentBank, MeshBank
from actionmesh.pipeline import ActionMeshPipeline
from actionmesh.scheduler.guidance import ClassifierFreeGuidance
from actionmesh.scheduler.scheduler import SchedulerFlow


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
@pytest.mark.parametrize('mode,split', [('direct', False), ('residual', True)])
def test_temporal_latent_and_mesh_stages(mode, split):
    torch.manual_seed(786)
    denoiser = ActionMeshDenoiser(in_channels=8, width=32, num_layers=3,
        num_attention_heads=4, cross_attention_dim=16, inflated_layers=(0, 2)).eval()
    vae = ActionMeshAutoencoder(latent_channels=8, width=32, num_layers=2,
        num_attention_heads=4, verbose=False, prediction_mode=mode).eval()
    context = torch.randn(31, 7, 16)
    anchor = torch.randn(1, 5, 8)
    mesh = trimesh.creation.icosphere(subdivisions=0, radius=.3)
    inputs = ActionMeshInput([Image.new('RGB', (8, 8))]*31, torch.arange(31, dtype=torch.float32))
    results = []
    for device in ('cpu', 'mps'):
        pipeline = ActionMeshPipeline.__new__(ActionMeshPipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline._target_device = torch.device(device)
        pipeline._denoiser_latent_shape = (5, 8)
        pipeline.cfg = OmegaConf.create({'anchor_idx': 0, 'sliding_window_denoiser': 15,
            'sliding_window_autoencoder': 15, 'subsampling_level': 1,
            'model': {'temporal_3D_denoiser': {'temporal_context_size': 16},
                      'temporal_3D_vae': {'temporal_context_size': 16}}})
        pipeline.temporal_3D_denoiser = copy.deepcopy(denoiser).to(device)
        pipeline.temporal_3D_vae = copy.deepcopy(vae).to(device)
        pipeline.scheduler = SchedulerFlow(num_inference_steps=3, is_additive=True, split_cfg_batch=split)
        pipeline.cf_guidance = ClassifierFreeGuidance(guidance_scales=[1.5, 2.])
        # Feed identical noise to both backends; their RNG sequences differ.
        native_noise = pipeline.scheduler.get_noise
        def controlled_noise(*, generator, device, **kwargs):
            generator = torch.Generator().manual_seed(generator.initial_seed())
            return native_noise(generator=generator, device='cpu', **kwargs).to(device)
        pipeline.scheduler.get_noise = controlled_noise
        bank = LatentBank(empty_dims=(5, 8))
        bank.update(inputs.timesteps[:1], anchor.to(device))
        meshes = MeshBank()
        meshes.update(inputs.timesteps[:1], [mesh.copy()])
        callbacks = []
        with torch.no_grad():
            bank = pipeline.generate_3d_latents(inputs, context.to(device), bank,
                step_callback=lambda *args: callbacks.append(args))
            meshes = pipeline.generate_mesh_animation(bank, meshes)
        latent, times = bank.get_ordered()
        torch.testing.assert_close(latent[0].cpu(), anchor[0], atol=0, rtol=0)
        torch.testing.assert_close(times.cpu(), inputs.timesteps)
        assert len(callbacks) == 6 and callbacks[-1][2:] == (1, 2)
        meshes, times = meshes.get_ordered('cpu')
        torch.testing.assert_close(times, inputs.timesteps)
        for m in meshes:
            np.testing.assert_array_equal(m.faces, mesh.faces)
            assert np.isfinite(m.vertices).all()
        results.append((latent.cpu(), np.stack([m.vertices for m in meshes])))
    torch.testing.assert_close(results[1][0], results[0][0], atol=5e-5, rtol=5e-4)
    np.testing.assert_allclose(results[1][1], results[0][1], atol=8e-6, rtol=3e-4)
