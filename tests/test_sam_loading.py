import os
os.environ['LIDRA_SKIP_INIT']='1'
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline


@pytest.mark.parametrize('device',['cpu']+(['mps'] if torch.backends.mps.is_available() else []))
@pytest.mark.parametrize('format',['pt','legacy_pt','safetensors'])
def test_pipeline_checkpoint_loading(tmp_path,device,format):
    expected=torch.nn.Linear(3,2)
    path=tmp_path/('test.'+format)
    if format in ('pt', 'legacy_pt'):
        torch.save({'state_dict':expected.state_dict()},path,
                   _use_new_zipfile_serialization=format == 'pt')
    else: save_file(expected.state_dict(),path)
    pipeline=InferencePipeline.__new__(InferencePipeline)
    pipeline.device=torch.device(device)
    config=OmegaConf.create({'_target_':'torch.nn.Linear','in_features':3,'out_features':2})
    model=pipeline.instantiate_and_load_from_pretrained(config,str(path))
    assert next(model.parameters()).device.type==device and not model.training
    for key,value in expected.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key].cpu(),value)


def test_complete_sam_checkpoint_skips_dino_pretraining(tmp_path, monkeypatch):
    from sam3d_objects.model.backbone.dit.embedder import dino
    from hydra.utils import instantiate

    def local_backbone(*, pretrained=True, **kwargs):
        assert pretrained is False, "Complete checkpoint must not fetch DINO pretraining weights"
        model = torch.nn.Linear(3, 4)
        model.embed_dim = 4
        model.patch_embed = torch.nn.Identity()
        return model

    monkeypatch.setattr(dino.local_dinov2, 'dinov2_vits14', local_backbone)
    config = OmegaConf.create({
        '_target_': 'sam3d_objects.model.backbone.dit.embedder.dino.Dino',
        'dino_model': 'dinov2_vits14', 'backbone_kwargs': {'pretrained': False},
    })
    expected = instantiate(config)
    path = tmp_path / 'complete.ckpt'
    torch.save({'state_dict': expected.state_dict()}, path)
    config.backbone_kwargs.pretrained = True
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.device = torch.device('cpu')
    loaded = pipeline.instantiate_and_load_from_pretrained(config, str(path))
    assert config.backbone_kwargs.pretrained is True
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
