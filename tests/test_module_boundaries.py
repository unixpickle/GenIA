"""Import compatibility and initialization boundaries after splitting large modules."""
import importlib
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("legacy,owner,name", [
    ("genia.core.finetuning", "genia.core.finetune.strategies", "finetune_canonical_tokens"),
    ("genia.core.finetuning", "genia.core.finetune.strategies", "finetune_perframe_tokens"),
    ("genia.core.finetuning", "genia.core.finetune.decoder", "_prepare_finetuning_decoder"),
    ("genia.core.finetuning", "genia.core.finetune.losses", "compute_total_loss"),
    ("genia.core.utils.refinement", "genia.core.utils.pose_refinement.sequence", "refine_poses_for_sequence"),
    ("genia.core.utils.refinement", "genia.core.utils.pose_refinement.decoding", "differentiable_pose_decode"),
    ("genia.core.utils.refinement", "genia.core.utils.pose_refinement.rendering", "apply_pose_to_gaussian"),
    ("genia.core.utils.render_final_results", "genia.core.utils.final_rendering.run", "FinalRun"),
    ("genia.core.utils.render_final_results", "genia.core.utils.final_rendering.geometry", "load_gaussian_ply"),
    ("genia.core.utils.render_final_results", "genia.core.utils.final_rendering.cli", "process_run"),
])
def test_existing_imports_resolve_to_implementation(legacy, owner, name):
    assert getattr(importlib.import_module(legacy), name) is getattr(importlib.import_module(owner), name)


@pytest.mark.parametrize("module", [
    "genia.core.utils.render_final_results", "genia.core.utils.final_rendering.cli",
])
def test_renderer_registry_and_cpu_cli_do_not_load_gpu_stack(module):
    # A fresh interpreter detects eager torch/model imports regardless of test order.
    code = f"""
import importlib, json, sys
r = importlib.import_module({module!r})
r._lazy_imports(gpu=False)
args = r.build_parser().parse_args(['--list'])
assert args.list
assert r.build_parser().description == 'Render visualizations from a pipeline ``final/`` directory.'
from genia.core.utils.final_rendering.builtins import register_builtin_renderers
register_builtin_renderers()
assert 'torch' not in sys.modules
assert 'sam3d_objects' not in sys.modules
print(json.dumps(list(r.RENDERERS)))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [
        "train_views", "track_overlay", "synth_nvs", "track_overlay_nvs",
        "track_overlay_test", "orbit", "world_space", "contact_sheet",
    ]


def test_frame_sampler_is_shared_and_retains_epoch_order(monkeypatch):
    from genia.core import finetuning
    from genia.core.utils import frame_sampling, refinement
    assert finetuning.EpochFrameSampler is refinement.EpochFrameSampler is frame_sampling.EpochFrameSampler
    monkeypatch.setattr(frame_sampling.random, "shuffle", lambda values: values.reverse())
    sampler = frame_sampling.EpochFrameSampler([0, 1, 2])
    assert sampler.sample(5) == [2, 1, 0, 2, 1]
    assert sampler.sample(2) == [0, 2]


def test_renderer_plugins_share_one_registry(monkeypatch):
    from genia.core.utils import render_final_results as legacy
    from genia.core.utils.final_rendering import cli, registry
    assert legacy.RENDERERS is cli.RENDERERS is registry.RENDERERS
    monkeypatch.setitem(registry.RENDERERS, "test_plugin", None)
    @legacy.renderer("test_plugin", "Test registration", needs_gpu=False)
    def plugin(run, ctx):
        pass
    assert registry.RENDERERS["test_plugin"].fn is plugin
    assert not registry.RENDERERS["test_plugin"].needs_gpu
