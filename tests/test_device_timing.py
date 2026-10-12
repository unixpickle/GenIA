import pytest
import torch

from genia.core.utils.timing import PipelineTimer
from genia.core.actionmesh_video import RunTimer


@pytest.mark.parametrize('device', ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []))
def test_timing_on_selected_device_and_unavailable_peaks(monkeypatch, device):
    monkeypatch.setenv('GENIA_DEVICE', device)
    x = torch.randn(512, 512, device=device)
    timer = PipelineTimer()
    timer.start_block('inference')
    with timer.exclude():
        value = x @ x
    value = value @ x
    timer.end_block('inference')
    report = timer.to_dict()
    block = report['blocks']['inference']
    assert block['total_seconds'] >= block['excluded_seconds'] >= 0
    if device == 'mps':
        assert report['peak_alloc_mb'] is None and block['peak_reserved_mb'] is None
        assert block['baseline_alloc_mb'] >= x.numel()*x.element_size()/1024**2
    else:
        assert report['peak_alloc_mb'] == 0
    dynamic = RunTimer()
    with dynamic.phase('inference'):
        value = value @ x
    report = dynamic.to_dict()
    assert dynamic.total_core_seconds > 0
    if device == 'mps':
        assert report['peak_alloc_mb'] is None and report['peak_reserved_mb'] is None
    assert value.isfinite().all()
