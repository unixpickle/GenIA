import pytest
import torch

from genia.core.device import empty_cache, get_device, synchronize


def test_cpu_override(monkeypatch):
    monkeypatch.setenv("GENIA_DEVICE", "cpu")
    assert get_device() == torch.device("cpu")
    synchronize()
    empty_cache()


def test_available_default(monkeypatch):
    monkeypatch.delenv("GENIA_DEVICE", raising=False)
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert get_device().type == expected
    synchronize()


def test_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        get_device("cuda")
