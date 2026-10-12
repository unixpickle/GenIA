import pytest
import torch

from genia.core import shape_inversion


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_mps_inversion_oom_reduces_chunks_and_preserves_order(monkeypatch):
    batches = []
    def invert(decoder, occupancy, **kwargs):
        batches.append(len(occupancy))
        if len(occupancy) > 1:
            raise RuntimeError('MPS backend out of memory (simulated allocation failure)')
        z = occupancy.mean((1, 2, 3, 4))[:, None, None, None, None].expand(-1, 8, 16, 16, 16).clone()
        return z, []
    monkeypatch.setattr(shape_inversion, 'invert_decoder', invert)
    occupancy = torch.zeros(3, 8, device='mps')
    for i in range(3):
        occupancy[i, :i+1] = 1
    occupancy = occupancy.reshape(3, 1, 2, 2, 2)
    decoder = lambda z: z[:, :1, :2, :2, :2]
    tokens, ious = shape_inversion.occupancies_to_shape_tokens(occupancy, decoder, chunk_size=3)
    assert batches == [3, 1, 1, 1]
    torch.testing.assert_close(tokens[:, 0, 0].cpu(), torch.tensor([.125, .25, .375]))
    assert ious == [.125, .25, .375]


def test_non_memory_runtime_errors_are_not_retried(monkeypatch):
    calls = []
    def invert(*args, **kwargs):
        calls.append(True)
        raise RuntimeError('shape mismatch')
    monkeypatch.setattr(shape_inversion, 'invert_decoder', invert)
    with pytest.raises(RuntimeError, match='shape mismatch'):
        shape_inversion.occupancies_to_shape_tokens(torch.ones(3, 1, 2, 2, 2), None, chunk_size=3)
    assert len(calls) == 1
