"""Dimension-wise unique using device-native stable sorts on MPS."""
import torch
import math


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if input.device.type != 'mps' or dim is None:
        return torch.unique(input, sorted=sorted, return_inverse=return_inverse,
                            return_counts=return_counts, dim=dim)
    if input.requires_grad:
        raise ValueError('Dimension-wise unique does not support gradients')
    values = input.movedim(dim, 0)
    n = values.shape[0]
    if any(size == 0 for size in values.shape[1:]):
        raise RuntimeError('unique cannot process zero-sized unselected dimensions')
    rows = values.reshape(n, math.prod(values.shape[1:]))
    order = torch.arange(n, device=input.device)
    # Sort from least to most significant column: lexicographic ordering without
    # packing coordinates into a potentially overflowing integer hash.
    for column in range(rows.shape[1]-1, -1, -1):
        order = order[rows[order,column].argsort(stable=True)]
    ordered = values[order]
    different = torch.ones(n, dtype=torch.bool, device=input.device)
    if n>1:
        different[1:] = (rows[order[1:]] != rows[order[:-1]]).any(1)
    starts = different.nonzero().flatten()
    output = ordered[different].movedim(0, dim)
    result = [output]
    if return_inverse:
        inverse = torch.empty(n, dtype=torch.long, device=input.device)
        inverse[order] = different.long().cumsum(0)-1
        result.append(inverse)
    if return_counts:
        ends = torch.cat([starts[1:], starts.new_tensor([n])])
        result.append(ends-starts if n else starts)
    return tuple(result) if len(result)>1 else output
