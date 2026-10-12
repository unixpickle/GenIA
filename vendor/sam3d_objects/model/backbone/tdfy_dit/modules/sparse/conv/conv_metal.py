"""Checkpoint-compatible SAM sparse convolutions using GenIA's local operators."""
import math

import torch
from torch import nn

from genia.core.ops.sparse import neighbor_indices, submanifold_conv3d, triple


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, padding=None, bias=True, indice_key=None):
        super().__init__()
        if triple(stride) != (1, 1, 1) or padding is not None:
            raise NotImplementedError("Metal SparseConv3d currently implements submanifold convolution")
        self.kernel_size = triple(kernel_size)
        self.dilation = triple(dilation)
        self.stride = (1, 1, 1)
        self.padding = padding
        # Preserve the original conv.weight / conv.bias checkpoint names.
        self.conv = nn.Module()
        self.conv.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        self.conv.register_parameter("bias", nn.Parameter(torch.empty(out_channels)) if bias else None)
        self.conv.in_channels, self.conv.out_channels = in_channels, out_channels
        bound = 1 / math.sqrt(in_channels * math.prod(self.kernel_size))
        nn.init.uniform_(self.conv.weight, -bound, bound)
        if bias:
            nn.init.uniform_(self.conv.bias, -bound, bound)

    def forward(self, x):
        coords = x.coords
        key = ("metal_neighbors", coords.data_ptr(), coords._version, str(coords.device),
               self.kernel_size, self.dilation)
        neighbors = x.get_spatial_cache(key)
        if neighbors is None:
            neighbors = neighbor_indices(coords.to(torch.int32), self.kernel_size, self.dilation)
            x.register_spatial_cache(key, neighbors)
        features = x.feats.to(self.conv.weight.dtype)
        return x.replace(submanifold_conv3d(features, self.conv.weight, neighbors, self.conv.bias))


class SparseInverseConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("SAM decoders use SparseSubdivide; inverse convolutions are not yet ported")
