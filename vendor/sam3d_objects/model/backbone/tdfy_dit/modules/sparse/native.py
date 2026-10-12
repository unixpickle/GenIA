"""Sparse tensor storage independent of CUDA extension packages."""
import torch


class SparseTensorData:
    def __init__(self, features, indices):
        self.features = features
        self.indices = indices

    def dense(self):
        if len(self.indices) == 0:
            raise ValueError("cannot infer a dense spatial extent from an empty sparse grid")
        shape = tuple((self.indices.max(0).values + 1).cpu().tolist())
        out = self.features.new_zeros((*shape, *self.features.shape[1:]))
        out[tuple(self.indices.long().T)] = self.features
        return out.permute(0, 4, 1, 2, 3)
