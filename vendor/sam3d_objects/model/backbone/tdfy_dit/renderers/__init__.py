# Copyright (c) Meta Platforms, Inc. and affiliates.
from .gaussian_render import GaussianRenderer


def __getattr__(name):
    # Gaussian rendering should not import the unrelated octree CUDA stack.
    if name == "OctreeRenderer":
        from .octree_renderer import OctreeRenderer
        return OctreeRenderer
    if name == "MeshRenderer":
        try:
            from .mesh_renderer import MeshRenderer
            return MeshRenderer
        except ImportError:
            return None
    raise AttributeError(name)
