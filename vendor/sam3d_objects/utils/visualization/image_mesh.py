# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections import namedtuple
from typing import Tuple, Optional, Union
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex

try:
    # utils3d 0.0.2 (sam3d-objects env)
    from utils3d.numpy import (
        depth_edge,
        normals_edge,
        points_to_normals,
        image_uv,
        image_mesh,
    )
except ImportError:
    # utils3d >= 1.x (pinned by MoGe-2) renamed these maps
    # helpers. Provide the 0.0.2 names as thin adapters. Only
    # SceneVisualizer.object_pointcloud is on the pipeline's critical path and it
    # uses none of these; the adapters keep the depth->mesh viz importable/working.
    from utils3d.numpy import (
        depth_map_edge as depth_edge,
        normal_map_edge as normals_edge,
        point_map_to_normal_map as _point_map_to_normal_map,
        uv_map as _uv_map,
        build_mesh_from_map as _build_mesh_from_map,
    )

    def points_to_normals(point, mask=None):
        # 0.0.2 returned (normals, valid_mask); 1.x returns only normals.
        normals = _point_map_to_normal_map(point, mask=mask)
        if mask is None:
            mask = np.ones(point.shape[:-1], dtype=bool)
        return normals, mask

    def image_uv(height, width, **kwargs):
        return _uv_map((height, width))

    def image_mesh(*image_attrs, **kwargs):
        return _build_mesh_from_map(*image_attrs, **kwargs)

np.acos = np.arccos  # sam3d_objects-3dfy version of numpy doesn't have acos

MeshAndTexture = namedtuple(
    "mesh_and_texture", ["faces", "vertices", "vertex_colors", "vertex_uvs"]
)


def mesh_from_pointmap(
    pointmap: np.ndarray,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    filter_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    depth_edge_tol: float = 5,
) -> MeshAndTexture:
    """
    Create a mesh from pointmap and image.
    Returns:
        faces, vertices, vertex_colors, vertex_uvs: Mesh components
    """
    assert pointmap.ndim == 3, pointmap.shape
    assert pointmap.shape[-1] == 3, pointmap.shape
    assert image.ndim == 3, image.shape
    assert image.shape[-1] == 3, image.shape

    if mask is None:
        mask = np.ones_like(pointmap[..., 2], dtype=np.float32) > 0

    if depth is None:
        depth = pointmap[..., 2]

    height, width = image.shape[:2]
    normals, normals_mask = points_to_normals(pointmap, mask=mask)

    if filter_edges:
        mask = mask & ~(
            depth_edge(depth, rtol=depth_edge_rtol, mask=mask)
            & normals_edge(normals, tol=depth_edge_tol, mask=normals_mask)
        )

    faces, vertices, vertex_colors, vertex_uvs = image_mesh(
        pointmap,
        image.astype(np.float32),
        image_uv(width=width, height=height),
        mask=mask,
        tri=True,
    )
    vertices, vertex_uvs = vertices * [1, 1, 1], vertex_uvs * [1, -1] + [0, 1]
    return MeshAndTexture(faces, vertices, vertex_colors, vertex_uvs)


def create_textured_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    vert_colors: torch.Tensor,
) -> Meshes:
    tex = TexturesVertex(verts_features=[vert_colors])
    mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
    return mesh
