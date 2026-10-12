"""Native autograd replacement for PyTorch3D's extension-backed face normals."""
import torch


def mesh_face_areas_normals(verts, faces):
    if verts.ndim != 2 or verts.shape[-1] != 3 or faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError('Expected vertices (V,3) and triangle indices (F,3)')
    triangle = verts[faces.long()]
    normals = torch.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0], dim=-1)
    length = torch.linalg.vector_norm(normals, dim=-1)
    return length*.5, normals/length.clamp_min(1e-6)[:,None]
