"""SAM's legacy utils3d mesh API restored from revision 36a9d5cab85b2913b2db11092d6d4eb121480a73.

Upstream MIT licence is retained in vendor/licenses/utils3d.LICENSE.
Dimension-wise unique and grouping are adapted for Metal and empty inputs.
"""
from typing import *
import torch
from torch import Tensor
from genia.core.ops.unique import unique

__all__ = ['compute_edges', 'compute_connected_components', 'compute_edge_connected_components', 'compute_dual_graph', 'remove_unreferenced_vertices', 'rasterize_triangle_faces', 'perspective_from_fov_xy', 'intrinsics_from_fov_xy']

def _group(
    values: Tensor,
    required_group_size: Optional[int] = None,
    return_values: bool = False
) -> Tuple[Union[List[Tensor], Tensor], Optional[Tensor]]:
    """
    Group values into groups with identical values.
    
    ## Parameters
        values (Tensor): [N] values to group
        required_group_size (int, optional): required group size. Defaults to None.
        return_values (bool, optional): return values of groups. Defaults to False.
        
    ## Returns
        group (Union[List[Tensor], Tensor]): list of groups or group indices. It will be a list of groups if required_group_size is None, otherwise a tensor of group indices.
        group_values (Optional[Tensor]): values of groups. Only returned if return_values is True.
    """
    if not values.numel():
        groups = () if required_group_size is None else values.new_empty((0, required_group_size), dtype=torch.long)
        return (groups, values) if return_values else groups
    sorted_values, indices = torch.sort(values, stable=True)
    nondupe = torch.cat([torch.tensor([True], dtype=torch.bool, device=values.device), sorted_values[1:] != sorted_values[:-1]])
    nondupe_indices = torch.cumsum(nondupe, dim=0) - 1
    starts = nondupe.nonzero().flatten()
    counts = torch.cat([starts[1:], starts.new_tensor([len(values)])]) - starts
    if required_group_size is None:
        groups = torch.split(indices, counts.tolist())
        if return_values:
            group_values = sorted_values[nondupe]
            return groups, group_values
        else:
            return groups
    else:
        counts = counts[nondupe_indices]
        groups = indices[counts == required_group_size].reshape(-1, required_group_size)
        if return_values:
            group_values = sorted_values[nondupe][counts[nondupe] == required_group_size]
            return groups, group_values
        else:
            return groups

def compute_edges(
    faces: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute edges of a mesh.
    
    ## Parameters
        faces (Tensor): [T, 3] triangular face indices
        
    ## Returns
        edges (Tensor): [E, 2] edge indices
        face2edge (Tensor): [T, 3] mapping from face to edge
        counts (Tensor): [E] degree of each edge
    """
    T = faces.shape[0]
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)    # [3T, 2]
    edges = torch.sort(edges, dim=1).values
    edges, inv_map, counts = unique(edges, return_inverse=True, return_counts=True, dim=0)
    face2edge = inv_map.view(3, T).T
    return edges, face2edge, counts

def compute_connected_components(
    faces: Tensor,
    edges: Tensor=None,
    face2edge: Tensor=None
) -> List[Tensor]:
    """
    Compute connected faces of a mesh.

    ## Parameters
        faces (Tensor): [T, 3] triangular face indices
        edges (Tensor, optional): [E, 2] edge indices. Defaults to None.
        face2edge (Tensor, optional): [T, 3] mapping from face to edge. Defaults to None.
            NOTE: If edges and face2edge are not provided, they will be computed.

    ## Returns
        components (List[Tensor]): list of connected faces
    """
    T = faces.shape[0]
    if edges is None or face2edge is None:
        edges, face2edge, _ = compute_edges(faces)
    E = edges.shape[0]

    labels = torch.arange(T, dtype=torch.int32, device=faces.device)
    while True:
        edge_labels = torch.scatter_reduce(
            torch.zeros(E, dtype=torch.int32, device=faces.device),
            0,
            face2edge.flatten().long(),
            labels.view(-1, 1).expand(-1, 3).flatten(),
            reduce='amin',
            include_self=False
        )
        new_labels = torch.min(edge_labels[face2edge], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components = _group(labels)

    return components

def compute_edge_connected_components(
    edges: Tensor,
) -> List[Tensor]:
    """
    Compute connected edges of a mesh.

    ## Parameters
        edges (Tensor): [E, 2] edge indices

    ## Returns
        components (List[Tensor]): list of connected edges
    """
    E = edges.shape[0]

    # Re-index edges
    verts, edges = torch.unique(edges.flatten(), return_inverse=True)
    edges = edges.view(-1, 2)
    V = verts.shape[0]

    labels = torch.arange(E, dtype=torch.int32, device=edges.device)
    while True:
        vertex_labels = torch.scatter_reduce(
            torch.zeros(V, dtype=torch.int32, device=edges.device),
            0,
            edges.flatten().long(),
            labels.view(-1, 1).expand(-1, 2).flatten(),
            reduce='amin',
            include_self=False
        )
        new_labels = torch.min(vertex_labels[edges], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components = _group(labels)

    return components

def compute_dual_graph(
    face2edge: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Compute dual graph of a mesh.
    
    ## Parameters
        face2edge (Tensor): [T, 3] mapping from face to edge.
            
    ## Returns
        dual_edges (Tensor): [DE, 2] face indices of dual edges
        dual_edge2edge (Tensor): [DE] mapping from dual edge to edge
    """
    all_edge_indices = face2edge.flatten()  # [3T]
    dual_edges, dual_edge2edge = _group(all_edge_indices, required_group_size=2, return_values=True)
    dual_edges = dual_edges // face2edge.shape[1]
    return dual_edges, dual_edge2edge


def remove_unreferenced_vertices(faces, *attributes, **kwargs):
    from .mesh import remove_unused_vertices
    return remove_unused_vertices(faces, *attributes, **kwargs)


def rasterize_triangle_faces(ctx, vertices, faces, width, height, **kwargs):
    from .rasterization import rasterize_triangles
    return rasterize_triangles(ctx, width, height, vertices=vertices, faces=faces, **kwargs)


def perspective_from_fov_xy(fov_x, fov_y, near, far):
    from .transforms import perspective_from_fov
    reference = fov_x if isinstance(fov_x, Tensor) else fov_y
    if isinstance(reference, Tensor):
        near = torch.as_tensor(near, device=reference.device, dtype=reference.dtype)
        far = torch.as_tensor(far, device=reference.device, dtype=reference.dtype)
    return perspective_from_fov(fov_x=fov_x, fov_y=fov_y, near=near, far=far)


def intrinsics_from_fov_xy(fov_x, fov_y):
    from .transforms import intrinsics_from_fov
    return intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)
