"""Differentiable triangle rasterization with local Metal visibility selection.

Coordinates, barycentrics and signed squared edge distances follow PyTorch3D
0.7.9. Visibility is discrete; selected face values are recomputed with autograd.
The upstream wrapper handles near-plane clipping and camera projection.
"""
import math
import torch
from .metal import library
from .gaussian import gaussian_tile_intersections


def _edge(p, a, b):
    return (p[..., 0]-a[..., 0])*(b[..., 1]-a[..., 1]) - (p[..., 1]-a[..., 1])*(b[..., 0]-a[..., 0])


def _line_distance(p, a, b):
    delta = b-a
    length = delta.square().sum(-1)
    t = ((p-a)*delta).sum(-1) / length.clamp_min(1e-8)
    nearest = a + t.clamp(0, 1)[..., None]*delta
    nearest = torch.where((length <= 1e-8)[..., None], b, nearest)
    return (p-nearest).square().sum(-1)


def _values(faces, pixels, perspective, clip):
    a, b, c = faces.unbind(-2)
    area = _edge(c[..., :2], a[..., :2], b[..., :2])
    denom = area + 1e-8
    # Selected faces cannot be degenerate; padded pixels may refer to one.
    denom = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
    bary = torch.stack([_edge(pixels, b[..., :2], c[..., :2]),
                        _edge(pixels, c[..., :2], a[..., :2]),
                        _edge(pixels, a[..., :2], b[..., :2])], -1) / denom[..., None]
    if perspective:
        z = faces[..., 2]
        products = torch.stack([z[..., 1]*z[..., 2], z[..., 0]*z[..., 2], z[..., 0]*z[..., 1]], -1)
        bary = bary*products
        norm = bary.sum(-1, keepdim=True)
        bary = bary / torch.where(norm.abs() > 1e-12, norm, torch.ones_like(norm))
    inside = (bary > 0).all(-1)
    if clip:
        bary = bary.clamp_min(0)
        bary = bary / bary.sum(-1, keepdim=True).clamp_min(1e-5)
    depth = (bary*faces[..., 2]).sum(-1)
    distance = torch.minimum(torch.minimum(_line_distance(pixels, a[..., :2], b[..., :2]),
        _line_distance(pixels, a[..., :2], c[..., :2])), _line_distance(pixels, b[..., :2], c[..., :2]))
    return depth, bary, torch.where(inside, -distance, distance), inside, area


def _pixel_grid(h, w, device, dtype):
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                          torch.arange(w, device=device, dtype=dtype), indexing='ij')
    return torch.stack([(w-1-2*x)/min(h,w), (h-1-2*y)/min(h,w)], -1).reshape(-1, 2)


@torch.no_grad()
def _select(faces, neighbors, h, w, blur, k, perspective, clip, cull, inclusive=False):
    device = faces.device
    if not len(faces):
        return torch.full((h*w, k), -1, dtype=torch.long, device=device)
    if device.type == 'mps':
        # The Gaussian bin builder accepts conservative pixel-space rectangles.
        xy = faces[..., :2]
        lo, hi = xy.amin(1), xy.amax(1)
        centers = (torch.tensor([w,h],device=device)-(lo+hi)*(.5*min(h,w)))/2
        radii = torch.ceil((hi-lo)*(.25*min(h,w)) + math.sqrt(blur)*min(h,w)/2 + 1).int()
        ranges, ids = gaussian_tile_intersections(centers.float(), radii,
            torch.arange(len(faces), device=device, dtype=torch.float32), w, h)
        result = torch.empty((h*w,k), dtype=torch.int32, device=device)
        library('mesh').select_faces(faces.float().contiguous(), neighbors.int().contiguous(),
            ranges, ids, result, w, h, k, float(blur), int(perspective), int(clip), int(cull), int(inclusive), threads=h*w)
        return result.long()
    pixels = _pixel_grid(h,w,device,faces.dtype)
    result=[]
    # Bound the temporary face-by-pixel arrays on the CPU reference path.
    step=max(1, min(512, 100000//len(faces)))
    for p in pixels.split(step):
        depth, _, distance, inside, area = _values(faces[None], p[:,None], perspective, clip)
        if inclusive:
            _, bary, _, _, _ = _values(faces[None], p[:,None], perspective, False)
            inside = (bary >= 0).all(-1)
        valid = ((area.abs()>1e-8) & (faces[...,2].amin(-1)[None]>=1e-8) &
                 (depth>=0) & (inside | (distance<blur)))
        if cull: valid &= area>=0
        if (neighbors>=0).any():
            ni=neighbors.clamp_min(0).long()
            other=distance[:,ni].abs()
            better = (other<distance.abs()) | ((other==distance.abs()) & (ni<torch.arange(len(faces),device=device)))
            valid &= ~((neighbors>=0)[None] & valid[:,ni] & better)
        order=torch.where(valid,depth,torch.inf).argsort(dim=-1,stable=True)[:,:k]
        order=torch.where(valid.gather(1,order),order,-1)
        if order.shape[1]<k: order=torch.nn.functional.pad(order,(0,k-order.shape[1]),value=-1)
        result.append(order)
    return torch.cat(result)


def rasterize_face_verts(faces, first, counts, neighbors, image_size, blur_radius,
                        faces_per_pixel, bin_size=0, max_faces_per_bin=0,
                        perspective_correct=False, clip_barycentric_coords=False,
                        cull_backfaces=False):
    h,w=image_size
    if h<1 or w<1 or blur_radius<0 or not 1<=faces_per_pixel<=64:
        raise ValueError('positive image size, nonnegative blur, and 1..64 faces_per_pixel required')
    if faces.ndim!=3 or faces.shape[1:]!=(3,3) or not faces.is_floating_point():
        raise ValueError('faces must be floating point (F,3,3)')
    pixels=_pixel_grid(h,w,faces.device,faces.dtype)
    all_ids=[]; all_z=[]; all_bary=[]; all_dist=[]
    for start,count in zip(first.tolist(),counts.tolist()):
        face=faces[start:start+count]
        neighbor=neighbors[start:start+count]
        neighbor=torch.where(neighbor>=0,neighbor-start,neighbor)
        ids=_select(face,neighbor,h,w,blur_radius,faces_per_pixel,perspective_correct,
                    clip_barycentric_coords,cull_backfaces)
        mask=ids>=0
        if count:
            z,bary,dist,_,_=_values(face[ids.clamp_min(0)], pixels[:,None],
                                   perspective_correct,clip_barycentric_coords)
        else:
            anchor=faces.sum()*0
            z=faces.new_zeros((h*w,faces_per_pixel))+anchor
            dist=z.clone(); bary=z[...,None].expand(-1,-1,3)
        all_ids.append(torch.where(mask,ids+start,-1).reshape(h,w,-1))
        all_z.append(torch.where(mask,z,-1).reshape(h,w,-1))
        all_bary.append(torch.where(mask[...,None],bary,-1).reshape(h,w,faces_per_pixel,3))
        all_dist.append(torch.where(mask,dist,-1).reshape(h,w,-1))
    if not len(counts):
        shape=(0,h,w,faces_per_pixel)
        return (torch.empty(shape,dtype=torch.long,device=faces.device),faces.new_empty(shape),
                faces.new_empty((*shape,3)),faces.new_empty(shape))
    return tuple(torch.stack(x) for x in (all_ids,all_z,all_bary,all_dist))
