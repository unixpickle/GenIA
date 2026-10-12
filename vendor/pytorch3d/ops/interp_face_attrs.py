# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def interpolate_face_attributes(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate arbitrary face attributes using the barycentric coordinates
    for each pixel in the rasterized output.

    Args:
        pix_to_face: LongTensor of shape (...) specifying the indices
            of the faces (in the packed representation) which overlap each
            pixel in the image. A value < 0 indicates that the pixel does not
            overlap any face and should be skipped.
        barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
            the barycentric coordinates of each pixel
            relative to the faces (in the packed
            representation) which overlap the pixel.
        face_attributes: packed attributes of shape (total_faces, 3, D),
            specifying the value of the attribute for each
            vertex in the face.

    Returns:
        pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
        value of the face attribute for each pixel.
    """
    # Check shapes
    F, FV, D = face_attributes.shape
    if FV != 3:
        raise ValueError("Faces can only have three vertices; got %r" % FV)
    N, H, W, K, _ = barycentric_coords.shape
    if pix_to_face.shape != (N, H, W, K):
        msg = "pix_to_face must have shape (batch_size, H, W, K); got %r"
        raise ValueError(msg % (pix_to_face.shape,))

    return interpolate_face_attributes_python(pix_to_face, barycentric_coords, face_attributes)


def interpolate_face_attributes_python(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    F, FV, D = face_attributes.shape
    N, H, W, K, _ = barycentric_coords.shape

    if F == 0:
        return face_attributes.new_zeros((*pix_to_face.shape, D)) + face_attributes.sum()*0 + barycentric_coords.sum()*0

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = pix_to_face < 0
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    return pixel_vals
