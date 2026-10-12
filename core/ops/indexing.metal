#include <metal_stdlib>
using namespace metal;

kernel void unravel_indices(
    device const long *indices, device const long *dims, device long *coordinates,
    constant uint &ndim, constant uint &count, uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    long index = indices[i];
    for (int axis = int(ndim) - 1; axis >= 0; --axis) {
        coordinates[i * ndim + axis] = index % dims[axis];
        index /= dims[axis];
    }
}
