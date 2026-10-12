#include <metal_stdlib>
using namespace metal;

// One thread per query. Memory is O(B * P * K), independent of reference count.
// Equal distances retain reference order, making ties deterministic.
kernel void knn_indices(
    const device float *query, const device float *reference,
    const device int *length1, const device int *length2,
    device int *indices, constant uint &P, constant uint &Q,
    constant uint &D, constant uint &K, constant uint &norm,
    uint tid [[thread_position_in_grid]]) {
    uint b = tid / P, p = tid % P;
    float best[32];
    int selected[32];
    for (uint k = 0; k < K; ++k) { best[k] = INFINITY; selected[k] = 0; }
    if (p < uint(length1[b])) {
        for (uint q = 0; q < uint(length2[b]); ++q) {
            float dist = 0;
            for (uint d = 0; d < D; ++d) {
                float delta = query[tid * D + d] - reference[(b * Q + q) * D + d];
                dist += norm == 2 ? delta * delta : abs(delta);
            }
            for (uint k = 0; k < K; ++k) {
                if (dist < best[k]) {
                    for (uint j = K - 1; j > k; --j) {
                        best[j] = best[j - 1]; selected[j] = selected[j - 1];
                    }
                    best[k] = dist; selected[k] = int(q); break;
                }
            }
        }
    }
    for (uint k = 0; k < K; ++k) indices[tid * K + k] = selected[k];
}

// A SIMD group cooperatively scans a 3D cloud for one nearest neighbor.
// Reference loads are adjacent across lanes; ties retain the first index.
kernel void knn1_3d(
    const device packed_float3 *query, const device packed_float3 *reference,
    const device int *length1, const device int *length2,
    device int *indices, device float *distances,
    constant uint &P, constant uint &Q, constant uint &norm,
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = tid / 32, b = row / P, p = row % P;
    float best = INFINITY;
    uint selected = 0xffffffff;
    if (p < uint(length1[b])) {
        float3 point = float3(query[row]);
        for (uint q = lane; q < uint(length2[b]); q += 32) {
            float3 delta = point - float3(reference[b * Q + q]);
            float3 terms = norm == 2 ? delta * delta : abs(delta);
            float dist = (terms.x + terms.y) + terms.z;
            if (dist < best) {best = dist; selected = q;}
        }
    }
    float nearest = simd_min(best);
    uint index = simd_min(best == nearest ? selected : 0xffffffff);
    if (lane == 0) {
        indices[row] = index == 0xffffffff ? 0 : int(index);
        if (p < uint(length1[b]) && length2[b] > 0) {
            // Preserve the generic path's distance for all-NaN/overflow clouds.
            if (index == 0xffffffff) {
                float3 delta = float3(query[row]) - float3(reference[b * Q]);
                float3 terms = norm == 2 ? delta * delta : abs(delta);
                nearest = (terms.x + terms.y) + terms.z;
            }
            distances[row] = nearest;
        } else distances[row] = 0.0f;
    }
}
