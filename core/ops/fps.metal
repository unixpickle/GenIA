// PyTorch3D-compatible FPS. BSD-3-Clause; vendor/licenses/pytorch3d.LICENSE.
#include <metal_stdlib>
using namespace metal;

kernel void farthest_points(device const float* points [[buffer(0)]],
                            device const long* lengths [[buffer(1)]],
                            device const long* counts [[buffer(2)]],
                            device const long* starts [[buffer(3)]],
                            device float* distances [[buffer(4)]],
                            device long* indices [[buffer(5)]],
                            constant int& p [[buffer(6)]],
                            constant int& d [[buffer(7)]],
                            constant int& max_k [[buffer(8)]],
                            uint batch [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float best_distance[256];
    threadgroup int best_index[256];
    threadgroup int selected;
    int n = int(lengths[batch]), k = min(n, int(counts[batch]));
    if (k == 0) return;
    if (lane == 0) {selected = int(starts[batch]); indices[batch*max_k] = selected;}
    for (int i=int(lane); i<n; i+=256) distances[batch*p+i] = 1e10f;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (int s=1; s<k; ++s) {
        float farthest = -1;
        int farthest_index = 2147483647;
        for (int i=int(lane); i<n; i+=256) {
            float dist = 0;
            for (int c=0; c<d; ++c) {
                float diff = points[(batch*p+i)*d+c] - points[(batch*p+selected)*d+c];
                dist += diff*diff;
            }
            float nearest = min(distances[batch*p+i], dist);
            distances[batch*p+i] = nearest;
            if (nearest > farthest) {farthest = nearest; farthest_index = i;}
        }
        best_distance[lane] = farthest; best_index[lane] = farthest_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int stride=128; stride>0; stride>>=1) {
            if (lane < uint(stride)) {
                float other = best_distance[lane+stride]; int ix = best_index[lane+stride];
                if (other > best_distance[lane] || (other == best_distance[lane] && ix < best_index[lane])) {
                    best_distance[lane] = other; best_index[lane] = ix;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {selected=best_index[0]; indices[batch*max_k+s]=selected;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
