// GenIA Metal implementation of the 3D Gaussian splatting equations.
// Compositing thresholds and pixel conventions match gsplat v1.5.3.
#include <metal_stdlib>
using namespace metal;

kernel void tile_counts(const device float2 *means, const device int2 *radii,
                        device int4 *bounds, device int *counts,
                        constant uint &tile_size, constant uint &tiles_x,
                        constant uint &tiles_y, uint i [[thread_position_in_grid]]) {
    float2 m = means[i];
    int2 r = radii[i];
    int2 lo = clamp(int2(floor((m - float2(r)) / float(tile_size))), int2(0), int2(tiles_x, tiles_y));
    int2 hi = clamp(int2(ceil((m + float2(r)) / float(tile_size))), int2(0), int2(tiles_x, tiles_y));
    if (any(r <= 0)) hi = lo;
    bounds[i] = int4(lo, hi);
    counts[i] = (hi.x - lo.x) * (hi.y - lo.y);
}

kernel void tile_emit(const device int4 *bounds, const device int *offsets,
                      const device int *ranks, device long *keys, device int *ids,
                      constant uint &N, constant uint &tiles_x,
                      uint i [[thread_position_in_grid]]) {
    int4 b = bounds[i];
    int out = offsets[i];
    for (int y = b.y; y < b.w; ++y) {
        for (int x = b.x; x < b.z; ++x) {
            keys[out] = long(y * tiles_x + x) * long(N) + long(ranks[i]);
            ids[out++] = int(i);
        }
    }
}

kernel void tile_ranges(const device long *keys, device int *ranges,
                        constant uint &N, constant uint &count,
                        uint i [[thread_position_in_grid]]) {
    long tile = keys[i] / N;
    // Scalar stores are essential: vector-component stores can become a
    // read/modify/write of int2 and race when different threads set the ends.
    if (i == 0 || keys[i - 1] / N != tile) ranges[tile * 2] = int(i);
    if (i + 1 == count || keys[i + 1] / N != tile) ranges[tile * 2 + 1] = int(i + 1);
}

inline float gaussian_alpha(float2 delta, float3 conic, float opacity, thread float &raw) {
    float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y)
                  + conic.y * delta.x * delta.y;
    raw = opacity * exp(-sigma);
    float a = min(0.999f, raw);
    return (sigma < 0 || a < 1.0f / 255.0f) ? 0.0f : a;
}

kernel void splat_forward(
    const device float2 *means, const device packed_float3 *conics,
    const device float4 *colors, const device float *opacities,
    const device float4 *background, const device int2 *ranges,
    const device int *ids, device float4 *output, device float *transmittance,
    device int *last_ids, constant uint &width, constant uint &tile_size,
    constant uint &tiles_x, uint pixel [[thread_position_in_grid]]) {
    uint x = pixel % width, y = pixel / width;
    int2 range = ranges[(y / tile_size) * tiles_x + x / tile_size];
    float2 pos = float2(x, y) + 0.5f;
    float T = 1.0f;
    float4 result = 0;
    int last = range.x - 1;
    for (int j = range.x; j < range.y; ++j) {
        int i = ids[j];
        float raw;
        float a = gaussian_alpha(means[i] - pos, float3(conics[i]), opacities[i], raw);
        if (a == 0) continue;
        float next = T * (1.0f - a);
        if (next <= 1e-4f) break;
        result += T * a * colors[i];
        T = next;
        last = j;
    }
    output[pixel] = result + T * background[0];
    transmittance[pixel] = T;
    last_ids[pixel] = last;
}

// Portable float accumulation, including Apple GPUs without atomic_float add.
inline void add_float(device atomic_uint *target, float value) {
    if (value == 0) return;
    uint old = atomic_load_explicit(target, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(target, &old,
               as_type<uint>(as_type<float>(old) + value),
               memory_order_relaxed, memory_order_relaxed)) {}
}

kernel void splat_backward(
    const device float2 *means, const device packed_float3 *conics,
    const device float4 *colors, const device float *opacities,
    const device float4 *background, const device int2 *ranges,
    const device int *ids, const device float *transmittance, const device int *last_ids,
    const device float4 *grad_output, const device float *grad_alpha,
    device atomic_uint *grad_means, device atomic_uint *grad_conics,
    device atomic_uint *grad_colors, device atomic_uint *grad_opacities,
    constant uint &width, constant uint &tile_size, constant uint &tiles_x,
    uint pixel [[thread_position_in_grid]]) {
    uint x = pixel % width, y = pixel / width;
    int2 range = ranges[(y / tile_size) * tiles_x + x / tile_size];
    float2 pos = float2(x, y) + 0.5f;
    float final_T = transmittance[pixel], T = final_T;
    float4 behind = background[0], grad = grad_output[pixel];
    for (int j = last_ids[pixel]; j >= range.x; --j) {
        int i = ids[j];
        float raw;
        float2 delta = means[i] - pos;
        float3 conic = float3(conics[i]);
        float a = gaussian_alpha(delta, conic, opacities[i], raw);
        if (a == 0) continue;
        T /= (1.0f - a);
        float da = T * dot(grad, colors[i] - behind)
                   + grad_alpha[pixel] * final_T / (1.0f - a);
        float4 dc = T * a * grad;
        for (uint c = 0; c < 4; ++c) add_float(grad_colors + 4 * i + c, dc[c]);
        if (raw < 0.999f) {
            float ds = -raw * da;
            add_float(grad_means + 2 * i, ds * (conic.x * delta.x + conic.y * delta.y));
            add_float(grad_means + 2 * i + 1, ds * (conic.y * delta.x + conic.z * delta.y));
            add_float(grad_conics + 3 * i, ds * 0.5f * delta.x * delta.x);
            add_float(grad_conics + 3 * i + 1, ds * delta.x * delta.y);
            add_float(grad_conics + 3 * i + 2, ds * 0.5f * delta.y * delta.y);
            add_float(grad_opacities + i, da * raw / opacities[i]);
        }
        behind = a * colors[i] + (1.0f - a) * behind;
    }
}

// One 16x16 image tile per threadgroup. Gaussian records are cooperatively
// loaded once per batch and reused by all pixels in the tile.
kernel void splat_forward_tiled(
    const device float2 *means, const device packed_float3 *conics,
    const device float4 *colors, const device float *opacities,
    const device float4 *background, const device int2 *ranges,
    const device int *ids, device float4 *output, device float *transmittance,
    device int *last_ids, constant uint &width, constant uint &height,
    constant uint &tiles_x, uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 xy[256];
    threadgroup float3 co[256];
    threadgroup float4 rgb[256];
    threadgroup float op[256];
    threadgroup uint active[8];
    uint x = (tile % tiles_x) * 16 + lane % 16;
    uint y = (tile / tiles_x) * 16 + lane / 16;
    bool inside = x < width && y < height, done = !inside;
    int2 range = ranges[tile];
    float2 pos = float2(x,y) + .5f;
    float T = 1; float4 result = 0; int last = range.x - 1;
    for (int start = range.x; start < range.y; start += 256) {
        uint live = simd_sum(uint(!done));
        if (lane % 32 == 0) active[lane / 32] = live;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint total = 0; for (uint s=0; s<8; ++s) total += active[s];
        if (total == 0) break;
        int count = min(256, range.y-start);
        if (lane < uint(count)) {
            int i = ids[start + lane];
            xy[lane]=means[i]; co[lane]=float3(conics[i]); rgb[lane]=colors[i]; op[lane]=opacities[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j=0; j<count && !done; ++j) {
            float raw; float a=gaussian_alpha(xy[j]-pos,co[j],op[j],raw);
            if (a == 0) continue;
            float next=T*(1-a);
            if (next <= 1e-4f) {done=true; break;}
            result += T*a*rgb[j]; T=next; last=start+j;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inside) {
        uint pixel=y*width+x;
        output[pixel]=result+T*background[0]; transmittance[pixel]=T; last_ids[pixel]=last;
    }
}

// Accumulate each SIMD group's pixel gradients before updating global memory.
// This reduces atomic traffic by up to 32x and keeps inactive lanes in reductions.
kernel void splat_backward_tiled(
    const device float2 *means, const device packed_float3 *conics,
    const device float4 *colors, const device float *opacities,
    const device float4 *background, const device int2 *ranges,
    const device int *ids, const device float *transmittance, const device int *last_ids,
    const device float4 *grad_output, const device float *grad_alpha,
    device atomic_uint *grad_means, device atomic_uint *grad_conics,
    device atomic_uint *grad_colors, device atomic_uint *grad_opacities,
    constant uint &width, constant uint &height, constant uint &tiles_x,
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 xy[256]; threadgroup float3 co[256];
    threadgroup float4 rgb[256]; threadgroup float op[256]; threadgroup int gi[256];
    threadgroup int ends[8];
    uint x=(tile%tiles_x)*16+lane%16, y=(tile/tiles_x)*16+lane/16;
    bool inside=x<width && y<height;
    uint pixel=y*width+x; int2 range=ranges[tile];
    int last=inside ? last_ids[pixel] : range.x-1;
    int end=simd_max(last);
    if (lane%32==0) ends[lane/32]=end;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=0;s<8;++s) end=max(end,ends[s]);
    float2 pos=float2(x,y)+.5f;
    float final_T=inside ? transmittance[pixel] : 1, T=final_T;
    float4 behind=background[0], grad=inside ? grad_output[pixel] : float4(0);
    float ga=inside ? grad_alpha[pixel] : 0;
    for (int stop=end+1;stop>range.x;stop-=256) {
        int start=max(range.x,stop-256), count=stop-start;
        if (lane<uint(count)) {
            int i=ids[start+lane]; gi[lane]=i;
            xy[lane]=means[i];co[lane]=float3(conics[i]);rgb[lane]=colors[i];op[lane]=opacities[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j=count-1;j>=0;--j) {
            float4 dc=0; float2 dm=0; float3 dv=0; float dop=0;
            if (inside && start+j<=last) {
                float raw; float2 delta=xy[j]-pos; float3 conic=co[j];
                float a=gaussian_alpha(delta,conic,op[j],raw);
                if (a!=0) {
                    T/=1-a;
                    float da=T*dot(grad,rgb[j]-behind)+ga*final_T/(1-a);
                    dc=T*a*grad;
                    if (raw<.999f) {
                        float ds=-raw*da;
                        dm=ds*float2(conic.x*delta.x+conic.y*delta.y,conic.y*delta.x+conic.z*delta.y);
                        dv=ds*float3(.5f*delta.x*delta.x,delta.x*delta.y,.5f*delta.y*delta.y);
                        dop=da*raw/op[j];
                    }
                    behind=a*rgb[j]+(1-a)*behind;
                }
            }
            dc=simd_sum(dc);dm=simd_sum(dm);dv=simd_sum(dv);dop=simd_sum(dop);
            if (lane%32==0) {
                int i=gi[j];
                for (uint c=0;c<4;++c) add_float(grad_colors+4*i+c,dc[c]);
                for (uint c=0;c<2;++c) add_float(grad_means+2*i+c,dm[c]);
                for (uint c=0;c<3;++c) add_float(grad_conics+3*i+c,dv[c]);
                add_float(grad_opacities+i,dop);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
