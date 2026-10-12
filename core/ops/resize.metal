#include <metal_stdlib>
using namespace metal;

// Pillow/PyTorch antialias filters. Cubic uses a=-0.5, not the a=-0.75
// polynomial used by non-antialiased PyTorch bicubic interpolation.
inline float aa_filter(float x, uint cubic) {
    x = abs(x);
    if (!cubic) return max(1.0f - x, 0.0f);
    if (x < 1) return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    if (x < 2) return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    return 0;
}

inline int2 span(uint dst, uint input_size, float scale, uint cubic) {
    float support = (cubic ? 2.0f : 1.0f) * max(scale, 1.0f);
    float center = scale * (float(dst) + 0.5f);
    return int2(max(int(center - support + .5f), 0), min(int(center + support + .5f), int(input_size)));
}

inline float weight(uint dst, int src, float scale, uint cubic) {
    float center = scale * (float(dst) + .5f);
    return aa_filter((float(src) - center + .5f) / max(scale, 1.0f), cubic);
}

kernel void resize_axis(const device float *input, device float *output,
                        constant uint &input_size, constant uint &output_size,
                        constant uint &inner, constant float &scale,
                        constant uint &cubic, uint tid [[thread_position_in_grid]]) {
    uint dst = (tid / inner) % output_size;
    uint base = (tid / (inner * output_size)) * (inner * input_size) + tid % inner;
    int2 range = span(dst, input_size, scale, cubic);
    float sum = 0, norm = 0;
    for (int src = range.x; src < range.y; ++src) {
        float w = weight(dst, src, scale, cubic);
        sum += w * input[base + src * inner];
        norm += w;
    }
    output[tid] = norm == 0 ? 0 : sum / norm;
}

// Adjoint gather: one thread owns each input gradient, avoiding atomic races.
kernel void resize_axis_backward(const device float *grad_output, device float *grad_input,
                                 constant uint &input_size, constant uint &output_size,
                                 constant uint &inner, constant float &scale,
                                 constant uint &cubic, uint tid [[thread_position_in_grid]]) {
    uint src = (tid / inner) % input_size;
    uint base = (tid / (inner * input_size)) * (inner * output_size) + tid % inner;
    float support = (cubic ? 2.0f : 1.0f) * max(scale, 1.0f);
    int lo = scale == 0 ? 0 : max(0, int(floor((float(src) + .5f - support) / scale - .5f)) - 1);
    int hi = scale == 0 ? int(output_size) : min(int(output_size), int(ceil((float(src) + .5f + support) / scale - .5f)) + 2);
    float sum = 0;
    for (int dst = lo; dst < hi; ++dst) {
        int2 range = span(dst, input_size, scale, cubic);
        if (int(src) < range.x || int(src) >= range.y) continue;
        float norm = 0;
        for (int s = range.x; s < range.y; ++s) norm += weight(dst, s, scale, cubic);
        if (norm != 0) sum += grad_output[base + dst * inner] * weight(dst, src, scale, cubic) / norm;
    }
    grad_input[tid] = sum;
}
