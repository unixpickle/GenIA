// Adapted from Microsoft TRELLIS vox2seq (MIT); vendor/licenses/vox2seq.LICENSE.
#include <metal_stdlib>
using namespace metal;

inline uint expand_bits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    return (v * 0x00000005u) & 0x49249249u;
}
inline uint extract_bits(uint v) {
    v &= 0x49249249u;
    v = (v ^ (v >> 2)) & 0x030C30C3u;
    v = (v ^ (v >> 4)) & 0x0300F00Fu;
    v = (v ^ (v >> 8)) & 0x030000FFu;
    return (v ^ (v >> 16)) & 0x000003FFu;
}
kernel void encode_voxels(device const uint* coords [[buffer(0)]],
                          device uint* codes [[buffer(1)]],
                          constant int& hilbert [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint point[3] = {coords[3*id], coords[3*id+1], coords[3*id+2]};
    if (hilbert) {
        for (uint q=512; q>1; q>>=1) {
            uint p=q-1;
            for (int i=0; i<3; ++i) {
                if (point[i]&q) point[0]^=p;
                else {uint t=(point[0]^point[i])&p; point[0]^=t; point[i]^=t;}
            }
        }
        point[1]^=point[0]; point[2]^=point[1];
        uint t=0;
        for (uint q=512; q>1; q>>=1) if (point[2]&q) t^=q-1;
        for (int i=0; i<3; ++i) point[i]^=t;
    }
    codes[id] = expand_bits(point[0])*4 + expand_bits(point[1])*2 + expand_bits(point[2]);
}
kernel void decode_voxels(device const uint* codes [[buffer(0)]],
                          device uint* coords [[buffer(1)]],
                          constant int& hilbert [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint point[3] = {extract_bits(codes[id]>>2), extract_bits(codes[id]>>1), extract_bits(codes[id])};
    if (hilbert) {
        uint t=point[2]>>1;
        point[2]^=point[1]; point[1]^=point[0]; point[0]^=t;
        for (uint q=2; q<1024; q<<=1) {
            uint p=q-1;
            for (int i=2; i>=0; --i) {
                if (point[i]&q) point[0]^=p;
                else {uint t=(point[0]^point[i])&p; point[0]^=t; point[i]^=t;}
            }
        }
    }
    for (int i=0; i<3; ++i) coords[3*id+i]=point[i];
}
