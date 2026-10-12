#include <metal_stdlib>
using namespace metal;

kernel void area_pool2d(
    device const float *input, device float *output,
    constant uint &ih, constant uint &iw, constant uint &oh, constant uint &ow, constant uint &planes,
    uint i [[thread_position_in_grid]]) {
    if (i>=planes*oh*ow) return;
    uint plane=i/(oh*ow), y=(i/ow)%oh, x=i%ow;
    uint y0=y*ih/oh, y1=((y+1)*ih+oh-1)/oh;
    uint x0=x*iw/ow, x1=((x+1)*iw+ow-1)/ow;
    float sum=0;
    for (uint a=y0; a<y1; ++a)
    for (uint b=x0; b<x1; ++b)
        sum+=input[(plane*ih+a)*iw+b];
    output[i]=sum/float(y1-y0)/float(x1-x0);
}

kernel void area_pool2d_backward(
    device const float *grad_output, device float *grad_input,
    constant uint &ih, constant uint &iw, constant uint &oh, constant uint &ow, constant uint &planes,
    uint i [[thread_position_in_grid]]) {
    if (i>=planes*ih*iw) return;
    uint plane=i/(ih*iw), y=(i/iw)%ih, x=i%iw;
    uint y0=y*oh/ih, y1=((y+1)*oh+ih-1)/ih;
    uint x0=x*ow/iw, x1=((x+1)*ow+iw-1)/iw;
    float sum=0;
    for (uint a=y0; a<y1; ++a)
    for (uint b=x0; b<x1; ++b) {
        uint h=((a+1)*ih+oh-1)/oh-a*ih/oh;
        uint w=((b+1)*iw+ow-1)/ow-b*iw/ow;
        sum+=grad_output[(plane*oh+a)*ow+b]/float(h)/float(w);
    }
    grad_input[i]=sum;
}

// Exact PyTorch CPU max_pool3d argmax, including lowest-index finite ties.
kernel void volume_argmax(
    device const float *volume, device long *indices,
    constant uint &nx, constant uint &ny, constant uint &nz,
    constant uint &radius, uint i [[thread_position_in_grid]]) {
    if (i>=nx*ny*nz) return;
    int x=i/(ny*nz), y=(i/nz)%ny, z=i%nz, r=int(radius);
    float best=-INFINITY; long selected=-1;
    for (int a=max(0,x-r); a<=min(int(nx)-1,x+r); ++a)
    for (int b=max(0,y-r); b<=min(int(ny)-1,y+r); ++b)
    for (int c=max(0,z-r); c<=min(int(nz)-1,z+r); ++c) {
        long at=(long(a)*ny+b)*nz+c;
        float value=volume[at];
        if (selected<0 || value>best || isnan(value)) {
            best=value; selected=at;
        }
    }
    indices[i]=selected;
}
