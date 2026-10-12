// DISO 0.1.4 dual topology, ported from CUDA (CC BY-NC 4.0).
#include <metal_stdlib>
using namespace metal;

kernel void classify_dual_cells(
    device const float *grid, device const int *corners,
    device uchar *codes, device uchar *bits,
    constant uint &nx, constant uint &ny, constant uint &nz,
    constant float &iso, uint i [[thread_position_in_grid]]) {
    if (i >= nx*ny*nz) return;
    uint x=i/(ny*nz), y=(i/nz)%ny, z=i%nz;
    codes[i]=0; bits[i]=0;
    if (x+1>=nx || y+1>=ny || z+1>=nz) return;
    int code=0;
    for (int c=0; c<8; ++c) {
        uint at=i+corners[3*c]*ny*nz+corners[3*c+1]*nz+corners[3*c+2];
        if (grid[at]>=iso) code |= 1<<c;
    }
    codes[i]=uchar(code);
    if (code==0 || code==255) return;
    float a=grid[i];
    uint strides[3]={ny*nz,nz,1};
    int b=0;
    for (int d=0; d<3; ++d) {
        float v=grid[i+strides[d]];
        // Keep the original NaN behavior used by the adaptive TripoSG grid.
        if ((a<iso && v>=iso) || (v<iso && a>=iso)) b |= 1<<d;
    }
    bits[i]=uchar(b);
}

kernel void resolve_dual_cells(
    device const long *cells, device const uchar *codes,
    device const int *problematic, device const int *first_patch,
    device int *good_codes, device int *counts,
    constant uint &nx, constant uint &ny, constant uint &nz,
    constant uint &n, uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    long cell=cells[i]; int code=codes[cell];
    int direction=problematic[code];
    if (direction!=255) {
        int axis=direction>>1, delta=(direction&1) ? 1:-1;
        long strides[3]={long(ny)*nz,long(nz),1};
        int xyz[3]={int(cell/(ny*nz)),int((cell/nz)%ny),int(cell%nz)};
        int dims[3]={int(nx),int(ny),int(nz)};
        int next=xyz[axis]+delta;
        if (next>=0 && next<dims[axis]-1 && problematic[codes[cell+delta*strides[axis]]]!=255)
            code ^= 255;
    }
    good_codes[i]=code;
    counts[i]=first_patch[code+1]-first_patch[code];
}

kernel void emit_dual_quads(
    device const long *origins, device const long *axes, device const float *grid,
    device const int *cell_to_used, device const int *good_codes,
    device const long *vertex_first, device const int *quad_table,
    device const int *edge_offset, device long *quads,
    constant uint &ny, constant uint &nz, constant float &iso,
    constant uint &n, uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    long origin=origins[i]; int type=int(axes[i])+(grid[origin]>=iso ? 3:0);
    for (int j=0; j<4; ++j) {
        int at=(type*4+j)*4;
        long cell=origin+long(quad_table[at])*ny*nz+long(quad_table[at+1])*nz+quad_table[at+2];
        int used=cell_to_used[cell];
        quads[4*i+j]=vertex_first[used]+edge_offset[good_codes[used]*12+quad_table[at+3]];
    }
}
