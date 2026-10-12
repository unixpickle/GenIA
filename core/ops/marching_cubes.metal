// DISO 0.1.4 topology rewritten for Metal. Xinyue Wei, CC BY-NC 4.0.
// See vendor/licenses/diso.LICENSE. Vertex interpolation uses native autograd.
#include <metal_stdlib>
using namespace metal;

kernel void classify_cells(
    device const float *grid, device const int *corners,
    device int *codes, device int *bits, device int *counts,
    constant uint &nx, constant uint &ny, constant uint &nz,
    constant float &iso, uint i [[thread_position_in_grid]]) {
    if (i >= nx*ny*nz) return;
    uint x=i/(ny*nz), y=(i/nz)%ny, z=i%nz;
    codes[i]=0; bits[i]=0; counts[i]=0;
    if (x+1>=nx || y+1>=ny || z+1>=nz) return;
    int code=0;
    for (uint c=0; c<8; ++c) {
        uint at=(x+corners[3*c])*ny*nz+(y+corners[3*c+1])*nz+z+corners[3*c+2];
        if (grid[at]>=iso) code |= 1<<c;
    }
    codes[i]=code;
    if (code==0 || code==255) return;
    bool sign=grid[i]>=iso;
    int b=((grid[i+ny*nz]>=iso)!=sign ? 1:0)
         |((grid[i+nz]>=iso)!=sign ? 2:0)
         |((grid[i+1]>=iso)!=sign ? 4:0);
    bits[i]=b; counts[i]=popcount(uint(b));
}

kernel void emit_faces(
    device const long *cells, device const int *codes, device const int *bits,
    device const long *vertex_first, device const long *face_first,
    device const int *first, device const int *triangles, device const int *edges,
    device long *faces, constant uint &ny, constant uint &nz,
    constant uint &n, uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    long cell=cells[i]; int code=codes[cell];
    int begin=first[code], end=first[code+1];
    for (int j=begin; j<end; ++j) {
        int e=triangles[j];
        long origin=cell+edges[4*e]*ny*nz+edges[4*e+1]*nz+edges[4*e+2];
        int axis=edges[4*e+3];
        faces[face_first[i]+j-begin]=vertex_first[origin]+popcount(uint(bits[origin]&((1<<axis)-1)));
    }
}
