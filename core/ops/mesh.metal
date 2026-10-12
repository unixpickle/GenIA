#include <metal_stdlib>
using namespace metal;

inline float edge(float2 p, float2 a, float2 b) {
    return (p.x-a.x)*(b.y-a.y)-(p.y-a.y)*(b.x-a.x);
}
inline float line_dist(float2 p, float2 a, float2 b) {
    float2 v=b-a;
    float l=dot(v,v);
    if(l<=1e-8f) return dot(p-b,p-b);
    float2 d=p-(a+clamp(dot(p-a,v)/l,0.0f,1.0f)*v);
    return dot(d,d);
}
inline bool candidate(const device packed_float3 *faces, int id, float2 p,
                      float blur, uint perspective, uint clip, uint cull, uint inclusive,
                      thread float &z, thread float &distance) {
    float3 a=faces[id*3], b=faces[id*3+1], c=faces[id*3+2];
    float area=edge(c.xy,a.xy,b.xy);
    if(abs(area)<=1e-8f || min(a.z,min(b.z,c.z))<1e-8f || (cull && area<0)) return false;
    float3 bary=float3(edge(p,b.xy,c.xy),edge(p,c.xy,a.xy),edge(p,a.xy,b.xy))/(area+1e-8f);
    if(perspective) {
        bary*=float3(b.z*c.z,a.z*c.z,a.z*b.z);
        float total=bary.x+bary.y+bary.z;
        if(abs(total)<=1e-12f) return false;
        bary/=total;
    }
    bool inside=inclusive ? all(bary>=0) : all(bary>0);
    if(clip) { bary=max(bary,0.0f); bary/=max(bary.x+bary.y+bary.z,1e-5f); }
    z=dot(bary,float3(a.z,b.z,c.z));
    distance=min(line_dist(p,a.xy,b.xy), min(line_dist(p,a.xy,c.xy),line_dist(p,b.xy,c.xy)));
    return z>=0 && (inside || distance<blur);
}

kernel void select_faces(const device packed_float3 *faces, const device int *neighbors,
                         const device int2 *ranges, const device int *ids,
                         device int *output, constant uint &width, constant uint &height,
                         constant uint &K, constant float &blur, constant uint &perspective,
                         constant uint &clip, constant uint &cull, constant uint &inclusive, uint pixel [[thread_position_in_grid]]) {
    uint x=pixel%width,y=pixel/width;
    float2 p=float2(float(width)-1-2*float(x),float(height)-1-2*float(y))/float(min(width,height));
    int2 range=ranges[(y/16)*((width+15)/16)+x/16];
    float depths[64]; int selected[64];
    for(uint j=0;j<K;++j) {depths[j]=INFINITY; selected[j]=-1;}
    for(int j=range.x;j<range.y;++j) {
        int id=ids[j]; float z,d;
        if(!candidate(faces,id,p,blur,perspective,clip,cull,inclusive,z,d)) continue;
        int neighbor=neighbors[id];
        if(neighbor>=0) {
            float nz,nd;
            if(candidate(faces,neighbor,p,blur,perspective,clip,cull,inclusive,nz,nd) &&
               (nd<d || (nd==d && neighbor<id))) continue;
        }
        uint pos=K;
        for(uint i=0;i<K;++i) {
            if(z<depths[i] || (z==depths[i] && id<selected[i])) {pos=i;break;}
        }
        if(pos==K) continue;
        for(uint i=K-1;i>pos;--i) {depths[i]=depths[i-1];selected[i]=selected[i-1];}
        depths[pos]=z;selected[pos]=id;
    }
    for(uint j=0;j<K;++j) output[pixel*K+j]=selected[j];
}
