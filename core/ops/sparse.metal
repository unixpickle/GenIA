#include <metal_stdlib>
using namespace metal;

inline uint coord_hash(int4 c) {
    uint h = uint(c.x) * 0x8da6b343u ^ uint(c.y) * 0xd8163841u;
    h ^= uint(c.z) * 0xcb1ab31fu ^ uint(c.w) * 0x165667b1u;
    h ^= h >> 16;
    return h;
}

kernel void build_table(const device int4 *coords, device atomic_int *table,
                        constant uint &capacity, uint tid [[thread_position_in_grid]]) {
    uint slot = coord_hash(coords[tid]) & (capacity - 1);
    for (uint probe = 0; probe < capacity; ++probe) {
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(table + slot, &expected, int(tid),
                                                  memory_order_relaxed, memory_order_relaxed)) return;
        // Weak CAS can fail spuriously without a competing insertion.
        if (expected == -1) { --probe; continue; }
        slot = (slot + 1) & (capacity - 1);
    }
}

kernel void neighbor_table(const device int4 *coords, const device int4 *offsets,
                           const device int *table, device int *neighbors,
                           constant uint &N, constant uint &capacity,
                           uint tid [[thread_position_in_grid]]) {
    uint point = tid % N, offset = tid / N;
    int4 query = coords[point] + offsets[offset];
    uint slot = coord_hash(query) & (capacity - 1);
    int result = -1;
    for (uint probe = 0; probe < capacity; ++probe) {
        int candidate = table[slot];
        if (candidate < 0) break;
        if (all(coords[candidate] == query)) { result = candidate; break; }
        slot = (slot + 1) & (capacity - 1);
    }
    neighbors[tid] = result;
}
