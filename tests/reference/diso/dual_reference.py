"""Scalar transcription of DISO 0.1.4 CUDA dual MC for independent tests.

Tables are read from the unchanged upstream source, not production Python.
Xinyue Wei, CC BY-NC 4.0; see LICENSE in this directory.
"""
import ast
from functools import lru_cache
from pathlib import Path
import re

import numpy as np


@lru_cache(None)
def tables():
    source = re.sub(r'//[^\n]*', '', Path(__file__).with_name('cudualmc.cu').read_text())
    result = []
    for name in ['mcCorners', 'mcEdgeLocations', 'mcFirstPatchIndex', 'mcFirstEdgeIndex',
                 'mcEdgeIndex', 'problematicConfigs', 'dmcEdgeOffset', 'dmcQuad']:
        value = re.search(r'__constant__[^;]*?\b'+name+r'[^=]*=\s*(\{.*?\});', source, re.S).group(1)
        result.append(ast.literal_eval(value.replace('{', '[').replace('}', ']')))
    return result


def extract(grid, deform=None, iso=0., normalize=True):
    corners, edges, first_patch, first_edge, edge_index, problematic, edge_offset, quad_table = tables()
    data = np.pad(grid, 1, constant_values=iso+1)
    d = None if deform is None else np.pad(deform, [(1, 1)]*3+[(0, 0)])
    codes = {}
    for xyz in np.ndindex(tuple(n-1 for n in data.shape)):
        codes[xyz] = sum(1 << c for c, delta in enumerate(corners)
                         if data[tuple(a+b for a, b in zip(xyz, delta))] >= iso)
    vertices, crossings = [], []
    mapping = {}
    for xyz, raw_code in codes.items():
        if raw_code in (0, 255):
            continue
        code = raw_code
        direction = problematic[code]
        if direction != 255:
            neighbor = list(xyz)
            neighbor[direction >> 1] += 1 if direction & 1 else -1
            neighbor = tuple(neighbor)
            if neighbor in codes and problematic[codes[neighbor]] != 255:
                code ^= 255
        mapping[xyz] = (len(vertices), code)
        for patch in range(first_patch[code], first_patch[code+1]):
            position = np.zeros(3, dtype=grid.dtype)
            n = 0
            for edge in edge_index[first_edge[patch]:first_edge[patch+1]]:
                *delta, axis = edges[edge]
                start = tuple(a+b for a, b in zip(xyz, delta))
                end = list(start); end[axis] += 1; end = tuple(end)
                a, b = data[start], data[end]
                t = np.clip((iso-a)/(b-a), 0, 1) if b != a else .5
                if np.isnan(t):
                    t = 0.
                p0, p1 = np.array(start, dtype=grid.dtype), np.array(end, dtype=grid.dtype)
                if d is not None:
                    p0 += d[start]; p1 += d[end]
                position += p0+(p1-p0)*t
                n += 1
            vertices.append(position/n)
        for axis in range(3):
            end = list(xyz); end[axis] += 1
            a, b = data[xyz], data[tuple(end)]
            if (a < iso and b >= iso) or (b < iso and a >= iso):
                crossings.append((xyz, axis+(3 if a >= iso else 0)))
    quads = []
    for xyz, kind in crossings:
        face = []
        for dx, dy, dz, eid in quad_table[kind]:
            start, code = mapping[(xyz[0]+dx, xyz[1]+dy, xyz[2]+dz)]
            face.append(start+edge_offset[code][eid])
        quads.append(face)
    vertices = np.array(vertices).reshape(-1, 3)-1
    if normalize:
        vertices = vertices/(np.array(grid.shape)-1)
    return vertices, np.array(quads, dtype=np.int64).reshape(-1, 4)


def triangulate(vertices, quads):
    groups = [[], []]
    for q in quads:
        scores = []
        for pattern in (((0, 1, 3), (1, 2, 3)), ((0, 1, 2), (0, 2, 3))):
            cosines = []
            for face in pattern:
                triangle = vertices[q[list(face)]]
                for i in range(3):
                    a, b = triangle[(i+1)%3]-triangle[i], triangle[(i+2)%3]-triangle[i]
                    a /= max(np.linalg.norm(a), 1e-12)
                    b /= max(np.linalg.norm(b), 1e-12)
                    cosines.append((a*b).sum())
            scores.append(max(cosines))
        first = scores[0] < scores[1]
        groups[0 if first else 1].extend(q[[0, 1, 3, 1, 2, 3] if first else [0, 1, 2, 0, 2, 3]].reshape(2, 3))
    return np.array(groups[0]+groups[1], dtype=np.int64).reshape(-1, 3)
