"""Run the actual viewer's GLB parser in JavaScriptCore or Node, without a browser."""
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "core/webui/static"
JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


def run_js(tmp_path, code):
    node = shutil.which("node")
    if node:
        executable = node
        prelude = "const readText = p => require('fs').readFileSync(p, 'utf8'); const readBytes = p => new Uint8Array(require('fs').readFileSync(p)); const print=console.log;\n"
    elif JSC.exists():
        executable = str(JSC)
        prelude = """const readText = p => readFile(p); const readBytes = p => readFile(p,'binary');
class TextDecoder { decode(bytes) { return decodeURIComponent(Array.from(bytes,b=>'%'+b.toString(16).padStart(2,'0')).join('')); } }
"""
    else:
        pytest.skip("Node or JavaScriptCore required for viewer tests")
    script = tmp_path / "viewer-check.js"
    script.write_text(prelude + "\nconst assert=(x,m)=>{if(!x)throw new Error(m || 'Assertion failed');};\n" + code)
    result = subprocess.run([executable, str(script)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def tiny_glb(tmp_path, **updates):
    # Interleaved POSITION and COLOR_0, with normalized uint8 colors. A nested
    # node applies translation/scale before the viewer centers the whole scene.
    binary = b"".join(struct.pack("<fffBBBB", *pos, *color) for pos, color in [
        ((0, 0, 0), (255, 0, 0, 255)), ((1, 0, 0), (0, 255, 0, 255)),
        ((0, 1, 0), (0, 0, 255, 255))]) + struct.pack("<HHH", 0, 1, 2) + b"\0\0"
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48, "byteStride": 16},
                           {"buffer": 0, "byteOffset": 48, "byteLength": 6}],
           "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                         {"bufferView": 0, "byteOffset": 12, "componentType": 5121, "count": 3, "type": "VEC4", "normalized": True},
                         {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
           "nodes": [{"children": [1], "translation": [20, 30, 40]}, {"mesh": 0, "scale": [2, 2, 2]}],
           "scenes": [{"nodes": [0]}], "scene": 0, **updates}
    payload = json.dumps(doc).encode()
    payload += b" " * (-len(payload) % 4)
    blob = struct.pack("<III", 0x46546C67, 2, 28 + len(payload) + len(binary))
    blob += struct.pack("<II", len(payload), 0x4E4F534A) + payload
    blob += struct.pack("<II", len(binary), 0x004E4942) + binary
    path = tmp_path / "test.glb"; path.write_bytes(blob)
    return path


def parser_code():
    return f"const parseGLB = new Function(readText({json.dumps(str(STATIC / 'viewer.js'))}).replaceAll('export ', '') + '; return parseGLB;')();\n"


def test_javascript_syntax(tmp_path):
    run_js(tmp_path, "\n".join(f"new Function(readText({json.dumps(str(path))}).replaceAll('export ', ''));" for path in STATIC.glob("*.js")))


def test_actual_glb_parser_colors_normals_and_scene_transform(tmp_path):
    path = tiny_glb(tmp_path)
    run_js(tmp_path, parser_code() + f"""
const meshes=parseGLB(readBytes({json.dumps(str(path))}).buffer);
assert(meshes.length===1);
const m=meshes[0];
assert(JSON.stringify(Array.from(m.positions))==='[-1,-1,0,1,-1,0,-1,1,0]');
assert(JSON.stringify(Array.from(m.colors))==='[1,0,0,0,1,0,0,0,1]');
assert(JSON.stringify(Array.from(m.normals))==='[0,0,1,0,0,1,0,0,1]');
assert(JSON.stringify(Array.from(m.indices))==='[0,1,2]');
""")


@pytest.mark.parametrize("updates", [{"extensionsRequired": ["KHR_draco_mesh_compression"]},
                                     {"nodes": [{"children": [0]}]}, {"scenes": []}])
def test_glb_rejects_unsupported_or_invalid_scenes(tmp_path, updates):
    path = tiny_glb(tmp_path, **updates)
    run_js(tmp_path, parser_code() + f"""
let threw=false;
try {{ parseGLB(readBytes({json.dumps(str(path))}).buffer); }} catch(e) {{ threw=true; }}
assert(threw, 'Invalid GLB must be rejected');
""")


def test_glb_rejects_truncated_files(tmp_path):
    path = tiny_glb(tmp_path)
    run_js(tmp_path, parser_code() + f"""
const bytes=readBytes({json.dumps(str(path))});
for(const length of [0,12,19,bytes.length-1]) {{
 let threw=false; try {{ parseGLB(bytes.slice(0,length).buffer); }} catch(e) {{ threw=true; }}
 assert(threw,'Truncated GLB must be rejected');
}}
""")
