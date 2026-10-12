"""Compare browser decoding with the actual exporter, and exercise publication/sorting."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_webui_viewer import STATIC, run_js
from test_webui import enqueue
from genia.core.webui.app import create_app
from genia.core.webui.artifacts import collect
from gsconverter.formats.compressed_ply import CompressedPlyFormat
from gsconverter.utils.ply_utils import PlyData, PlyElement


def data_code():
    return f"const SplatData=new Function(readText({json.dumps(str(STATIC / 'splat-data.js'))})+';return SplatData;')();\n"


def gaussian_data(count=517):
    rng = np.random.default_rng(19)
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    data = np.zeros(count, dtype=[(n, '<f4') for n in names])
    for n in ['x', 'y', 'z']: data[n] = rng.uniform(-1, 1, count)
    for n in ['f_dc_0', 'f_dc_1', 'f_dc_2']: data[n] = rng.uniform(-1.7, 1.7, count)
    data['opacity'] = rng.uniform(-2, 3, count)
    for n in ['scale_0', 'scale_1', 'scale_2']: data[n] = rng.uniform(-4, -2, count)
    q = rng.normal(size=(count, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(4): data[f'rot_{i}'] = q[:, i]
    return data


def reference_texture(data):
    positions = np.stack([data[n] for n in ['x', 'y', 'z']], -1).astype(float)
    q = np.stack([data[f'rot_{i}'] for i in range(4)], -1).astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                         2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                         2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)
    scales = np.exp(np.stack([data[f'scale_{i}'] for i in range(3)], -1).astype(float))
    matrix = rotation * scales[:, None, :]
    cov = matrix @ matrix.transpose(0, 2, 1)
    alpha = 1 / (1 + np.exp(-data['opacity'].astype(float)))
    visible = alpha >= 1/255
    radii = 3 * np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    low = (positions-radii)[visible].min(0); high = (positions+radii)[visible].max(0)
    center = (low+high)/2; factor = 2/max((high-low).max(), 1e-6)
    positions = (positions-center)*factor; cov *= factor**2
    texture = np.zeros((len(data), 16), np.float32)
    texture[:, :3] = positions; texture[:, 3] = alpha
    texture[:, 4:7] = cov[:, 0, :]
    texture[:, 8:11] = cov[:, [1, 1, 2], [1, 2, 2]]
    texture[:, 12:15] = np.clip(np.stack([data[f'f_dc_{i}'] for i in range(3)], -1)*.28209479177387814+.5, 0, 1)
    return texture


@pytest.mark.parametrize('compressed', [False, True])
def test_decoder_matches_python_exporter(tmp_path, compressed):
    data = gaussian_data()
    path = tmp_path / 'test.ply'
    if compressed:
        codec = CompressedPlyFormat(); codec.write(data, str(path)); data = codec.read(str(path))
    else:
        PlyData([PlyElement.describe(data, 'vertex')], text=False, byte_order='<').write(path)
    actual = json.loads(run_js(tmp_path, data_code() + f"""
const data=SplatData.decode(readBytes({json.dumps(str(path))}).buffer);
assert(data.count===517 && data.hasSH===false);
print(JSON.stringify(Array.from(data.texture.slice(0,data.count*16))));
"""))
    np.testing.assert_allclose(np.array(actual).reshape(-1,16), reference_texture(data), atol=3e-7, rtol=2e-4)


def test_covariance_axes_and_depth_sort_order(tmp_path):
    run_js(tmp_path, data_code() + """
const q=[Math.cos(Math.PI/8),0,0,Math.sin(Math.PI/8)];
const c=SplatData.covariance(q,[2,1,3]);
assert(Math.abs(c[0]-2.5)<1e-7 && Math.abs(c[1]-1.5)<1e-7 && c[5]===9);
const sort=SplatData.createSorter(new Float32Array([0,0,3, 0,0,-2, 1,0,-2, 0,0,0]));
assert(JSON.stringify(Array.from(sort([0,0,1])))==='[1,2,3,0]','Must draw far to near, with stable ties');
assert(JSON.stringify(Array.from(sort([0,0,-1])))==='[0,3,1,2]','Camera reversal must reverse depth order');
assert(JSON.stringify(Array.from(sort([0,1,0])))==='[0,1,2,3]','All equal depths must not produce NaNs');
""")


def test_decoder_rejects_truncated_and_nonfinite_data(tmp_path):
    path = tmp_path / 'test.ply'
    CompressedPlyFormat().write(gaussian_data(3), str(path))
    run_js(tmp_path, data_code() + f"""
const bytes=readBytes({json.dumps(str(path))});
const invalid=buffer=>{{let failed=false;try{{SplatData.decode(buffer);}}catch(e){{failed=true;}}assert(failed);}};
for(const n of [0,10,100,bytes.length-1])invalid(bytes.slice(0,n).buffer);
const bad=bytes.slice();
const text=String.fromCharCode(...bad.slice(0,2000));
const offset=text.indexOf('end_header')+11;
assert(offset>11,'Fixture header not found');
new DataView(bad.buffer).setFloat32(offset,NaN,true);invalid(bad.buffer);
""")


def test_old_and_new_jobs_publish_splats_with_ranges(tmp_path):
    app = create_app(tmp_path); store = app.extensions['genia_store']; client = app.test_client()
    job = enqueue(store); directory = store.jobs_dir/job['id']; final = directory/'output/final'
    (final/'meshes').mkdir(parents=True); (final/'meshes/001.glb').write_bytes(b'glTF')
    # Simulate a history entry created before splats were added to artifact metadata.
    store.claim(); store.finish(job['id'], 'succeeded', assets=collect(directory))
    (final/'gaussians').mkdir()
    payload = b'ply\n' + b'0'*100
    for name in ['world.compressed.ply', '001.compressed.ply', 'background.compressed.ply']:
        (final/'gaussians'/name).write_bytes(payload)
    outside = tmp_path/'secret.ply'; outside.write_bytes(b'secret')
    (final/'gaussians/escape.ply').symlink_to(outside)
    data = client.get(f"/api/jobs/{job['id']}").json
    splats = [a for a in data['assets'] if a['kind']=='splat']
    assert [a['name'] for a in splats] == ['001.compressed.ply', 'world.compressed.ply']
    assert all('path' not in a for a in splats)
    for asset in splats:
        response = client.get(asset['url'], headers={'Range':'bytes=0-3'})
        assert response.status_code == 206 and response.data == b'ply\n'
        assert response.mimetype == 'application/octet-stream'
        response = client.get(asset['url']+'?download=1')
        assert 'attachment' in response.headers['Content-Disposition']
    collected = collect(directory)
    assert len([a for a in collected.values() if a['kind']=='splat']) == 2
    # Partial output of a failed job is never discovered/published.
    failed = enqueue(store); store.claim(); store.finish(failed['id'], 'failed')
    (store.jobs_dir/failed['id']/'output').symlink_to(directory/'output', target_is_directory=True)
    assert client.get(f"/api/jobs/{failed['id']}").json['assets'] == []
    assert client.get(f"/api/jobs/{failed['id']}/assets/splat-001.compressed.ply").status_code == 404
    for name in ['splat-viewer.js', 'splat-worker.js', 'splat-data.js']:
        assert client.get('/static/'+name).status_code == 200
    assert "worker-src 'self'" in client.get('/').headers['Content-Security-Policy']


def test_saved_camera_orientation_and_corrupt_pose_fallback(tmp_path):
    from genia.core.webui.artifacts import splat_orientations
    (tmp_path/'output/final').mkdir(parents=True)
    path = tmp_path/'output/final/poses.json'
    q = [np.cos(.3), 0, np.sin(.3), 0]
    path.write_text(json.dumps({'objects':[{'ply':'gaussians/001.compressed.ply',
        'transforms':[{'rotation':q}]}], 'cameras':[{'c2w':np.eye(4).tolist()}]}))
    rotations=splat_orientations(tmp_path)
    c=np.cos(.6); s=np.sin(.6)
    expected=np.diag([-1,1,-1])@np.array([[c,0,s],[0,1,0],[-s,0,c]]).T
    np.testing.assert_allclose(np.array(rotations['001.compressed.ply']).reshape(3,3),expected)
    assert rotations['world.compressed.ply']==[1,0,0,0,-1,0,0,0,-1]
    run_js(tmp_path, f"""
const cameraView=new Function(readText({json.dumps(str(STATIC/'splat-viewer.js'))}).replaceAll('export ','')+';return cameraView;')();
const base={json.dumps(rotations['001.compressed.ply'])};
const v=cameraView(0,0,3.5,base);
for(let r=0;r<3;r++)for(let c=0;c<3;c++)assert(Math.abs(v[c*4+r]-base[r*3+c])<1e-6);
assert(v[14]===-3.5 && v[15]===1);
""")
    path.write_text('{bad')
    assert splat_orientations(tmp_path)=={}


def test_viewer_worker_lifetime_abort_error_and_gpu_cleanup(tmp_path):
    # Exercise the real viewer with controllable I/O. Shader behavior is separately
    # checked by test_webui_splat_render against an actual GLES implementation.
    code = r'''
class DOMException extends Error {constructor(message,name){super(message);this.name=name;}}
class Signal {
  constructor(){this.aborted=false;this.handlers=new Set();}
  addEventListener(_,fn){this.handlers.add(fn);}
  removeEventListener(_,fn){this.handlers.delete(fn);}
  throwIfAborted(){if(this.aborted)throw new DOMException('Aborted','AbortError');}
  abort(){this.aborted=true;for(const h of [...this.handlers])h();}
}
let contexts=0,textureDeletes=0,bufferDeletes=0,draws=0,observerStops=0;
const gl=new Proxy({}, {get(_,key){
  if(key==='getShaderParameter'||key==='getProgramParameter')return ()=>true;
  if(key==='getParameter')return ()=>2048;
  if(key==='getError')return ()=>0;
  if(key==='NO_ERROR')return 0;
  if(key==='getExtension')return ()=>({loseContext(){contexts++;}});
  if(key==='deleteTexture')return ()=>textureDeletes++;
  if(key==='deleteBuffer')return ()=>bufferDeletes++;
  if(key==='drawArraysInstanced')return ()=>draws++;
  if(key==='getAttribLocation')return (_,name)=>name==='corner'?0:1;
  if(/^[A-Z_0-9]+$/.test(key))return 1;
  return ()=>({});
}});
class Node {
  constructor(){this.children=[];this.parentNode=null;this.events={};}
  setAttribute(){} getContext(){return gl;}
  addEventListener(type,fn){this.events[type]=fn;}
  append(...nodes){for(const node of nodes){node.parentNode=this;this.children.push(node);}}
  remove(){if(this.parentNode){this.parentNode.children=this.parentNode.children.filter(n=>n!==this);this.parentNode=null;}}
  getBoundingClientRect(){return {width:300,height:300};}
}
const document={createElement:()=>new Node()};
const workers=[];
class Worker {
  constructor(url){assert(url==='/static/splat-worker.js');this.messages=[];this.terminated=false;workers.push(this);}
  postMessage(data){this.messages.push(data);}
  terminate(){this.terminated=true;}
  emit(data){this.onmessage({data});}
}
class ResizeObserver {observe(){} disconnect(){observerStops++;}}
const requestAnimationFrame=()=>1,cancelAnimationFrame=()=>{};
const fetch=async()=>({ok:true,headers:{get:()=>16},arrayBuffer:async()=>new ArrayBuffer(16)});
const drain=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
'''
    code += f"const source=readText({json.dumps(str(STATIC/'splat-viewer.js'))}).replaceAll('export ','');\n"
    code += r'''
const createViewer=new Function('document','Worker','ResizeObserver','requestAnimationFrame','cancelAnimationFrame','fetch','DOMException',source+';return createViewer;')(
 document,Worker,ResizeObserver,requestAnimationFrame,cancelAnimationFrame,fetch,DOMException);
(async()=>{
 const stage=new Node(),signal=new Signal();
 const result=createViewer(stage,'/asset',signal);await drain();
 const worker=workers[0];assert(worker.messages[0].type==='load');
 worker.emit({type:'loaded',count:1,width:2048,height:1,texture:new Float32Array(8192)});
 assert(worker.messages.length===2 && worker.messages[1].type==='sort');
 worker.emit({type:'sorted',direction:worker.messages[1].direction,order:new Uint32Array([0])});
 const viewer=await result;assert(stage.children.length===2 && draws===1);
 worker.emit({type:'error',message:'Sort failed'});
 assert(stage.children[1].textContent.includes('Sort failed') && worker.terminated);
 viewer.dispose();viewer.dispose();
 assert(worker.terminated && contexts===1 && textureDeletes===1 && bufferDeletes===2);
 assert(stage.children.length===0 && observerStops===1 && signal.handlers.size===0);
 const cancel=new Signal();
 const loading=createViewer(stage,'/other',cancel).then(()=>false,e=>e.name==='AbortError');await drain();
 cancel.abort();assert(await loading,'Closing during decoding did not reject');
 assert(workers[1].terminated && stage.children.length===0);
 workers[1].emit({type:'loaded'});assert(stage.children.length===0,'Late message revived disposed viewer');
 const errors=new Signal();
 const failed=createViewer(stage,'/broken',errors).then(()=>'',e=>e.message);await drain();
 workers[2].emit({type:'error',message:'Invalid PLY'});
 assert(await failed==='Invalid PLY','Worker error was masked by cancellation');
 assert(workers[2].terminated && errors.handlers.size===0);
 print('Viewer lifecycle checks passed');
})().catch(error=>{print(error.stack);if(typeof quit==='function')quit(1);else process.exit(1);});
'''
    assert 'Viewer lifecycle checks passed' in run_js(tmp_path, code)
