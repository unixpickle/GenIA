/* Offline viewer for GenIA's uncompressed, vertex-colored GLB mesh exports.
 * No external libraries, decoders, telemetry, or background model requests.
 */
const widths = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16};
const types = {5120: [1, "getInt8", 127], 5121: [1, "getUint8", 255], 5122: [2, "getInt16", 32767],
  5123: [2, "getUint16", 65535], 5125: [4, "getUint32", 4294967295], 5126: [4, "getFloat32", 1]};
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  }
  return out;
}
function nodeMatrix(node) {
  if (node.matrix) return node.matrix;
  const [x, y, z, w] = node.rotation || [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale || [1, 1, 1];
  const [tx, ty, tz] = node.translation || [0, 0, 0];
  return [(1 - 2*y*y - 2*z*z)*sx, (2*x*y + 2*z*w)*sx, (2*x*z - 2*y*w)*sx, 0,
    (2*x*y - 2*z*w)*sy, (1 - 2*x*x - 2*z*z)*sy, (2*y*z + 2*x*w)*sy, 0,
    (2*x*z + 2*y*w)*sz, (2*y*z - 2*x*w)*sz, (1 - 2*x*x - 2*y*y)*sz, 0, tx, ty, tz, 1];
}

export function parseGLB(buffer) {
  const header = new DataView(buffer);
  if (buffer.byteLength < 20 || header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2 || header.getUint32(8, true) !== buffer.byteLength) throw new Error("Invalid GLB file.");
  let doc, binary;
  for (let offset = 12; offset < buffer.byteLength;) {
    if (offset + 8 > buffer.byteLength) throw new Error("Truncated GLB chunk.");
    const length = header.getUint32(offset, true), type = header.getUint32(offset + 4, true);
    offset += 8;
    if (offset + length > buffer.byteLength) throw new Error("Truncated GLB data.");
    if (type === 0x4e4f534a) doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset, length)));
    if (type === 0x004e4942) binary = new DataView(buffer, offset, length);
    offset += length;
  }
  if (!doc || !binary) throw new Error("GLB has no mesh data.");
  if (doc.extensionsRequired?.length) throw new Error("This mesh needs an unsupported GLB extension. Download it to use an external viewer.");
  function accessor(index) {
    const a = doc.accessors[index], view = doc.bufferViews[a.bufferView];
    const type = types[a.componentType], width = widths[a.type];
    if (!type || !width || a.sparse || view.buffer !== 0) throw new Error("Unsupported mesh accessor.");
    const offset = (view.byteOffset || 0) + (a.byteOffset || 0), stride = view.byteStride || type[0] * width;
    if (!Number.isSafeInteger(a.count) || a.count < 0 || offset + (a.count - 1) * stride + width * type[0] > binary.byteLength) throw new Error("Invalid mesh buffer bounds.");
    const out = new Float32Array(a.count * width);
    for (let i = 0; i < a.count; i++) for (let c = 0; c < width; c++) {
      const value = binary[type[1]](offset + i * stride + c * type[0], true);
      out[i * width + c] = a.normalized ? Math.max(-1, value / type[2]) : value;
    }
    return {data: out, width, count: a.count};
  }
  const meshes = [], low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  function visit(index, parent, ancestors = new Set()) {
    if (ancestors.has(index)) throw new Error("Cyclic GLB scene.");
    const node = doc.nodes[index], matrix = multiply(parent, nodeMatrix(node));
    if (node.mesh !== undefined) for (const primitive of doc.meshes[node.mesh].primitives) {
      if ((primitive.mode ?? 4) !== 4) continue;
      const pos = accessor(primitive.attributes.POSITION);
      if (pos.width !== 3) throw new Error("Invalid vertex positions.");
      const positions = pos.data;
      const indices = primitive.indices === undefined ? Uint32Array.from({length: pos.count}, (_, i) => i) : Uint32Array.from(accessor(primitive.indices).data);
      if (indices.length % 3 || indices.some(i => i >= pos.count)) throw new Error("Invalid triangle indices.");
      const color = primitive.attributes.COLOR_0 === undefined ? null : accessor(primitive.attributes.COLOR_0);
      if (color && (color.count !== pos.count || ![3,4].includes(color.width))) throw new Error("Invalid vertex colors.");
      const material = doc.materials?.[primitive.material]?.pbrMetallicRoughness;
      if (material?.baseColorTexture) throw new Error("This viewer supports GenIA vertex-colored meshes. Download textured meshes to view externally.");
      const base = material?.baseColorFactor || [0.72, 0.79, 0.65, 1];
      const colors = new Float32Array(pos.count * 3), normals = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        const x = positions[i*3], y = positions[i*3+1], z = positions[i*3+2];
        for (let c = 0; c < 3; c++) {
          const value = matrix[c]*x + matrix[4+c]*y + matrix[8+c]*z + matrix[12+c];
          if (!Number.isFinite(value)) throw new Error("Invalid mesh coordinate.");
          positions[i*3+c] = value; low[c] = Math.min(low[c], value); high[c] = Math.max(high[c], value);
          colors[i*3+c] = color ? color.data[i*color.width+c] * (material?.baseColorFactor?.[c] ?? 1) : base[c];
        }
      }
      for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i]*3, b = indices[i+1]*3, c = indices[i+2]*3;
        const ux = positions[b]-positions[a], uy = positions[b+1]-positions[a+1], uz = positions[b+2]-positions[a+2];
        const vx = positions[c]-positions[a], vy = positions[c+1]-positions[a+1], vz = positions[c+2]-positions[a+2];
        const normal = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx];
        for (const index of [a, b, c]) for (let j = 0; j < 3; j++) normals[index+j] += normal[j];
      }
      for (let i = 0; i < normals.length; i += 3) {
        const length = Math.hypot(normals[i], normals[i+1], normals[i+2]) || 1;
        for (let c = 0; c < 3; c++) normals[i+c] /= length;
      }
      meshes.push({positions, normals, colors, indices});
    }
    const next = new Set(ancestors); next.add(index);
    for (const child of node.children || []) visit(child, matrix, next);
  }
  const scene = doc.scenes?.[doc.scene || 0];
  if (!scene) throw new Error("GLB has no scene.");
  for (const root of scene.nodes || []) visit(root, identity());
  if (!meshes.length) throw new Error("GLB has no triangles.");
  const center = low.map((v, i) => (v + high[i]) / 2);
  const scale = Math.max(...high.map((v, i) => v - low[i])) || 1;
  for (const mesh of meshes) for (let i = 0; i < mesh.positions.length; i++) mesh.positions[i] = (mesh.positions[i] - center[i % 3]) * 2 / scale;
  return meshes;
}

const vertexShader = `#version 300 es
in vec3 position; in vec3 normal; in vec3 color;
uniform mat4 transform; uniform mat4 rotation;
out vec3 vNormal; out vec3 vColor;
void main(){ gl_Position = transform * vec4(position,1.); vNormal = mat3(rotation)*normal; vColor = color; }`;
const fragmentShader = `#version 300 es
precision highp float; in vec3 vNormal; in vec3 vColor; out vec4 pixel;
void main(){ vec3 n=normalize(vNormal); if(!gl_FrontFacing)n=-n;
float light=.63+.27*max(dot(n,normalize(vec3(-.5,.8,1.))),0.)+.1*max(dot(n,normalize(vec3(.8,.2,-.5))),0.);
pixel=vec4(vColor*light,1.); }`;

export async function createViewer(container, url, signal) {
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error("Could not download this mesh.");
  const meshes = parseGLB(await response.arrayBuffer());
  signal.throwIfAborted();
  const canvas = document.createElement("canvas"); canvas.className = "viewer-canvas";
  canvas.tabIndex = 0; canvas.setAttribute("aria-label", "3D object. Drag or use arrow keys to rotate; scroll or use plus and minus to zoom. Double click or press R to reset.");
  const gl = canvas.getContext("webgl2", {antialias: true, alpha: true});
  if (!gl) throw new Error("WebGL 2 is unavailable in this browser. You can still download the GLB.");
  const resources = [], shaders = [];
  let program, observer;
  let disposed = false;
  function dispose() {
    if (disposed) return; disposed = true;
    observer?.disconnect(); for (const buffer of resources) gl.deleteBuffer(buffer);
    for (const shader of shaders) gl.deleteShader(shader);
    if (program) gl.deleteProgram(program);
    gl.getExtension("WEBGL_lose_context")?.loseContext(); canvas.remove(); hint.remove();
  }
  const hint = document.createElement("span"); hint.className = "viewer-label"; hint.textContent = "Drag to orbit · Scroll to zoom · Double-click to reset";
  try {
    function compile(type, source) {
      const shader = gl.createShader(type); shaders.push(shader); gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error("Could not initialize the 3D renderer.");
      return shader;
    }
    program = gl.createProgram(); gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexShader));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentShader)); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("Could not initialize the 3D renderer.");
    const positions = gl.getAttribLocation(program, "position"), normals = gl.getAttribLocation(program, "normal"), colors = gl.getAttribLocation(program, "color");
    const transform = gl.getUniformLocation(program, "transform"), rotation = gl.getUniformLocation(program, "rotation");
    function buffer(target, data) {
      const value = gl.createBuffer(); resources.push(value); gl.bindBuffer(target, value); gl.bufferData(target, data, gl.STATIC_DRAW); return value;
    }
    const geometry = meshes.map(m => ({position: buffer(gl.ARRAY_BUFFER, m.positions), normal: buffer(gl.ARRAY_BUFFER, m.normals), color: buffer(gl.ARRAY_BUFFER, m.colors), indices: buffer(gl.ELEMENT_ARRAY_BUFFER, m.indices), count: m.indices.length}));
    let yaw = 0.35, pitch = 0.12, distance = 4.8;
    function draw() {
      if (disposed) return;
      const rect = container.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * ratio)); canvas.height = Math.max(1, Math.round(rect.height * ratio));
      gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.useProgram(program);
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const rot = multiply([1,0,0,0, 0,cp,sp,0, 0,-sp,cp,0, 0,0,0,1], [cy,0,-sy,0, 0,1,0,0, sy,0,cy,0, 0,0,0,1]);
      const view = identity(); view[14] = -distance;
      const aspect = canvas.width/canvas.height, f = 1/Math.tan(Math.PI/8);
      const projection = [f/aspect,0,0,0, 0,f,0,0, 0,0,-100.01/99.99,-1, 0,0,-2/99.99,0];
      gl.uniformMatrix4fv(transform, false, multiply(projection, multiply(view, rot))); gl.uniformMatrix4fv(rotation, false, rot);
      for (const mesh of geometry) {
        for (const [location, data] of [[positions, mesh.position], [normals, mesh.normal], [colors, mesh.color]]) {
          gl.bindBuffer(gl.ARRAY_BUFFER, data); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
        }
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.indices); gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
      }
    }
    function zoom(delta) { distance = Math.max(2, Math.min(16, distance * Math.exp(delta))); draw(); }
    const pointers = new Map();
    const separation = () => {
      const [a,b] = [...pointers.values()];
      return a && b ? Math.hypot(a[0]-b[0], a[1]-b[1]) : 0;
    };
    canvas.addEventListener("pointerdown", event => { pointers.set(event.pointerId, [event.clientX, event.clientY]); canvas.setPointerCapture(event.pointerId); });
    canvas.addEventListener("pointermove", event => {
      const previous = pointers.get(event.pointerId);
      if (!previous) return;
      const before = separation();
      pointers.set(event.pointerId, [event.clientX, event.clientY]);
      const after = separation();
      if (before > 0 && after > 0) zoom(Math.log(before/after));
      else { yaw += (event.clientX-previous[0])*.01; pitch += (event.clientY-previous[1])*.01; draw(); }
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(type, event => pointers.delete(event.pointerId));
    canvas.addEventListener("wheel", event => { event.preventDefault(); zoom(Math.max(-.3, Math.min(.3, event.deltaY*.002))); }, {passive: false});
    function reset() { yaw=.35; pitch=.12; distance=4.8; draw(); }
    canvas.addEventListener("dblclick", reset);
    canvas.addEventListener("keydown", event => {
      const actions = {ArrowLeft:()=>{yaw-=.1;},ArrowRight:()=>{yaw+=.1;},ArrowUp:()=>{pitch-=.1;},ArrowDown:()=>{pitch+=.1;},"+":()=>zoom(-.1),"-":()=>zoom(.1),r:reset};
      if (actions[event.key]) { event.preventDefault(); actions[event.key](); draw(); }
    });
    container.append(canvas, hint); observer = new ResizeObserver(draw); observer.observe(container); draw();
    return {dispose};
  } catch (error) { dispose(); throw error; }
}
