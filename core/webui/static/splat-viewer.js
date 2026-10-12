/* WebGL2 Gaussian renderer for GenIA exports. Projection follows core/ops/gaussian.py.
 * Colors are the optimized splat colors, with no mesh lighting or tone mapping.
 * Sorting and decoding live in splat-worker.js; all assets are served locally.
 */
const vertexShader = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 corner;
in uint splatIndex;
uniform sampler2D splats;
uniform int textureWidth;
uniform mat4 view;
uniform vec2 viewport;
uniform float focal;
out vec2 gaussian;
out vec4 color;
vec4 readSplat(int offset) {
  int address = int(splatIndex)*4+offset;
  return texelFetch(splats,ivec2(address%textureWidth,address/textureWidth),0);
}
void main() {
  gl_Position=vec4(2.,2.,2.,1.); gaussian=vec2(0.); color=vec4(0.);
  vec4 p=readSplat(0);
  vec3 camera=(view*vec4(p.xyz,1.)).xyz;
  float depth=-camera.z;
  if(depth<.01 || depth>1000. || p.w<1./255.) return;
  vec3 a=readSplat(1).xyz, b=readSplat(2).xyz;
  mat3 covariance=mat3(a.x,a.y,a.z, a.y,b.x,b.y, a.z,b.y,b.z);
  // Pin-hole Jacobian, clamped outside the image as in the training renderer.
  vec2 slope=clamp(camera.xy/depth,-.65*viewport/focal,.65*viewport/focal);
  vec3 jx=transpose(mat3(view))*vec3(focal/depth,0.,focal*slope.x/depth);
  vec3 jy=transpose(mat3(view))*vec3(0.,focal/depth,focal*slope.y/depth);
  float xx=dot(jx,covariance*jx)+.3;
  float xy=dot(jx,covariance*jy);
  float yy=dot(jy,covariance*jy)+.3;
  float middle=.5*(xx+yy), spread=length(vec2(.5*(xx-yy),xy));
  vec2 eigenvalues=max(vec2(middle+spread,middle-spread),vec2(.0001));
  float angle=.5*atan(2.*xy,xx-yy);
  // atan(0,0) is undefined in GLSL: isotropic footprints have arbitrary axes.
  if(abs(xy)+abs(xx-yy)<1e-7) angle=0.;
  vec2 major=vec2(cos(angle),sin(angle)), minor=vec2(-major.y,major.x);
  float extent=min(3.33,sqrt(max(0.,2.*log(255.*p.w))));
  vec2 radius=extent*sqrt(eigenvalues);
  vec2 center=focal*camera.xy/depth;
  vec2 bounds=abs(major)*radius.x+abs(minor)*radius.y;
  if(any(greaterThan(abs(center)-bounds,viewport*.5))) return;
  vec2 delta=major*radius.x*corner.x+minor*radius.y*corner.y;
  gl_Position=vec4(2.*(center+delta)/viewport,0.,1.);
  gaussian=corner*extent;
  color=vec4(readSplat(3).rgb,p.w);
}`;
const fragmentShader = `#version 300 es
precision highp float;
in vec2 gaussian;
in vec4 color;
out vec4 pixel;
void main() {
  float alpha=min(.99,color.a*exp(-.5*dot(gaussian,gaussian)));
  if(alpha<1./255.) discard;
  pixel=vec4(color.rgb*alpha,alpha);
}`;

// Column-major orbit around the saved input-camera orientation, when available.
export function cameraView(yaw,pitch,distance,base=[1,0,0,0,1,0,0,0,1]) {
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const orbit=[cy,0,sy, sp*sy,cp,-sp*cy, -cp*sy,sp,cp*cy];
  const view=new Float32Array(16);view[14]=-distance;view[15]=1;
  for(let row=0;row<3;row++)for(let col=0;col<3;col++)
    for(let k=0;k<3;k++)view[col*4+row]+=orbit[row*3+k]*base[k*3+col];
  return view;
}

export async function createViewer(container,url,signal,onProgress=()=>{},base) {
  signal.throwIfAborted();
  const canvas=document.createElement('canvas'); canvas.className='viewer-canvas'; canvas.tabIndex=0;
  canvas.setAttribute('aria-label','Gaussian splats. Drag or use arrow keys to orbit, scroll or pinch to zoom. Double-click or press R to reset.');
  const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,depth:false,stencil:false});
  if(!gl) throw new Error('WebGL 2 is unavailable. You can still download the splats.');
  const hint=document.createElement('span'); hint.className='viewer-label';
  hint.textContent='Splats · Drag to orbit · Pinch or scroll to zoom';
  const buffers=[],shaders=[];
  let texture,program,worker,observer,frame=0,disposed=false,contextLost=false;
  let count=0,sortedCount=0,busy=false,lastDirection=null,readyToView=false;
  if(base && (base.length!==9 || !base.every(Number.isFinite)))base=undefined;
  const initialYaw=base?0:.35,initialPitch=base?0:.12,initialDistance=3.5;
  let yaw=initialYaw,pitch=initialPitch,distance=initialDistance;
  let resolveReady,rejectReady;
  const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
  // A worker/abort can reject while fetch is pending. Attach a handler immediately.
  ready.catch(()=>{});
  function dispose() {
    if(disposed)return; disposed=true;
    signal.removeEventListener('abort',abort);
    rejectReady(new DOMException('Viewer closed.','AbortError'));
    cancelAnimationFrame(frame); observer?.disconnect(); worker?.terminate();
    for(const b of buffers)gl.deleteBuffer(b);
    for(const s of shaders)gl.deleteShader(s);
    if(texture)gl.deleteTexture(texture);
    if(program)gl.deleteProgram(program);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    canvas.remove(); hint.remove();
  }
  function abort(){dispose();}
  signal.addEventListener('abort',abort,{once:true});
  function fail(error){
    if(readyToView){
      contextLost=true;cancelAnimationFrame(frame);worker?.terminate();
      hint.textContent=`${error.message} Close 3D and load again.`;
    }else{rejectReady(error);dispose();}
  }
  try {
    function compile(type,source){
      const shader=gl.createShader(type); shaders.push(shader); gl.shaderSource(shader,source); gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error('Could not compile the splat renderer: '+gl.getShaderInfoLog(shader));
      return shader;
    }
    program=gl.createProgram(); gl.attachShader(program,compile(gl.VERTEX_SHADER,vertexShader));
    gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fragmentShader)); gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error('Could not link the splat renderer.');
    gl.useProgram(program);
    function buffer(data,usage){const b=gl.createBuffer();buffers.push(b);gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,usage);return b;}
    buffer(new Float32Array([-1,-1, 1,-1, -1,1, 1,1]),gl.STATIC_DRAW);
    const corner=gl.getAttribLocation(program,'corner');gl.enableVertexAttribArray(corner);gl.vertexAttribPointer(corner,2,gl.FLOAT,false,0,0);
    const indices=buffer(0,gl.DYNAMIC_DRAW),index=gl.getAttribLocation(program,'splatIndex');
    gl.enableVertexAttribArray(index);gl.vertexAttribIPointer(index,1,gl.UNSIGNED_INT,0,0);gl.vertexAttribDivisor(index,1);
    const viewUniform=gl.getUniformLocation(program,'view'),viewportUniform=gl.getUniformLocation(program,'viewport');
    const focalUniform=gl.getUniformLocation(program,'focal');
    const maxTextureSize=gl.getParameter(gl.MAX_TEXTURE_SIZE),width=Math.min(2048,maxTextureSize);
    worker=new Worker('/static/splat-worker.js');
    function direction(){const v=cameraView(yaw,pitch,distance,base);return [v[2],v[6],v[10]];}
    function requestSort(){
      if(disposed || contextLost || busy || !count)return;
      const d=direction();
      if(lastDirection && d.every((v,i)=>Math.abs(v-lastDirection[i])<1e-7))return;
      busy=true;worker.postMessage({type:'sort',direction:d});
    }
    function schedule(){if(!disposed && !contextLost && !frame)frame=requestAnimationFrame(draw);}
    function draw(){
      cancelAnimationFrame(frame);frame=0;if(disposed || contextLost)return;
      const rect=container.getBoundingClientRect();
      // Cap fill rate on dense phone screens, without dropping any splats.
      const ratio=Math.min(globalThis.devicePixelRatio||1,1.5,1400/Math.max(1,rect.width,rect.height));
      const w=Math.max(1,Math.round(rect.width*ratio)),h=Math.max(1,Math.round(rect.height*ratio));
      if(canvas.width!==w)canvas.width=w;if(canvas.height!==h)canvas.height=h;
      gl.viewport(0,0,w,h);gl.clearColor(1,1,1,1);gl.clear(gl.COLOR_BUFFER_BIT);
      if(sortedCount){
        gl.useProgram(program);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);
        gl.uniformMatrix4fv(viewUniform,false,cameraView(yaw,pitch,distance,base));
        gl.uniform2f(viewportUniform,w,h);gl.uniform1f(focalUniform,Math.min(w,h)/(2*Math.tan(Math.PI/8)));
        gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,sortedCount);
      }
      requestSort();
    }
    worker.onmessage=({data})=>{
      if(disposed)return;
      try {
        if(data.type==='error')throw new Error(data.message);
        if(data.type==='loaded'){
          if(data.height>maxTextureSize)throw new Error('This splat model exceeds the device texture limit.');
          texture=gl.createTexture();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,data.width,data.height,0,gl.RGBA,gl.FLOAT,data.texture);
          if(gl.getError()!==gl.NO_ERROR)throw new Error('Not enough GPU memory to display these splats.');
          gl.uniform1i(gl.getUniformLocation(program,'splats'),0);gl.uniform1i(gl.getUniformLocation(program,'textureWidth'),data.width);
          count=data.count;
          if(data.hasSH)hint.textContent+=' · Base colors';
          onProgress('Sorting splats…');requestSort();
        }else if(data.type==='sorted'){
          busy=false;lastDirection=data.direction;
          gl.bindBuffer(gl.ARRAY_BUFFER,indices);gl.bufferData(gl.ARRAY_BUFFER,data.order,gl.DYNAMIC_DRAW);
          sortedCount=data.order.length;
          if(!canvas.parentNode){container.append(canvas,hint);observer=new ResizeObserver(schedule);observer.observe(container);}
          draw();readyToView=true;resolveReady();
          // draw() requests the newest direction if the camera moved during this sort.
        }
      }catch(error){fail(error);}
    };
    worker.onerror=event=>{event.preventDefault();fail(new Error('The splat worker could not run. Close 3D and try again.'));};
    canvas.addEventListener('webglcontextlost',event=>{
      event.preventDefault();if(disposed)return;contextLost=true;worker.terminate();
      hint.textContent='Graphics context lost. Close 3D and load again.';
      rejectReady(new Error(hint.textContent));
    });
    function zoom(delta){distance=Math.max(.3,Math.min(30,distance*Math.exp(delta)));schedule();}
    function reset(){yaw=initialYaw;pitch=initialPitch;distance=initialDistance;schedule();}
    const pointers=new Map();
    const separation=()=>{const [a,b]=[...pointers.values()];return a&&b?Math.hypot(a[0]-b[0],a[1]-b[1]):0;};
    canvas.addEventListener('pointerdown',e=>{pointers.set(e.pointerId,[e.clientX,e.clientY]);canvas.setPointerCapture(e.pointerId);canvas.focus({preventScroll:true});});
    canvas.addEventListener('pointermove',e=>{
      const old=pointers.get(e.pointerId);if(!old)return;
      const before=separation();pointers.set(e.pointerId,[e.clientX,e.clientY]);const after=separation();
      if(before>0&&after>0)zoom(Math.log(before/after));
      else{yaw+=(e.clientX-old[0])*.01;pitch+=(e.clientY-old[1])*.01;schedule();}
    });
    for(const type of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(type,e=>pointers.delete(e.pointerId));
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.max(-.3,Math.min(.3,e.deltaY*.002)));},{passive:false});
    canvas.addEventListener('dblclick',reset);
    canvas.addEventListener('keydown',e=>{
      const actions={ArrowLeft:()=>{yaw-=.1;},ArrowRight:()=>{yaw+=.1;},ArrowUp:()=>{pitch-=.1;},ArrowDown:()=>{pitch+=.1;},'+':()=>zoom(-.1),'-':()=>zoom(.1),r:reset,R:reset};
      if(actions[e.key]){e.preventDefault();actions[e.key]();schedule();}
    });
    onProgress('Downloading splats…');
    const response=await fetch(url,{signal});
    if(!response.ok)throw new Error('Could not download these splats.');
    if(Number(response.headers.get('Content-Length'))>256*1024*1024)throw new Error('Splat file is too large for this viewer.');
    const payload=await response.arrayBuffer();signal.throwIfAborted();
    if(disposed)await ready;
    onProgress('Decoding splats…');worker.postMessage({type:'load',buffer:payload,width},[payload]);
    await ready;signal.throwIfAborted();
    return {dispose};
  }catch(error){dispose();throw error;}
}
