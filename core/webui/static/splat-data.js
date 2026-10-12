/* PLY decoding and depth ordering for the WebGL splat viewer.
 * Runs in a classic worker. No DOM, model libraries, or network dependencies.
 * Compressed layout matches vendor/gsconverter/formats/compressed_ply.py.
 */
const SplatData = (() => {
  const types = {float: [4, 'getFloat32'], float32: [4, 'getFloat32'],
    double: [8, 'getFloat64'], uchar: [1, 'getUint8'], uint8: [1, 'getUint8'],
    uint: [4, 'getUint32'], uint32: [4, 'getUint32'], int: [4, 'getInt32']};
  const maxSplats = 2_000_000;
  function header(buffer) {
    const bytes = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 65536));
    let end = -1;
    // Decode only ASCII header bytes; binary payload can contain arbitrary UTF-8.
    for (let i = 0; i < bytes.length - 10; i++) {
      if ((i === 0 || bytes[i-1] === 10) && bytes[i] === 101 &&
          String.fromCharCode(...bytes.subarray(i, i+10)) === 'end_header') {
        let n = i + 10;
        if (bytes[n] === 13) n++;
        if (bytes[n] === 10) { end = n + 1; break; }
      }
    }
    if (end < 0) throw new Error('Invalid or oversized PLY header.');
    const lines = new TextDecoder().decode(bytes.subarray(0, end)).trim().split(/\r?\n/);
    if (lines[0] !== 'ply' || lines[1] !== 'format binary_little_endian 1.0') {
      throw new Error('This viewer needs a binary little-endian Gaussian PLY.');
    }
    const elements = []; let current;
    for (const line of lines.slice(2)) {
      const [tag, a, b, c] = line.trim().split(/\s+/);
      if (tag === 'element') {
        const count = Number(b);
        if (!Number.isSafeInteger(count) || count < 0 || count > maxSplats || elements.some(e => e.name === a)) throw new Error('Invalid PLY element count.');
        current = {name: a, count, stride: 0, props: {}}; elements.push(current);
      } else if (tag === 'property') {
        if (!current || !types[a] || c || current.props[b]) throw new Error('Unsupported PLY property.');
        current.props[b] = {offset: current.stride, type: types[a]}; current.stride += types[a][0];
      }
    }
    let offset = end;
    for (const e of elements) {
      e.offset = offset; offset += e.count * e.stride;
      if (!e.stride || !Number.isSafeInteger(offset) || offset > buffer.byteLength) throw new Error('Truncated PLY data.');
    }
    const vertex = elements.find(e => e.name === 'vertex');
    if (!vertex?.count) throw new Error('PLY contains no splats.');
    return {vertex, chunk: elements.find(e => e.name === 'chunk'),
      hasSH: elements.some(e => e.name === 'sh' && e.count) || Object.keys(vertex.props).some(n => n.startsWith('f_rest_'))};
  }
  function covariance(q, scale) {
    const length = Math.hypot(...q);
    if (!length) throw new Error('Invalid splat rotation.');
    const [w,x,y,z] = q.map(v => v / length);
    // Columns of quaternion rotation R, scaled by each Gaussian standard deviation.
    const a = [(1-2*(y*y+z*z))*scale[0], 2*(x*y+w*z)*scale[0], 2*(x*z-w*y)*scale[0]];
    const b = [2*(x*y-w*z)*scale[1], (1-2*(x*x+z*z))*scale[1], 2*(y*z+w*x)*scale[1]];
    const c = [2*(x*z+w*y)*scale[2], 2*(y*z-w*x)*scale[2], (1-2*(x*x+y*y))*scale[2]];
    const dot = (i,j) => a[i]*a[j]+b[i]*b[j]+c[i]*c[j];
    return [dot(0,0),dot(0,1),dot(0,2),dot(1,1),dot(1,2),dot(2,2)];
  }
  function decode(buffer, width = 2048) {
    if (buffer.byteLength > 256 * 1024 * 1024) throw new Error('Splat file is too large for this viewer.');
    if (!Number.isInteger(width) || width < 4 || width % 4) throw new Error('Invalid texture width.');
    const {vertex: v, chunk: ch, hasSH} = header(buffer), count = v.count;
    const view = new DataView(buffer);
    function reader(e, name) {
      const p = e?.props[name];
      if (!p) throw new Error(`Missing Gaussian property: ${name}.`);
      return i => view[p.type[1]](e.offset + i * e.stride + p.offset, true);
    }
    const positions = new Float32Array(count * 3);
    const height = Math.ceil(count * 4 / width), texture = new Float32Array(width * height * 4);
    const low = [Infinity,Infinity,Infinity], high = [-Infinity,-Infinity,-Infinity];
    let read;
    if (ch) {
      if (ch.count !== Math.ceil(count/256)) throw new Error('Invalid compressed PLY chunks.');
      const chunkNames = ['min_x','min_y','min_z','max_x','max_y','max_z',
        'min_scale_x','min_scale_y','min_scale_z','max_scale_x','max_scale_y','max_scale_z'];
      if (ch.props.min_r) chunkNames.push('min_r','min_g','min_b','max_r','max_g','max_b');
      const chunkReaders = chunkNames.map(n => reader(ch,n));
      const [rp,rq,rs,rc] = ['packed_position','packed_rotation','packed_scale','packed_color'].map(n => reader(v,n));
      let bounds;
      const unpack = n => [n >>> 21, (n >>> 11) & 1023, n & 2047];
      read = i => {
        if (i % 256 === 0) {
          bounds = chunkReaders.map(r => r(i >>> 8));
          if (!bounds.every(Number.isFinite)) throw new Error('Invalid compressed PLY bounds.');
        }
        const lerp = (j,t) => bounds[j] + t * (bounds[j+3]-bounds[j]);
        const p = unpack(rp(i)).map((n,j) => lerp(j,n/(j===1?1023:2047)));
        const s = unpack(rs(i)).map((n,j) => Math.exp(lerp(6+j,n/(j===1?1023:2047))));
        const packed = rq(i), largest = packed >>> 30;
        const small = [20,10,0].map(shift => (((packed >>> shift)&1023)/1023-.5)*Math.SQRT2);
        const q = small.slice(); q.splice(largest,0,Math.sqrt(Math.max(0,1-small.reduce((a,b)=>a+b*b,0))));
        const color = rc(i);
        const rgb = [24,16,8].map((shift,j) => {
          const t = ((color >>> shift)&255)/255; return bounds.length === 18 ? lerp(12+j,t) : t;
        });
        return {p,s,q,rgb,alpha:(color&255)/255};
      };
    } else {
      const xyz = ['x','y','z'].map(n => reader(v,n));
      const scales = [0,1,2].map(n => reader(v,`scale_${n}`));
      const quats = [0,1,2,3].map(n => reader(v,`rot_${n}`));
      const colors = [0,1,2].map(n => reader(v,`f_dc_${n}`));
      const opacity = reader(v,'opacity');
      read = i => ({p:xyz.map(r=>r(i)), s:scales.map(r=>Math.exp(r(i))), q:quats.map(r=>r(i)),
        rgb:colors.map(r=>.5+.28209479177387814*r(i)), alpha:1/(1+Math.exp(-opacity(i)))});
    }
    for (let i = 0; i < count; i++) {
      const {p,s,q,rgb,alpha} = read(i), offset = i*16;
      if (![...p,...s,...q,...rgb,alpha].every(Number.isFinite)) throw new Error('Non-finite Gaussian data.');
      positions.set(p,i*3); texture.set([...p,alpha],offset);
      const cov = covariance(q,s);
      texture.set(cov.slice(0,3),offset+4); texture.set(cov.slice(3),offset+8);
      texture.set(rgb.map(x=>Math.max(0,Math.min(1,x))),offset+12);
      // Invisible splats must not spoil framing; include 3 sigma of visible ones.
      if (alpha >= 1/255) for (let j=0;j<3;j++) {
        const radius = 3 * Math.sqrt(cov[[0,3,5][j]]);
        low[j] = Math.min(low[j],p[j]-radius); high[j] = Math.max(high[j],p[j]+radius);
      }
    }
    if (!low.every(Number.isFinite)) throw new Error('PLY contains no visible splats.');
    const center = low.map((n,j)=>(n+high[j])/2);
    const extent = Math.max(...high.map((n,j)=>n-low[j]));
    const factor = 2 / Math.max(extent,1e-6);
    for (let i=0;i<count;i++) {
      for (let j=0;j<3;j++) texture[i*16+j] = positions[i*3+j] = (positions[i*3+j]-center[j])*factor;
      for (const j of [4,5,6,8,9,10]) texture[i*16+j] *= factor*factor;
    }
    if (!texture.every(Number.isFinite)) throw new Error('Gaussian data exceeds the viewer numeric range.');
    return {positions,texture,count,width,height,hasSH,center,factor};
  }
  function createSorter(positions) {
    const count = positions.length/3, depths = new Float32Array(count), bins = new Uint16Array(count);
    const offsets = new Uint32Array(65536);
    return direction => {
      if (direction.length !== 3 || !direction.every(Number.isFinite)) throw new Error('Invalid camera direction.');
      let low=Infinity, high=-Infinity;
      for (let i=0;i<count;i++) {
        const d=positions[i*3]*direction[0]+positions[i*3+1]*direction[1]+positions[i*3+2]*direction[2];
        depths[i]=d; low=Math.min(low,depths[i]); high=Math.max(high,depths[i]);
      }
      offsets.fill(0);
      const factor=high>low?65535/(high-low):0;
      for(let i=0;i<count;i++){const b=Math.max(0,Math.min(65535,Math.floor((depths[i]-low)*factor)));bins[i]=b;offsets[b]++;}
      let offset=0;
      for(let i=0;i<offsets.length;i++){const n=offsets[i];offsets[i]=offset;offset+=n;}
      const order=new Uint32Array(count);
      // Camera looks along -Z: ascending camera Z is back-to-front.
      for(let i=0;i<count;i++) order[offsets[bins[i]]++]=i;
      return order;
    };
  }
  return {decode,covariance,createSorter};
})();
