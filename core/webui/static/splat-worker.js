/* Decode once and sort off the UI thread. Main thread allows one sort in flight. */
importScripts('./splat-data.js');
let sort;
self.onmessage = ({data}) => {
  try {
    if (data.type === 'load') {
      const decoded = SplatData.decode(data.buffer, data.width);
      sort = SplatData.createSorter(decoded.positions);
      const {texture,count,width,height,hasSH} = decoded;
      self.postMessage({type:'loaded',texture,count,width,height,hasSH}, [texture.buffer]);
    } else if (data.type === 'sort' && sort) {
      const order = sort(data.direction);
      self.postMessage({type:'sorted',order,direction:data.direction}, [order.buffer]);
    }
  } catch (error) { self.postMessage({type:'error',message:error.message}); }
};
