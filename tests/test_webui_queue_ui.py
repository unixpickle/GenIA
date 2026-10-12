"""Exercise the real queue UI in JS with a small DOM adapter (no browser needed)."""
import json
from pathlib import Path

from test_webui_viewer import run_js

APP_JS = Path(__file__).resolve().parents[1] / "core/webui/static/app.js"


DOM_ADAPTER = r'''
class Node {
  constructor(tag) {
    this.tag=tag; this.children=[]; this.dataset={}; this.handlers={};
    this.hidden=false; this.className=''; this.parent=null; this.attributes={};
    this._text=''; this._scroll=0; this.clientHeight=100; this.textUpdates=0;
    this.classList={add(){},remove(){}};
  }
  get textContent(){return this._text;}
  set textContent(value){this._text=String(value);this.textUpdates++;}
  get scrollHeight(){return Math.max(this.clientHeight,this._text.split('\n').length*15);}
  get scrollTop(){return this._scroll;}
  set scrollTop(value){this._scroll=Math.max(0,Math.min(value,this.scrollHeight-this.clientHeight));}
  get isConnected(){return this===root || Boolean(this.parent?.isConnected);}
  append(...nodes){for(const node of nodes){node.remove();node.parent=this;this.children.push(node);}}
  remove(){if(this.parent){this.parent.children.splice(this.parent.children.indexOf(this),1);this.parent=null;}}
  insertBefore(node,before){node.remove();const index=before?this.children.indexOf(before):this.children.length;node.parent=this;this.children.splice(index,0,node);}
  setAttribute(key,value){this.attributes[key]=value;}
  removeAttribute(key){delete this.attributes[key];delete this[key];}
  addEventListener(name,handler){this.handlers[name]=handler;}
  click(){return this.handlers.click();}
  querySelector(selector){
    for(const node of this.children){
      if(node.className.split(' ').includes(selector.slice(1)))return node;
      const nested=node.querySelector(selector);if(nested)return nested;
    }return null;
  }
}
'''


def test_queue_refresh_keeps_log_nodes_and_reader_scroll_position(tmp_path):
    code = DOM_ADAPTER + r'''
const root=new Node('div'), count=new Node('span');
const document={createElement:tag=>new Node(tag),querySelector:selector=>selector==='#queue'?root:count};
const getComputedStyle=()=>({lineHeight:'15px'});
let logText=Array.from({length:200},(_,i)=>'line '+i).join('\n');
const fetch=async()=>({ok:true,json:async()=>({text:logText})});
'''
    # Load the real functions, excluding page bootstrap and unrelated gallery code.
    code += f"\nconst source=readText({json.dumps(str(APP_JS))}).split('function makeCard')[0];\n"
    code += r'''
const ui=new Function('document','fetch','getComputedStyle',source+';return {renderQueue,refreshLog};')(document,fetch,getComputedStyle);
const drain=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
(async()=>{
  const job={id:'a',title:'Object',status:'running',created:1,started:1,input_url:'/input',progress:{label:'Generating the 3D shape'}};
  ui.renderQueue([job]);
  const item=root.children[0],log=item.querySelector('.logs'),toggle=item.querySelector('.log-button');
  await toggle.click();
  assert(!log.hidden);
  assert(log.scrollTop===log.scrollHeight-log.clientHeight,'Opening log should follow latest output');
  log.scrollTop=300;
  logText+='\nnew output';
  ui.renderQueue([job]);await drain();
  assert(root.children[0]===item,'Polling replaced the queue item');
  assert(item.querySelector('.logs')===log,'Polling replaced the log');
  assert(!log.hidden && log.scrollTop===300,'Polling reset reader scroll position');
  const changes=log.textUpdates;
  ui.renderQueue([job]);await drain();
  assert(log.textUpdates===changes,'Unchanged log should not be rewritten');
  log.scrollTop=log.scrollHeight;
  logText+='\nmore output';
  job.progress.label='Optimizing shape and appearance';
  ui.renderQueue([job]);await drain();
  assert(log.scrollTop===log.scrollHeight-log.clientHeight,'Bottom reader should follow new output');
  assert(item.querySelector('.queue-stage').textContent===job.progress.label);
  assert(item.querySelector('.queue-progress').hidden===false);
  await toggle.click();
  ui.renderQueue([job]);await drain();
  assert(log.hidden,'Polling reopened a collapsed log');
  job.status='cancelling';ui.renderQueue([job]);
  assert(item.querySelector('.queue-progress').hidden,'Cancelling job still shows activity');
  assert(item.querySelector('.cancel').disabled);
  print('Queue and scrolling checks passed');
})().catch(error=>{print(error.stack);if(typeof quit==='function')quit(1);else process.exit(1);});
'''
    assert "Queue and scrolling checks passed" in run_js(tmp_path, code)


def test_delete_card_closes_viewer_and_does_not_reappear(tmp_path):
    code = DOM_ADAPTER + r'''
const nodes=new Map(['history','queue','queue-count','total','empty','notice','more','connection'].map(id=>['#'+id,new Node('div')]));
const root=nodes.get('#history');
const document={createElement:tag=>new Node(tag),querySelector:selector=>nodes.get(selector)};
const getComputedStyle=()=>({lineHeight:'15px'});
const job={id:'a',seq:1,title:'Object',status:'succeeded',created:1,started:1,finished:2,input_url:'/input',seed:42,assets:[]};
let records=[job], fail=true, closed=0;
const requests=[];
const fetch=async(url,options)=>{
  requests.push([url,options.method]);
  if(options.method==='DELETE'){
    if(fail)return {ok:false,status:500,json:async()=>({error:'Could not delete'})};
    records=[];return {ok:true,json:async()=>({deleted:'a'})};
  }
  return {ok:true,json:async()=>({queue:[],history:records,history_ids:records.map(j=>j.id),counts:{succeeded:records.length},next:null,worker:'idle'})};
};
'''
    code += f"\nconst source=readText({json.dumps(str(APP_JS))}).split('$(\"#more\").addEventListener')[0];\n"
    code += r'''
const ui=new Function('document','fetch','getComputedStyle','setTimeout','clearTimeout',
  source+';return {renderHistory,refresh,cards,setViewer:v=>{viewer=v;}};'
)(document,fetch,getComputedStyle,()=>0,()=>{});
(async()=>{
  ui.renderHistory([job]);nodes.get('#total').textContent='1';
  const card=ui.cards.get('a');
  const remove=card.querySelector('.delete-button');
  ui.setViewer({card,close:()=>{closed++;ui.setViewer(null);}});
  await remove.click();
  assert(root.children[0]===card && closed===0,'Failed deletion removed the card or viewer');
  assert(remove.disabled===false,'Failed deletion cannot be retried');
  fail=false;await remove.click();
  assert(root.children.length===0 && closed===1,'Deleted card or viewer remained');
  assert(nodes.get('#empty').hidden===false);
  assert(nodes.get('#total').textContent==='0');
  ui.renderHistory([job]);
  assert(root.children.length===0,'Stale response resurrected a deleted card');
  const other={...job,id:'b',seq:2};records=[other];ui.renderHistory(records);
  records=[];await ui.refresh();
  assert(root.children.length===0,'Deletion in another tab was not reflected');
  assert(requests.filter(([,method])=>method==='DELETE').length===2);
  print('Deletion UI checks passed');
})().catch(error=>{print(error.stack);if(typeof quit==='function')quit(1);else process.exit(1);});
'''
    assert "Deletion UI checks passed" in run_js(tmp_path, code)


def test_iteration_progress_works_with_log_closed_and_resets_between_stages(tmp_path):
    code = DOM_ADAPTER + r'''
const root=new Node('div'), count=new Node('span');
const document={createElement:tag=>new Node(tag),querySelector:selector=>selector==='#queue'?root:count};
const getComputedStyle=()=>({lineHeight:'15px'});
let logText='\r  Finetune obj 1:  24%|██▍       | 24/100 [04:02<12:12, 9.64s/it]\r\x1b[32m  Finetune obj 1:  25%|██▌       | 25/100 [04:12<12:02, loss=0.0955]\x1b[0m';
let hold=false, respond;
const fetch=async()=>{
  if(hold)await new Promise(resolve=>{respond=resolve;});
  return {ok:true,json:async()=>({text:logText})};
};
'''
    code += f"\nconst source=readText({json.dumps(str(APP_JS))}).split('function makeCard')[0];\n"
    code += r'''
const ui=new Function('document','fetch','getComputedStyle',source+';return {renderQueue,optimizationProgress};')(document,fetch,getComputedStyle);
const drain=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
(async()=>{
  const job={id:'a',title:'Object',status:'running',created:1,started:1,input_url:'/input',progress:{stage:'finetune',label:'Optimizing shape and appearance'}};
  ui.renderQueue([job]);await drain();
  const item=root.children[0],log=item.querySelector('.logs'),bar=item.querySelector('.queue-progress'),counter=item.querySelector('.queue-counter');
  assert(log.hidden && log.textContent==='','Progress polling opened or populated a hidden log');
  assert(bar.value===25 && bar.max===100,'Counter did not use the latest iteration');
  assert(counter.textContent==='25 / 100 iterations · 25%' && !counter.hidden);
  logText='\rFinetune obj 1: 50%|#####     | 50/100 [08:00<08:00, loss=0.1]';
  ui.renderQueue([job]);
  assert(bar.value===25,'Polling reset the progress bar before the next response');
  await drain();assert(bar.value===50);
  assert(ui.optimizationProgress('Finetune obj 1: 0%| | 0/100 [00:00<?, ?it/s]').completed===0);
  assert(ui.optimizationProgress('Finetune obj 1: 100%|##########| 100/100 [16:00<00:00]').completed===100);
  assert(ui.optimizationProgress('Stage 2: 50%|#####| 5/10 [00:10<00:10]')===null);
  assert(ui.optimizationProgress('Finetune obj 1: 0%| | 0/0 [00:00]')===null);
  assert(ui.optimizationProgress('Finetune obj 1: 101%|#| 101/100 [00:00]')===null);
  assert(ui.optimizationProgress('Finetune obj 1: 50%|#####| 50/')===null);
  hold=true;ui.renderQueue([job]);
  job.progress={stage:'exporting',label:'Exporting the model and renders'};
  ui.renderQueue([job]);
  assert(counter.hidden && bar.value===undefined,'New stage retained old iteration progress');
  respond();await drain();
  assert(counter.hidden && bar.value===undefined,'Late log response restored stale progress');
  print('Iteration progress checks passed');
})().catch(error=>{print(error.stack);if(typeof quit==='function')quit(1);else process.exit(1);});
'''
    assert "Iteration progress checks passed" in run_js(tmp_path, code)


def test_splats_are_default_and_models_only_load_on_click(tmp_path):
    code = DOM_ADAPTER + r'''
Node.prototype.replaceChildren=function(...nodes){for(const n of [...this.children])n.remove();this.append(...nodes);};
class AbortController {constructor(){this.signal={aborted:false};}abort(){this.signal.aborted=true;}}
const root=new Node('div'),noticeNode=new Node('div');
const document={createElement:tag=>new Node(tag),querySelector:()=>noticeNode};
const modules=[],requests=[];let disposed=0,release;
const loadRenderer=async kind=>{
 modules.push(kind);
 return {createViewer:async(stage,url,signal,onProgress,rotation)=>{
   requests.push({url,signal,rotation});onProgress('Decoding splats…');
   await new Promise(resolve=>{release=resolve;});return {dispose(){disposed++;}};
 }};
};
const drain=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
'''
    code += f"const source=readText({json.dumps(str(APP_JS))}).split('$(\"#more\").addEventListener')[0].replace('await import(\"./splat-viewer.js\")','await loadRenderer(\"splat\")').replace('await import(\"./viewer.js\")','await loadRenderer(\"mesh\")');\n"
    code += r'''
const ui=new Function('document','loadRenderer','AbortController',source+';return {makeCard};')(document,loadRenderer,AbortController);
(async()=>{
 const rotation=[1,0,0,0,1,0,0,0,1];
 const job={id:'a',title:'cake',status:'succeeded',created:1,seed:42,input_url:'/input',assets:[
   {name:'001.glb',kind:'model',bytes:100,url:'/mesh'},
   {name:'001.compressed.ply',kind:'splat',bytes:80,url:'/splats',view_rotation:rotation}]};
 const card=ui.makeCard(job);root.append(card);
 assert(!requests.length && !modules.length,'Gallery prefetched a model or renderer');
 const load=card.querySelector('.load-model'),select=card.querySelector('.mesh-select'),download=card.querySelector('.download');
 assert(select.children[0].textContent.startsWith('Splats'));
 assert(download.href==='/splats?download=1' && download.textContent==='Download splats ↓');
 const pending=load.click();await drain();
 assert(modules[0]==='splat' && requests[0].url==='/splats' && requests[0].rotation===rotation);
 assert(load.disabled);
 await card.querySelector('.close-model').click();
 assert(requests[0].signal.aborted && !load.disabled);
 release();await pending;assert(disposed===1,'Closed load leaked its late renderer');
 select.value='1';select.handlers.change();
 assert(download.href==='/mesh?download=1' && download.textContent==='Download GLB ↓');
 assert(requests.length===1,'Selection change fetched a model');
 const mesh=load.click();await drain();assert(modules[1]==='mesh' && requests[1].url==='/mesh');
 release();await mesh;
 await card.querySelector('.close-model').click();assert(disposed===2);
 print('Lazy splat selection checks passed');
})().catch(error=>{print(error.stack);if(typeof quit==='function')quit(1);else process.exit(1);});
'''
    assert 'Lazy splat selection checks passed' in run_js(tmp_path, code)
