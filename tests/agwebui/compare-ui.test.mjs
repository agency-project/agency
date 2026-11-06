import assert from 'node:assert/strict';
import {test} from 'node:test';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {loadModule} from './module-loader.mjs';
const model = await loadModule('compare-model.js');
const cards = await loadModule('event-details.js');
const source = (await readFile(new URL('../../agency/observability/agwebui/static/compare.js', import.meta.url), 'utf8')).replace(/^import .*;$/mg, '');
const settle = async () => {for (let i=0;i<8;i++) await new Promise(resolve=>setImmediate(resolve));};
function page() {
  const nodes = new Map(), handlers = {}, requests=[];
  const node = id => {if (!nodes.has(id)) nodes.set(id, {id, innerHTML:'', value:'', disabled:false});return nodes.get(id);};
  const document={getElementById:node, addEventListener:(type, handler)=>{handlers[type]=handler;}, querySelectorAll:()=>[]};
  const runs={a:{id:'a',title:'A',metadata:{id:'a'},metrics:{agents:2,requests:3},agents:[],extent:10,history_url:'/profiler?run=a',coverage:{}},b:{id:'b',title:'B',metadata:{id:'b'},metrics:{agents:1,requests:6},agents:[],extent:20,history_url:'/profiler?run=b',coverage:{}}};
  const pending = [];
  let delay=false, failure=false;
  const context = vm.createContext({...model,...cards,esc:cards.escapeHTML,document, URL, URLSearchParams, AbortController, setTimeout,clearTimeout,
    requestAnimationFrame:fn=>fn(), location:{href:'http://localhost/compare', search:''},history:{replaceState:()=>{}},window:{addEventListener:()=>{}},
    fetch:async(url,{signal})=>{
      requests.push(url);
      if(url==='/api/compare/runs') return {ok:true,json:async()=>({runs:[{id:'a',title:'A'},{id:'b',title:'B'}]})};
      if(delay) await new Promise(resolve=>pending.push(resolve));
      if(signal.aborted) throw Object.assign(new Error('aborted'),{name:'AbortError'});
      if(failure) return {ok:false,status:422,json:async()=>({detail:'Recorded trace is unavailable'})};
      return {ok:true,json:async()=>runs[url.split('/').pop()]};
    }});
  vm.runInContext(source,context);
  return {node,requests,runs,handlers,context,pending,setDelay:value=>{delay=value;},setFailure:value=>{failure=value;},
    select:(id,value)=>{node(id).value=value;handlers.change({target:node(id)});},state:()=>vm.runInContext('state',context)};
}
test('real selection and swap handlers preserve records and reverse baseline without fetching again',async()=>{
  const p=page();await settle();
  assert.match(p.node('compare-content').innerHTML,/Select Run A and Run B/);
  p.select('run-a','a');await settle();p.select('run-b','b');await settle();
  assert.match(p.node('compare-content').innerHTML,/Recorded model requests/);
  assert.equal(p.state().runs[0],p.runs.a);
  const fetched=p.requests.length;
  p.node('swap').onclick();await settle();
  assert.equal(p.node('run-a').value,'b');assert.equal(p.state().runs[0],p.runs.b);
  assert.equal(p.requests.length,fetched);
  assert(p.requests.every(url=>url.startsWith('/api/compare/')));
});
test('changing selection aborts stale reads and keeps the latest selected run',async()=>{
  const p=page();await settle();p.setDelay(true);
  p.select('run-a','a');await settle();p.select('run-a','b');await settle();
  p.setDelay(false);p.pending.splice(0).forEach(resolve=>resolve());await settle();
  assert.equal(p.state().ids[0],'b');assert.equal(p.state().runs[0],p.runs.b);
});
test('a failed historical read exposes a retry state without invented comparison values',async()=>{
  const p=page();await settle();p.setFailure(true);p.select('run-a','a');await settle();
  assert.match(p.node('compare-content').innerHTML,/Not available/);
  assert.match(p.node('compare-content').innerHTML,/Recorded trace is unavailable/);
  assert.match(p.node('compare-content').innerHTML,/data-reselect/);
  assert.equal(p.state().runs[0],null);
});
