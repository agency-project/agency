import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../../agency/observability/agwebui/static/metrics-model.js',import.meta.url),'utf8');
const {projectMetrics,percentile,peak,union,readPreferences}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const call=(id,start,duration,outcome='success',kind='tool')=>({id,agent:'a',start,duration,outcome,kind});
test('identity deduplication, active calls and missing usage',()=>{const a=call('a',0,2);const m=projectMetrics({duration:10,actions:[a,a,call('b',4,6,'running'),call('m',2,1,'success','model')],agents:[]});assert.equal(m.totals.tools,2);assert.equal(m.dt.length,1);assert.equal(m.totals.input,null);assert.equal(m.oldest.id,'b');});
test('nearest rank quantiles, missing observations, overlapping coverage',()=>{assert.equal(percentile([],0.95),null);assert.equal(percentile([1,2,3,4],.5),2);assert.equal(percentile([1,2,3,4],.95),4);const calls=[call('a',0,5),call('b',2,5)];assert.equal(peak(calls,7),2);assert.equal(union(calls,7),7);});
test('rolling rates and execution totals survive client retention',()=>{const m=projectMetrics({duration:100,actions:[call('a',10,1),call('b',90,1)],metrics_totals:{tools:800},agents:[]});assert.equal(m.toolRate,1/60);assert.equal(m.totals.tools,800);});
test('preferences persist six unique slots and tolerate unavailable storage',()=>{assert.equal(readPreferences({getItem(){throw Error();}}).pins.length,6);assert.deepEqual(readPreferences({getItem:()=>JSON.stringify({pins:['tools','tools'],category:'Tools',expanded:true})}),{pins:['tools'],category:'Tools',expanded:true});});
for(const n of [3,10,32])test(`${n} simultaneous agents with 32000 calls`,()=>{const agents=Array.from({length:n},(_,i)=>({id:String(i),status:'running'}));const actions=Array.from({length:32000},(_,i)=>({...call(String(i),i/10,1),agent:String(i%n)}));const m=projectMetrics({agents,actions,duration:3201});assert.equal(m.totals.tools,32000);assert.equal(m.agents.length,n);});
const {effectiveScope}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const scopedRun=()=>({duration:100,agents:[{id:'workflow'},{id:'a',label:'A',status:'running'},{id:'b',label:'B',status:'completed'},{id:'c',label:'C',status:'running'}],
 actions:[{...call('a1',0,1),agent:'a'},{...call('a2',0,9),agent:'a'},{...call('b1',0,2,'failed'),agent:'b'},{...call('c1',0,100),agent:'c'},
  {...call('am',1,4,'success','model'),agent:'a',tokens:100,output_tokens:10},{...call('bm',1,8,'success','model'),agent:'b',tokens:50},{...call('cm',1,6,'success','model'),agent:'c'}]});
test('all agents is the default scope and keeps execution-wide totals',()=>{
 const run={...scopedRun(),metrics_totals:{tools:900,models:90},metrics_by_agent:{a:{tools:2},b:{tools:1},c:{tools:1}}},m=projectMetrics(run);
 assert.equal(m.scoped,false);assert.equal(m.totals.tools,900);assert.equal(m.agents.length,3);assert.equal(m.actions.length,7);
 assert.equal(projectMetrics(run,null).totals.tools,900);
});
test('an individual agent or any subset sums additive counters over the selected agents only',()=>{
 const run=scopedRun(),one=projectMetrics(run,new Set(['a'])),two=projectMetrics(run,new Set(['a','c']));
 assert.equal(one.scoped,true);assert.deepEqual(one.agents.map(a=>a.id),['a']);assert.equal(one.roster.length,3);
 assert.equal(one.totals.tools,2);assert.equal(one.totals.input,100);assert.equal(one.totals.output,10);assert.equal(one.totals.tool_errors,0);
 assert.equal(two.totals.tools,3);assert.equal(two.totals.models,2);assert.equal(two.totals.input,100);assert.equal(two.totals.input_samples,1);
 assert.deepEqual(two.states,{running:2});
 assert.equal(projectMetrics(run,new Set(['b'])).totals.tool_errors,1);
});
test('scoped totals use execution-wide per-agent counters when the client retains only a window',()=>{
 const run={...scopedRun(),metrics_totals:{tools:900},metrics_by_agent:{a:{tools:400,input:9000,input_samples:30,models:30},b:{tools:300},c:{tools:200,models:5}}};
 const m=projectMetrics(run,new Set(['a','c']));
 assert.equal(m.totals.tools,600);assert.equal(m.totals.input,9000);assert.equal(m.totals.models,35);assert.equal(m.totals.output_samples,0);
 // An agent with no counted call yet contributes zero, not NaN.
 assert.equal(projectMetrics({...run,agents:[...run.agents,{id:'new'}]},new Set(['new'])).totals.tools,0);
});
test('latency percentiles pool the selected agents observations, never an average of per-agent values',()=>{
 const run=scopedRun(),durations=scope=>projectMetrics(run,scope).dt.map(a=>a.duration);
 // a: 1s and 9s, c: 100s. The pooled median is 9s; averaging per-agent medians would give 50.5s or 52.5s.
 assert.equal(percentile(durations(new Set(['a','c'])),.5),9);
 assert.equal(percentile(durations(new Set(['a'])),.5),1);
 assert.equal(percentile(projectMetrics(run,new Set(['b','c'])).dm.map(a=>a.duration),.95),8);
});
test('a selection survives spawning and termination and ignores vanished agents',()=>{
 const run=scopedRun(),scope=new Set(['a','b']);
 const grown={...run,agents:[...run.agents,{id:'d',status:'running'}]};
 assert.deepEqual([...effectiveScope(scope,grown)],['a','b']);assert.equal(projectMetrics(grown,scope).agents.length,2);
 assert.equal(projectMetrics(grown).agents.length,4);
 const ended={...run,agents:run.agents.map(a=>a.id==='a'?{...a,status:'completed'}:a)};
 assert.deepEqual(projectMetrics(ended,scope).states,{completed:2});
 assert.deepEqual([...effectiveScope(new Set(['a','gone']),run)],['a']);
 assert.equal(effectiveScope(null,run),null);
});
test('no agents selected is a valid scope: every projection is empty, finite and never falls back to all agents',()=>{
 const run={...scopedRun(),metrics_totals:{tools:900,models:90},metrics_by_agent:{a:{tools:2,input:100,input_samples:1},b:{tools:1},c:{tools:1}}};
 for(const scope of [new Set(),new Set(['gone']),new Set(['workflow'])]){
  assert.equal(effectiveScope(scope,run).size,0);
  const m=projectMetrics(run,scope);
  assert.equal(m.scoped,true);assert.equal(m.agents.length,0);assert.equal(m.roster.length,3);assert.equal(m.actions.length,0);
  assert.deepEqual(m.states,{});assert.deepEqual(m.perAgent,[]);assert.equal(m.oldest,undefined);
  for(const value of Object.values(m.totals))assert.equal(value,0);
  assert.equal(m.toolRate,0);assert.equal(m.modelRate,0);
  assert.equal(percentile(m.dt.map(a=>a.duration),.5),null);assert.equal(peak(m.tools,m.end),0);assert.equal(union(m.tools,m.end),0);
 }
 // Without per-agent counters the retained-call fallback is just as empty.
 const retained=projectMetrics(scopedRun(),new Set());
 assert.equal(retained.totals.tools,0);assert.equal(retained.totals.input,null);assert.equal(retained.totals.input_samples,0);
});
