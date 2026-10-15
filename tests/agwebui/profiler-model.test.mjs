import {test} from 'node:test';
import assert from 'node:assert/strict';
import {loadModule} from './module-loader.mjs';
const {attributeTime, profilerData, operationState, callStats, sampleExtrema, performanceFindings, processResources} = await loadModule('profiler-model.js');
const {systemMeasurements} = await loadModule('investigator-model.js');
const call = (id, start, duration, kind='tool', outcome='success', agent='a') => ({id, start, duration, kind, outcome, agent, name:'bash'});
const run = (actions=[], extra={}) => ({id:'test', duration:20, status:'completed', actions, agents:[{id:'a',label:'A',status:'completed'},{id:'b',label:'B',status:'completed'}], intervals:[], counters:{}, coverage:{}, ...extra});
test('same-category overlapping calls are unioned and conflicting categories remain unknown', () => {
 const r=run([call('a',0,10),call('b',5,10),call('m',8,4,'model')]);
 const attribution=attributeTime(r.actions,[],[0,20],r);
 assert.deepEqual(attribution.totals,{tools:11,llm:0,waiting:0,unknown:9});
 assert.equal(Object.values(attribution.totals).reduce((a,b)=>a+b),20);
});
test('waiting needs recorded intervals; unclassified gaps are never waiting',()=>{
 const r=run([call('m',0,5,'model')]);
 const a=attributeTime(r.actions,[{start:7,duration:2,kind:'queue',source:'recorded'}],[0,20],r);
 assert.deepEqual(a.totals,{llm:5,tools:0,waiting:2,unknown:13});
 assert.equal(attributeTime(r.actions,[],[0,20],r).totals.waiting,0);
});
test('missing starts, ends and synthetic timestamps cannot create measured attribution',()=>{
 const calls=[{...call('a',0,10),missing_start:true},{...call('b',5,10),missing_end:true},{...call('c',1,2),source:'timestamp_missing'}];
 const r=run(calls);assert.equal(callStats(calls).samples,0);
 assert.deepEqual(attributeTime(calls,[],[0,20],r).totals,{llm:0,tools:0,waiting:0,unknown:20});
});
test('live, disconnected, completed, old unmatched starts and incomplete traces have distinct confidence',()=>{
 const a=call('open',15,5,'tool','running'),r=run([a],{status:'running',agents:[{id:'a',status:'running'}]});
 assert.equal(operationState(a,r,{transport:'connected'}),'running');
 assert.equal(operationState(a,r,{transport:'disconnected'}),'incomplete');
 assert.equal(operationState(a,r,{transport:'disconnected',frozen:true}),'running');
 assert.equal(operationState(a,{...r,status:'completed'}),'incomplete');
 assert.equal(operationState(a,{...r,duration:100}),'incomplete');
 assert.equal(operationState(a,{...r,coverage:{gaps:['missing frame']}}),'incomplete');
 assert.equal(profilerData(r,{transport:'disconnected'}).active.length,0);
});
test('agent, time, model, tool and status filters select canonical references',()=>{
 const a=call('tool',0,4),b=call('other',10,2,'tool','failed','b'),m={...call('model',1,2,'model'),model:'x',tokens:100,output_tokens:30};
 const r=run([a,b,m]);
 const d=profilerData(r,{agent:'a',range:[1,3],model:'x'});
 assert.deepEqual(d.actions,[a,m]);assert.equal(d.actions[0],a);
 assert.equal(d.agents.length,1);assert.equal(d.llm.input.value,100);
 assert.equal(profilerData(r,{agent:'b',status:'failed',tool:'bash'}).tools[0],b);
 assert.equal(profilerData(r,{model:'missing'}).models.length,0);
});
test('nearest-rank quantiles use completed timed samples and throughput is output / E2E',()=>{
 const a={...call('m',0,2,'model'),tokens:0,output_tokens:20},b={...call('n',3,3,'model'),tokens:null},open=call('o',4,9,'model','running');
 const stats=callStats([a,b,open]);
 assert.equal(stats.p50,2);assert.equal(stats.p95,3);assert.equal(stats.throughput,10);
 assert.deepEqual(stats.input,{value:0,samples:1});assert.equal(stats.output.samples,1);
 assert.equal(callStats([b]).output.value,null);
});
test('selection clips attribution, not full request latency',()=>{
 const r=run([call('a',0,10)]),d=profilerData(r,{range:[2,5]});
 assert.equal(d.agents[0].attribution.totals.tools,3);assert.equal(d.tool.p50,10);
});
test('process identities preserve PID reuse, missing samples and explicit ownership',()=>{
 const r=run([{...call('a',1,3),metadata:{'agency.process_identity':'12-100',program:'python'}}],{counters:{'proc:12-100:rss_mb':[[1,20],[2,30]],'proc:12-200:rss_mb':[[15,40]],'proc:12-200:vms_mb':[[15,500]]}});
 const processes=processResources(r,systemMeasurements(r));
 assert.equal(processes.length,2);assert.equal(processes[0].agent,'a');assert.equal(processes[1].agent,'');
 assert.equal(processes[0].peak,30);assert.equal(processes[1].rss,40);assert.match(processes[0].status,/exit unknown/);
 assert.equal(processResources(r,systemMeasurements(r),[10,20]).length,1);
});
test('conflicting process owners are unavailable rather than guessed',()=>{
 const r=run([...'ab'].map((agent,i)=>({...call(String(i),0,1,'tool','success',agent),metadata:{process_identity:'12-100'}})),{counters:{'proc:12-100:rss_mb':[[1,20]]}});
 assert.equal(processResources(r,systemMeasurements(r))[0].agent,'');
});
test('bucket extrema retain a one-point spike with bounded marks on large traces',()=>{
 const points=Array.from({length:100000},(_,i)=>[i,i===53117?10000:1]),sampled=sampleExtrema(points);
 assert.ok(sampled.length<=480);assert.ok(sampled.some(p=>p[1]===10000));assert.equal(sampled[0],points[0]);assert.equal(sampled.at(-1),points.at(-1));
});
test('findings use deterministic evidence and missing coverage, never causation',()=>{
 const actions=Array.from({length:6},(_,i)=>call(String(i),i, i===5?30:1,'model'));
 const r=run(actions,{duration:40}),findings=performanceFindings(profilerData(r),r);
 assert.equal(findings[0].title,'Model latency spike');assert.equal(findings[0].action,'5');
 assert.doesNotMatch(JSON.stringify(findings),/caused|responsible/);
 const empty=run([]);assert.deepEqual(performanceFindings(profilerData(empty),empty),[]);
});
test('32,000 operations across simultaneous agents have exclusive attribution per row',()=>{
 const agents=Array.from({length:32},(_,i)=>({id:String(i),status:'completed'}));
 const actions=Array.from({length:32000},(_,i)=>call(String(i),Math.floor(i/32),2,i%3===0?'model':'tool','success',String(i%32)));
 const d=profilerData(run(actions,{agents,duration:1002}));
 assert.equal(d.actions.length,32000);assert.equal(d.agents.length,32);
 for(const a of d.agents)assert.equal(Object.values(a.attribution.totals).reduce((n,v)=>n+v),1002);
});
test('recorded pressure, host capacity and network rates preserve actual semantics',()=>{
 const r=run([],{counters:{'host:cpu_util_pct':[[1,50]],'host:cpu_count':[[1,8]],'host:memory_psi_full_pct':[[1,2]],'host:net_rx_mb_s':[[1,3]],'host:disk_read_mb_s':[[1,4]],'host:mem_used_mb':[[1,600]]}});
 const metrics=systemMeasurements(r);
 assert.equal(metrics.find(m=>m.label==='Host utilization').latest[1],50);
 assert.equal(metrics.find(m=>m.label==='CPU capacity').unit,'cores');
 assert.equal(metrics.find(m=>m.label==='Memory pressure (full)').unit,'%');
 assert.equal(metrics.find(m=>m.label==='Network RX').latest[1],3);
 assert.equal(metrics.find(m=>m.label==='Disk read').latest[1],4);
 assert.equal(metrics.find(m=>m.label==='RAM').latest[1],600);
});
test('saved process commands and agent ownership use explicit sampler sandbox metadata',()=>{
 const r=run([],{agents:[{id:'a',sandbox:'box'},{id:'b'}],sandbox_owners:{box:['a']},summary:{process_metrics:[{identity:'12-100',pid:12,cmdline:'python task.py',sandbox:'box'}]},counters:{'process:12-100:rss_mb':[[1,20]]}});
 const p=processResources(r,systemMeasurements(r))[0];assert.equal(p.agent,'a');assert.equal(p.command,'python task.py');
 r.sandbox_owners.box.push('b');assert.equal(processResources(r,systemMeasurements(r))[0].agent,'');
});
