import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source = await readFile(new URL('../../agency/observability/agwebui/static/investigator-model.js', import.meta.url),'utf8');
const {overlapDuration, matchesAction, trajectoryActions, packActionTracks, groupCallStack, executionIdentity, readableExecutionTitle, executionMeasurements, systemMeasurements, systemActivity} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('syscalls remain system telemetry while trajectory groups and call counts exclude them',()=>{
  const run={duration:30,actions:[
    {id:'send',agent:'a',kind:'syscall',name:'sendto',start:1,duration:20,outcome:'running'},
    {id:'exec',agent:'a',kind:'syscall',name:'execve',start:2,duration:1,outcome:'success'},
    {id:'model',agent:'a',kind:'model',name:'model',start:3,outcome:'running'},
    {id:'tool',agent:'a',kind:'tool',name:'execve',start:4,outcome:'success'},
  ]};
  assert.deepEqual(trajectoryActions(run).map(a=>a.id),['model','tool']);
  assert.deepEqual(groupCallStack(run.actions).flatMap(g=>g.actions).map(a=>a.id).sort(),['model','tool']);
  assert.equal(executionMeasurements(run).primary.find(m=>m[0]==='Calls')[1],2);
  assert.equal(systemActivity(run).count,2);
  assert.deepEqual(systemActivity(run).groups.map(g=>g.name).sort(),['execve','sendto']);
  assert.equal(run.actions.length,4);
});

test('live cumulative CPU and disk counters become rates without double-counting recorded CPU percentages',()=>{
  const metrics=systemMeasurements({duration:2,counters:{
    'workload:cpu_us':[[0,1000000],[2,5000000]],
    'workload:io_r':[[0,100],[2,4194404]],
    'proc:42-100:cpu_s':[[0,1],[2,2]],
    'proc:42-100:cpu_pct':[[2,50]],
  }});
  assert.equal(metrics.find(m=>m.name==='workload:cpu_us').latest[1],200);
  assert.equal(metrics.find(m=>m.name==='workload:io_r').latest[1],2);
  assert.equal(metrics.filter(m=>m.scope==='Process 42-100').length,1);
  assert.equal(metrics.find(m=>m.scope==='Process 42-100').aggregate,false);
});
test('rates reject counter resets, missing baselines, bad points and future replay samples',()=>{
  const counters={'workload:cpu_us':[[3,4000000],[0,1000000],[1,0],[2,1000000],[4,9000000],[null,1],[1,null]],'workload:io_r':[[2,100]],'broken':null};
  const before=structuredClone(counters),metrics=systemMeasurements({duration:3,counters});
  assert.deepEqual(metrics[0].points,[[2,100],[3,300]]);
  assert.equal(metrics.length,1);
  assert.deepEqual(counters,before);
});
test('saved rates and GPU gauges retain recorded units and scopes',()=>{
  const metrics=systemMeasurements({counters:{
    'workload_total cpu %':[[1,0]],'workload_total memory_mb':[[1,512]],
    'workload_total io read MB/s':[[1,4]],'sandbox:worker:cpu_pct':[[1,175]],
    'gpu0:mem_mb':[[1,1024]],'gpu1:util_pct':[[1,20]],'gpu0:worker:power_w_est':[[1,12]],
  }});
  assert.deepEqual(metrics.filter(m=>m.aggregate).map(m=>[m.label,m.unit,m.scope]),[
    ['CPU','%','Workload'],['RAM','MB','Workload'],['Disk read','MB/s','Workload'],['VRAM','MB','gpu0'],['GPU','%','gpu1'],
  ]);
  assert.equal(metrics.find(m=>m.name==='sandbox:worker:cpu_pct').scope,'Sandbox worker');
  assert.equal(metrics.find(m=>m.label==='Power estimate').aggregate,false);
});

test('history order depends on starts and stable IDs, never completion duration',()=>{
  const calls=[{id:'slow',agent:'a',start:0,duration:50,outcome:'running'},
    {id:'b',agent:'a',start:1,duration:1,outcome:'running'},
    {id:'a',agent:'a',start:1,duration:3,outcome:'success'}];
  assert.equal(groupCallStack(calls)[0].running,true);
  calls[1].outcome='success';calls[0].outcome='success';
  assert.deepEqual(groupCallStack(calls).map(g=>g.id),['a','b','slow']);
  calls[0].duration=10000;calls[1].duration=500;
  assert.deepEqual(groupCallStack(calls).map(g=>g.id),['a','b','slow']);
});
test('selection execution keys survive reconnect epochs and distinguish recorded origins',()=>{
  assert.equal(executionIdentity({id:'legacy'}),'legacy');
  assert.equal(executionIdentity({id:'live',time_origin:1,epoch:'a'}),executionIdentity({id:'live',time_origin:1,epoch:'b'}));
  assert.notEqual(executionIdentity({id:'live',time_origin:1}),executionIdentity({id:'live',time_origin:2}));
});

test('readable titles prefer a recorded task over a technical execution heading',()=>{
  assert.equal(readableExecutionTitle({id:'live',title:'Current execution · 1234',task:'Repair the scheduler\nFurther instructions'}),'Repair the scheduler');
  assert.equal(readableExecutionTitle({id:'1234',title:'1234',task:'Inspect traces'}),'Inspect traces');
  assert.equal(readableExecutionTitle({id:'a',title:'Named investigation',task:'Task'}),'Named investigation');
  assert.equal(readableExecutionTitle({id:'live',title:'Current execution · 1234'}),'Current execution · 1234');
  assert.equal(readableExecutionTitle({}),'Execution');
});
test('empty and older recordings omit absent measurements rather than showing zero',()=>{
  const metrics=executionMeasurements({duration:0,actions:[],agents:[],counters:{cpu:[]}});
  assert.deepEqual(metrics.primary,[]);assert.deepEqual(metrics.secondary,[]);
  assert.match(metrics.coverage,/No model calls recorded/);assert.match(metrics.coverage,/No system samples recorded/);
  const older=executionMeasurements({duration:10,actions:[{kind:'tool'}],agents:[{id:'worker'}]});
  assert.deepEqual(older.primary.map(m=>m[0]),['Wall time','Calls','Agents']);
  assert(!older.secondary.some(m=>m[0]==='Active agents'));
});
test('reported zero and partial model usage remain explicit without filling missing usage',()=>{
  const metrics=executionMeasurements({duration:10,actions:[{kind:'model',tokens:0,start:0,duration:1},{kind:'model',tokens:null,output_tokens:5,start:1,duration:1},{kind:'model',tokens:NaN,start:2,duration:1}]});
  assert.equal(metrics.primary.find(m=>m[0]==='Input tokens')[1],0);
  assert.equal(metrics.primary.find(m=>m[0]==='Output tokens')[1],5);
  assert.match(metrics.coverage,/1\/3 input and 1\/3 output/);
  assert.match(metrics.secondary.find(m=>m[0]==='Total tokens')[3],/missing usage excluded/);
});
test('secondary samples retain scope, exclude invalid samples and use memory units',()=>{
  const metrics=executionMeasurements({counters:{'workload:cpu_pct':[[0,0],[1,NaN]],'gpu0:mem_mb':[[1,128]],'empty':[]}});
  assert.equal(metrics.secondary.find(m=>m[0]==='CPU')[1],0);
  assert.match(metrics.secondary.find(m=>m[0]==='CPU')[3],/workload:cpu_pct/);
  assert.equal(metrics.secondary.find(m=>m[0]==='VRAM')[2],'MB');
  assert(!metrics.secondary.some(m=>m[0]==='RAM'));
  assert.match(metrics.coverage,/2 sampled resource series/);
});

test('parallel nested model spans do not double count wall time',()=>{
  const intervals=[{start:1,duration:5},{start:2,duration:2},{start:5,duration:4}];
  assert.equal(overlapDuration(intervals,0,10),8);
  assert.equal(overlapDuration(intervals,3,7),4);
});
test('search and agent filters use the same event projection',()=>{
  const action={agent:'worker',name:'Bash',intent:'Locate scheduler',command:'rg dispatch',result:'found',files:['orchestrator.py']};
  assert(matchesAction(action,'worker','ORCHESTRATOR'));
  assert(!matchesAction(action,'supervisor','scheduler'));
  assert(matchesAction(action,'','found'));
});
test('nested actions stay visible and adjacent actions reuse a track',()=>{
  const actions=[{id:'next',start:10,duration:2},{id:'nested',start:2,duration:3},{id:'long',start:0,duration:10},{id:'parallel',start:3,duration:1}];
  const tracks=packActionTracks(actions);
  assert.deepEqual(tracks.map(track=>track.map(a=>a.id)),[['long','next'],['nested'],['parallel']]);
  assert.equal(actions[0].id,'next');
});
test('instant events reserve their visible width without losing actions',()=>{
  const actions=[{id:'one',start:0,duration:0},{id:'two',start:0,duration:0},{id:'three',start:1,duration:0}];
  assert.deepEqual(packActionTracks(actions,1).map(track=>track.map(a=>a.id)),[['one','three'],['two']]);
  assert.deepEqual(packActionTracks([]),[]);
});


test('summary resource projection reads latest samples without scanning history',()=>{
  let reads=0;
  const samples=new Proxy(Array.from({length:10000},(_,i)=>[i,i]),{get(target,key,receiver){if(/^\d+$/.test(String(key)))reads++;return Reflect.get(target,key,receiver);}});
  const run={duration:10000,counters:{'workload:memory_mb':samples,'workload:cpu_us':[[0,0],[2,2000000],[4,3000000]]}};
  const latest=systemMeasurements(run,{latestOnly:true});
  assert.equal(latest.find(m=>m.label==='RAM').latest[1],9999);
  assert.equal(latest.find(m=>m.label==='CPU').latest[1],50);
  assert(reads<20,`read ${reads} historical samples for a summary`);
  run.duration++;
  assert.equal(systemMeasurements(run,{latestOnly:true}),latest);
});
test('summary resources respect held time, appended samples and counter resets',()=>{
  const run={duration:2,counters:{'workload:cpu_us':[[0,0],[2,2000000],[4,3000000]],'workload:memory_mb':[[0,10],[2,20],[4,30]]}};
  assert.deepEqual(systemMeasurements(run,{latestOnly:true}).map(m=>m.latest),[[2,100],[2,20]]);
  run.duration=4;
  assert.deepEqual(systemMeasurements(run,{latestOnly:true}).map(m=>m.latest),[[4,50],[4,30]]);
  run.counters['workload:cpu_us'].push([5,0]);run.duration=5;
  assert.deepEqual(systemMeasurements(run,{latestOnly:true}).find(m=>m.label==='CPU').latest,[4,50]);
  run.counters['workload:cpu_us'].push([6,1000000]);run.duration=6;
  assert.deepEqual(systemMeasurements(run,{latestOnly:true}).find(m=>m.label==='CPU').latest,[6,100]);
});

const {agentTokens, agentResources} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
test('agent tokens are reported input plus output, prefer execution counters and stay absent without usage',()=>{
  const run={actions:[{id:'m1',agent:'a',kind:'model',tokens:1000,output_tokens:200},{id:'m2',agent:'a',kind:'model',tokens:500},{id:'m3',agent:'a',kind:'model'},{id:'m4',agent:'b',kind:'model'},{id:'t',agent:'a',kind:'tool',tokens:99}]};
  assert.deepEqual(agentTokens(run,'a'),{total:1700,input:1500,output:200,reported:2,models:3});
  assert.equal(agentTokens(run,'b'),null);assert.equal(agentTokens(run,'nobody'),null);
  const counted={...run,metrics_by_agent:{a:{models:40,input:8000,output:200,input_samples:40,output_samples:40},b:{models:2,input:0,output:0,input_samples:0,output_samples:0}}};
  assert.equal(agentTokens(counted,'a').total,8200);assert.equal(agentTokens(counted,'b'),null);assert.equal(agentTokens(counted,'spawned'),null);
  // Zero is valid reported usage, not a missing measurement.
  assert.equal(agentTokens({metrics_by_agent:{a:{models:1,input:0,output:0,input_samples:1,output_samples:1}}},'a').total,0);
});
test('agent CPU is the recorded sandbox container only, never host or workload, and is summed per selected sandbox',()=>{
  const run={duration:10,agents:[{id:'a',sandbox:'sandbox_a_0000'},{id:'b',sandbox:'sandbox_b_0000'},{id:'c'}],counters:{
    'host:cpu_pct':[[10,640]],'workload:cpu_pct':[[10,300]],'workload:memory_mb':[[10,9000]],'cg:conmon:cpu_pct':[[10,5]],
    'cg:sandbox_a_0000:cpu_pct':[[9,10],[10,12.5]],'sandbox:sandbox_a_0000:mem_mb':[[10,256]],
    'cg:sandbox_b_0000:cpu_pct':[[10,150]],'sandbox:sandbox_b_0000:mem_mb':[[10,100]]}};
  assert.deepEqual(agentResources(run,['a']).cpu,{value:12.5,sandboxes:1});assert.equal(agentResources(run,['a']).memory.value,256);
  assert.deepEqual(agentResources(run,['a','b']).cpu,{value:162.5,sandboxes:2});assert.equal(agentResources(run,['a','b']).memory.value,356);
  // No sandbox recorded: unavailable, and a similarly named series is not guessed.
  const unlinked=agentResources({...run,counters:{...run.counters,'cg:sandbox_c_0000:cpu_pct':[[10,77]]}},['c']);
  assert.equal(unlinked.cpu,null);assert.equal(unlinked.memory,null);assert.equal(unlinked.sandboxes,0);
  assert.equal(agentResources(run,['a','c']).cpu.value,12.5);
});
test('shared sandboxes, stopped containers and cumulative-only counters are handled without invented attribution',()=>{
  const shared={duration:10,agents:[{id:'a',sandbox:'box'},{id:'b',sandbox:'box'}],counters:{'cg:box:cpu_pct':[[10,80]]}};
  assert.equal(agentResources(shared,['a']).cpu,null);assert.equal(agentResources(shared,['a']).shared,1);
  assert.equal(agentResources(shared,['a','b']).cpu.value,80);
  const stopped={duration:60,agents:[{id:'a',sandbox:'box'}],counters:{'cg:box:cpu_pct':[[10,80]]}};
  assert.equal(agentResources(stopped,['a']).cpu,null);assert.equal(agentResources(stopped,['a']).sandboxes,1);
  // A saved cumulative counter becomes a rate: 0.5 CPU-seconds over one second is 50%.
  const cumulative={duration:2,agents:[{id:'a',sandbox:'box'}],counters:{'cg:box:cpu_us':[[1,1000000],[2,1500000]]}};
  assert.equal(agentResources(cumulative,['a']).cpu.value,50);
  assert.equal(agentResources({duration:1,agents:[{id:'a',sandbox:'box'}]},['a']).cpu,null);
});
