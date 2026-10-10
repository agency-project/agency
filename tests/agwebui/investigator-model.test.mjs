import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source = await readFile(new URL('../../agency/observability/agwebui/static/investigator-model.js', import.meta.url),'utf8');
const {alignSequences, overlapDuration, matchesAction, packActionTracks, groupCallStack, executionIdentity, readableExecutionTitle, executionMeasurements} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

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

test('insertion identifies divergence and the next matched reconvergence',()=>{
  const left=['search','read','edit','test'].map(kind=>({kind}));
  const right=['search','read','read','edit','test'].map(kind=>({kind}));
  const rows=alignSequences(left,right);
  assert.equal(rows.filter(r=>!r.match).length,1);
  assert.equal(rows[2].left,null);
  assert.equal(rows[3].reconverged,true);
});
test('deletion and empty sequences retain all unmatched actions',()=>{
  assert.equal(alignSequences([{kind:'edit'}],[])[0].right,null);
  assert.equal(alignSequences([],[{kind:'test'}])[0].left,null);
  assert.deepEqual(alignSequences([],[]),[]);
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
