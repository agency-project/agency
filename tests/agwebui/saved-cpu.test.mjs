import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../../agency/observability/agwebui/static/investigator-model.js',import.meta.url),'utf8');
const {agentResources}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const agents=[{id:'a',sandbox:'shared'},{id:'b',sandbox:'shared'},{id:'c',sandbox:'own'}];

for(const format of ['live','saved','legacy'])test(`${format} CPU preserves core percentages for individual and subset selections`,()=>{
  const key=box=>format==='live'?`cg:${box}:cpu_pct`:format==='saved'?`sandbox:${box}:cpu_pct`:`cg:${box}:cpu_us`;
  const points=value=>format==='legacy'?[[0,1000000],[2,1000000+value*20000]]:[[2,value]];
  const run={duration:2,agents,counters:{[key('shared')]:points(200),[key('own')]:points(50)}};
  assert.equal(agentResources(run,['a']).cpu,null);
  assert.equal(agentResources(run,['a']).shared,1);
  assert.deepEqual(agentResources(run,['c']).cpu,{value:50,sandboxes:1});
  assert.deepEqual(agentResources(run,['a','b']).cpu,{value:200,sandboxes:1});
  assert.deepEqual(agentResources(run,['a','c']).cpu,{value:50,sandboxes:1});
  assert.deepEqual(agentResources(run,['a','b','c']).cpu,{value:250,sandboxes:2});
});

test('live/saved aliases count once and use the newest valid observation',()=>{
  const run={duration:2,agents,counters:{'cg:own:cpu_pct':[[1,50]],'sandbox:own:cpu_pct':[[2,75]],'cg:own:cpu_us':[[0,0],[1,500000]]}};
  assert.deepEqual(agentResources(run,['c','c']).cpu,{value:75,sandboxes:1});
});

test('zero is measured; missing ownership or counter data remains unavailable',()=>{
  const run={duration:2,agents:[{id:'a'}],counters:{'sandbox:a:cpu_pct':[[2,999]],'sandbox:box:cpu_pct':[[2,0]]}};
  assert.equal(agentResources(run,['a']).cpu,null);
  run.agents[0].sandbox='box';
  assert.equal(agentResources(run,['a']).cpu.value,0);
  run.counters={};
  assert.equal(agentResources(run,['a']).cpu,null);
});

test('all explicit historical owners and presentation-only owners constrain attribution',()=>{
  const run={duration:2,agents:[{id:'a',sandboxes:['box','own']},{id:'c',sandbox:'own'}],sandbox_owners:{box:['a','b'],own:['a','c']},counters:{'sandbox:box:cpu_pct':[[2,200]],'sandbox:own:cpu_pct':[[2,50]]}};
  assert.equal(agentResources(run,['a']).cpu,null);
  assert.equal(agentResources(run,['a','c']).cpu.value,50);
  assert.equal(agentResources(run,['a','b','c']).cpu.value,250);
});

test('saved stale measurements are not converted into zero',()=>{
  const run={duration:20,agents:[{id:'a',sandbox:'box'}],counters:{'sandbox:box:cpu_pct':[[1,80]]}};
  assert.equal(agentResources(run,['a']).cpu,null);
});
