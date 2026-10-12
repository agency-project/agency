import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const root=new URL('../../agency/observability/agwebui/static/',import.meta.url);
const asURL=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const modelURL=asURL(await readFile(new URL('investigator-model.js',root),'utf8'));
const {systemMeasurements,executionMeasurements}=await import(modelURL);
const prefix=pid=>`gpu0:proc:${pid}-100:sandbox:shared:device:GPU-test`;
const run={duration:2,counters:{
  [prefix(11)+':mem_mb']:[[2,100]], [prefix(12)+':mem_mb']:[[2,200]],
  [prefix(11)+':util_pct']:[[1.4,20],[1.6,30]], [prefix(12)+':util_pct']:[[1.6,40]],
  'gpu0:processes:mem_mb':[[2,300]],'gpu0:sandbox:shared:process_mem_mb':[[2,300]],
  'gpu0:mem_mb':[[2,900]],'gpu0:util_pct':[[2,75]],
}};

test('live and saved gauge identities preserve distinct processes and independent memory totals',()=>{
  for(const latestOnly of [false,true]){
    const metrics=systemMeasurements(run,{latestOnly});
    assert.equal(metrics.filter(m=>m.processIdentity).length,4);
    assert.equal(metrics.find(m=>m.name===prefix(11)+':mem_mb').latest[1],100);
    assert.equal(metrics.find(m=>m.name===prefix(12)+':mem_mb').latest[1],200);
    assert.equal(metrics.find(m=>m.name==='gpu0:processes:mem_mb').latest[1],300);
    assert.equal(metrics.find(m=>m.name==='gpu0:processes:mem_mb').label,'Process VRAM');
    assert.equal(metrics.find(m=>m.name==='gpu0:mem_mb').latest[1],900);
    assert.equal(metrics.find(m=>m.name===prefix(11)+':util_pct').label,'Process SM activity');
    assert.match(metrics.find(m=>m.processIdentity==='11-100').scope,/Process 11-100 · Sandbox shared/);
    assert.equal(metrics.filter(m=>m.aggregate&&m.label==='GPU').length,1);
  }
  const summary=executionMeasurements(run);
  assert.equal(summary.secondary.find(m=>m[0]==='GPU')[1],75);
  assert.equal(summary.secondary.find(m=>m[0]==='VRAM')[1],900);
});

test('zero, no new samples, unsupported, failure and stale measurements remain distinct',()=>{
  const key=prefix(11)+':util_pct';
  const zero={duration:2,counters:{[key]:[[2,0]],[key+':state']:[[2,0]]}};
  assert.equal(systemMeasurements(zero)[0].latest[1],0);
  for(const [code,status] of [[1,'no_new_samples'],[2,'unsupported'],[3,'collection_failure'],[4,'stale'],[5,'unavailable']]){
    const metrics=systemMeasurements({duration:3,counters:{[key]:[[2,80]],[key+':state']:[[2,0],[3,code]]}});
    assert.equal(metrics.length,1);
    assert.equal(metrics[0].status,status);
    assert.equal(metrics[0].latest[1],null);
    assert.deepEqual(metrics[0].points,[[2,80]]);
  }
  assert.equal(systemMeasurements({...zero,duration:13})[0].status,'stale');
  const missing=systemMeasurements({duration:2,counters:{'gpu0:mem_mb:state':[[2,2]]}})[0];
  assert.equal(missing.status,'unsupported');
  assert.equal(missing.latest[1],null);
  assert.deepEqual(missing.points,[]);
});

test('replay respects state timestamps and cached live gauges expire without new counters',()=>{
  const recording={duration:2,counters:{'gpu0:util_pct':[[1,90]],'gpu0:util_pct:state':[[1,0],[3,3]]}};
  assert.equal(systemMeasurements(recording,{latestOnly:true})[0].latest[1],90);
  recording.duration=4;
  assert.equal(systemMeasurements(recording,{latestOnly:true})[0].status,'collection_failure');
  const stalled={duration:2,counters:{'gpu0:util_pct':[[1,90]]}};
  assert.equal(systemMeasurements(stalled,{latestOnly:true})[0].latest[1],90);
  stalled.duration=12;
  assert.equal(systemMeasurements(stalled,{latestOnly:true})[0].status,'stale');
});

test('legacy GPU counters remain readable without fabricating process identities',()=>{
  const metrics=systemMeasurements({duration:1,counters:{'gpu0:worker:mem_mb':[[1,100]],'gpu0:worker:util_pct':[[1,20]],'gpu0:worker:power_w_est':[[1,5]]}});
  assert.equal(metrics.length,3);
  assert.ok(metrics.every(m=>m.scope==='gpu0 · worker'&&m.processIdentity===null));
});

test('summary renders unsupported GPU gauges as unavailable with a reason',async()=>{
  globalThis.localStorage={getItem:()=>null,setItem:()=>{}};
  const metricsURL=asURL(await readFile(new URL('metrics-model.js',root),'utf8'));
  const source=(await readFile(new URL('metrics-ui.js',root),'utf8')).replace('./metrics-model.js',metricsURL).replace('./investigator-model.js',modelURL);
  const {metricsSummary,handleMetricEvent}=await import(asURL(source));
  const target={dataset:{toggle:true},hasAttribute:key=>key==='data-metric-toggle'&&target.dataset.toggle};
  handleMetricEvent({target:{closest:()=>target}},{render:()=>{}});
  target.dataset={metricCategory:'Systems'};target.hasAttribute=()=>false;
  handleMetricEvent({target:{closest:()=>target}},{render:()=>{}});
  const html=metricsSummary({duration:2,agents:[],actions:[],counters:{'gpu0:util_pct:state':[[2,2]]}});
  assert.match(html,/Unavailable/);
  assert.match(html,/Unsupported metric/);
});
