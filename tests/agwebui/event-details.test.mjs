import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source=await readFile(new URL('../../agency/observability/agwebui/static/event-details.js',import.meta.url),'utf8');
const {eventCard,actionState,eventRelationships,eventMeasurements,payloadValue,payloadWindow,renderPayloadWindow,renderEventCard}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('tool cards promote commands and specific recorded failure without changing agent health',()=>{
  const agent={id:'a',status:'running'};
  const action={id:'t',agent:'a',kind:'test',name:'Bash',arguments:'{"command":"pytest tests/test_scheduler.py -q"}',command:'{"command":"pytest tests/test_scheduler.py -q"}',result:'{"exit_code":1,"stdout":"initialization\\nFAILED tests/test_scheduler.py::test_queue - AssertionError"}',outcome:'failed',duration:1.2};
  const card=eventCard(action,{agents:[agent]});
  assert.equal(card.primary,'pytest tests/test_scheduler.py -q');assert.equal(card.secondary,'Bash');
  assert.equal(card.state.label,'Tool failed');assert.equal(card.state.icon,'!');
  assert.match(card.preview,/Exit 1.*FAILED.*AssertionError/);assert.equal(agent.status,'running');
  assert.equal(card.elapsedText,'Wall interval 1.2s');
});

test('tool cards retain multiline messages and useful arguments with old recording fallback',()=>{
  assert.equal(eventCard({kind:'tool',name:'send',arguments:{message:'First line\nSecond line'},outcome:'running'}).primary,'First line\nSecond line');
  assert.equal(eventCard({kind:'read',name:'read_file',arguments:'{"path":"src/orchestrator.py"}'}).primary,'src/orchestrator.py');
  const old=eventCard({kind:'tool',name:'Bash',command:'echo old',duration:0,outcome:'unknown'});
  assert.equal(old.primary,'echo old');assert.equal(old.elapsed,0);assert.equal(old.state.label,'Outcome unknown');
  assert.equal(eventCard({kind:'tool',name:'Bash'}).primary,'Arguments not captured');
});

test('model cards show captured model identity, emitted output and partial available usage',()=>{
  const card=eventCard({id:'m',kind:'model',agent:'a',model:'recorded-model',outcome:'success',duration:0.9,result:'I will check the scheduler.',tokens:0,output_tokens:7},{model:'default-model'});
  assert.equal(card.primary,'recorded-model');assert.equal(card.preview,'I will check the scheduler.');
  assert.equal(card.previewLabel,'Emitted output');assert.deepEqual(card.usage,['0 input tokens','7 output tokens']);
  const absent=eventCard({kind:'model',outcome:'running'},{model:'Recorded in event metadata'});
  assert.equal(absent.primary,'Model identity unavailable');assert.equal(absent.elapsed,null);assert.deepEqual(absent.usage,[]);
  assert.equal(absent.preview,'No emitted text captured');
  assert.equal(eventCard({kind:'model',tool_call_ids:['t1','t2']}).preview,'2 recorded tool requests');
});

test('old request model fallback is visibly qualified rather than claiming event-time identity',()=>{
  const old={id:'old',kind:'model',agent:'a',outcome:'success',result:'Older output'};
  const run={model:'execution-default',agents:[{id:'a',model:'later-agent-model'}]};
  const card=eventCard(old,run);
  assert.equal(card.primary,'Configured later-agent-model · request identity unrecorded');
  assert.equal(card.secondary,'Model request · agent metadata');assert.equal(card.modelSource,'agent metadata');
  const execution=eventCard({...old,agent:'unknown-agent'},run);
  assert.equal(execution.primary,'Configured execution-default · request identity unrecorded');assert.equal(execution.modelSource,'execution metadata');
  const recorded=eventCard({...old,model:'event-time-model'},run);
  assert.equal(recorded.primary,'event-time-model');assert.equal(recorded.modelSource,'event');
  assert.match(eventCard(old,{model:'long-'.repeat(200)}).primary,/request identity unrecorded$/);
});

test('control cards distinguish requested, accepted and actual known outcomes',()=>{
  const action={kind:'control',agent:'a',target_agent:'a',requested_action:'pause',outcome:'accepted',result:''};
  const card=eventCard(action);
  assert.equal(card.secondary,'pause → a');assert.equal(card.preview,'Accepted · application unconfirmed');
  assert.equal(actionState({...action,outcome:'requested'}).label,'Requested · not acknowledged');
  assert.equal(actionState({...action,outcome:'partial'}).label,'Partial outcome');
  assert.equal(actionState({...action,outcome:'delivered'}).label,'Input delivered · response unconfirmed');
  assert.equal(actionState({...action,outcome:'context_committed'}).label,'Retained context committed · model consumption unknown');
  assert.equal(actionState({...action,outcome:'submitted'}).label,'Invocation submitted · completion unconfirmed');
  assert.equal(eventCard({...action,requested_action:'run',arguments:{task:'review',input:'Review\nthen report'}}).primary,'Run task review · Review\nthen report');
  assert.equal(eventCard({...action,input:'Please inspect\nwithout changing files'}).primary,'Please inspect\nwithout changing files');
});

test('captured output keeps both errors and returned output, and model input uses prompt evidence',()=>{
  assert.equal(payloadValue({error:'specific error',result:'captured result'},'output'),'Recorded error\nspecific error\n\nCaptured output\ncaptured result');
  assert.equal(payloadValue({kind:'model',context:{blocks:[{preview:'first prompt'},{preview:'second prompt'}]}},'input'),'first prompt\n\nsecond prompt');
  assert.equal(payloadValue({arguments:'{}',command:'echo old'},'input'),'echo old');
  assert.match(payloadValue({kind:'model',result:'short preview',model_response:{role:'assistant',tool_calls:[{id:'captured',function:{name:'read',arguments:'{}'}}]}},'output'),/captured/);
});

test('multi-megabyte text is displayed in bounded windows without dropping access to its end',()=>{
  const raw='<script>window.compromised=true</script>\n'+ 'large output\n'.repeat(200000);
  const first=payloadWindow(raw,{limit:4000});assert.equal(first.text.length,4000);assert.equal(first.next,true);assert.equal(first.previous,false);
  const later=payloadWindow(raw,{offset:first.end,limit:4000});assert.equal(later.offset,4000);assert.equal(later.previous,true);
  const hugeRequest=payloadWindow(raw,{limit:Infinity});assert.equal(hugeRequest.text.length,4000);
  assert.equal(payloadWindow(raw,{limit:999999}).text.length,32000);
  const last=payloadWindow(raw,{offset:raw.length-100});assert.equal(last.next,false);assert.equal(last.text,raw.slice(-100));
  const html=renderPayloadWindow(raw);assert(!html.includes('<script>'));assert(html.includes('&lt;script&gt;'));assert(html.length<5000);
  const largeJSON=JSON.stringify({stdout:'recorded text '.repeat(200000),exit_code:1});
  const card=eventCard({kind:'test',outcome:'failed',result:largeJSON,result_preview:'Exit 1 · FAILED specific test'});
  assert.equal(card.preview,'Exit 1 · FAILED specific test');
});

test('all card content is escaped and no raw executable HTML is emitted',()=>{
  const html=renderEventCard({id:'t',kind:'tool',name:'<img onerror=x>',command:'<script>bad()</script>',result:'<iframe src=javascript:bad()>',outcome:'failed'});
  assert(!html.includes('<script>'));assert(!html.includes('<img'));assert(!html.includes('<iframe'));assert(html.includes('&lt;script&gt;'));
});

test('one model response explicitly links several concurrent tool executions in either start order',()=>{
  const model={id:'a:model:m',kind:'model',agent:'a',tool_call_ids:['x','y']};
  const first={id:'a:t1',kind:'read',agent:'a',model_tool_call_id:'x',model_id:model.id,command:'read first'};
  const second={id:'a:t2',kind:'test',agent:'a',model_tool_call_id:'y',command:'pytest second'};
  const otherAgent={id:'b:t',kind:'tool',agent:'b',model_tool_call_id:'x',model_id:model.id};
  const run={actions:[second,otherAgent,model,first]};
  assert.deepEqual(eventRelationships(model,run).links.map(link=>link.id),['a:t2','a:t1']);
  assert.equal(eventRelationships(first,run).links[0].id,model.id);
  assert.equal(eventRelationships(second,run).links[0].evidence,'Matching recorded tool-call ID');
  assert.equal(eventRelationships(otherAgent,run).links.length,0);
});

test('absent metadata and temporal adjacency never produce definite relationships',()=>{
  const model={id:'m',kind:'model',agent:'a',start:1,duration:1,request_id:'r'};
  const tool={id:'t',kind:'tool',agent:'a',start:2,request_id:'r',parent_id:'m'};
  for(const action of [tool,model]){
    const relationships=eventRelationships(action,{actions:[model,tool]});assert.deepEqual(relationships.links,[]);assert.match(relationships.unavailable,/No explicit.*Timing and adjacency/);
  }
});

test('ambiguous and retained-window correlations remain unavailable rather than linking the wrong event',()=>{
  const tool={id:'t',kind:'tool',agent:'a',model_tool_call_id:'same'};
  const models=['m1','m2'].map(id=>({id,kind:'model',agent:'a',tool_call_ids:['same']}));
  const ambiguous=eventRelationships(tool,{actions:[tool,...models]});assert.equal(ambiguous.links.length,0);assert.match(ambiguous.unavailable,/ambiguous/);
  const retained=eventRelationships({...tool,model_id:'missing'},{actions:[tool]});assert.deepEqual(retained.unresolved,['missing']);
});

test('historical measurements never substitute a newer current host sample',()=>{
  const action={id:'a',kind:'test',start:10,duration:2,outcome:'failed',timing:'host_event_boundary'};
  const metrics=eventMeasurements(action,{counters:{'host:cpu_pct':[[5,10],[11,15],[100,90]],'workload_total:memory_mb':[[10,64],[100,1024]],'gpu0:util_pct':[[99,70]]}});
  assert.equal(metrics.samples.find(row=>row.key==='host:cpu_pct').at,11);
  assert.equal(metrics.samples.find(row=>row.key==='host:cpu_pct').value,15);
  assert.match(metrics.samples[0].scope,/Host-wide; not an agent/);
  assert.equal(metrics.samples.find(row=>row.key==='workload_total:memory_mb').at,10);
  assert(!metrics.samples.some(row=>row.key==='gpu0:util_pct'));
  assert.equal(metrics.rows[0].provenance,'calculated from recorded boundaries');
  assert.deepEqual(metrics.rows[0].interval,[10,12]);
});

test('interval, thread and model metrics retain scope, provenance and absent values',()=>{
  const action={kind:'model',start:2,duration:4,execution_duration:3,metadata:{cpu_ms:0,runqueue_ms:17,blocked_ms:'n/a'},tokens:0};
  const metrics=eventMeasurements(action,{source:'synthetic'});
  assert.equal(metrics.rows.find(row=>row.key==='cpu_ms').value,0);
  assert(!metrics.rows.some(row=>row.key==='blocked_ms'));assert(!metrics.rows.some(row=>row.key==='output_tokens'));
  assert.equal(metrics.rows.find(row=>row.key==='tokens').scope,'This model request');
  assert.equal(metrics.rows.find(row=>row.key==='tokens').provenance,'synthetic example');
  assert.match(metrics.rows.find(row=>row.key==='active').provenance,/calculated/);
  assert.match(metrics.rows.find(row=>row.key==='active').note,/remote operation/);
});

test('missing starts, metrics and old recordings remain explicitly unavailable',()=>{
  const metrics=eventMeasurements({kind:'tool',duration:0,missing_start:true});
  assert.deepEqual(metrics.rows,[]);assert.deepEqual(metrics.samples,[]);assert.match(metrics.coverage,/No thread CPU/);assert.match(metrics.coverage,/No recorded resource samples/);
  assert.equal(eventCard({kind:'tool',duration:0,missing_start:true}).elapsedText,'Elapsed unavailable');
  assert.deepEqual(eventMeasurements({kind:'tool',start:10,duration:0,missing_start:true},{counters:{'host:cpu_pct':[[10,20]]}}).samples,[]);
  const missingClock=eventMeasurements({kind:'tool',start:10,duration:1,source:'timestamp_missing_ordered_at_last_observation'},{counters:{'host:cpu_pct':[[10,20]]}});
  assert.deepEqual(missingClock.samples,[]);assert.match(missingClock.coverage,/Start measurement unavailable/);assert(!missingClock.rows.some(row=>row.key==='wall'));
  assert.equal(eventCard({kind:'tool',start:10,duration:1,source:'timestamp_missing_ordered_at_last_observation'}).elapsed,null);
  assert.equal(eventMeasurements({kind:'tool',start:0,duration:1,source:'reconstructed_trace_boundary'}).rows[0].provenance,'reconstructed from recorded span');
  const unspecified=eventMeasurements({kind:'tool',start:0,duration:2},{counters:{cpu:[[0,12]]}});
  assert.equal(unspecified.samples[0].scope,'Recorded series; scope not recorded');
});

test('overlapping calls expose individual intervals and never sum wall-clock elapsed',()=>{
  const run={actions:[{id:'one',kind:'tool',start:0,duration:10},{id:'two',kind:'tool',start:2,duration:8}]};
  const one=eventMeasurements(run.actions[0],run),two=eventMeasurements(run.actions[1],run);
  assert.equal(one.rows[0].value,10);assert.equal(two.rows[0].value,8);
  assert.match(one.rows[0].note,/not a sum/);assert(!one.rows.some(row=>row.value===18));
});
