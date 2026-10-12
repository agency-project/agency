import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source=await readFile(new URL('../../agency/observability/agwebui/static/trajectory-stream.js',import.meta.url),'utf8');
const {changedTrajectoryActions,mergeTrajectory,mergeTrajectorySnapshot,boundTrajectory,TrajectoryStream}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('reconnect snapshots merge existing identities and duplicated samples without reviving completed calls',()=>{
  const action={id:'a',start:1,outcome:'success',event_ids:['start','end']};
  const run={actions:[action],episodes:[],counters:{cpu:[[1,2]]},coverage:{}};
  const snapshot={actions:[{id:'a',start:1,outcome:'running',event_ids:['start']},{id:'b',start:2,outcome:'running',event_ids:['b']},{id:'b',start:2,outcome:'running',event_ids:['b']}],counters:{cpu:[[1,2],[2,3]]}};
  mergeTrajectorySnapshot(run,snapshot);mergeTrajectorySnapshot(run,snapshot);
  assert.equal(run.actions[0],action);assert.equal(action.outcome,'success');
  assert.deepEqual(run.actions.map(a=>a.id),['a','b']);
  assert.deepEqual(run.counters.cpu,[[1,2],[2,3]]);
  mergeTrajectory(run,{actions:[{id:'a',outcome:'success',start:0,event_ids:['start','end','late-start']}]});
  assert.equal(action.start,0);assert.equal(action.outcome,'success');
});
test('live retention bounds calls and samples while pinning selected old evidence',()=>{
  const run={actions:[],episodes:[],counters:{cpu:[]}};
  for(let batch=0;batch<20;batch++){
    mergeTrajectory(run,{actions:Array.from({length:500},(_,i)=>({id:`a${batch*500+i}`,start:batch*500+i,outcome:'success'})),counter_samples:{cpu:Array.from({length:500},(_,i)=>[batch*500+i,i])}});
    boundTrajectory(run,['a0']);
    assert(run.actions.length<=2001);assert(run.counters.cpu.length<=2000);
  }
  assert(run.actions.some(a=>a.id==='a0'));assert.match(run.coverage.client_window,/column paging/);
  mergeTrajectory(run,{actions:[{id:'a0',outcome:'failed',event_ids:['late-result']}]});
  assert.equal(run.actions.filter(a=>a.id==='a0').length,1);
  mergeTrajectory(run,{meta:{coverage:{events:20000}}});boundTrajectory(run,['a0']);
  assert.match(run.coverage.client_window,/column paging/); // Coverage survives ordinary backend metadata ticks.
});

test('in-flight retention marks only clipped active coverage and preserves inspected calls',()=>{
  const run={actions:Array.from({length:300},(_,i)=>({id:`active-${i}`,start:i,outcome:'running'})),coverage:{}};
  boundTrajectory(run,['active-0']);
  assert.equal(run.actions.length,201);assert(run.actions.some(a=>a.id==='active-0'));
  assert.equal(run.coverage.client_inflight_windowed,true);
  mergeTrajectory(run,{meta:{coverage:{events:300}}});boundTrajectory(run,['active-0']);
  assert.equal(run.coverage.client_inflight_windowed,true);
  const completed={actions:Array.from({length:2100},(_,i)=>({id:`completed-${i}`,start:i,outcome:'success'})),coverage:{}};
  boundTrajectory(completed);assert.equal(completed.coverage.client_inflight_windowed,undefined);
});
test('updates retain collection, action and episode objects while deduplicating IDs',()=>{
  const run={actions:[{id:'a',outcome:'running',event_ids:['start']}],episodes:[{id:'e',actions:['a']}],agents:[],edges:[]};
  const actions=run.actions,action=actions[0],episode=run.episodes[0];
  const patch={actions:[{id:'a',outcome:'success',event_ids:['start','end']}],episodes:[{id:'e',status:'completed'}],meta:{status:'running'}};
  assert.deepEqual(mergeTrajectory(run,patch),['a']);
  assert.deepEqual(mergeTrajectory(run,patch),[]);
  assert.equal(run.actions,actions);
  assert.equal(run.actions[0],action);
  assert.equal(run.episodes[0],episode);
  assert.equal(run.episodes[0].status,'completed');
});
test('duration ticks do not count as unread meaningful activity',()=>{
  const run={actions:[{id:'a',outcome:'running',event_ids:['start'],duration:1}]};
  assert.deepEqual(mergeTrajectory(run,{actions:[{id:'a',outcome:'running',event_ids:['start'],duration:20}]}),[]);
  assert.deepEqual(changedTrajectoryActions(run.actions,[{id:'a',outcome:'running',event_ids:['start'],duration:30}]),[]);
  assert.deepEqual(changedTrajectoryActions(run.actions,[{id:'a',outcome:'success',event_ids:['start','end']},{id:'b',outcome:'running',event_ids:['new']}]),['a','b']);
});
test('attention signals resolve in place without replacing retained evidence',()=>{
  const signal={id:'input:a',active:true,evidence:['event:1']},run={signals:[signal]};
  mergeTrajectory(run,{signals:[{id:'input:a',active:false,evidence:['event:1']}]});
  assert.equal(run.signals[0],signal);
  assert.equal(signal.active,false);
  assert.deepEqual(signal.evidence,['event:1']);
});
test('multi-thousand-event projections accept bounded incremental batches',()=>{
  const run={actions:[]};
  for(let batch=0;batch<10;batch++)mergeTrajectory(run,{actions:Array.from({length:500},(_,i)=>({id:`a${batch*500+i}`,outcome:'running',event_ids:['start']}))});
  const original=run.actions[0];
  mergeTrajectory(run,{actions:[{id:'a0',outcome:'success',event_ids:['start','end']}]});
  assert.equal(run.actions.length,5000);
  assert.equal(run.actions[0],original);
});
test('reconnect includes recovery cursor and ignores stale frames from the previous socket',()=>{
  const previousSocket=globalThis.WebSocket,previousLocation=globalThis.location;
  const sockets=[];
  class Socket {static OPEN=1;readyState=1;constructor(url){this.url=url;sockets.push(this);}close(){}send(value){this.sent=value;}}
  globalThis.WebSocket=Socket;globalThis.location={protocol:'http:',host:'localhost'};
  let received=0;
  const stream=new TrajectoryStream({runId:'live',replay:false,onMessage:()=>received++,onTransport:()=>{}});
  try {
    sockets[0].onmessage({data:JSON.stringify({type:'snapshot',cursor:12,epoch:'one',run:{}})});
    sockets[0].onmessage({data:JSON.stringify({type:'updates',cursor:12,epoch:'one',patches:[]})});
    assert.equal(received,1);
    stream.reconnect();
    assert(sockets[1].url.includes('cursor=12'));
    assert(sockets[1].url.includes('epoch=one'));
    sockets[0].onmessage({data:JSON.stringify({type:'updates',cursor:15,epoch:'one',patches:[]})});
    assert.equal(received,1);
    sockets[1].onmessage({data:JSON.stringify({type:'updates',cursor:14,epoch:'one',patches:[]})});
    assert.equal(received,2);
    sockets[1].onmessage({data:JSON.stringify({type:'snapshot',cursor:13,epoch:'one',run:{}})});
    assert.equal(received,2); // A delayed snapshot cannot roll back newer incremental evidence.
  } finally {stream.close();globalThis.WebSocket=previousSocket;globalThis.location=previousLocation;}
});


test('execution acknowledgements retain the recovery cursor and command paths stay separate',()=>{
  const previousSocket=globalThis.WebSocket,previousLocation=globalThis.location;
  const sockets=[];
  class Socket {static OPEN=1;readyState=1;constructor(){this.sent=[];sockets.push(this);}close(){}send(value){this.sent.push(JSON.parse(value));}}
  globalThis.WebSocket=Socket;globalThis.location={protocol:'http:',host:'localhost'};
  const received=[];
  const live=new TrajectoryStream({runId:'live',replay:false,onMessage:m=>received.push(m),onTransport:()=>{}});
  const replay=new TrajectoryStream({runId:'saved',replay:true,onMessage:()=>{},onTransport:()=>{}});
  try {
    sockets[0].onmessage({data:JSON.stringify({type:'snapshot',cursor:12,epoch:'one',run:{}})});
    assert.equal(live.sendExecution('pause',{agname:'worker',id:'request'}),true);
    assert.deepEqual(sockets[0].sent[0],{type:'execution_command',command:'pause',agname:'worker',id:'request'});
    sockets[0].onmessage({data:JSON.stringify({type:'execution_command_result',id:'request',queued:true})});
    assert.equal(live.cursor,12);assert.equal(live.epoch,'one');assert.equal(received.length,2);
    assert.equal(live.sendReplay('pause'),false);
    assert.equal(replay.sendExecution('pause_all'),false);
    assert.equal(replay.sendReplay('pause'),true);
    assert.deepEqual(sockets[1].sent,[{type:'pause'}]);
    sockets[0].readyState=3;
    assert.equal(live.sendExecution('resume_all'),false);
    sockets[0].readyState=1;sockets[0].send=()=>{throw Error('send failed');};
    assert.equal(live.sendExecution('resume_all'),false);
  } finally {live.close();replay.close();globalThis.WebSocket=previousSocket;globalThis.location=previousLocation;}
});

test('late counter samples preserve series identity and temporal ordering',()=>{
  const points=[[2,20]],run={counters:{cpu:points}};
  mergeTrajectory(run,{counter_samples:{cpu:[[1,10],[3,30]]}});
  assert.equal(run.counters.cpu,points);
  assert.deepEqual(points,[[1,10],[2,20],[3,30]]);
});

test('measurement commands use their own path, stay out of replay and keep the recovery cursor',()=>{
  const previousSocket=globalThis.WebSocket,previousLocation=globalThis.location;
  const sockets=[];
  class Socket {static OPEN=1;readyState=1;constructor(){this.sent=[];sockets.push(this);}close(){}send(value){this.sent.push(JSON.parse(value));}}
  globalThis.WebSocket=Socket;globalThis.location={protocol:'http:',host:'localhost'};
  const received=[];
  const live=new TrajectoryStream({runId:'live',replay:false,onMessage:m=>received.push(m),onTransport:()=>{}});
  const replay=new TrajectoryStream({runId:'live',replay:true,onMessage:()=>{},onTransport:()=>{}});
  const saved=new TrajectoryStream({runId:'saved',replay:false,onMessage:()=>{},onTransport:()=>{}});
  try {
    sockets[0].onmessage({data:JSON.stringify({type:'snapshot',cursor:7,epoch:'one',run:{}})});
    assert.equal(live.sendMeasurement('end',{id:'m1',elapsed:20}),true);
    assert.deepEqual(sockets[0].sent[0],{type:'measurement_command',command:'end',id:'m1',elapsed:20});
    assert.equal(live.sendMeasurement('hold',{id:'m2'}),true);assert.equal(live.sendMeasurement('reset',{id:'m3'}),true);
    // Agent controls and measurement controls cannot be confused for one another.
    for(const command of ['stop','kill_all','pause','end-at',''])assert.equal(live.sendMeasurement(command,{id:'x'}),false);
    assert.equal(live.sendExecution('end',{id:'x'}),false);
    assert.equal(sockets[0].sent.every(message=>message.type==='measurement_command'),true);assert.equal(sockets[0].sent.length,3);
    sockets[0].onmessage({data:JSON.stringify({type:'measurement_command_result',id:'m1',recorded:true})});
    assert.equal(live.cursor,7);assert.equal(live.epoch,'one');assert.equal(received.at(-1).type,'measurement_command_result');
    assert.equal(replay.sendMeasurement('end',{id:'m4'}),false);assert.equal(saved.sendMeasurement('end',{id:'m5'}),false);
    assert.equal(sockets[1].sent.length+sockets[2].sent.length,0);
    sockets[0].readyState=3;assert.equal(live.sendMeasurement('end',{id:'m6'}),false);
  } finally {live.close();replay.close();saved.close();globalThis.WebSocket=previousSocket;globalThis.location=previousLocation;}
});
