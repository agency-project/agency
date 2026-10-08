import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source=await readFile(new URL('../../agency/observability/agwebui/static/trajectory-stream.js',import.meta.url),'utf8');
const {changedTrajectoryActions,mergeTrajectory,TrajectoryStream}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

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
