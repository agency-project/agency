import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source=fs.readFileSync(new URL('../../agency/observability/agwebui/static/app.js',import.meta.url),'utf8');
const helpers=source.slice(source.indexOf('function dashboardExecutionAvailable()'),source.indexOf('// Config editor modal'));
function fixture(){
  const sent=[],logs=[],confirmations=[],timers=[];
  let identifier=0;
  const state={timeline:{liveMode:true},executionControls:{available:true,relay_id:'relay-a',execution:{id:'live',time_origin:100},agents:['alpha','beta']},controlRequests:new Map(),agents:new Map([['alpha',{state:'running'}]])};
  const context=vm.createContext({state,ws:{readyState:1,send:text=>sent.push(JSON.parse(text))},WebSocket:{OPEN:1},crypto:{randomUUID:()=>`command-${++identifier}`},requestControlConfirmation:options=>{confirmations.push(options);if(!options.validate())options.onConfirm();},appendLog:text=>logs.push(text),updateAgentActionsBar:()=>{},setTimeout:callback=>timers.push(callback)});
  vm.runInContext(helpers,context);
  return{state,sent,logs,confirmations,timers,context};
}

test('dashboard sends fenced targets once and never fabricates paused state',()=>{
  const fixtureState=fixture();
  vm.runInContext("sendDashboardControl('pause','alpha');sendDashboardControl('pause','alpha')",fixtureState.context);
  assert.equal(fixtureState.sent.length,1);
  assert.deepEqual(fixtureState.sent[0].targets,['alpha']);
  assert.deepEqual(fixtureState.sent[0].execution,{id:'live',time_origin:100});
  assert.equal(fixtureState.sent[0].relay_id,'relay-a');
  assert.equal(fixtureState.state.agents.get('alpha').state,'running');
  vm.runInContext("receiveDashboardControlResult({id:'command-1',queued:true,state:'queued',targets:['alpha']})",fixtureState.context);
  assert.equal(fixtureState.state.controlRequests.size,1);
  vm.runInContext("receiveDashboardControlResult({id:'command-1',state:'accepted',targets:[{agname:'alpha',state:'accepted'}]})",fixtureState.context);
  assert.equal(fixtureState.state.controlRequests.size,0);
  assert.equal(fixtureState.state.agents.get('alpha').state,'running');
});

test('dashboard historical, disconnected, and standalone contexts cannot control agents',()=>{
  for(const change of ["state.timeline.liveMode=false","ws.readyState=3","state.executionControls.available=false"]){
    const fixtureState=fixture();
    vm.runInContext(`${change};sendDashboardControl('pause_all')`,fixtureState.context);
    assert.equal(fixtureState.sent.length,0);
  }
});

test('dashboard destructive confirmations name scope and signal while pause has no confirmation',()=>{
  const fixtureState=fixture();
  vm.runInContext("sendDashboardControl('kill_all')",fixtureState.context);
  assert.equal(fixtureState.confirmations[0].force,true);
  assert.deepEqual([...fixtureState.confirmations[0].targets],['alpha','beta']);
  const routine=fixture();
  vm.runInContext("sendDashboardControl('resume','alpha')",routine.context);
  assert.equal(routine.confirmations.length,0);
});

test('dashboard timeout preserves uncertain request and does not retry',()=>{
  const fixtureState=fixture();
  vm.runInContext("sendDashboardControl('stop','alpha')",fixtureState.context);
  fixtureState.timers[0]();
  assert.match(fixtureState.logs.at(-1),/Outcome unknown; request will not be retried/);
  assert.equal(fixtureState.sent.length,1);
  assert.equal(fixtureState.state.controlRequests.size,1);
});

test('dashboard retains pending application separately from API acceptance',()=>{
  const f=fixture();vm.runInContext("sendDashboardControl('pause','alpha');receiveDashboardControlResult({id:'command-1',targets:[{agname:'alpha',state:'accepted',application:{state:'pending'}}]})",f.context);
  assert.equal(f.state.controlRequests.size,1);assert.match(f.logs.at(-1),/local application pending/);
  vm.runInContext("receiveDashboardControlResult({id:'command-1',targets:[{agname:'alpha',state:'accepted',application:{state:'confirmed'}}]})",f.context);
  assert.equal(f.state.controlRequests.size,0);assert.match(f.logs.at(-1),/local application confirmed/);
});
