import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source = await readFile(new URL('../../agency/observability/agwebui/static/agent-roster.js', import.meta.url), 'utf8');
const {agentRoster, longRunningCalls, rosterAgents, reconcilePinnedAgents, pinAgent, MAX_AGENT_COLUMNS} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('raw syscalls do not become active, failed, or long-running agent calls',()=>{
  const run={duration:60,agents:[{id:'a'}],actions:[
    {id:'send',agent:'a',kind:'syscall',start:0,outcome:'running'},
    {id:'exec',agent:'a',kind:'syscall',start:0,outcome:'failed'},
    {id:'model',agent:'a',kind:'model',start:0,outcome:'running'},
  ]};
  const row=agentRoster(run)[0];
  assert.equal(row.activeCount,1);
  assert.equal(row.failedActionCount,0);
  assert.deepEqual(longRunningCalls(run).map(x=>x.action.id),['model']);
});

function fixture(count) {
  return {id: `fixture-${count}`, duration: 90,
    agents: Array.from({length: count}, (_, index) => ({id: `agent-${index}`, label: `Worker ${index}`, status: index === 1 ? 'completed' : 'running', parent: null})),
    actions: Array.from({length: count}, (_, index) => ({id: `call-${index}`, agent: `agent-${index}`, start: index, duration: 2, outcome: index === count - 1 ? 'failed' : 'success'})),
  };
}
for (const count of [3, 10, 32]) {
  test(`${count}-agent recording defaults to three readable columns and retains every roster actor`, () => {
    const run = fixture(count), pinned = reconcilePinnedAgents(run);
    assert.deepEqual(pinned, ['agent-0', 'agent-1', 'agent-2']);
    const rows = agentRoster(run, {pinnedIds: pinned});
    assert.equal(rows.length, count);
    assert.equal(rows.filter(row => row.pinned).length, 3);
    const failed = rows.at(-1);
    assert.equal(failed.failedActionCount, 1); assert.equal(failed.status, 'running');
    assert.equal(failed.latestFailedAction.id, `call-${count - 1}`);
    assert(rows.every(row => row.parentId === null && row.childIds.length === 0));
  });
}

test('pinning preserves chosen order and cannot replace a focused column at capacity', () => {
  const run = fixture(32), ids = ['agent-8', 'agent-2', 'agent-0'];
  run.agents.reverse();run.agents[0].status = 'failed';
  assert.deepEqual(reconcilePinnedAgents(run, ids, {customized: true}), ids);
  const added = pinAgent(run, ids, 'agent-31');
  assert.equal(added.changed, true); assert.deepEqual(added.ids, [...ids, 'agent-31']);
  const full = ['agent-8', 'agent-2', 'agent-0', 'agent-31', 'agent-6', 'agent-7'];
  const blocked = pinAgent(run, full, 'agent-12');
  assert.equal(MAX_AGENT_COLUMNS, 6); assert.equal(blocked.changed, false);
  assert.deepEqual(blocked.ids, full); assert.match(blocked.reason, /Hide a column/);
  assert.deepEqual(pinAgent(run, full, 'agent-8').ids, full);
  assert.equal(pinAgent(run, full, 'unknown').changed, false);
});
test('a hidden recorded failure is searchable without claiming its agent failed', () => {
  const run = fixture(32), pinned = reconcilePinnedAgents(run);
  const failures = agentRoster(run, {pinnedIds: pinned, status: 'failed-actions'});
  assert.equal(failures.length, 1); assert.equal(failures[0].pinned, false);
  assert.equal(failures[0].status, 'running'); assert.equal(failures[0].id, 'agent-31');
  assert.equal(agentRoster(run, {query: 'worker 31'})[0].id, 'agent-31');
  assert.equal(agentRoster(run, {query: 'AGENT-31'})[0].id, 'agent-31');
  assert.equal(agentRoster(run, {status: 'failed'}).length, 0);
  assert.equal(agentRoster(run, {status: 'completed'})[0].id, 'agent-1');
  assert.equal(agentRoster(run, {pinnedIds: pinned, status: 'hidden'}).length, 29);
});
test('only explicit known parent metadata establishes hierarchy', () => {
  const run = fixture(10);
  run.agents[1].parent = 'agent-0'; run.agents[2].parent = 'agent-0';
  run.agents[3].parent = 'missing-parent'; run.agents[4].label = 'agent-0.child';
  const rows = agentRoster(run), byId = new Map(rows.map(row => [row.id, row]));
  assert.deepEqual(byId.get('agent-0').childIds, ['agent-1', 'agent-2']);
  assert.equal(byId.get('agent-2').parentId, 'agent-0');
  assert.equal(byId.get('agent-3').parentId, null); assert.equal(byId.get('agent-3').recordedParentId, 'missing-parent');
  assert.equal(byId.get('agent-4').parentId, null);
  run.agents[0].parent = 'agent-1';
  const cyclic = agentRoster(run);
  assert.equal(cyclic[0].parentId, null); assert.equal(cyclic[1].parentId, null);
  assert.equal(cyclic[2].parentId, null);
});
test('older action-only recordings retain unknown agent state and no inferred hierarchy', () => {
  const run = {actions: [{id: 'one', agent: 'team.child', outcome: 'failed'}, {id: 'two', agent: 'workflow', outcome: 'running'}]};
  assert.deepEqual(rosterAgents(run), [{id: 'team.child'}]);
  const row = agentRoster(run)[0];
  assert.equal(row.status, 'unknown'); assert.equal(row.parentId, null); assert.equal(row.failedActionCount, 1);
  assert.deepEqual(reconcilePinnedAgents(run), ['team.child']);
});
test('concurrent long calls use recorded starts and the displayed clock with explicit calculated provenance', () => {
  const run = fixture(3);
  run.actions.push({id: 'slow', agent: 'agent-0', outcome: 'running', start: 10},
    {id: 'parallel', agent: 'agent-0', outcome: 'running', start: 20},
    {id: 'finished', agent: 'agent-0', outcome: 'success', start: 0, duration: 80},
    {id: 'missing', agent: 'agent-1', outcome: 'running', start: 0, missing_start: true},
    {id: 'missing-ts', agent: 'agent-1', outcome: 'running', start: 0, source: ['timestamp_missing']});
  const long = longRunningCalls(run, {agent: 'agent-0', clock: 60});
  assert.deepEqual(long.map(item => [item.action.id, item.elapsed]), [['slow', 50], ['parallel', 40]]);
  assert.match(long[0].provenance, /calculated/);
  const row = agentRoster(run, {clock: 60})[0];
  assert.equal(row.activeCount, 2); assert.equal(row.longCalls.length, 2);
  assert.equal(longRunningCalls(run, {clock: 25}).length, 0);
  assert.equal(longRunningCalls(run, {clock: NaN}).length, 0);
});
