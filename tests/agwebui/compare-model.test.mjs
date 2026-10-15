import assert from 'node:assert/strict';
import {test} from 'node:test';
import {loadModule} from './module-loader.mjs';
const {difference, format, pairAgents, matchable, compatibleSeries, timeWindow, choosePair, historyURL} = await loadModule('compare-model.js');

test('neutral deltas and percentages safely handle zero, missing and nonnumeric baselines', () => {
  assert.equal(difference(10, 15), 5);
  assert.equal(difference(10, 5), -5);
  assert.equal(difference(10, 15, 'percent'), 50);
  assert.equal(difference(0, 0), 0);
  assert.equal(difference(0, 5, 'percent'), null);
  for (const missing of [null, undefined, NaN, Infinity, 'completed']) {
    assert.equal(difference(missing, 5), null);assert.equal(difference(5, missing), null);
  }
  assert.equal(format(null), 'Not available');assert.equal(format(0), '0');assert.equal(format(-0), '0');
});

test('selection stays explicit and stale/live IDs are excluded', () => {
  const runs = [{id: 'a'}, {id: 'b'}];
  assert.deepEqual(choosePair(runs), ['', '']);
  assert.deepEqual(choosePair(runs, 'a', 'b'), ['a', 'b']);
  assert.deepEqual(choosePair(runs, 'live', 'b'), ['', 'b']);
  assert.deepEqual(choosePair(runs, 'a', 'a'), ['a', 'a']);
});

test('swapping reverses delta without recomputing historical data', () => {
  const selected = [{id: 'a', duration: 10}, {id: 'b', duration: 15}];
  assert.equal(difference(...selected.map(run => run.duration)), 5);
  selected.reverse();assert.equal(difference(...selected.map(run => run.duration)), -5);
});

test('different agent counts and unmatched explicit identities remain visible', () => {
  const a = [{id: 'worker'}, {id: 'a-only'}], b = [{id: 'worker'}, {id: 'b-only'}, {id: 'other'}];
  const rows = pairAgents(a, b);
  assert.equal(rows.length, 4);assert.equal(rows[0].a, a[0]);assert.equal(rows[0].b, b[0]);
  assert.equal(rows[1].b, null);assert.equal(rows[2].a, null);
});

test('operation names, indices, lengths and times never establish correspondence', () => {
  assert.equal(matchable({id: 'same', name: 'exec'}, {id: 'same', name: 'exec'}), false);
  assert.equal(matchable({match: ['task', 'invocation']}, {match: ['task', 'invocation']}), true);
  assert.equal(matchable({match: ['task', '1']}, {match: ['task', '2']}), false);
});

test('shared time windows align different recording lengths and clamp pan', () => {
  assert.deepEqual(timeWindow(5, 20), {extent: 20, start: 0, end: 20, span: 20});
  assert.deepEqual(timeWindow(5, 20, 18, 5), {extent: 20, start: 15, end: 20, span: 5});
  assert.deepEqual(timeWindow(null, 20, -4, 5), {extent: 20, start: 0, end: 5, span: 5});
});

test('systems overlays require identical explicit measurement scopes and units', () => {
  const a = {scope: 'host', metric: 'cpu', unit: '%', comparable: true};
  assert.equal(compatibleSeries(a, {...a}), true);
  assert.equal(compatibleSeries(a, {...a, scope: 'workload'}), false);
  assert.equal(compatibleSeries({...a, comparable: false}, a), false);
  assert.equal(compatibleSeries(a, {...a, unit: 'core%'}), false);
});

test('historical trace navigation escapes recorded identities', () => {
  const url = new URL(historyURL('run / a', 'worker & b'), 'http://localhost');
  assert.equal(url.pathname, '/profiler');assert.equal(url.searchParams.get('run'), 'run / a');
  assert.equal(url.searchParams.get('agent'), 'worker & b');assert.equal(url.searchParams.get('view'), 'trajectory');
});
