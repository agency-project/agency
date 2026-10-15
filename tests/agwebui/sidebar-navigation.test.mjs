import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import vm from 'node:vm';

const root = new URL('../../agency/observability/agwebui/static/', import.meta.url);
const html = await readFile(new URL('investigator.html', root), 'utf8');
const script = (await readFile(new URL('investigator.js', root), 'utf8')).replace(/^import .*;\n/gm, '');
const labels = ['Execution Trajectory', 'Profiler', 'Critical Path Profiler', 'Compare Runs'];

test('the initial document has the entire sidebar before JavaScript or run data loads', () => {
  const nav = html.match(/<nav id="views"[^>]*>(.*?)<\/nav>/s)[1];
  for (const label of labels) assert.ok(nav.includes(label), label);
  assert.match(nav, /data-view="trajectory"/);
  assert.match(nav, /data-view="profiler"/);
  assert.match(nav, /href="\/critical-path"/);
  assert.match(nav, /href="\/compare"/);
});

for (const view of ['trajectory', 'profiler']) {
  test(`returning from Compare Runs to ${view} keeps the sidebar during a pending catalog request`, () => {
    const nodes = new Map();
    const node = id => {
      if (!nodes.has(id)) nodes.set(id, {innerHTML: '', dataset: {}, addEventListener() {}});
      return nodes.get(id);
    };
    const requests = [];
    const context = vm.createContext({
      URLSearchParams, URL,
      location: {search: `?view=${view}`},
      document: {getElementById: node, addEventListener() {}},
      window: {addEventListener() {}},
      fetch: url => {
        requests.push(url);
        return new Promise(() => {});
      },
    });
    vm.runInContext(script, context);

    assert.deepEqual(requests, ['/api/investigator/runs']);
    assert.equal(node('refresh').disabled, true);
    for (const label of labels) assert.ok(node('views').innerHTML.includes(label), label);
    assert.match(node('views').innerHTML, new RegExp(`data-view="${view}" class="active" aria-current="page"`));
    assert.equal(node('workspace').dataset.view, view);
    assert.equal(node('view-label').textContent, view === 'profiler' ? 'Profiler' : 'Execution Trajectory');

    // The same chrome remains available while a selected execution is loading.
    const nav = node('views').innerHTML;
    vm.runInContext('state.loading = true; render();', context);
    assert.equal(node('views').innerHTML, nav);
  });
}
