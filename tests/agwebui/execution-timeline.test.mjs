import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import vm from 'node:vm';
const source = await readFile(new URL('../../agency/observability/agwebui/static/execution-timeline.js', import.meta.url), 'utf8');
const {eventTimeRange, measurementMarkers} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('timeline intervals use recorded timing and never total overlapping duration', () => {
  assert.deepEqual(eventTimeRange({start: 10, duration: 5, outcome: 'success'}, 90), {start: 10, end: 15});
  assert.deepEqual(eventTimeRange({start: 10, duration: 1, outcome: 'running'}, 90), {start: 10, end: 90});
  assert.deepEqual(eventTimeRange({start: 10, duration: 0, outcome: 'failed'}, 90), {start: 10, end: 10});
  for (const action of [{duration: 5}, {start: 10}, {start: 10, duration: -1}, {start: -1, duration: 3},
    {start: 0, duration: 1, missing_start: true}, {start: 0, duration: 1, source: ['timestamp_missing']}]) {
    assert.equal(eventTimeRange(action, 90), null);
  }
  assert.equal(eventTimeRange({start: 10, outcome: 'running'}, NaN), null);
  assert.equal(eventTimeRange({start: 10, outcome: 'running'}, 9), null);
});

function viewer(navigation = {scale: null, left: 0, top: 0, lock: false}) {
  const document = {activeElement: null};
  class Element {
    constructor() {this.children = [];this.dataset = {};this.style = {};this.handlers = {};this._html = '';}
    set innerHTML(value) {
      this._html = value;this.buttons = [];
      for (const match of value.matchAll(/data-timeline-action="([^"]+)"/g)) {
        const button = new Element();button.dataset.timelineAction = match[1];button.parent = this;this.buttons.push(button);
      }
    }
    get innerHTML() {return this._html;}
    get firstElementChild() {return this.children[0];}
    append(node) {this.insertBefore(node, null);}
    insertBefore(node, before) {node.remove();const index = before ? this.children.indexOf(before) : this.children.length;this.children.splice(index, 0, node);node.parent = this;}
    remove() {if (this.parent) {this.parent.children.splice(this.parent.children.indexOf(this), 1);this.parent = null;}}
    addEventListener(type, fn) {this.handlers[type] = fn;}
    querySelectorAll(selector) {return selector === '[data-track]' ? this.children.filter(node => node.dataset.track) : [];}
    querySelector(selector) {
      if (selector === '.execution-time-axis') return this.children.find(node => node.className === 'execution-time-axis') || null;
      const id = selector.match(/\[data-timeline-action="([^"]+)"\]/)?.[1];
      if (id) return (this.buttons || []).find(node => node.dataset.timelineAction === id) || this.children.map(node => node.querySelector(selector)).find(Boolean) || null;
      return null;
    }
    closest(selector) {return selector === '[data-track]' ? this.parent : selector === '[data-timeline-action]' ? this : null;}
    focus() {document.activeElement = this;}
    getBoundingClientRect() {
      if (!this.parent) return {top: 0};
      const rows = this.parent.children, index = rows.indexOf(this);
      return {top: rows.slice(0, index).reduce((height, row) => height + (parseInt(row.style.height) || 36), 0) - (scroll.scrollTop || 0)};
    }
  }
  document.createElement = () => new Element();
  const canvas = new Element(), scroll = new Element();scroll.append(canvas);
  scroll.clientWidth = 900;scroll.clientHeight = 250;scroll.scrollTop = 0;
  let left = 0;
  Object.defineProperties(scroll, {
    scrollWidth: {get: () => parseFloat(canvas.style.width) || scroll.clientWidth},
    scrollLeft: {get: () => left, set: value => {left = Math.max(0, Math.min(value, scroll.scrollWidth - scroll.clientWidth));}},
  });
  const parts = new Map([['.execution-timeline-scroll', scroll]]), host = new Element();
  host.querySelector = selector => {if (!parts.has(selector)) parts.set(selector, new Element());return parts.get(selector);};
  let historyCount = 0, selected = null;
  const context = vm.createContext({document, CSS: {escape: value => value}, ResizeObserver: class {observe() {}disconnect() {}},
    requestAnimationFrame: () => 1, cancelAnimationFrame: () => {}});
  vm.runInContext(source.replace(/export function/g, 'function').replace('export class ExecutionTimeline', 'class ExecutionTimeline') + '\nglobalThis.ExecutionTimeline = ExecutionTimeline;', context);
  const timeline = new context.ExecutionTimeline(host, navigation, id => {selected = id;}, () => {historyCount++;});
  return {timeline, scroll, canvas, document, nav: navigation, host, historyCount: () => historyCount, selected: () => selected};
}
function run(count = 3) {
  return {duration: 120, agents: Array.from({length: count}, (_, index) => ({id: `a-${index}`})),
    actions: Array.from({length: count}, (_, index) => ({id: `call-${index}`, agent: `a-${index}`, name: `Tool ${index}`, kind: 'tool', start: 10 + index, duration: 4, outcome: 'success'})), counters: {}};
}

test('reveal frames the actual selected interval and focuses its recorded agent row', () => {
  const p = viewer(), execution = run(32);p.timeline.update(execution);
  const result = p.timeline.revealEvent('call-31');
  assert.equal(result.revealed, true);assert.equal(result.range.start, 41);assert.equal(result.range.end, 45);
  assert.equal(p.nav.selected, 'call-31');assert.equal(p.nav.lock, false);assert.equal(p.historyCount(), 1);
  assert.equal(p.document.activeElement.dataset.timelineAction, 'call-31');
  assert(p.scroll.scrollTop > 1000);assert(p.scroll.scrollLeft > 0);
  assert(p.canvas.querySelector('[data-timeline-action="call-31"]').parent.innerHTML.includes('aria-pressed="true"'));
});
test('filtered and unknown events explain why reveal cannot navigate without changing filters', () => {
  const p = viewer(), execution = run(10);p.timeline.update(execution, [execution.actions[0]], 'a-0');
  const before = {...p.nav};
  assert.match(p.timeline.revealEvent('call-8').reason, /filters/);
  assert.match(p.timeline.revealEvent('missing').reason, /not retained/);
  assert.deepEqual(p.nav, before);assert.equal(p.historyCount(), 0);
  execution.actions[0].missing_start = true;
  assert.match(p.timeline.revealEvent('call-0').reason, /timing is unavailable/);
  assert.equal(p.historyCount(), 0);
});
test('timeline navigation retains both axes and zoom across reconstruction', () => {
  const nav = {scale: 30, left: 800, top: 900, lock: false}, first = viewer(nav), execution = run(32);
  first.timeline.update(execution);
  assert.equal(first.scroll.scrollLeft, 800);assert.equal(first.scroll.scrollTop, 900);
  first.scroll.scrollLeft = 1100;first.scroll.scrollTop = 1200;first.timeline.destroy();
  const second = viewer(nav);second.timeline.update(execution);
  assert.equal(second.scroll.scrollLeft, 1100);assert.equal(second.scroll.scrollTop, 1200);assert.equal(second.nav.scale, 30);
});
test('concurrent calls occupy separate lanes and retain selected focus after delayed completion', () => {
  const p = viewer({scale: 5, left: 0, top: 0, lock: false}), execution = run();
  execution.actions.push({id: 'parallel', agent: 'a-0', name: '<script>unsafe</script>', start: 12, duration: 1, outcome: 'running', kind: 'tool'});
  p.timeline.update(execution);p.timeline.revealEvent('parallel');
  const row = p.canvas.querySelector('[data-timeline-action="parallel"]').parent;
  assert.match(row.innerHTML, /top:38px/);assert.match(row.innerHTML, /&lt;script&gt;/);assert(!row.innerHTML.includes('<script>'));
  execution.actions.at(-1).outcome = 'success';execution.actions.at(-1).duration = 4;
  p.timeline.update(execution);
  assert.equal(p.nav.selected, 'parallel');assert.equal(p.document.activeElement.dataset.timelineAction, 'parallel');
});
test('frozen timeline disables live locking while explicit reveal remains local navigation', () => {
  const p = viewer({scale: 5, left: 0, top: 0, lock: true}), execution = run();p.timeline.update(execution);
  p.timeline.setFrozen(true);assert.equal(p.nav.lock, false);assert.equal(p.host.querySelector('[data-lock]').disabled, true);
  assert.equal(p.timeline.revealEvent('call-1', {focus: false}).revealed, true);
  assert.equal(p.document.activeElement, null);assert.equal(p.nav.lock, false);
  p.timeline.setFrozen(false);assert.equal(p.host.querySelector('[data-lock]').disabled, false);assert.equal(p.nav.lock, false);
});

test('unknown timing omits blocks without inventing spans or losing evidence selection', () => {
  const p = viewer({scale: 5, left: 0, top: 0, lock: false}), execution = run();
  execution.actions.push(
    {id: 'missing-start', agent: 'a-0', name: 'End-only evidence', start: 0, duration: 2, outcome: 'success', missing_start: true},
    {id: 'missing-timestamp', agent: 'a-0', name: 'Ordering placeholder', start: 0, duration: 2, outcome: 'success', source: 'timestamp_missing_ordered_at_last_observation'},
    {id: 'absent-start', agent: 'a-0', name: 'No timestamp', duration: 1, outcome: 'failed'},
    {id: 'unknown-end', agent: 'a-0', name: 'No interval', start: 1, outcome: 'success'},
    {id: 'recorded-instant', agent: 'a-0', name: 'Observed instant', start: 0, duration: 0, outcome: 'success'},
  );
  p.timeline.update(execution);p.timeline.setSelection('missing-start');
  for (const id of ['missing-start', 'missing-timestamp', 'absent-start', 'unknown-end']) {
    assert.equal(p.canvas.querySelector(`[data-timeline-action="${id}"]`), null);
  }
  assert(p.canvas.querySelector('[data-timeline-action="recorded-instant"]'));
  const coverage = p.host.querySelector('[data-timing-coverage]');
  assert.equal(coverage.hidden, false);assert.match(coverage.textContent, /4 recorded calls are omitted/);
  assert.match(coverage.textContent, /captured evidence remains inspectable/);
  const before = {...p.nav};
  assert.match(p.timeline.revealEvent('missing-start').reason, /timing is unavailable/);
  assert.deepEqual(p.nav, before);assert.equal(p.nav.selected, 'missing-start');
  assert.equal(p.historyCount(), 0);
  // A late captured start makes this same identity renderable again.
  const recovered = execution.actions.find(action => action.id === 'missing-start');
  recovered.missing_start = false;recovered.start = 6;
  p.timeline.update(execution);
  assert.equal(p.nav.selected, 'missing-start');
  assert(p.canvas.querySelector('[data-timeline-action="missing-start"]'));
  assert.match(coverage.textContent, /3 recorded calls are omitted/);
});

test('open and incomplete tooltip boundaries never claim a confirmed completion', () => {
  const p = viewer({scale: 5, left: 0, top: 0, lock: false}), execution = run();
  execution.actions.push(
    {id: 'open', agent: 'a-0', name: 'Active request', start: 1, duration: 2, outcome: 'running'},
    {id: 'unfinished', agent: 'a-0', name: 'Incomplete capture', start: 1, duration: 8, outcome: 'incomplete'},
  );
  p.timeline.update(execution);
  const html = p.canvas.querySelector('[data-timeline-action="unfinished"]').parent.innerHTML;
  const incompleteTip = html.match(/data-timeline-action="unfinished" title="([^"]+)"/)[1];
  const openTip = html.match(/data-timeline-action="open" title="([^"]+)"/)[1];
  assert.match(incompleteTip, /end not recorded/);assert(!incompleteTip.includes('→'));
  assert.match(openTip, /open through displayed clock/);
  assert.match(openTip, /1.0s → 120.0s/);
});

test('the selected wall-clock end is marked without hiding or changing later calls', () => {
  const automatic = {policy: 'automatic', end_offset: 50, automatic_end_offset: 50};
  assert.deepEqual(measurementMarkers({measurement: automatic}, 120), [{kind: 'automatic', at: 50, label: 'Wall clock end · automatic · 50.0s'}]);
  const manual = measurementMarkers({measurement: {policy: 'manual', end_offset: 20, automatic_end_offset: 50}}, 120);
  assert.deepEqual(manual.map(marker => [marker.kind, marker.at]), [['superseded', 50], ['manual', 20]]);
  assert.match(manual[1].label, /Wall clock end · manual · 20\.0s/);
  // Kept running: no selected end yet; the original automatic boundary stays visible.
  assert.deepEqual(measurementMarkers({measurement: {policy: 'manual', held: true, end_offset: null, automatic_end_offset: 50}}, 120).map(marker => marker.kind), ['superseded']);
  // A boundary chosen after the trace ended is pinned to its edge and says so.
  const late = measurementMarkers({measurement: {policy: 'manual', end_offset: 400, automatic_end_offset: 50}}, 120)[1];
  assert.equal(late.at, 120);assert.match(late.label, /400\.0s · beyond the trace/);
  for (const absent of [{}, {measurement: {policy: 'automatic', end_offset: null, automatic_end_offset: null}}, null]) assert.deepEqual(measurementMarkers(absent, 120), []);

  const p = viewer(), execution = run(3);
  execution.measurement = {policy: 'manual', end_offset: 11, automatic_end_offset: 60};
  p.timeline.update(execution);
  const blocks = () => p.canvas.children.filter(node => node.dataset.track?.startsWith('agent:')).map(row => row.innerHTML).join('');
  // call-1 and call-2 start after the 11s boundary and are still drawn.
  for (const id of ['call-0', 'call-1', 'call-2']) assert.match(blocks(), new RegExp(`data-timeline-action="${id}"`));
  const axis = p.canvas.querySelector('.execution-time-axis').innerHTML;
  assert.match(axis, /timeline-measure-flag manual/);assert.match(axis, /timeline-measure-flag superseded/);
  const lines = () => p.canvas.children.filter(node => String(node.className).startsWith('timeline-measure-line'));
  assert.deepEqual(lines().map(line => line.className), ['timeline-measure-line superseded', 'timeline-measure-line manual']);
  const scale = p.nav.scale, kept = lines()[1];
  assert.equal(kept.style.left, `${180 + 11 * scale}px`);
  // Tracks keep their order ahead of the overlay lines.
  assert.deepEqual(p.canvas.children.slice(1, 4).map(node => node.dataset.track), ['agent:a-0', 'agent:a-1', 'agent:a-2']);
  execution.measurement = {policy: 'automatic', end_offset: 60, automatic_end_offset: 60};
  p.timeline.update(execution);
  assert.deepEqual(lines().map(line => line.className), ['timeline-measure-line automatic']);
  assert.equal(execution.actions.length, 3);assert.match(blocks(), /call-2/);
  delete execution.measurement;p.timeline.update(execution);assert.equal(lines().length, 0);
});
