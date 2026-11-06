import {profileIndex, profileBounds, graphSelection, swimlaneLayout, processSelection, NODE_LIMIT} from './critical-path-model.js';
import {esc, seconds, renderOverview, renderLaneLabels, renderTimeline, renderInspector, renderProcessTable, graphNotice} from './critical-path-ui.js';

const $ = id => document.getElementById(id), params = new URLSearchParams(location.search);
const state = {id:params.get('run') || '', profile:null, index:null, catalog:[], mode:'critical', agent:params.get('agent') || '', query:'', offset:0, expanded:new Set(), selection:null, crossOnly:false, relationships:false, scale:null, limit:5000, processAgent:'', processQuery:'', processOffset:0};
let generation = 0, controller = null, selection = null, layout = null, frame = null, drag = null, searchTimer = null;
async function json(url, signal) {
  const response = await fetch(url, {signal, cache:'no-store'}), data = await response.json();
  if (!response.ok) throw new Error(data.error || data.detail || `Historical query failed (${response.status})`);
  return data;
}
function status(text, error = false) {$('cp-status').textContent = text; $('cp-status').classList.toggle('error', error);}
function links() {
  const query = new URLSearchParams({run:state.id});
  $('trajectory-link').href = `/profiler?${query}`;
  query.set('view', 'profiler'); $('profiler-link').href = `/profiler?${query}`;
  const url = new URL(location.href); url.searchParams.set('run', state.id);
  if (state.agent) url.searchParams.set('agent', state.agent); else url.searchParams.delete('agent');
  history.replaceState(null, '', url);
}
function fitScale() {
  const bounds = profileBounds(state.profile), width = Math.max(320, $('cp-scroll').clientWidth - 40);
  return width / Math.max(.001, bounds.end - bounds.start);
}
function graphOptions() {return {mode:state.mode, agent:state.agent, query:state.query, offset:state.offset, selected:state.selection?.type === 'operation' ? state.selection.id : '', expanded:state.expanded, crossOnly:state.crossOnly, relationships:state.relationships};}
function paintGraph() {
  frame = null; if (!layout || !selection || !state.index) return;
  const scroller = $('cp-scroll');
  const viewport = {left:scroller.scrollLeft - 60, right:scroller.scrollLeft + scroller.clientWidth + 60, top:scroller.scrollTop - 50, bottom:scroller.scrollTop + scroller.clientHeight + 50};
  $('cp-canvas').innerHTML = renderTimeline(state.index, selection, layout, viewport, state.selection?.id);
  $('cp-lane-labels').style.transform = `translateY(${-scroller.scrollTop}px)`;
}
function schedulePaint() {if (frame === null) frame = requestAnimationFrame(paintGraph);}
function renderGraph(resetPosition = false) {
  if (!state.index) return;
  const scroller = $('cp-scroll');
  selection = graphSelection(state.index, graphOptions()); state.offset = selection.offset;
  state.scale ||= fitScale();
  layout = swimlaneLayout(state.index, selection, state.scale, state.selection?.type === 'operation' ? state.selection.id : '');
  $('cp-canvas').style.width = `${layout.width}px`; $('cp-canvas').style.height = `${layout.height}px`;
  $('cp-lane-labels').innerHTML = renderLaneLabels(layout, state.index, state);
  $('cp-graph-notice').textContent = graphNotice(selection);
  $('cp-page-position').textContent = selection.sequence.length ? `${selection.offset + 1}–${Math.min(selection.offset + NODE_LIMIT, selection.sequence.length)} / ${selection.sequence.length}` : 'No matching operations';
  $('cp-previous').disabled = !selection.offset; $('cp-next').disabled = selection.offset + NODE_LIMIT >= selection.sequence.length;
  $('cp-operation-list').innerHTML = selection.nodes.map(node => `<button data-cp-select="${esc(node.id)}" title="${esc(node.label || node.id)}">${esc(node.agent || 'Unattributed')} · ${esc(node.label || node.id)} · ${seconds(node.duration)}</button>`).join('');
  if (resetPosition) {scroller.scrollLeft = 0; scroller.scrollTop = 0;}
  paintGraph();
}
function renderProcesses() {
  if (!state.index) return;
  const page = processSelection(state.index, {agent:state.processAgent, query:state.processQuery, offset:state.processOffset});
  state.processOffset = page.offset;
  $('cp-process-table').innerHTML = renderProcessTable(page);
  $('cp-process-position').textContent = page.count ? `${page.offset + 1}–${Math.min(page.offset + 50, page.count)} / ${page.count}` : 'No matching processes';
  $('cp-process-previous').disabled = !page.offset; $('cp-process-next').disabled = page.offset + 50 >= page.count;
}
function inspect(value, reveal = false) {
  state.selection = value;
  const inspector = $('cp-inspector'); inspector.hidden = !value;
  $('cp-workspace').classList.toggle('inspecting', Boolean(value));
  inspector.innerHTML = value ? renderInspector(state.index, value) : '';
  renderGraph();
  if (reveal && value?.type === 'operation') {
    const p = layout.positions.get(value.id); if (p) {
      const scroller = $('cp-scroll'); scroller.scrollLeft = Math.max(0, p.x - scroller.clientWidth * .3); scroller.scrollTop = Math.max(0, p.y - scroller.clientHeight * .3); paintGraph();
    }
  }
  if (reveal && value) inspector.querySelector('button')?.focus({preventScroll:true});
}
function zoom(factor, anchor = null) {
  if (!layout) return;
  const scroll = $('cp-scroll'), x = anchor ?? scroll.clientWidth / 2;
  const instant = (scroll.scrollLeft + x - 16) / state.scale;
  const bounds = profileBounds(state.profile), extent = bounds.end - bounds.start;
  // Browser layout sizes are bounded even for multi-day recordings.
  state.scale = Math.max(fitScale() / 4, Math.min(Math.max(fitScale(), 200000 / extent), 300, state.scale * factor));
  renderGraph(); scroll.scrollLeft = Math.max(0, 16 + instant * state.scale - x); paintGraph();
}
function renderProfile() {
  const p = state.profile;
  $('cp-overview').innerHTML = renderOverview(p);
  if ((p.truncation?.operations || 0) > 0 && state.limit < 20000) $('cp-overview').insertAdjacentHTML('beforeend', '<button id="cp-load-more" class="cp-more">Load more span summaries · up to 20,000</button>');
  const oldAgent = state.agent;
  $('cp-agent').innerHTML = '<option value="">All agents</option>' + (p.agents || []).map(a => `<option value="${esc(a.id)}">${esc(a.label || a.id)}</option>`).join('');
  state.agent = [...$('cp-agent').options].some(o => o.value === oldAgent) ? oldAgent : '';
  $('cp-agent').value = state.agent; $('cp-workspace').hidden = false;
  $('cp-process-agent').innerHTML = '<option value="">All agents and unattributed processes</option>' + (p.agents || []).map(a => `<option value="${esc(a.id)}">${esc(a.label || a.id)}</option>`).join('');
  state.processAgent = ''; state.processOffset = 0; state.processQuery = ''; $('cp-process-query').value = '';
  $('cp-process-title').textContent = `Recorded system processes · ${p.processes?.length || 0} retained summaries`;
  $('cp-process-explorer').open = false; renderProcesses();
  $('cp-inspector').hidden = true; $('cp-workspace').classList.remove('inspecting');
  state.selection = null; state.offset = 0; state.expanded.clear(); state.scale = null;
  renderGraph(true); links();
  status(`${p.run?.title || state.id} · ${p.operations?.length || 0} operation summaries · ${p.edges?.length || 0} recorded relationships. Analysis uses the historical recording.`);
}
async function loadRun(id, limit = 5000) {
  if (!id) return;
  const current = ++generation; controller?.abort(); controller = new AbortController();
  state.id = id; state.limit = limit; $('cp-run').value = id;
  $('cp-workspace').hidden = true; $('cp-overview').innerHTML = ''; status('Computing historical dependency analysis…');
  try {
    const p = await json(`/api/critical-path/${encodeURIComponent(id)}?limit=${limit}`, controller.signal);
    if (current !== generation) return;
    state.profile = p; state.index = profileIndex(p); renderProfile();
    const selected = params.get('operation'); if (selected && (state.index.operations.has(selected) || state.index.processes.has(selected))) inspect({type:'operation', id:selected}, true);
  } catch (error) {if (error.name !== 'AbortError' && current === generation) status(error.message, true);}
}
async function refresh() {
  $('cp-refresh').disabled = true;
  const before = generation;
  try {
    const result = await json('/api/critical-path/runs'); state.catalog = result.runs || [];
    $('cp-run').innerHTML = state.catalog.length ? state.catalog.map(run => `<option value="${esc(run.id)}">${esc(run.title || run.id)} · ${esc(run.subtitle || run.id)}${run.source === 'synthetic' ? ' · synthetic fixture' : ''}${run.status ? ' · ' + esc(run.status) : ''}</option>`).join('') : '<option value="">No historical executions found</option>';
    if (!state.catalog.length) {status('No completed recordings found. Record an Agency execution, then refresh this catalog.'); return;}
    const id = state.catalog.some(r => r.id === state.id) ? state.id : state.catalog[0].id;
    $('cp-run').value = id;
    if (before === generation) await loadRun(id, state.limit);
  } catch (error) {status(error.message, true);} finally {$('cp-refresh').disabled = false;}
}
$('cp-run').addEventListener('change', event => loadRun(event.target.value));
$('cp-refresh').addEventListener('click', refresh);
for (const [id, property] of [['cp-mode','mode'], ['cp-agent','agent'], ['cp-cross','crossOnly'], ['cp-relationships','relationships']]) $(id).addEventListener('change', event => {
  state[property] = event.target.type === 'checkbox' ? event.target.checked : event.target.value; state.offset = 0; renderGraph(true); links();
});
$('cp-query').addEventListener('input', event => {state.query = event.target.value; clearTimeout(searchTimer); searchTimer = setTimeout(() => {state.offset = 0; renderGraph(true);}, 160);});
$('cp-fit').addEventListener('click', () => {if (state.profile) {state.scale = fitScale(); renderGraph(true);}});
$('cp-zoom-out').addEventListener('click', () => zoom(.7)); $('cp-zoom-in').addEventListener('click', () => zoom(1.4));
$('cp-previous').addEventListener('click', () => {state.offset = Math.max(0, state.offset - NODE_LIMIT); renderGraph(true);});
$('cp-next').addEventListener('click', () => {state.offset += NODE_LIMIT; renderGraph(true);});
$('cp-process-agent').addEventListener('change', event => {state.processAgent = event.target.value; state.processOffset = 0; renderProcesses();});
$('cp-process-query').addEventListener('input', event => {state.processQuery = event.target.value; state.processOffset = 0; renderProcesses();});
$('cp-process-previous').addEventListener('click', () => {state.processOffset = Math.max(0, state.processOffset - 50); renderProcesses();});
$('cp-process-next').addEventListener('click', () => {state.processOffset += 50; renderProcesses();});
$('cp-scroll').addEventListener('scroll', schedulePaint, {passive:true});
$('cp-scroll').addEventListener('wheel', event => {if (event.ctrlKey || event.metaKey) {event.preventDefault(); zoom(event.deltaY > 0 ? .88 : 1.12, event.clientX - $('cp-scroll').getBoundingClientRect().left);}}, {passive:false});
$('cp-scroll').addEventListener('pointerdown', event => {
  if (event.button !== 0 || event.target.closest('[data-cp-select],[data-cp-edge],[data-cp-group]')) return;
  const scroller = $('cp-scroll'); drag = {x:event.clientX, y:event.clientY, left:scroller.scrollLeft, top:scroller.scrollTop, id:event.pointerId};
  scroller.setPointerCapture(event.pointerId); scroller.classList.add('dragging');
});
$('cp-scroll').addEventListener('pointermove', event => {if (drag?.id !== event.pointerId) return; const scroller = $('cp-scroll'); scroller.scrollLeft = drag.left - event.clientX + drag.x; scroller.scrollTop = drag.top - event.clientY + drag.y;});
for (const name of ['pointerup', 'pointercancel']) $('cp-scroll').addEventListener(name, () => {drag = null; $('cp-scroll').classList.remove('dragging');});
document.addEventListener('click', event => {
  const target = event.target.closest('[data-cp-select],[data-cp-agent],[data-cp-focus],[data-cp-edge],[data-cp-group],[data-cp-group-zoom],[data-cp-close],[data-cp-expand],[data-cp-process-agent],#cp-load-more');
  if (!target || !state.index) return;
  if (target.id === 'cp-load-more') {loadRun(state.id, 20000); return;}
  if (target.hasAttribute('data-cp-group-zoom')) {const first = state.index.operations.get(state.selection?.members?.[0]);zoom(4);if(first){$('cp-scroll').scrollLeft=Math.max(0,16+(first.start-layout.bounds.start)*state.scale-100);paintGraph();}return;}
  if (target.dataset.cpGroup) {const group = layout.displayNodes.find(n => n.id === target.dataset.cpGroup); if (group) inspect({type:'group',id:group.id,members:group.members});return;}
  if (target.hasAttribute('data-cp-close')) {inspect(null); return;}
  if (target.dataset.cpProcessAgent) {state.processAgent = target.dataset.cpProcessAgent; state.processOffset = 0; state.processQuery = ''; $('cp-process-agent').value = state.processAgent; $('cp-process-query').value = ''; $('cp-process-explorer').open = true; renderProcesses(); return;}
  if (target.dataset.cpFocus) {state.agent = target.dataset.cpFocus; state.mode = 'all'; state.offset = 0; $('cp-agent').value = state.agent; $('cp-mode').value = state.mode; renderGraph(true); links(); return;}
  if (target.dataset.cpExpand) {const id = target.dataset.cpExpand; state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id); renderGraph(); return;}
  if (target.dataset.cpSelect) inspect({type:'operation', id:target.dataset.cpSelect}, true);
  if (target.dataset.cpAgent) inspect({type:'agent', id:target.dataset.cpAgent});
  if (target.dataset.cpEdge) inspect({type:'edge', id:target.dataset.cpEdge});
});
document.addEventListener('keydown', event => {if (['Enter', ' '].includes(event.key) && event.target.matches('g[data-cp-select],g[data-cp-group],path[data-cp-edge]')) {event.preventDefault(); event.target.dispatchEvent(new MouseEvent('click', {bubbles:true}));} if (event.key === 'Escape' && state.selection) inspect(null);});
new ResizeObserver(schedulePaint).observe($('cp-scroll'));
refresh();
