import {renderEventCard, escapeHTML as esc} from './event-details.js';
import {ExecutionTimeline} from './execution-timeline.js';
import {available, difference, format, metrics, views, pairAgents, compatibleSeries, timeWindow, choosePair, historyURL} from './compare-model.js';

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = {ids: ['', ''], runs: [null, null], view: views.includes(params.get('view')) ? params.get('view') : 'Overview', mode: 'absolute',
  agent: ['', ''], kind: '', query: '', skip: [0, 0], pages: [null, null], alignment: 'independent', sync: false,
  window: {start: 0, span: null}, series: '', chart: 'overlay', agentSkip: 0};
let catalog = [], generation = 0, detailGeneration = 0, controller = new AbortController(), detailController = null;
let timelines = [], searchTimer = null, scrollGuard = false, chartData = null, detailState = null;
const NA = 'Not available';
const runLabel = i => i === 0 ? 'Run A' : 'Run B';
const endpoint = (i, suffix = '') => `/api/compare/runs/${encodeURIComponent(state.ids[i])}${suffix}`;
const value = (v, unit = '') => esc(format(v, unit));
const notice = text => `<p class="compare-note">${esc(text)}</p>`;
const unavailable = text => `<div class="compare-unavailable">Not available.${text ? ' ' + esc(text) : ''}</div>`;
const table = (headers, rows) => `<div class="compare-table-wrap"><table class="compare-table"><thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows || `<tr><td colspan="${headers.length}">${NA}</td></tr>`}</tbody></table></div>`;
async function json(url, signal = controller.signal) {
  const response = await fetch(url, {signal});
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || data.error || `Historical query failed (${response.status}).`);
  return data;
}
function updateURL() {
  const url = new URL(location.href);
  ['a', 'b'].forEach((key, i) => state.ids[i] ? url.searchParams.set(key, state.ids[i]) : url.searchParams.delete(key));
  url.searchParams.set('view', state.view);
  history.replaceState(null, '', url);
}
function cleanup() {
  controller.abort();controller = new AbortController();
  timelines.forEach(timeline => timeline.destroy());timelines = [];
  state.pages = [null, null];chartData = null;
  clearTimeout(searchTimer);searchTimer = null;
  closeInspector();
}
function closeInspector() {
  ++detailGeneration;detailController?.abort();detailController = null;detailState = null;
  $('compare-inspector').hidden = true;$('compare-inspector').innerHTML = '';
}
function header() {
  $('swap').disabled = !state.ids.some(Boolean);
  $('compare-views').innerHTML = views.map(view => `<button data-view="${esc(view)}" class="${view === state.view ? 'active' : ''}" aria-current="${view === state.view ? 'page' : 'false'}">${esc(view)}</button>`).join('');
  state.runs.forEach((run, i) => {
    const host = $(`meta-${i ? 'b' : 'a'}`);
    if (!run) {host.innerHTML = state.ids[i] ? 'Reading recorded metadata…' : 'No run selected';return;}
    const m = run.metadata, met = run.metrics;
    host.innerHTML = `<div><strong>ID</strong> ${esc(run.id)} · <a href="${esc(run.history_url)}">Historical execution ↗</a></div>
      <div><strong>Task</strong> ${value(m.task)} · <strong>Started</strong> ${available(m.execution_started_at) ? esc(new Date(m.execution_started_at * 1000).toISOString()) : NA}</div>
      <div><strong>Execution duration</strong> ${value(met.makespan, 's')} · <strong>Agents</strong> ${value(met.agents)}</div>
      <div><strong>Model</strong> ${value(m.model)} · <strong>Harness</strong> ${value(m.harness)}</div>
      <details><summary>Recorded configuration</summary>${m.configuration ? `<pre>${esc(JSON.stringify(m.configuration, null, 2))}</pre>` : NA}</details>`;
  });
}
function modes() {
  return `<label>Display<select id="metric-mode"><option value="absolute"${state.mode === 'absolute' ? ' selected' : ''}>Absolute values</option><option value="delta"${state.mode === 'delta' ? ' selected' : ''}>Absolute delta · B − A</option><option value="percent"${state.mode === 'percent' ? ' selected' : ''}>Percentage change</option></select></label>`;
}
function compareTable(rows, a, b) {
  return table(state.mode === 'absolute' ? ['Recorded metric', 'Run A', 'Run B'] : ['Recorded metric', state.mode === 'percent' ? '(B − A) / A × 100%' : 'B − A'], rows.map(([key, label, unit]) => {
    const av = a?.[key], bv = b?.[key];
    return `<tr><th scope="row">${esc(label)}</th>${state.mode === 'absolute' ? `<td>${value(av, unit)}</td><td>${value(bv, unit)}</td>` : `<td>${value(difference(av, bv, state.mode), state.mode === 'percent' ? '%' : unit)}</td>`}</tr>`;
  }).join(''));
}
function overview() {
  return `<div class="compare-toolbar">${modes()}</div>${compareTable(metrics, state.runs[0].metrics, state.runs[1].metrics)}${notice('Counts describe recorded operations. Missing token usage makes a total unavailable. Zero baselines have no percentage change; status has no numerical delta. Resource totals retain their recorded workload summary scope. Profiler recording duration includes setup and finalization and is not execution makespan.')}
    ${state.runs.map((run, i) => `<details><summary>${runLabel(i)} · recording coverage</summary><pre>${esc(JSON.stringify(run.coverage, null, 2))}</pre></details>`).join('')}`;
}
function metadataView() {
  const fields = {id: 'Run ID', task: 'Workload / task identifier', execution_started_at: 'Start timestamp', time_origin: 'Recorded time origin', model: 'Model', harness: 'Harness', status: 'Completion status', configuration: 'Recorded configuration', scheduler: 'Scheduler configuration', hardware: 'Hardware', commit: 'Commit hash', environment: 'Environment', parameters: 'Execution parameters', condition: 'Recorded condition', repetition: 'Repetition'};
  const a = state.runs[0].metadata, b = state.runs[1].metadata;
  let rows = Object.entries(fields).map(([key, label]) => {
    // Optional fields absent from both runs are not invented configuration fields.
    if (!['id', 'task', 'execution_started_at', 'model', 'harness', 'status', 'configuration'].includes(key) && !(key in a) && !(key in b)) return '';
    return `<tr class="${a[key] != null && b[key] != null && JSON.stringify(a[key]) !== JSON.stringify(b[key]) ? 'differing' : ''}"><th scope="row">${esc(label)}</th><td>${value(a[key])}</td><td>${value(b[key])}</td></tr>`;
  }).join('');
  rows += `<tr><th scope="row">Agent count</th>${state.runs.map(run => `<td>${value(run.metrics.agents)}</td>`).join('')}</tr>`;
  return table(['Recorded field', 'Run A', 'Run B'], rows) + notice('Differing recorded values have a neutral background. Absent configurations are unavailable; generated-at timestamps are not execution start times.');
}
function filterControls(includeAlignment = false) {
  return `<div class="compare-toolbar"><label>Events<select id="event-kind"><option value="">All recorded operations</option><option value="model"${state.kind === 'model' ? ' selected' : ''}>LLM requests</option><option value="tool"${state.kind === 'tool' ? ' selected' : ''}>Tool calls</option></select></label><label>Search<input id="event-query" type="search" maxlength="200" value="${esc(state.query)}" placeholder="Command, tool, event ID, content"></label>${includeAlignment ? `<label>Alignment<select id="alignment"><option value="independent"${state.alignment === 'independent' ? ' selected' : ''}>Independent</option><option value="time"${state.alignment === 'time' ? ' selected' : ''}>Relative-time window</option></select></label><label><input id="sync-scroll" type="checkbox"${state.sync ? ' checked' : ''}> Synchronize scroll position</label>` : ''}</div>`;
}
function agentFilter(i) {
  return `<label>Agent<input data-agent-filter="${i}" list="agents-${i}" value="${esc(state.agent[i])}" placeholder="All / exact recorded ID" aria-label="${runLabel(i)} agent identifier"><datalist id="agents-${i}">${state.runs[i].agents.map(a => `<option value="${esc(a.id)}">${esc(a.label || a.id)}</option>`).join('')}</datalist></label>`;
}
function sharedWindow() {
  return timeWindow(state.runs[0]?.extent, state.runs[1]?.extent, state.window.start, state.window.span);
}
function rangeControls() {
  const window = sharedWindow();
  return `<div class="compare-toolbar"><button data-window="beginning">Beginning</button><button data-window="back">← Pan</button><button data-window="forward">Pan →</button><button data-window="end">End</button><button data-window="out">− Zoom</button><button data-window="in">+ Zoom</button><button data-window="fit">Fit recordings</button><label>Elapsed offset<input id="range-pan" type="range" min="0" max="${Math.max(0, window.extent - window.span)}" step="any" value="${window.start}"${window.extent === window.span ? ' disabled' : ''}></label><output>${format(window.start, 's')} – ${format(window.end, 's')}</output></div>`;
}
const clockNote = () => notice(`Shared offsets use each recording's ${state.runs.map(run => run.time_basis).join(' / ')}. Execution-start alignment is unavailable when its boundary was not recorded. Events are not matched by sequence or proximity.`);
function pager(i, page) {
  return `<div class="compare-pager"><button data-page="${i}" data-direction="first"${!state.skip[i] ? ' disabled' : ''}>Beginning</button><button data-page="${i}" data-direction="previous"${!state.skip[i] ? ' disabled' : ''}>Previous</button><button data-page="${i}" data-direction="next"${page.next == null ? ' disabled' : ''}>Next</button><button data-page="${i}" data-direction="last">End</button><span>${page.offset + (page.operations.length ? 1 : 0)}–${page.offset + page.operations.length} · ${page.total} captured total</span></div>`;
}
function operationHTML(action, i) {
  return `<details class="compare-operation" data-operation="${esc(action.id)}"><summary>${renderEventCard(action, state.runs[i])}</summary><div class="compare-operation-body"><div>Event ${esc(action.id)} · agent ${esc(action.agent)}</div><div>Start offset ${value(action.start, 's')} · duration ${value(action.duration, 's')} · status ${value(action.outcome)}</div><div>Input / output tokens ${value(action.tokens)} / ${value(action.output_tokens)} · error ${value(action.error)}</div><button data-detail="${esc(action.id)}" data-run="${i}">Inspect recorded details</button>${action.match ? `<button data-match="${esc(action.id)}" data-run="${i}">Inspect explicit task / invocation match</button>` : `<span> · Cross-run correspondence: ${NA}</span>`}</div></details>`;
}
async function operationPages(signal) {
  const window = sharedWindow();
  return Promise.all(state.ids.map((_, i) => {
    const query = new URLSearchParams({skip: Math.max(0, state.skip[i]), tail: state.skip[i] === -1, agent: state.agent[i], kind: state.kind, query: state.query});
    if (state.alignment === 'time' || state.view === 'Timeline') {query.set('start', window.start);query.set('end', window.end);}
    return json(endpoint(i, `/operations?${query}`), signal);
  }));
}
async function trajectory(signal) {
  const pages = await operationPages(signal);
  if (signal.aborted) return;
  state.pages = pages;state.skip = pages.map(page => page.offset);
  $('compare-content').innerHTML = filterControls(true) + (state.alignment === 'time' ? rangeControls() : '') + clockNote() + `<div class="compare-panes">${pages.map((page, i) => {
    const groups = new Map();for (const action of page.operations) {if (!groups.has(action.agent)) groups.set(action.agent, []);groups.get(action.agent).push(action);}
    return `<section class="compare-pane"><header class="compare-pane-header"><strong>${runLabel(i)}</strong>${agentFilter(i)}<a href="${esc(state.runs[i].history_url)}">Historical trace ↗</a></header><div class="compare-feed" data-feed="${i}">${groups.size ? [...groups].map(([agent, actions]) => `<section class="compare-agent-group"><h3>${esc(agent)}</h3>${actions.map(action => operationHTML(action, i)).join('')}</section>`).join('') : '<p class="compare-empty">No recorded operations in this page / filter.</p>'}</div>${pager(i, page)}</section>`;
  }).join('')}</div>` + notice('Each page retains at most 100 operations per run; paging replaces prior rows. Event order is preserved within each agent. Synchronized scrolling shares viewport position only, and does not imply matched operations. Complete captured details load on demand.');
  document.querySelectorAll('[data-feed]').forEach(feed => feed.addEventListener('scroll', () => {
    if (!state.sync || scrollGuard) return;
    const other = document.querySelector(`[data-feed="${1 - Number(feed.dataset.feed)}"]`);
    if (!other) return;
    scrollGuard = true;other.scrollTop = feed.scrollTop;requestAnimationFrame(() => {scrollGuard = false;});
  }));
}
async function timeline(signal) {
  const pages = await operationPages(signal);
  if (signal.aborted) return;
  state.pages = pages;state.skip = pages.map(page => page.offset);
  $('compare-content').innerHTML = filterControls() + rangeControls() + clockNote() + pages.map((page, i) => `<section class="compare-pane"><header class="compare-pane-header"><strong>${runLabel(i)}</strong>${agentFilter(i)}<a href="${esc(state.runs[i].history_url)}">Historical trace ↗</a></header><div id="timeline-${i}" class="compare-timeline"></div>${pager(i, page)}</section>`).join('') + notice('Shared horizontal scale and pan position. Each lane shows up to 100 operations intersecting the window; use Next for additional intervals. Missing end/start timestamps are omitted from interval blocks and remain inspectable in Tool Trajectory. Recorded queue/dependency intervals are shown; unrecorded gaps are not labeled idle.');
  const window = sharedWindow();
  const width = Math.max(100, ($('timeline-0').clientWidth || 800) - 210);
  const scale = width / Math.max(.01, window.span);
  pages.forEach((page, i) => {
    // A small timeline facade references the current page and roster; no trace copy.
    const run = {id: state.ids[i], duration: state.runs[i].extent, agents: state.runs[i].agents, actions: page.operations, counters: {}};
    const navigation = {scale, left: window.start * scale, top: 0, lock: false};
    const viewer = new ExecutionTimeline($(`timeline-${i}`), navigation, id => inspect(i, id));
    viewer.update(run, page.operations, state.agent[i], window.extent);viewer.setFrozen(true);timelines.push(viewer);
  });
  timelines.forEach((viewer, i) => viewer.scroll.addEventListener('scroll', () => {
    if (scrollGuard || !timelines[1 - i]) return;
    scrollGuard = true;timelines[1 - i].scroll.scrollLeft = viewer.scroll.scrollLeft;
    requestAnimationFrame(() => {scrollGuard = false;});
  }));
}
async function agents(signal) {
  const pages = await Promise.all(state.ids.map((_, i) => json(endpoint(i, `/agents?skip=${state.agentSkip}`), signal)));
  if (signal.aborted) return;
  const ids = [...new Set(pages.flatMap(page => page.agents.map(agent => agent.id)))];
  const paired = await Promise.all(state.ids.map((_, i) => json(endpoint(i, `/agents?${new URLSearchParams({identifiers: JSON.stringify(ids)})}`), signal)));
  if (signal.aborted) return;
  const rows = pairAgents(paired[0].agents, paired[1].agents);
  const fields = [['duration', 'Duration (s)'], ['status', 'Status'], ['requests', 'Model requests'], ['tools', 'Tool calls'], ['input', 'Input tokens'], ['output', 'Output tokens'], ['cpu', 'CPU time (s)'], ['rss', 'Peak RSS (MB)'], ['errors', 'Errors']];
  $('compare-content').innerHTML = `<div class="compare-toolbar"><button data-agent-page="previous"${!state.agentSkip ? ' disabled' : ''}>Previous agents</button><button data-agent-page="next"${pages.every(page => page.next == null) ? ' disabled' : ''}>Next agents</button><span>Up to 100 per run · ${pages.map((page, i) => `${runLabel(i)}: ${page.total}`).join(' · ')}</span></div>` + table(['Explicit agent ID', 'Run', ...fields.map(([, label]) => label), 'Trajectory'], rows.map(row => [0, 1].map(i => {
    const agent = i ? row.b : row.a;
    return `<tr><th scope="row">${esc(row.id)}</th><td>${runLabel(i)}${agent ? '' : ' · unmatched'}</td>${fields.map(([key]) => `<td>${value(agent?.[key])}</td>`).join('')}<td>${agent ? `<button data-inspect-agent="${esc(agent.id)}" data-run="${i}">Compare trajectory</button> <a href="${esc(agent.history_url)}">Historical trace ↗</a>` : NA}</td></tr>`;
  }).join('')).join('')) + notice('Only exact recorded agent identifiers establish correspondence. Agents without a counterpart stay visible. Each page selects up to 100 agent identifiers per run and finds their counterparts across the complete recorded roster. Agent CPU/RSS attribution and lifecycle duration are unavailable unless explicitly recorded.');
}
async function llmTools(signal) {
  const data = await Promise.all(state.ids.map((_, i) => json(endpoint(i, '/activity'), signal)));
  if (signal.aborted) return;
  const fields = [['count', 'Recorded calls'], ['input', 'Input tokens'], ['output', 'Output tokens'], ['duration', 'Sum of measured durations (s)'], ['errors', 'Errors'], ['retries', 'Retries']];
  const rows = kind => {
    const names = new Set(data.flatMap(run => run[kind].map(row => row.name)));
    return [...names].map(name => [0, 1].map(i => {
      const row = data[i][kind].find(row => row.name === name);
      return `<tr><th scope="row">${esc(name)}</th><td>${runLabel(i)}</td>${fields.map(([key]) => `<td>${value(row?.[key])}</td>`).join('')}<td>${kind === 'models' ? `${value(row?.ttft_ms?.value)} (${row?.ttft_ms?.samples || 0} samples)` : NA}</td><td>${kind === 'models' ? `${value(row?.throughput?.value)} (${row?.throughput?.samples || 0} samples)` : NA}</td><td>${row ? `<button data-drill="${kind === 'models' ? 'model' : 'tool'}" data-query="${name === NA ? '' : esc(name)}">Recorded calls ↗</button>` : NA}</td></tr>`;
    }).join('')).join('');
  };
  const headers = ['Recorded identity', 'Run', ...fields.map(([, label]) => label), 'Mean recorded TTFT (ms)', 'Mean recorded throughput (tokens/s)', 'Details'];
  const bins = ['<10ms', '10–100ms', '0.1–1s', '1–10s', '10–60s', '60–300s', '≥300s'];
  const distribution = data.flatMap((run, i) => run.tools.map(row => `<tr><th scope="row">${esc(row.name)}</th><td>${runLabel(i)}</td>${row.distribution.map(n => `<td>${n}</td>`).join('')}</tr>`)).join('');
  const aggregate = kind => data.map(run => ({...run.totals[kind], ttft: run.totals[kind].ttft_ms.value, throughput: run.totals[kind].throughput.value}));
  const aggregateMetrics = [['count', 'Recorded calls', ''], ['input', 'Input tokens', ''], ['output', 'Output tokens', ''], ['duration', 'Sum of recorded durations', 's'], ['errors', 'Errors', ''], ['retries', 'Retries', ''], ['ttft', 'Mean recorded TTFT', 'ms'], ['throughput', 'Mean recorded throughput', 'tokens/s']];
  $('compare-content').innerHTML = `<div class="compare-toolbar">${modes()}</div><h2 class="compare-section-title">LLM · aggregate</h2>${compareTable(aggregateMetrics, ...aggregate('models'))}<h2 class="compare-section-title">Tools · aggregate</h2>${compareTable(aggregateMetrics.slice(0, 6), ...aggregate('tools'))}<h2 class="compare-section-title">LLM · per recorded request model</h2>${table(headers, rows('models'))}<h2 class="compare-section-title">Tools · per recorded tool type</h2>${table(headers, rows('tools'))}<h2 class="compare-section-title">Tool duration distribution · completed measured calls</h2>${table(['Tool', 'Run', ...bins], distribution)}${notice('Duration sums describe overlapping recorded operation intervals and are not wall-clock makespan. Partial token, duration, error and retry totals are unavailable. TTFT and throughput means include only explicitly reported samples, with coverage shown. Requests with no recorded model identity are kept separate from configured model metadata.')}<button data-tool-time>Tool duration over elapsed time ↗</button>`;
}
function chartSVG(series, {start, end}, high, unit, title) {
  const w = 1000, h = 215, left = 85, right = 18, top = 25, bottom = 36;
  const span = Math.max(.000001, end - start), height = h - top - bottom;
  let svg = `<svg viewBox="0 0 ${w} ${h}" class="compare-chart" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>`;
  for (let t = 0; t <= 4; t++) {
    const x = left + t / 4 * (w - left - right), y = top + t / 4 * height;
    svg += `<line class="axis" x1="${x}" x2="${x}" y1="${top}" y2="${h - bottom}"/><text x="${x}" y="${h - 12}" text-anchor="middle">${esc(format(start + t / 4 * span, 's'))}</text><text x="${left - 8}" y="${y + 4}" text-anchor="end">${esc(format(high * (1 - t / 4), unit))}</text>`;
  }
  for (const {points, i} of series) for (const [t, v, id] of points) {
    if (!available(t) || !available(v) || t < start || t > end || v < 0) continue;
    const x = left + (t - start) / span * (w - left - right), y = top + height - v / (high || 1) * height;
    svg += `<circle class="sample-${i ? 'b' : 'a'}" cx="${x}" cy="${y}" r="3" tabindex="0"${id ? ` data-detail="${esc(id)}" data-run="${i}"` : ''}><title>${runLabel(i)} · ${esc(format(t, 's'))} · ${esc(format(v, unit))}${id ? ' · ' + esc(id) : ''}</title></circle>`;
  }
  return svg + '</svg>';
}
function drawCharts() {
  const {series, infos, points} = chartData;
  const window = sharedWindow(), matched = compatibleSeries(infos[0], infos[1]);
  const high = points.reduce((max, runPoints) => runPoints.reduce((m, p) => Math.max(m, p[1]), max), 0);
  const header = series.length ? `<div class="compare-toolbar"><label>Recorded series<select id="resource-series" class="compare-series-select">${series.map(s => `<option value="${esc(s.name)}"${state.series === s.name ? ' selected' : ''}>${esc(s.name)} · ${esc(s.scope)}</option>`).join('')}</select></label><label>Charts<select id="chart-mode"><option value="overlay"${state.chart === 'overlay' ? ' selected' : ''}>Overlay</option><option value="stacked"${state.chart === 'stacked' ? ' selected' : ''}>Stacked</option></select></label></div>` : '';
  const charts = state.chart === 'overlay' && matched ? `<div class="compare-chart-shell"><h3>A ● · B ○ · ${esc(infos[0].scope)} · ${esc(infos[0].metric)}</h3>${chartSVG(points.map((p, i) => ({points: p, i})), window, high, infos[0].unit, 'Recorded samples · Run A and Run B')}</div>` : points.map((p, i) => `<div class="compare-chart-shell"><h3>${runLabel(i)} · ${esc(infos[i]?.scope || 'Scope not recorded')}</h3>${p.length ? chartSVG([{points: p, i}], window, matched ? high : p.reduce((m, point) => Math.max(m, point[1]), 0), infos[i]?.unit || '', runLabel(i) + ' recorded samples') : unavailable('No captured samples for this metric and scope in the current time window.')}</div>`).join('');
  $('compare-content').innerHTML = header + rangeControls() + clockNote() + (series.length ? charts : unavailable('No resource samples recorded.')) + (!matched && series.length ? notice('Overlay is unavailable for this selection: a shared metric, unit, and explicit host/workload scope must be recorded in both runs. Process and sandbox identities are kept in separate charts; matching names alone do not establish equivalent scope.') : '') + notice('Recorded samples only; no interpolation or integration. Existing historical sampling and min/max downsampling are retained. Zoom/pan apply to both charts; compatible charts share the Y-axis scale. Hover or focus dots for values. Run A uses blue dots; Run B uses outlined dots. No new CPU/RSS/I/O/stall measurements are collected.');
}
async function systems(signal) {
  const window = sharedWindow();
  const catalogs = await Promise.all(state.ids.map((_, i) => json(endpoint(i, '/systems'), signal)));
  if (signal.aborted) return;
  const series = [...new Map(catalogs.flatMap(run => run.series).map(s => [s.name, s])).values()];
  if (!series.some(s => s.name === state.series)) state.series = series[0]?.name || '';
  const data = state.series ? await Promise.all(state.ids.map((_, i) => json(endpoint(i, `/systems?${new URLSearchParams({name: state.series, start: window.start, end: window.end})}`), signal))) : catalogs;
  if (signal.aborted) return;
  chartData = {series, infos: catalogs.map(run => run.series.find(s => s.name === state.series)), points: data.map(run => run.points)};
  drawCharts();
}
async function critical(signal) {
  const data = await Promise.all(state.ids.map((_, i) => json(endpoint(i, '/critical'), signal)));
  if (signal.aborted) return;
  $('compare-content').innerHTML = `<div class="compare-panes">${data.map((run, i) => `<section class="compare-pane"><header class="compare-pane-header"><strong>${runLabel(i)}</strong><a href="${esc(run.history_url)}">Historical causal graph ↗</a></header>${unavailable(run.reason.replace(/^Not available\.\s*/, ''))}</section>`).join('')}</div>` + notice('The existing historical graph contains partial recorded dependency chains; those are not presented as a complete execution critical path.');
}
async function toolTime(signal) {
  const pages = await operationPages(signal);
  if (signal.aborted) return;
  state.pages = pages;state.skip = pages.map(page => page.offset);
  const points = pages.map(page => page.operations.filter(a => a.kind !== 'model' && available(a.start) && available(a.duration)).map(a => [a.start, a.duration, a.id]));
  const high = points.reduce((m, list) => list.reduce((n, p) => Math.max(n, p[1]), m), 0);
  $('compare-content').innerHTML = rangeControls() + clockNote() + chartSVG(points.map((p, i) => ({points: p, i})), sharedWindow(), high, 's', 'Recorded tool durations by start offset') + pages.map((page, i) => `<div>${runLabel(i)}${pager(i, page)}</div>`).join('') + notice('Measured tool duration by recorded start offset, up to 100 operations per run per page. Click a dot for captured details.');
}
let toolsOverTime = false;
async function render() {
  const ticket = ++generation;cleanup();const signal = controller.signal;
  header();updateURL();
  if (!state.runs.every(Boolean)) {
    $('compare-content').innerHTML = '<div class="compare-empty">Select Run A and Run B to compare their historical recordings.</div>';
    $('compare-status').textContent = catalog.length ? `${catalog.length} finalized historical recordings available` : 'No finalized historical recordings available';return;
  }
  $('compare-content').innerHTML = '<p class="loading-state">Reading recorded comparison data…</p>';
  $('compare-status').textContent = 'Reading historical data…';
  try {
    if (state.view === 'Overview') $('compare-content').innerHTML = overview();
    else if (state.view === 'Metadata') $('compare-content').innerHTML = metadataView();
    else if (state.view === 'Tool Trajectory') await trajectory(signal);
    else if (state.view === 'Timeline') await timeline(signal);
    else if (state.view === 'Agents') await agents(signal);
    else if (state.view === 'LLM & Tools') await (toolsOverTime ? toolTime(signal) : llmTools(signal));
    else if (state.view === 'Systems') await systems(signal);
    else if (state.view === 'Critical Path') await critical(signal);
    if (ticket === generation && !signal.aborted) $('compare-status').textContent = 'Historical recordings · unavailable measurements remain explicit';
  } catch (error) {
    if (signal.aborted || ticket !== generation) return;
    $('compare-content').innerHTML = unavailable(error.message) + '<button data-retry>Retry query</button>';
    $('compare-status').textContent = error.message;
  }
}
async function selectRuns() {
  const ticket = ++generation;cleanup();const signal = controller.signal;
  state.agentSkip = 0;state.runs = [null, null];state.skip = [0, 0];state.agent = ['', ''];state.window = {start: 0, span: null};header();updateURL();
  $('compare-content').innerHTML = '<p class="loading-state">Reading selected historical recordings…</p>';
  try {
    const runs = await Promise.all(state.ids.map((id, i) => id ? json(endpoint(i), signal) : null));
    if (signal.aborted || ticket !== generation) return;
    state.runs = runs;await render();
  } catch (error) {
    if (signal.aborted || ticket !== generation) return;
    $('compare-content').innerHTML = unavailable(error.message) + '<button data-reselect>Retry selection</button>';
    $('compare-status').textContent = error.message;
  }
}
async function refreshCatalog() {
  $('refresh').disabled = true;
  try {
    const data = await json('/api/compare/runs');catalog = data.runs;
    const requested = state.ids.some(Boolean) ? state.ids : [params.get('a'), params.get('b')];
    state.ids = choosePair(catalog, ...requested);
    ['run-a', 'run-b'].forEach((id, i) => {$(id).innerHTML = '<option value="">Select a historical run</option>' + catalog.map(run => `<option value="${esc(run.id)}">${esc(run.title)} · ${esc(run.id)}${run.task ? ' · ' + esc(run.task) : ''}</option>`).join('');$(id).value = state.ids[i];});
    await selectRuns();
  } catch (error) {
    if (error.name === 'AbortError') return;
    $('compare-content').innerHTML = unavailable(error.message) + '<button data-refresh>Retry catalog</button>';
    $('compare-status').textContent = error.message;
  } finally {$('refresh').disabled = false;}
}
async function inspect(i, id, matching = false, detailOffset = 0) {
  const ticket = ++detailGeneration;detailController?.abort();detailController = new AbortController();const signal = detailController.signal;
  const host = $('compare-inspector');host.hidden = false;
  host.innerHTML = '<div class="compare-inspector-header"><strong>Recorded operation</strong><button data-close-inspector>Close</button></div><p>Reading captured details…</p>';
  try {
    const result = await json(endpoint(i, `/operation?${new URLSearchParams({event_id: id, offset: detailOffset})}`), signal);
    let counterpart = null, reason = '';
    if (matching && result.operation.match) {
      const [task, invocation] = result.operation.match;
      // Both source and target identities must be unique, even outside current pages.
      const [self, other] = await Promise.all([i, 1 - i].map(index => json(endpoint(index, `/match?${new URLSearchParams({task, invocation})}`), signal)));
      counterpart = self.operation?.id === id ? other.operation : null;
      reason = self.operation?.id === id ? other.reason : 'The source task / invocation identity is ambiguous.';
    }
    if (signal.aborted || ticket !== detailGeneration) return;
    const action = result.operation;
    const a = i ? counterpart : action, b = i ? action : counterpart;
    const fields = [['agent', 'Agent', ''], ['name', 'Operation', ''], ['start', 'Start offset', 's'], ['duration', 'Duration', 's'], ['outcome', 'Status', ''], ['tokens', 'Input tokens', ''], ['output_tokens', 'Output tokens', ''], ['error', 'Error', '']];
    detailState = {i, id, matching, offset: result.offset, length: result.length, next: result.next};
    host.innerHTML = `<div class="compare-inspector-header"><strong>${runLabel(i)} · ${esc(id)}</strong><button data-close-inspector>Close</button></div>` + (matching ? table(['Field', 'Run A', 'Run B'], fields.map(([key, label, unit]) => `<tr><th scope="row">${esc(label)}</th><td>${value(a?.[key], unit)}</td><td>${value(b?.[key], unit)}</td></tr>`).join('')) + (reason ? unavailable(reason) : notice('Correspondence uses a unique, exact recorded task + invocation identifier in each run.')) : '') +
      `<p><a href="${esc(historyURL(state.ids[i], action.agent, action.id))}">Open historical execution and agent ↗</a>${counterpart ? ` · <button data-counterpart="${esc(counterpart.id)}" data-run="${1 - i}">Inspect matched operation</button>` : ''}</p><h3 class="compare-section-title">Complete captured details · paginated JSON</h3><pre>${esc(result.details)}</pre><div class="compare-details-pages"><button data-detail-page="previous"${!result.offset ? ' disabled' : ''}>Previous</button><button data-detail-page="next"${result.next == null ? ' disabled' : ''}>Next</button><span>${result.offset + 1}–${Math.min(result.offset + 8000, result.length)} of ${result.length} characters</span></div>`;
    host.querySelector('[data-close-inspector]').focus();
  } catch (error) {
    if (signal.aborted || ticket !== detailGeneration) return;
    host.innerHTML = `<div class="compare-inspector-header"><strong>Recorded operation</strong><button data-close-inspector>Close</button></div>${unavailable(error.message)}<button data-detail="${esc(id)}" data-run="${i}">Retry details</button>`;
  }
}
function pan(action) {
  const window = sharedWindow();let start = window.start, span = window.span;
  if (action === 'beginning') start = 0;
  else if (action === 'end') start = window.extent - span;
  else if (action === 'back') start -= span / 2;
  else if (action === 'forward') start += span / 2;
  else if (action === 'in') span = Math.max(.01, span / 2);
  else if (action === 'out') span *= 2;
  else if (action === 'fit') {start = 0;span = null;}
  state.window = {start, span};state.skip = [0, 0];render();
}
document.addEventListener('click', event => {
  const el = event.target.closest('[data-view],[data-page],[data-window],[data-detail],[data-match],[data-close-inspector],[data-inspect-agent],[data-agent-page],[data-drill],[data-tool-time],[data-counterpart],[data-detail-page],[data-retry],[data-reselect],[data-refresh]');
  if (!el) return;
  if (el.dataset.view) {state.view = el.dataset.view;state.skip = [0, 0];toolsOverTime = false;render();}
  else if (el.hasAttribute('data-page')) {
    const i = Number(el.dataset.page), direction = el.dataset.direction;
    state.skip[i] = direction === 'first' ? 0 : direction === 'previous' ? Math.max(0, state.skip[i] - 100) : direction === 'last' ? -1 : state.pages[i].next ?? state.skip[i];render();
  } else if (el.dataset.window) pan(el.dataset.window);
  else if (el.hasAttribute('data-detail')) inspect(Number(el.dataset.run), el.dataset.detail);
  else if (el.hasAttribute('data-match')) inspect(Number(el.dataset.run), el.dataset.match, true);
  else if (el.hasAttribute('data-counterpart')) inspect(Number(el.dataset.run), el.dataset.counterpart);
  else if (el.hasAttribute('data-close-inspector')) closeInspector();
  else if (el.hasAttribute('data-detail-page') && detailState) inspect(detailState.i, detailState.id, detailState.matching, el.dataset.detailPage === 'next' ? detailState.next : Math.max(0, detailState.offset - 8000));
  else if (el.hasAttribute('data-inspect-agent')) {state.agent[Number(el.dataset.run)] = el.dataset.inspectAgent;state.skip = [0, 0];state.view = 'Tool Trajectory';render();}
  else if (el.hasAttribute('data-agent-page')) {state.agentSkip = Math.max(0, state.agentSkip + (el.dataset.agentPage === 'next' ? 100 : -100));render();}
  else if (el.dataset.drill) {state.kind = el.dataset.drill;state.query = el.dataset.query || '';state.skip = [0, 0];state.view = 'Tool Trajectory';render();}
  else if (el.hasAttribute('data-tool-time')) {toolsOverTime = true;state.kind = 'tool';state.query = '';state.skip = [0, 0];render();}
  else if (el.hasAttribute('data-reselect')) selectRuns();
  else if (el.hasAttribute('data-refresh')) refreshCatalog();
  else if (el.hasAttribute('data-retry')) render();
});
document.addEventListener('change', event => {
  const el = event.target;
  if (el.id === 'run-a' || el.id === 'run-b') {state.ids[el.id === 'run-a' ? 0 : 1] = el.value;selectRuns();}
  else if (el.id === 'metric-mode') {state.mode = el.value;render();}
  else if (el.id === 'event-kind') {state.kind = el.value;state.skip = [0, 0];render();}
  else if (el.id === 'alignment') {state.alignment = el.value;state.skip = [0, 0];render();}
  else if (el.id === 'sync-scroll') state.sync = el.checked;
  else if (el.hasAttribute('data-agent-filter')) {state.agent[Number(el.dataset.agentFilter)] = el.value;state.skip = [0, 0];render();}
  else if (el.id === 'range-pan') {state.window.start = Number(el.value);state.skip = [0, 0];render();}
  else if (el.id === 'resource-series') {state.series = el.value;render();}
  else if (el.id === 'chart-mode') {state.chart = el.value;drawCharts();}
});
document.addEventListener('input', event => {
  if (event.target.id !== 'event-query') return;
  state.query = event.target.value;clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {state.skip = [0, 0];render();}, 350);
});
document.addEventListener('keydown', event => {if (event.key === 'Escape') closeInspector();});
$('swap').onclick = () => {
  state.ids.reverse();state.runs.reverse();state.agent.reverse();state.skip.reverse();
  $('run-a').value = state.ids[0];$('run-b').value = state.ids[1];render();
};
$('refresh').onclick = refreshCatalog;
window.addEventListener('pagehide', cleanup);
refreshCatalog();
