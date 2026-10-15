import {systemMeasurements, agentResources, systemActivity} from './investigator-model.js';
import {profilerData, breakdown, timed, failure, toolCall, hasStart, operationState, performanceFindings, processResources, sampleExtrema} from './profiler-model.js';

const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const num = v => v == null || !Number.isFinite(v) ? 'Unavailable' : new Intl.NumberFormat('en', v !== 0 && Math.abs(v) < .1 ? {maximumSignificantDigits:3} : {maximumFractionDigits: 1, notation: Math.abs(v) >= 10000 ? 'compact' : 'standard'}).format(v);
const sec = v => v == null ? 'Unavailable' : v > 0 && v < .1 ? `${new Intl.NumberFormat('en', {maximumSignificantDigits:3}).format(v * 1000)}ms` : `${num(v)}s`;
const latency = v => v == null ? 'Unavailable' : `${new Intl.NumberFormat('en', {maximumFractionDigits:2}).format(v * 1000)} ms / ${new Intl.NumberFormat('en', {maximumFractionDigits:3}).format(v)} s`;
const byte = v => v == null ? 'Unavailable' : `${num(v)} MiB`;
const largestKnown = values => {const known = values.filter(Number.isFinite); return known.length ? known.reduce((a, b) => Math.max(a, b)) : null;};
const colors = {llm:'var(--model-color)', tools:'var(--tool-color)', waiting:'var(--amber)', unknown:'var(--muted)'};
const labels = {llm:'LLM requests', tools:'Tool execution', waiting:'Measured waiting', unknown:'Unknown / unclassified'};
const tableState = new Map();
const tableLimit = 50;
const rowState = id => {if (!tableState.has(id)) tableState.set(id, {page:0, sort:'', direction:-1, query:''}); return tableState.get(id);};
export function resetProfilerTables() {tableState.clear();}
const empty = text => `<p class="prof-empty" role="status">${esc(text)}</p>`;
const metric = (label, value, note = '') => `<div title="${esc(note)}"><small>${esc(label)}</small><strong>${esc(value)}</strong>${note ? `<span>${esc(note)}</span>` : ''}</div>`;
const strip = rows => `<div class="prof-kpis">${rows.map(r => metric(...r)).join('')}</div>`;
const section = (title, content, note = '') => `<section class="prof-section"><h3>${esc(title)}</h3>${note ? `<p class="prof-definition">${esc(note)}</p>` : ''}${content}</section>`;
const actionLink = (a, text) => `<button data-prof-operation="${esc(a.id)}">${esc(text || a.name || a.id)} ↗</button>`;
const select = (key, title, values, selected) => `<label>${esc(title)} <select data-prof-filter="${key}" aria-label="${esc(title)}"><option value="">All ${esc(title.toLowerCase())}</option>${[...new Set(values)].sort().map(v => `<option value="${esc(v)}" ${selected === v ? 'selected' : ''}>${esc(v)}</option>`).join('')}</select></label>`;

// All large tables are bounded. Separate small state objects, no saved derived snapshots.
function table(id, rows, columns, {search = true, defaultSort = ''} = {}) {
  const state = rowState(id); if (!state.sort) state.sort = defaultSort;
  const query = state.query.toLowerCase(), matches = query ? rows.filter(r => columns.some(c => String(r[c.key] ?? '').toLowerCase().includes(query))) : rows;
  const ordered = [...matches];
  if (state.sort) ordered.sort((a, b) => {
    const av = a[state.sort], bv = b[state.sort];
    if (av == null) return bv == null ? 0 : 1; if (bv == null) return -1;
    return state.direction * (typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av).localeCompare(String(bv)));
  });
  state.page = Math.max(0, Math.min(state.page, Math.ceil(ordered.length / tableLimit) - 1));
  return `<div class="prof-table-controls">${search ? `<label>Search <input data-prof-search="${id}" aria-label="Search ${esc(id)}" value="${esc(state.query)}" placeholder="Filter recorded rows"></label>` : ''}<span>${num(ordered.length)} rows · ${state.page + 1}/${Math.max(1, Math.ceil(ordered.length / tableLimit))}</span><button data-prof-page="${id}" data-delta="-1" ${!state.page ? 'disabled' : ''}>Previous</button><button data-prof-page="${id}" data-delta="1" ${(state.page + 1) * tableLimit >= ordered.length ? 'disabled' : ''}>Next</button></div><div class="prof-table-scroll"><table class="prof-table"><thead><tr>${columns.map(c => `<th scope="col" aria-sort="${state.sort === c.key ? state.direction === 1 ? 'ascending' : 'descending' : 'none'}"><button data-prof-sort="${id}" data-key="${c.key}" title="${esc(c.note || 'Sort by ' + c.label)}">${esc(c.label)}${state.sort === c.key ? state.direction === 1 ? ' ↑' : ' ↓' : ''}</button></th>`).join('')}</tr></thead><tbody>${ordered.slice(state.page * tableLimit, (state.page + 1) * tableLimit).map(r => `<tr>${columns.map(c => `<td>${c.render ? c.render(r) : esc(r[c.key] ?? 'Unavailable')}</td>`).join('')}</tr>`).join('') || `<tr><td colspan="${columns.length}">No matching observations.</td></tr>`}</tbody></table></div>`;
}

function chart(title, observations, unit, data, {color = colors.llm, scatter = false, indexed = false, description = ''} = {}) {
  const points = observations.filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]) && (indexed || p[0] >= data.bounds[0] && p[0] <= data.bounds[1])).sort((a, b) => a[0] - b[0]);
  if (!points.length) return `<figure class="prof-chart"><figcaption>${esc(title)}</figcaption>${empty('No reliable observations in the selected interval.')}</figure>`;
  const [start, stop] = indexed ? [0, Math.max(1, points.at(-1)[0])] : data.bounds;
  const span = Math.max(.001, stop - start), max = Math.max(.001, points.reduce((n, p) => Math.max(n, p[1]), 0));
  const x = t => 64 + (t - start) / span * 636, y = v => 164 - v / max * 132, sampled = sampleExtrema(points);
  const ticks = [0, .5, 1], cursor = data.cursor;
  const spikes = data.threshold ?? Infinity;
  const marks = scatter ? sampled.map(p => `<circle tabindex="0" role="button" aria-label="${esc(`${p[2]?.agent || ''} ${p[2]?.name || ''} ${num(p[1])} ${unit} at ${sec(p[0])}`)}" ${p[2]?.id ? `data-prof-operation="${esc(p[2].id)}"` : ''} cx="${x(p[0]).toFixed(1)}" cy="${y(p[1]).toFixed(1)}" r="${p[1] > spikes ? 4 : 2.8}" fill="${p[1] > spikes ? 'var(--amber)' : color}"><title>${esc(`${p[2]?.agent || ''} · ${p[2]?.model || p[2]?.name || ''} · ${num(p[1])} ${unit} · at ${sec(p[0])}`)}</title></circle>`).join('') : `<polyline points="${sampled.map(p => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.5"/>${sampled.length === 1 ? `<circle cx="${x(sampled[0][0])}" cy="${y(sampled[0][1])}" r="3" fill="${color}"/>` : ''}`;
  return `<figure class="prof-chart"><figcaption>${esc(title)} <small>${num(points.length)} observations</small></figcaption><svg viewBox="0 0 720 212" role="img" aria-label="${esc(title)}" ${indexed ? '' : `data-prof-chart data-start="${start}" data-end="${stop}"`} style="--series-color:${color}">${ticks.map(t => `<line class="prof-grid" x1="64" x2="700" y1="${y(t * max)}" y2="${y(t * max)}"/><text x="57" y="${y(t * max) + 4}" text-anchor="end">${num(t * max)}</text><text x="${64 + t * 636}" y="184" text-anchor="${t === 0 ? 'start' : t === 1 ? 'end' : 'middle'}">${num(start + t * span)}</text>`).join('')}<text x="64" y="17">${esc(unit === 's' ? 'Latency (s)' : unit)}</text>${marks}${!indexed && Number.isFinite(cursor) && cursor >= start && cursor <= stop ? `<line class="prof-playhead" x1="${x(cursor)}" x2="${x(cursor)}" y1="30" y2="164"/>` : ''}<rect class="prof-drag-selection" y="30" height="134" width="0"/><text x="382" y="206" text-anchor="middle">${indexed ? 'Sorted observation index' : 'Time since recording start (s)'}</text></svg><p class="prof-chart-description">${esc(description)} ${sampled.length < points.length ? 'Bucket extrema preserve spikes; zoom or use the inspector table to reach every observation.' : ''}${indexed ? '' : ' Drag across the chart to select and zoom an interval.'}</p></figure>`;
}

function callTable(id, calls, run, options) {
  const rows = calls.map(a => ({...a, status:operationState(a, run, options), duration:timed(a) ? a.duration : null}));
  return table(id, rows, [
    {key:'agent', label:'Agent'}, {key:'name', label:id.startsWith('requests') ? 'Request / model' : 'Operation', render:a => actionLink(a, a.model || a.name)},
    {key:'status', label:'Status', render:a => esc(a.status === 'running' ? 'Running · observed' : a.status === 'incomplete' ? 'Unknown / incomplete' : a.status)},
    {key:'start', label:'Start / end', render:a => !hasStart(a) ? 'Start missing' : `${sec(a.start)} / ${timed(a) ? sec(a.start + a.duration) : 'End unobserved'}`},
    {key:'duration', label:'E2E duration', render:a => latency(a.duration)},
    ...(id.startsWith('requests') ? [{key:'tokens', label:'Input tokens', render:a => num(a.tokens)}, {key:'output_tokens', label:'Output tokens', render:a => num(a.output_tokens)}] : []),
    {key:'error', label:'Error / metadata', render:a => `${esc(String(a.error || '').slice(0, 180))} ${actionLink(a, 'Inspect')}`}
  ], {defaultSort:'start'});
}
function groupTable(id, groups, llm, dimension) {
  return table(id, groups, [
    {key:'key', label:dimension, render:g => `<button data-prof-${dimension === 'Agent' ? 'agent' : llm ? 'model' : 'tool'}="${esc(g.key)}">${esc(g.key)} ↗</button>`},
    {key:'calls', label:llm ? 'Requests' : 'Calls'}, {key:'p50', label:'P50', render:g => sec(g.p50)},
    {key:'p95', label:'P95', render:g => `${sec(g.p95)}${g.samples < 20 ? ' · sparse' : ''}`},
    ...(llm ? [{key:'inputValue', label:'Input tokens', render:g => `${num(g.input.value)} · ${g.input.samples}/${g.calls}`}, {key:'outputValue', label:'Output tokens', render:g => `${num(g.output.value)} · ${g.output.samples}/${g.calls}`}, {key:'throughput', label:'Output / E2E s', render:g => num(g.throughput)}] : [{key:'total', label:'Summed duration', render:g => sec(g.total)}]),
    {key:'failures', label:'Failures'}
  ], {defaultSort:llm ? 'calls' : 'total'});
}

function timeBreakdown(data) {
  const rows = data.agents.map(a => ({...a, key:a.id}));
  return table('agent-time', rows, [{key:'label', label:'Agent', render:a => `<button data-prof-agent="${esc(a.id)}">${esc(a.label || a.id)}</button>`},
    {key:'elapsed', label:'Selected wall interval', render:a => {
      const t = a.attribution, total = t.elapsed;
      if (!total || !t.measured) return empty('Incomplete attribution · no reliably classified intervals.');
      return `<div class="prof-time-bar" aria-label="${esc(a.label || a.id)} wall-time attribution">${Object.keys(labels).filter(k => t.totals[k] > 0).map(k => `<button class="prof-time-segment" style="width:${100 * t.totals[k] / total}%;background:${colors[k]}" data-prof-time-agent="${esc(a.id)}" data-kind="${k}" title="${esc(labels[k])}: ${sec(t.totals[k])}. Select supporting intervals."><span class="sr-only">${esc(labels[k])}: ${sec(t.totals[k])}</span></button>`).join('')}</div><div class="prof-time-values">${Object.keys(labels).filter(k => t.totals[k] > 0).map(k => `${esc(labels[k])} ${sec(t.totals[k])}`).join(' · ')}</div>`;
    }}], {search:false});
}
function overview(run, data, options, resources) {
  const totals = !options.agent && !options.range ? run.metrics_totals : options.agent && !options.range ? run.metrics_by_agent?.[options.agent] : null;
  const memory = resources.find(m => m.scope === 'Workload' && m.label === 'RAM'), peak = memory?.points.length ? memory.points.reduce((n, p) => Math.max(n, p[1]), 0) : null;
  const states = new Map(); for (const a of data.agents) states.set(a.status || 'unknown', (states.get(a.status || 'unknown') || 0) + 1);
  let html = strip([
    ['Wall clock', sec(Number.isFinite(run.execution_wall_time) ? run.execution_wall_time : data.bounds[1] - data.bounds[0]), 'Execution boundary; operations may overlap'],
    ['Agents', num(data.agents.length), [...states].map(([s, n]) => `${n} ${s}`).join(' · ')],
    ['LLM requests', num(totals?.models ?? data.llm.calls), 'Observed request identities'], ['Tool calls', num(totals?.tools ?? data.tool.calls), 'Observed tool identities'],
    ['Failed operations', num(data.actions.filter(failure).length), 'Retained failures / timeouts'], ['Peak workload memory', byte(peak), memory ? 'Recorded workload memory · selected observations' : 'Workload memory not recorded']
  ]);
  html += section('Where execution time went', timeBreakdown(data), 'Each row covers the same selected wall interval. Same-category overlaps are unioned; conflicting simultaneous categories and unobserved intervals are unknown. Agent lifecycle boundaries are unavailable; this is coverage of the execution window, not agent lifetime.');
  const findings = performanceFindings(data, run, resources);
  html += section('Performance findings', findings.length ? `<div class="prof-findings">${findings.map(f => `<article><strong>${esc(f.title)}</strong><p>${esc(f.evidence)}</p><button data-prof-finding="${f.tab}" data-agent="${esc(f.agent || '')}" data-tool="${esc(f.tool || '')}" data-operation="${esc(f.action || '')}" ${f.range ? `data-from="${f.range[0]}" data-to="${f.range[1]}"` : ''}>Inspect evidence →</button></article>`).join('')}</div>` : empty('No significant findings in the available observations.'));
  const health = [];
  if (options.transport && options.transport !== 'connected') health.push(`Live updates ${options.transport}; last received evidence cannot establish current call liveness.`);
  if (run.coverage?.gaps?.length) health.push(...run.coverage.gaps.map(String));
  if (run.coverage?.catching_up) health.push('Recovering events; trace coverage is provisional.');
  if (run.coverage?.client_window) health.push('Client history is retained in a bounded window. Full evidence remains in export and recorded inspection.');
  if (data.incomplete.length) health.push(`${data.incomplete.length} calls with missing completion or unknown state.`);
  const timeouts = data.actions.filter(a => a.outcome === 'timeout').length, failures = data.actions.filter(failure).length;
  if (failures) health.push(`${failures} failed operations, including ${timeouts} explicitly recorded timeouts.`);
  const retries = data.actions.filter(a => Number.isFinite(a.metadata?.retry_count)).reduce((n, a) => n + a.metadata.retry_count, 0);
  if (retries) health.push(`${retries} explicitly recorded retries.`);
  if (!resources.length) health.push('Resource samples unavailable.');
  html += section('Execution health', health.length ? `<ul class="prof-health">${health.slice(0, 12).map(h => `<li>${esc(h)}</li>`).join('')}</ul>` : empty('No execution or coverage problems observed.'), 'Open calls are labelled running only while execution and telemetry support it. Without per-call heartbeats, unmatched starts older than 60s are unresolved; this does not establish failure. Retry and timeout counts require explicit instrumentation.');
  return html + `<details class="prof-raw"><summary>Raw activity · ${num(data.actions.length)} retained operations</summary>${callTable('activity', data.actions, run, options)}</details>`;
}

function llmView(run, data, options) {
  const s = data.llm, totals = !options.range && !options.model ? options.agent ? run.metrics_by_agent?.[options.agent] : run.metrics_totals : null;
  const usage = (key, samples, retained) => totals ? `${num(totals[samples] ? totals[key] : null)} · ${totals[samples] || 0}/${totals.models || 0} reported` : `${num(retained.value)} · ${retained.samples}/${s.calls} reported`;
  let html = strip([['Requests', num(totals?.models ?? s.calls)], ['P50 E2E', latency(s.p50), `${s.samples} timed completions`], ['P95 E2E', latency(s.p95), s.samples < 20 ? 'Sparse tail estimate · fewer than 20 samples' : `${s.samples} samples`], ['Input tokens', usage('input', 'input_samples', s.input)], ['Output tokens', usage('output', 'output_samples', s.output)], ['Observed output / s', num(s.throughput), 'Reported output / summed E2E intervals; not decode speed'], ['Failures', num(s.failures)]]);
  html += `<div class="prof-local-filters">${select('model', 'Models', (run.actions || []).filter(a => a.kind === 'model').map(a => a.model || 'Unknown'), options.model)}</div>`;
  const done = data.models.filter(timed);
  html += chart('Request latency over time', done.map(a => [a.start + a.duration, a.duration, a]), 's', {...data, threshold:Math.max(5, (s.p50 || 0) * 3)}, {scatter:true, description:'Spikes show when request latency increased. Select a point to inspect the recorded request.'});
  html += section('Context and response growth', `<div class="prof-charts">${chart('Input tokens per request', data.models.filter(hasStart).map(a => [a.start, a.tokens, a]), 'tokens', data, {scatter:true, description:'Provider-reported prompt/context size. Compare with request latency; co-occurrence is not causation.'})}${chart('Output tokens per request', data.models.filter(hasStart).map(a => [a.start, a.output_tokens, a]), 'tokens', data, {scatter:true, description:'Provider-reported response size. Missing usage is excluded.'})}</div>`);
  for (const [dimension, title] of [['model', 'Model'], ['agent', 'Agent']]) {
    const groups = breakdown(data.models, dimension).map(g => ({...g, inputValue:g.input.value, outputValue:g.output.value}));
    html += section(`By ${title.toLowerCase()}`, groupTable('llm-' + dimension, groups, true, title));
  }
  html += section('Request inspector', callTable('requests', data.models, run, options), 'Inspect opens the existing evidence panel with captured payloads, errors, timing and operation metadata.');
  html += `<details class="prof-raw"><summary>Secondary analytics and timing coverage</summary>${chart('Duration distribution (sorted observations)', done.map(a => a.duration).sort((a, b) => a - b).map((v, i) => [i, v]), 's', data, {indexed:true})}${chart('Client E2E output throughput', done.filter(a => a.duration > 0 && Number.isFinite(a.output_tokens)).map(a => [a.start + a.duration, a.output_tokens / a.duration, a]), 'tokens/s', data, {scatter:true,description:'Reported output tokens / completed end-to-end request duration. This client-observed rate includes waiting; it is not model decode speed.'})}<p class="prof-definition">TTFT: unavailable. First-token boundary is not projected. Decode speed, server queue/prefill timings, cached tokens and reliable pricing are unavailable. First-chunk or E2E timestamps are never substituted for TTFT. Missing usage is excluded and reporting coverage appears beside totals.</p></details>`;
  return html;
}

function toolsView(run, data, options) {
  const s = data.tool, groups = breakdown(data.tools, 'name'), order = options.toolOrder || 'total';
  const ordered = [...groups].map(g => ({...g, average:g.samples ? g.total / g.samples : null, errorRate:g.calls ? g.failures / g.calls : 0})).sort((a, b) => (b[order] ?? -1) - (a[order] ?? -1));
  let html = strip([['Completed calls', num(s.completed)], ['Active observed', num(data.active.length), 'Observed starts within 60s; older unmatched calls are unresolved without heartbeats'], ['P50', latency(s.p50), `${s.samples} timed calls`], ['P95', latency(s.p95), s.samples < 20 ? 'Sparse tail estimate' : `${s.samples} samples`], ['Slowest completed', latency(s.slowest?.duration)], ['Failed calls', num(s.failures)], ['Tool types', num(groups.length)]]);
  html += `<div class="prof-local-filters">${select('tool', 'Tools', (run.actions || []).filter(toolCall).map(a => a.name || 'Unknown'), options.tool)}${select('status', 'Statuses', ['running', 'success', 'failed', 'timeout', 'cancelled', 'incomplete'], options.status)}<label>Order breakdown <select data-prof-filter="toolOrder" aria-label="Order tool breakdown">${[['total','Total duration'],['average','Average duration'],['p50','Median duration'],['p95','P95 latency'],['calls','Calls'],['errorRate','Error rate']].map(([key, label]) => `<option value="${key}" ${order === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label></div>`;
  const max = Math.max(1, ...ordered.slice(0, 24).map(g => g.total));
  html += section('Tool execution breakdown', ordered.length ? `<div class="prof-tool-bars">${ordered.slice(0, 24).map(g => `<button data-prof-tool="${esc(g.key)}" title="${esc(g.key)} · ${sec(g.total)} summed completed duration · median ${sec(g.p50)} · P95 ${sec(g.p95)} · ${g.calls} calls · ${num(g.errorRate * 100)}% failed"><span>${esc(g.key)}</span><span class="prof-bar-track"><i style="width:${100 * g.total / max}%"></i></span><strong>${sec(g.total)}</strong></button>`).join('')}</div>${ordered.length > 24 ? '<p class="prof-definition">Top 24 rows shown. All tools remain in the searchable breakdown table.</p>' : ''}` : empty('No tool calls recorded.'), 'Bar lengths show summed completed duration, not additive wall-clock contributions. Calls can run concurrently.');
  html += chart('Tool latency over time', data.tools.filter(timed).map(a => [a.start + a.duration, a.duration, a]), 's', {...data, threshold:Math.max(5, (s.p50 || 0) * 3)}, {color:colors.tools, scatter:true, description:'Amber points exceed three times the median and five seconds. Select a spike to inspect the call.'});
  html += section('By tool', groupTable('tool-breakdown', groups, false, 'Tool'));
  const slow = data.tools.filter(timed).sort((a, b) => b.duration - a.duration).slice(0, 10);
  const active = [...data.active].sort((a, b) => a.start - b.start).slice(0, 10);
  const failed = data.tools.filter(failure).slice(-10), missing = data.tools.filter(a => operationState(a, run, options) === 'incomplete').slice(-10);
  html += section('Slow and unresolved operations', `<div class="prof-operation-lists">${[['Slowest completed',slow],['Longest observed open calls',active],['Failures',failed],['Missing completion',missing]].map(([title, calls]) => `<div><h4>${title}</h4>${calls.length ? `<ul>${calls.map(a => `<li>${actionLink(a)} <span>${esc(a.agent)} · ${timed(a) ? sec(a.duration) : operationState(a, run, options) === 'running' ? sec(Math.max(0, data.bounds[1] - a.start)) + ' open at display boundary' : 'End unobserved'}</span></li>`).join('')}</ul>` : empty('None observed.')}</div>`).join('')}</div>`);
  return html + section('Call inspector', callTable('tools', data.tools, run, options), 'Searchable, sortable, 50 rows per page. Cancelled and unknown calls remain accessible.') + `<details class="prof-raw"><summary>Secondary latency distribution</summary>${chart('Tool duration distribution (sorted observations)', data.tools.filter(timed).map(a => a.duration).sort((a, b) => a - b).map((v, i) => [i, v]), 's', data, {indexed:true})}</details>`;
}

function systemsView(run, data, options, measurements) {
  const processes = processResources(run, measurements, options.range), scoped = options.agent ? processes.filter(p => p.agent === options.agent) : processes;
  const ownSandboxes = new Set();
  if (options.agent) {
    const agent = (run.agents || []).find(a => a.id === options.agent);
    for (const sandbox of new Set([agent?.sandbox, ...(agent?.sandboxes || [])])) {
      if (!sandbox) continue;
      const owners = new Set([...(run.sandbox_owners?.[sandbox] || []), ...(run.agents || []).filter(a => a.sandbox === sandbox || a.sandboxes?.includes(sandbox)).map(a => a.id)]);
      if (owners.size === 1 && owners.has(options.agent)) ownSandboxes.add(sandbox);
    }
  }
  const selectedMetrics = options.agent ? measurements.filter(m => [...ownSandboxes].some(id => m.name.startsWith(`sandbox:${id}:`) || m.name.startsWith(`cg:${id}:`)) || scoped.some(p => p.series.includes(m))) : measurements.filter(m => m.aggregate && (m.scope === 'Workload' || m.scope === 'Host' || /^gpu/.test(m.scope)));
  const current = label => selectedMetrics.find(m => m.label === label && (options.agent || m.scope === 'Workload'));
  const latest = m => m?.points.at(-1)?.[1] ?? null;
  const mem = current('RAM');
  let html = strip([[options.agent ? 'Sandbox CPU%' : 'Workload CPU%', num(latest(current('CPU'))), options.agent ? 'Selected owned sandbox CPU · 100% = one core' : '100% = one core; may exceed 100%'], [options.agent ? 'Sandbox memory' : 'Workload memory', byte(latest(mem)), 'Named workload/cgroup memory; not VMS or sum of RSS'], ['Peak memory', byte(mem?.points.length ? mem.points.reduce((n, p) => Math.max(n, p[1]), 0) : null)], ['Disk read', num(latest(current('Disk read'))), 'MiB/s · recorded counter deltas'], ['Disk write', num(latest(current('Disk write'))), 'MiB/s · recorded counter deltas']]);
  html += `<label class="prof-zero-toggle"><input type="checkbox" data-prof-filter="showZero" ${options.showZero ? 'checked' : ''}> Show recorded zero-value resource charts</label>`;
  const pressure = selectedMetrics.filter(m => m.label.includes('pressure') && m.points.length).map(m => ({metric:m, peak:largestKnown(m.points.map(p => p[1]))})).sort((a, b) => b.peak - a.peak)[0];
  html += `<p class="prof-definition">${pressure ? `Peak recorded pressure: ${num(pressure.peak)}% · ${esc(pressure.metric.scope)} · ${esc(pressure.metric.label)}. ` : ''}Host, workload and process scopes are independent and may overlap. Samples do not establish continuous utilization between observations; missing samples are not zero.</p>`;
  if (options.agent && !selectedMetrics.length) html += empty('No reliably owned resource series for this agent. Shared and unattributed measurements remain available after clearing the agent filter.');
  const nonzero = m => options.showZero || m.points.some(p => p[1] !== 0);
  const resourceGroups = /** @type {[string, string[], string][]} */ ([['CPU',['CPU','Host utilization'],'var(--chart-cpu)'], ['Memory',['RAM','RSS'],'var(--chart-memory)'], ['Disk',['Disk read','Disk write'],'var(--chart-disk)'], ['Network',['Network RX','Network TX'],'var(--chart-network)'], ['Resource pressure',['CPU pressure (some)','CPU pressure (full)','Memory pressure (some)','Memory pressure (full)','I/O pressure (some)','I/O pressure (full)'],'var(--chart-pressure)'], ['GPU',['GPU','VRAM','Power','Power estimate','Process VRAM'],'var(--chart-cpu)']]);
  for (const [title, labels, color] of resourceGroups) {
    const series = selectedMetrics.filter(m => labels.includes(m.label) && nonzero(m));
    if (series.length) html += section(title === 'Resource pressure' ? 'Resource pressure over time' : `${title} utilization over time`, `<div class="prof-charts prof-aligned">${series.slice(0, 8).map(m => chart(`${m.scope} · ${m.label}`, m.points, m.unit.replace('MB', 'MiB'), data, {color, description:`${m.name} · ${m.statusLabel || 'Recorded samples'} · latest at ${sec(m.points.at(-1)?.[0])}. ${m.label === 'Host utilization' ? '100% = all host CPUs.' : m.label.includes('pressure') ? 'Percentage of wall time stalled; some = at least one task, full = all non-idle tasks.' : ''}`})).join('')}</div>`);
  }
  if (!selectedMetrics.some(nonzero)) html += empty('No nonzero aggregate resource samples in this scope. Recorded zero values and individual processes remain inspectable below.');
  const agentRows = data.agents.map(a => {
    const own = processes.filter(p => p.agent === a.id), resource = agentResources({...run, duration:data.bounds[1]}, [a.id]);
    const sandboxes = [...new Set([a.sandbox, ...(a.sandboxes || [])])].filter(id => id && !(run.sandbox_owners?.[id] || []).some(owner => owner !== a.id) && !(run.agents || []).some(other => other.id !== a.id && (other.sandbox === id || other.sandboxes?.includes(id))));
    const sandboxRate = label => {
      const samples = sandboxes.map(id => measurements.filter(m => m.label === label && m.points.length && (m.name.startsWith(`sandbox:${id}:`) || m.name.startsWith(`cg:${id}:`))).sort((x, y) => y.latest[0] - x.latest[0])[0]?.latest).filter(p => p && Number.isFinite(p[1]) && data.bounds[1] - p[0] <= 10);
      return samples.length ? samples.reduce((n, p) => n + p[1], 0) : null;
    };
    const hasSelectedCPU = measurements.some(m => m.label === 'CPU' && m.points.length && sandboxes.some(id => m.name.startsWith(`sandbox:${id}:`) || m.name.startsWith(`cg:${id}:`)));
    return {key:a.id, label:a.label || a.id, cpu:hasSelectedCPU ? resource.cpu?.value ?? null : null,
      peak:largestKnown(own.map(p => p.peak)),
      read:sandboxRate('Disk read'), write:sandboxRate('Disk write'), count:own.length || null};
  });
  html += section('Resource usage by agent', table('agent-resources', agentRows, [{key:'label',label:'Agent',render:a => `<button data-prof-agent="${esc(a.key)}">${esc(a.label)}</button>`}, {key:'cpu',label:'Sandbox CPU%',render:a => num(a.cpu)}, {key:'peak',label:'Peak process RSS',render:a => byte(a.peak)}, {key:'read',label:'Read MiB/s',render:a => num(a.read)}, {key:'write',label:'Write MiB/s',render:a => num(a.write)}, {key:'count',label:'Known processes',render:a => num(a.count)}]), 'CPU and I/O use recent samples from explicitly owned sandboxes (10s freshness window). Process counts and RSS require explicit process identity → agent metadata. Shared sandboxes are excluded. RSS peaks are largest individual process observations.');
  html += section('Process explorer', table('processes', scoped, [{key:'key',label:'PID / process identity',render:p => `<button data-prof-process="${esc(p.key)}">${esc(p.key)} ↗</button>`}, {key:'agent',label:'Agent',render:p => esc(p.agent || 'Unattributed')}, {key:'command',label:'Command',render:p => `<span class="prof-command-preview" title="${esc(String(p.command || '').slice(0,600))}">${esc(p.command ? p.command.slice(0,180) + (p.command.length > 180 ? '…' : '') : 'Unavailable')}</span>`}, {key:'cpu',label:'Last CPU%',render:p => num(p.cpu)}, {key:'rss',label:'Last RSS',render:p => byte(p.rss)}, {key:'read',label:'Read MiB/s',render:p => num(p.read)}, {key:'write',label:'Write MiB/s',render:p => num(p.write)}, {key:'status',label:'Observation state'}], {defaultSort:'cpu'}), 'Values are the last observations in the interval. Stable collector identities preserve PID reuse and restarts. A missing sample does not prove process exit. Command and ownership are unavailable unless explicitly recorded.');
  const selected = scoped.find(p => p.key === options.process);
  if (selected) html += section(`Process ${selected.key} · recorded metrics`, `<details><summary>Recorded command</summary><pre class="prof-recorded-command">${esc(selected.command || 'Unavailable')}</pre></details><div class="prof-charts">${selected.series.map(m => chart(`${m.label} · ${m.scope}`, m.points, m.unit.replace('MB','MiB'), data, {color:'var(--chart-memory)', description:m.label === 'Virtual memory' ? 'Virtual address space; not physical memory.' : m.name})).join('')}</div>`);
  html += `<p class="prof-definition">Select or drag an interval to inspect resource samples and operations observed during that interval across all tabs. Coincident resource use and latency do not establish causation. ${selectedMetrics.some(m => m.label.includes('pressure')) ? 'Recorded PSI measurements show wall time stalled, with their original host or sandbox scope.' : 'CPU, memory and I/O pressure/stall measurements are unavailable in this selection.'}</p>`;
  const raw = measurements.filter(m => !options.agent || selectedMetrics.includes(m) || scoped.some(p => p.series.includes(m)));
  const selectedResource = raw.find(m => m.name === options.resource);
  html += `<details class="prof-raw" ${selectedResource ? 'open' : ''}><summary>All recorded resource series · including zero values and virtual memory</summary>${table('resource-series', raw.map(m => ({...m, key:m.name, value:m.latest[1], at:m.latest[0]})), [{key:'name',label:'Recorded series',render:m => `<button data-prof-resource="${esc(m.name)}">${esc(m.name)} ↗</button>`}, {key:'scope',label:'Scope'}, {key:'value',label:'Latest sample',render:m => `${num(m.value)} ${esc(m.unit.replace('MB','MiB'))}`}, {key:'at',label:'At',render:m => sec(m.at)}, {key:'statusLabel',label:'Collector state'}])}${selectedResource ? chart(selectedResource.name,selectedResource.points,selectedResource.unit.replace('MB','MiB'),data,{description:'Full recorded series for this selection. Virtual memory is address space, not physical memory.'}) : ''}</details>`;
  const syscalls = systemActivity({...run, actions:data.actions});
  html += `<details class="prof-raw"><summary>Syscall instrumentation · aggregated · ${num(syscalls.count)} events</summary><p class="prof-definition">${num(syscalls.unattributed)} without recorded ownership · ${sec(syscalls.duration)} summed measured time (may overlap) · ${num(syscalls.unmeasured)} untimed. Individual syscall evidence remains in raw activity inspection.</p>${table('syscalls',syscalls.groups,[{key:'name',label:'Syscall'},{key:'count',label:'Count'},{key:'duration',label:'Summed time',render:s => sec(s.duration)},{key:'unmeasured',label:'Untimed'}])}${callTable('syscall-calls',data.actions.filter(a => a.kind === 'syscall'),run,options)}</details>`;
  return html;
}

export function renderProfiler(run, category, options = {}) {
  const data = profilerData(run, options); data.cursor = options.cursor;
  // One canonical resource selector, filtered shallowly at the render boundary.
  const resources = category === 'Overview' || category === 'Activity' || category === 'Systems' ? systemMeasurements(run).map(m => {
    const points = m.points.filter(p => p[0] >= data.bounds[0] && p[0] <= data.bounds[1]);
    const latest = points.at(-1) || [data.bounds[1], null];
    return {...m, points, latest:m.status === 'valid' ? latest : [m.latest[0],null]};
  }) : [];
  const title = category === 'Activity' ? 'Overview' : category;
  const headline = {Overview:'Understand the execution, then follow the evidence.', LLM:'Find slow requests and track context growth.', Tools:'Find expensive calls, latency spikes and unresolved operations.', Systems:'Inspect resource use across the workload and recorded processes.'};
  return `<div class="prof-dashboard"><header class="prof-heading"><div><h2>${title} profiler</h2><p>${headline[title]}</p></div><span class="prof-definition">${esc(run.status || 'unknown')}${run.status === 'running' ? ' · provisional observations' : ' · recorded observations'}${options.frozen ? ' · frozen display' : ''}</span></header>${run.coverage?.client_window ? '<p class="prof-definition">Execution counters cover the full run; latency, breakdowns and findings cover retained observations.</p>' : ''}${title === 'Overview' ? overview(run,data,options,resources) : title === 'LLM' ? llmView(run,data,options) : title === 'Tools' ? toolsView(run,data,options) : systemsView(run,data,options,resources)}</div>`;
}

export function handleProfilerTableEvent(event, render) {
  const el = event.target.closest('[data-prof-page],[data-prof-sort],[data-prof-search]'); if (!el) return false;
  if (el.dataset.profPage) rowState(el.dataset.profPage).page += Number(el.dataset.delta);
  if (el.dataset.profSort) {const state = rowState(el.dataset.profSort); state.direction = state.sort === el.dataset.key ? -state.direction : -1; state.sort = el.dataset.key;}
  if (el.dataset.profSearch) {const state = rowState(el.dataset.profSearch); state.query = el.value; state.page = 0;}
  render(); return true;
}
