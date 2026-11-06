import {finite, kindOf, elapsed, visibleMarks, dependencyRoute, criticalContribution, historyURL, safeLink, NODE_LIMIT} from './critical-path-model.js';

export const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));
export const number = value => finite(value) ? new Intl.NumberFormat('en', {maximumFractionDigits:2}).format(value) : 'Unavailable';
export const seconds = value => finite(value) ? value > 0 && value < .1 ? `${number(value * 1000)}ms` : `${number(value)}s` : 'Unavailable';
const bandClass = node => kindOf(node).toLowerCase();
const label = node => node.label || node.name || node.id;
const field = (name, value) => `<div class="cp-field"><span>${esc(name)}</span><span>${esc(value ?? 'Unavailable')}</span></div>`;
const known = (values, keys) => {for (const key of keys) if (values?.[key] != null) return values[key]; return null;};
const evidence = values => values?.length ? `<ul class="cp-evidence">${values.slice(0, 30).map(value => `<li>${esc(typeof value === 'string' ? value : JSON.stringify(value))}</li>`).join('')}</ul>${values.length > 30 ? '<p class="cp-note">First 30 evidence items shown; full recording is available in the detailed profiler.</p>' : ''}` : '<p class="cp-note">No additional evidence recorded.</p>';
const link = (url, text) => safeLink(url) ? `<a href="${esc(safeLink(url))}">${esc(text)} ↗</a>` : '';
const pick = node => `<button data-cp-select="${esc(node.id)}">${esc(label(node))}</button>`;
function sandboxWindows(windows = []) {
  if (!windows.length) return '';
  return `<section class="cp-shared-sandbox"><h3>Recorded sandbox resource windows</h3><p class="cp-note">These measurements describe a shared sandbox during the recorded interval. They are not owned by this operation and do not establish a cause of latency.</p>${windows.slice(0, 6).map(window => `<div class="cp-resource-window">${field('Sandbox identity',window.sandbox_id)}${field('Measurement scope',window.scope || 'shared_sandbox')}${window.shared_with?.length ? field('Recorded sandbox agents',window.shared_with.join(', ')) : ''}${field('Sample window',`${seconds(window.start)} → ${seconds(window.end)}`)}${Object.entries(window.sampled_window || {}).slice(0, 16).map(([metric, observations]) => field(metric,`${number(observations.observed_min)}–${number(observations.observed_peak)} · ${number(observations.samples)} samples`)).join('')}<p class="cp-note">${esc(window.basis || 'Retained recorded samples; shared activity is not tool-owned.')}</p></div>`).join('')}${windows.length > 6 ? '<p class="cp-note">First six recorded sandbox windows shown; full metadata remains in the detailed profiler.</p>' : ''}</section>`;
}
export function renderOverview(profile) {
  const s = profile.summary || {}, c = profile.coverage || {}, path = profile.critical_path || {};
  const status = c.status || 'partial';
  const ratio = finite(s.critical_path_coverage) ? s.critical_path_coverage : c.ratio;
  const card = (name, value, note = '') => `<div><small>${esc(name)}</small><strong>${esc(value)}</strong>${note ? `<span>${esc(note)}</span>` : ''}</div>`;
  const breakdown = Object.entries(s.breakdown || {}).filter(([, value]) => finite(value) && value > 0);
  const total = breakdown.reduce((sum, [, value]) => sum + value, 0);
  const limitations = [...new Set([...(c.limitations || []), ...(profile.limitations || [])])];
  const truncation = Object.entries(profile.truncation || {}).filter(([, count]) => count > 0);
  return `<section class="cp-overview" aria-label="Execution overview"><div class="cp-result-state ${esc(status)}"><strong>${esc(status === 'invalid' ? 'Invalid dependency graph' : status === 'candidate' ? 'Candidate critical path' : c.complete ? 'Recorded critical path' : 'Partial critical path')}</strong><span>${esc(profile.run?.status || 'unknown')} execution · ${esc(profile.run?.source || 'recorded')} evidence</span></div>
    <div class="cp-kpis">${card('Recorded wall time', seconds(s.wall_time), 'Execution boundary when captured; otherwise trace extent')}${card('Critical path', status === 'invalid' ? 'Unavailable' : seconds(s.critical_path_duration), c.complete ? 'Recorded dependencies' : 'Recorded dependency chain; incomplete causality')}${card('Measured path coverage', finite(ratio) ? `${number(ratio * 100)}%` : 'Unavailable', `${seconds(s.critical_path_measured)} of measured contributing activity`)}${card('Aggregate observed activity', seconds(s.aggregate_work), 'Across agents and orchestration, including waits; may exceed wall time')}${card('Agents / processes', `${number(s.agent_count)} / ${number(s.process_count)}`, 'Recorded identities')}${card('Unattributed wall time', seconds(s.unattributed_time), 'No measured operation attribution')}</div>
    ${breakdown.length ? `<div class="cp-breakdown" aria-label="Critical path contribution breakdown">${breakdown.map(([kind, value]) => `<span class="${esc(kind.toLowerCase())}" style="flex:${Math.max(.01, value / total)}" title="${esc(kind)} · ${seconds(value)}">${esc(kind)} ${seconds(value)}</span>`).join('')}</div>` : ''}
    <p class="cp-note">Nested spans contribute once. Process measurements describe recorded resource use; they do not prove what caused latency.${finite(path.unexplained_wait) && path.unexplained_wait > 0 ? ` The chain includes ${seconds(path.unexplained_wait)} of unexplained elapsed gaps.` : ''}</p>
    ${truncation.length ? `<p class="cp-note">Response limits omitted ${truncation.map(([kind, count]) => `${number(count)} ${esc(kind)}`).join(', ')} from this view. Execution summary includes the analyzed recording; search, local counts and visible relationships use the retained summaries. Full evidence remains in the detailed profiler.</p>` : ''}
    ${(profile.joins || []).length ? `<div class="cp-join-gates"><span>Latest recorded join prerequisites</span>${profile.joins.slice(0, 6).map(join => (join.latest_prerequisite_ids || []).map(id => `<button data-cp-select="${esc(id)}">${esc((profile.operations || []).find(op => op.id === id)?.agent || id)} · ${seconds(join.latest_completion_time)}${join.candidate_path_matches_gate === false ? ' · candidate path differs' : ''}</button>`).join('')).join('')}</div>` : ''}
    ${(s.dominant_operations || []).length ? `<div class="cp-dominant"><span>Longest measured operations</span>${s.dominant_operations.slice(0, 5).map(op => `<button data-cp-select="${esc(op.id)}">${esc(op.label)} <b>${seconds(op.duration)}</b></button>`).join('')}</div>` : ''}
    ${limitations.length || (c.issues || []).length || (c.malformed_edges || []).length ? `<details class="cp-limitations"${status === 'invalid' ? ' open' : ''}><summary>Coverage and limitations · ${limitations.length + (c.issues || []).length + (c.malformed_edges || []).length} recorded findings</summary>${evidence(limitations)}${c.issues?.length ? `<h3>Incomplete observations</h3>${evidence(c.issues.map(item => `${item.operation_id || item.operation || 'Execution'}: ${item.reason || JSON.stringify(item)}`))}` : ''}${c.malformed_edges?.length ? `<h3>Rejected dependency evidence</h3>${evidence(c.malformed_edges.map(item => typeof item === 'string' ? item : JSON.stringify(item)))}` : ''}</details>` : ''}</section>`;
}
export function renderLaneLabels(layout, index, options = {}) {
  return `<div class="cp-lane-axis">Agent</div>${layout.lanes.map(lane => {
    const a = lane.agent, count = a?.operation_count ?? a?.operation_ids?.length ?? index.agentCounts.get(lane.id) ?? 0;
    const context = options.agent && lane.id !== options.agent;
    return `<div class="cp-lane-label ${context ? 'context' : ''}" style="top:${lane.top}px;height:${lane.height}px"><button data-cp-agent="${esc(lane.id)}" title="Inspect agent ${esc(lane.id)}"><strong>${esc(a?.label || lane.id)}</strong><code>${esc(lane.id)}</code></button><span>${count} operations${context ? ' · dependency context' : ''}</span>${!lane.count && count ? `<button data-cp-focus="${esc(lane.id)}">Show local activity →</button>` : ''}</div>`;
  }).join('')}`;
}
export function renderTimeline(index, selection, layout, viewport = {}, selected = '') {
  const visible = visibleMarks(selection, layout, viewport);
  const marks = visible.nodes.map(node => {
    const p = layout.positions.get(node.id), kind = node.container ? 'Execution envelope' : kindOf(node), complete = finite(node.start) && finite(node.end) && node.end >= node.start;
    const text = (node.aggregate ? `${node.agent} · ${label(node)} grouped for display at this zoom; individual timing and dependencies are preserved` : `${node.agent || 'Unattributed'} · ${kind} · ${label(node)} · ${complete ? seconds(elapsed(node)) : 'Incomplete timing'}`) + (node.gating ? ' · Latest recorded prerequisite at a join' : '');
    const width = Math.max(5, p.width), short = width < 85, name = label(node).slice(0, Math.max(2, Math.floor(width / 7) - 2));
    return `<g class="cp-node ${bandClass(node)} ${node.container ? 'envelope' : ''} ${node.critical ? 'critical' : 'parallel'} ${node.gating ? 'gating' : ''} ${node.aggregate ? 'aggregate' : ''} ${node.overlay ? 'process-overlay' : ''} ${complete ? '' : 'incomplete'} ${selected === node.id ? 'selected' : ''}" role="button" tabindex="0" ${node.aggregate ? 'data-cp-group' : 'data-cp-select'}="${esc(node.id)}" aria-label="${esc(text)}" transform="translate(${p.x},${p.y})"><title>${esc(text)}${node.overlay ? ' · process overlay; not added to path duration' : node.container ? ' · envelope boundaries; nested activity contributes separately' : ''}</title><rect width="${width}" height="30" rx="4"/>${node.critical && !node.aggregate && !node.container ? `<path class="cp-critical-mark" d="M 0 32 H ${width}"/>` : ''}${node.gating ? `<path class="cp-gate-mark" d="M ${width} -4 l 4 4 l -4 4 l -4 -4 z"/>` : ''}${node.aggregate ? `<text x="7" y="19">${node.members.length}</text>` : !short ? `<text x="7" y="19">${esc(name)}</text>` : ''}${!complete ? '<path d="M 1 4 V 26" class="cp-incomplete-mark"/>' : ''}</g>`;
  }).join('');
  const edges = visible.edges.map(edge => {
    const route = dependencyRoute(edge, layout); if (!route) return '';
  const from = index.operations.get(edge.source), to = index.operations.get(edge.target), cross = from?.agent !== to?.agent;
    return `<path class="cp-edge ${edge.critical ? 'critical' : ''} ${edge.gating ? 'gating' : ''} ${esc(edge.confidence || 'unknown')} ${cross ? 'cross-agent' : ''}" data-cp-edge="${esc(edge.id)}" role="button" tabindex="0" aria-label="${esc(`${edge.kind} · ${edge.confidence} · ${from?.agent || ''} to ${to?.agent || ''}`)}" d="${route.path}" marker-end="url(#cp-arrow)"><title>${esc(`${edge.kind} · ${edge.confidence} · ${edge.evidence || 'Recorded process ownership'}`)}</title></path>`;
  }).join('');
  const visibleWidth = Math.max(300, (viewport.right ?? 1000) - (viewport.left || 0));
  const duration = layout.bounds.end - layout.bounds.start, rough = visibleWidth / Math.max(.001, layout.scale) / 7;
  const base = 10 ** Math.floor(Math.log10(Math.max(.0001, rough))), step = [1, 2, 5, 10].find(v => v * base >= rough) * base;
  const first = Math.max(0, Math.floor((viewport.left || 0) / layout.scale / step));
  const last = Math.min(Math.ceil(duration / step), Math.ceil((viewport.right ?? layout.width) / layout.scale / step) + 1);
  const ticks = []; for (let i = first; i <= last && ticks.length < 30; i++) ticks.push(i * step);
  const lanes = layout.lanes.filter(l => l.top + l.height >= (viewport.top || 0) && l.top <= (viewport.bottom ?? Infinity));
  return `<svg width="${layout.width}" height="${layout.height}" aria-label="Historical execution temporal dependency graph"><defs><marker id="cp-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z"/></marker></defs>${lanes.map(l => `<rect class="cp-lane-band" x="0" y="${l.top}" width="${layout.width}" height="${l.height}"/><line class="cp-grid" x1="0" x2="${layout.width}" y1="${l.top + l.height}" y2="${l.top + l.height}"/>`).join('')}${ticks.map(t => `<line class="cp-grid" x1="${16 + t * layout.scale}" x2="${16 + t * layout.scale}" y1="32" y2="${layout.height}"/><text class="cp-tick" x="${19 + t * layout.scale}" y="22">${seconds(t)}</text>`).join('')}<g class="cp-edges">${edges}</g><g class="cp-nodes">${marks}</g></svg>`;
}
function relationships(index, id) {
  const entries = [...(index.incoming.get(id) || []), ...(index.outgoing.get(id) || [])].filter(edge => edge.source !== edge.target);
  return `<section><h3>Recorded relationships · ${entries.length}</h3>${entries.length ? entries.slice(0, 20).map(edge => {
    const inbound = edge.target === id, other = index.operations.get(inbound ? edge.source : edge.target);
    return `<div class="cp-relation"><small>${inbound ? 'Incoming' : 'Outgoing'} · ${esc(edge.kind)} · ${esc(edge.confidence)} · ${edge.blocking ? 'blocking' : 'relationship'}</small>${other ? pick(other) : esc(inbound ? edge.source : edge.target)}<p>${esc(edge.evidence || 'No additional evidence recorded.')}</p></div>`;
  }).join('') : '<p class="cp-note">No recorded dependency relationships. Parallel timing alone does not establish dependency.</p>'}${entries.length > 20 ? '<p class="cp-note">First 20 relationships shown.</p>' : ''}</section>`;
}
function metricFields(metrics, process = false) {
  const mapping = process ? [
    ['CPU time', ['cpu_time_s', 'cpu_seconds', 'cpu_time_seconds'], v => seconds(v)], ['CPU time', ['cpu_ms'], v => `${number(v)}ms`],
    ['CPU utilization', ['cpu_pct', 'cpu_peak_pct'], v => `${number(v)}%`],
    ['Average recorded CPU', ['cpu_average_percent'], v => `${number(v)}%`], ['Peak sampled CPU', ['cpu_peak_percent'], v => `${number(v)}%`],
    ['RSS', ['rss_mb', 'rss_peak_mb', 'peak_rss_mb'], v => `${number(v)} MiB`],
    ['Read bytes', ['io_read_bytes', 'read_bytes'], number], ['Write bytes', ['io_write_bytes', 'write_bytes'], number],
    ['Disk read', ['io_read_mb'], v => `${number(v)} MiB`], ['Disk write', ['io_write_mb'], v => `${number(v)} MiB`],
    ['Scheduling delay', ['scheduling_delay_s', 'sched_delay_s'], seconds], ['Exit status', ['exit_status', 'exit_code'], String],
    ['Command', ['command_line'], String],
  ] : [
    ['Model', ['model'], String], ['Input tokens', ['input_tokens', 'tokens'], number], ['Output tokens', ['output_tokens'], number],
    ['Time to first token', ['ttft_s', 'ttft_seconds'], seconds], ['Time to first token', ['ttft_ms'], v => `${number(v)}ms`],
    ['Streaming duration', ['streaming_duration_s', 'streaming_duration'], seconds], ['Retries', ['retry_count', 'retries'], number],
    ['Streaming duration', ['streaming_ms', 'stream_duration_ms'], v => `${number(v)}ms`], ['Attempt', ['attempt', 'attempt_index'], number],
    ['Post first-token interval', ['post_first_token_ms'], v => `${number(v)}ms`],
    ['First complete response', ['response_latency_ms'], v => `${number(v)}ms`],
    ['Run-queue delay', ['runqueue_ms'], v => `${number(v)}ms`], ['Blocked time', ['blocked_ms'], v => `${number(v)}ms`],
    ['Error', ['error', 'error_message'], String], ['Error type', ['error_type'], String], ['HTTP status', ['http_status'], number], ['Command', ['command_preview', 'command_line', 'command', 'program'], String],
  ];
  return mapping.map(([name, keys, format]) => {const v = known(metrics, keys); return v != null && (format === String || finite(v)) ? field(name, format(v)) : '';}).join('');
}
export function renderInspector(index, selection) {
  if (!selection) return '';
  const {profile, operations, processes, agents} = index;
  const top = '<div class="cp-inspector-top"><strong>Evidence inspector</strong><button data-cp-close aria-label="Close evidence inspector">Close</button></div>';
  if (selection.type === 'group') return top + `<h2>Grouped observations</h2><p class="cp-note">These short spans share a display bucket at this zoom. This group is a visual simplification; it creates no execution duration or causal relationship.</p><button data-cp-group-zoom>Zoom into these observations</button><section><h3>Recorded members · ${selection.members?.length || 0}</h3>${(selection.members || []).map(id => operations.get(id)).filter(Boolean).map(op => `<div class="cp-child-process">${pick(op)}<small>${esc(op.agent)} · ${seconds(op.start)} → ${seconds(op.end)}</small></div>`).join('')}</section>`;
  if (selection.type === 'agent') {
    const a = agents.get(selection.id); if (!a) return top + '<p>Agent evidence unavailable.</p>';
    const local = (a.operation_ids || []).map(id => operations.get(id)).filter(Boolean);
    const agentProcesses = [...processes.values()].filter(p => p.agent === a.id);
    const agentResources = a.metrics?.sandbox_resources || a.sandbox_resources || [];
    const children = [...agents.values()].filter(child => child.parent === a.id);
    const crossing = (profile.edges || []).filter(e => (operations.get(e.source)?.agent === a.id || operations.get(e.target)?.agent === a.id) && operations.get(e.source)?.agent !== operations.get(e.target)?.agent);
    return top + `<h2>${esc(a.label || a.id)}</h2>${field('Agent identity', a.id)}${field('Parent agent', a.parent || 'Unrecorded')}${field('Local path duration', seconds(a.critical_path_duration))}${field('Retained operations / on global path', `${local.length} / ${local.filter(op => op.critical).length}`)}${field('Cross-agent relationships', crossing.length)}<button data-cp-focus="${esc(a.id)}">Focus this agent and dependencies</button>${children.length ? `<section><h3>Recorded child agents</h3>${children.slice(0, 30).map(child => `<button data-cp-agent="${esc(child.id)}">${esc(child.label || child.id)}</button>`).join('')}</section>` : ''}<section><h3>Local operations</h3>${local.slice(0, 30).map(pick).join('')}${local.length > 30 ? '<p class="cp-note">First 30 shown; focus the agent to explore its timeline.</p>' : ''}</section>${sandboxWindows(agentResources)}${agentProcesses.length ? `<section><h3>Recorded agent processes · ${agentProcesses.length}</h3><button data-cp-process-agent="${esc(a.id)}">Explore this agent’s processes</button>${agentProcesses.slice(0, 15).map(pick).join('')}<p class="cp-note">Agent ownership does not establish ownership by a specific tool.</p></section>` : ''}${link(historyURL(profile, {agent:a.id}), 'Open agent trajectory')}`;
  }
  if (selection.type === 'edge') {
    const e = (profile.edges || []).find(edge => edge.id === selection.id);
    if (!e) return top + '<p>This overlay describes recorded process ownership. Inspect the process for attribution evidence.</p>';
    const from = operations.get(e.source), to = operations.get(e.target);
    return top + `<h2>${esc(e.kind)}</h2>${field('Evidence classification', e.confidence)}${field('Blocking dependency', e.blocking ? 'Yes' : 'No')}${field('On critical path', e.critical ? 'Yes' : 'No')}${from ? `<h3>Source · ${esc(from.agent)}</h3>${pick(from)}` : ''}${to ? `<h3>Target · ${esc(to.agent)}</h3>${pick(to)}` : ''}${field('Source / target boundary', `${seconds(e.source_time)} → ${seconds(e.target_time)}`)}<h3>Recorded evidence</h3><p>${esc(e.evidence || 'No additional evidence recorded.')}</p>`;
  }
  const process = processes.get(selection.id), op = process || operations.get(selection.id);
  if (!op) return top + '<p>Operation evidence unavailable.</p>';
  const m = op.metrics || {}, childProcesses = (op.process_ids || []).map(id => processes.get(id)).filter(Boolean);
  const childOperations = (op.child_ids || []).map(id => operations.get(id)).filter(Boolean);
  const contribution = criticalContribution(index, op.id), result = op.result_preview || m.result_preview || m.result || m.output;
  const joins = (profile.joins || []).filter(join => join.operation_id === op.id || join.latest_prerequisite_ids?.includes(op.id));
  const sampling = process?.lifecycle === 'sample_bounds';
  const resourceWindows = m.process_resources || [];
  return top + `<span class="cp-kind ${process ? 'system' : bandClass(op)}">${process ? 'System process' : op.container ? 'Execution envelope' : esc(kindOf(op))}${op.critical ? op.container ? ' · Critical boundary' : ' · Critical' : ''}${op.gating ? ' · Latest join prerequisite' : ''}</span><h2>${esc(label(op))}</h2>${field('Agent identity', op.agent || 'Unattributed')}${field(sampling ? 'First / last sample' : 'Start / end', `${seconds(op.start)} → ${seconds(op.end)}`)}${field(sampling ? 'Sample observation extent' : 'Recorded duration', seconds(op.duration))}${!process ? field('Exclusive path contribution', op.container ? 'Envelope boundary; nested activity is measured separately' : !op.critical && contribution == null ? 'Outside displayed chain' : seconds(contribution)) + field('Result status', op.status || 'Unknown') : ''}${finite(op.slack) ? field('Slack in recorded graph', seconds(op.slack)) : ''}${joins.length ? `<section class="cp-limitations"><h3>Observed join gate</h3>${joins.map(join => `${field('Latest prerequisite completion',seconds(join.latest_completion_time))}<p>${esc(join.evidence || 'Prerequisite completion identities recorded at join.')}</p>${join.candidate_path_matches_gate === false ? '<p>The longest measured candidate path differs from the latest join prerequisite. Missing causal history prevents a definitive global path.</p>' : ''}${(join.prerequisites || []).slice(0, 15).map(p => `<div>${operations.has(p.operation_id) ? pick(operations.get(p.operation_id)) : esc(p.operation_id)}${field('Observed completion slack',seconds(p.observed_slack))}</div>`).join('')}`).join('')}</section>` : ''}
    ${process ? `${field('Stable process identity', op.identity || m.process_identity || 'Unrecorded')}${field('PID / parent process', `${op.pid ?? m.pid ?? 'Unknown'} / ${op.parent_id || m.ppid || 'Unknown'}`)}${op.parent_id && processes.has(op.parent_id) ? pick(processes.get(op.parent_id)) : ''}${field('Attribution', typeof op.attribution === 'string' ? op.attribution : JSON.stringify(op.attribution || 'Unknown'))}${metricFields(m, true)}<p class="cp-note">Resource measurements are observations over this recorded interval. PID alone and coincident timing are not ownership evidence. Missing samples are not zero utilization.</p>${(op.owner_ids || []).length ? `<h3>Recorded owning tools</h3>${op.owner_ids.map(id => operations.get(id)).filter(Boolean).slice(0, 20).map(pick).join('')}` : ''}` : `${op.model ? field('Model', op.model) : ''}${metricFields(m)}${kindOf(op) === 'LLM' ? '<p class="cp-note">Unavailable first-token, streaming and retry measurements remain unknown. End-to-end latency is not decode speed.</p>' : ''}${op.parent_id ? field('Recorded parent operation', op.parent_id) : ''}`}
    ${op.request_id ? field('Request identity', op.request_id) : ''}${resourceWindows.length ? `<section><h3>Process samples during this tool</h3><p class="cp-note">Samples retain an explicitly attributed process identity. Values are observed minima and peaks inside the tool interval; whole-process totals are not allocated to the call.</p>${resourceWindows.slice(0, 10).map(window => `<div class="cp-resource-window">${processes.has(window.process_id) ? pick(processes.get(window.process_id)) : esc(window.process_id)}${field('Sample window',`${seconds(window.start)} → ${seconds(window.end)}`)}${Object.entries(window.sampled_window || {}).slice(0, 12).map(([metric, observations]) => field(metric,`${number(observations.observed_min)}–${number(observations.observed_peak)} · ${number(observations.samples)} samples`)).join('')}</div>`).join('')}</section>` : ''}
    ${sandboxWindows(m.sandbox_resources || [])}${childOperations.length ? `<section><h3>Recorded child operations · ${childOperations.length}</h3><p class="cp-note">Recorded parent relationships provide nested inspection. Containment alone does not prove a blocking dependency.</p>${childOperations.slice(0, 30).map(child => `<div class="cp-child-process">${pick(child)}<small>${esc(kindOf(child))} · ${esc(child.agent)} · ${seconds(child.duration)}</small></div>`).join('')}${childOperations.length > 30 ? '<p class="cp-note">First 30 child summaries shown; full operation evidence remains in the detailed profiler.</p>' : ''}</section>` : ''}
    ${result != null ? `<section><h3>Recorded result</h3><pre>${esc(String(typeof result === 'string' ? result : JSON.stringify(result)).slice(0, 2500))}</pre></section>` : ''}
    ${childProcesses.length ? `<section><h3>Attributed processes · ${childProcesses.length}</h3><button data-cp-expand="${esc(op.id)}">Expand / collapse process overlays</button>${childProcesses.slice(0, 25).map(p => `<div class="cp-child-process">${pick(p)}<small>${esc(p.identity || 'Identity unknown')} · ${seconds(p.duration)}</small></div>`).join('')}${childProcesses.length > 25 ? '<p class="cp-note">First 25 processes shown. Detailed Systems profiler contains recorded measurements.</p>' : ''}</section>` : ''}
    ${!process ? relationships(index, op.id) : ''}<section><h3>Attribution and timing evidence</h3>${evidence(op.evidence)}</section>${op.limitations?.length ? `<section class="cp-limitations"><h3>Limitations</h3>${evidence(op.limitations)}</section>` : ''}<details><summary>Captured compact metrics</summary><pre>${esc(JSON.stringify(m, null, 2).slice(0, 20000))}</pre></details>
    <div class="cp-inspector-links">${link(op.links?.detail || op.links?.profiler || historyURL(profile, op), 'Open recorded operation')}${link(historyURL(profile, op, process ? 'systems' : kindOf(op) === 'LLM' ? 'llm' : 'tools'), 'Open detailed profiler')}${link(profile.links?.trace, 'Download detailed trace')}${link(profile.links?.perfetto, 'Open Perfetto')}</div>`;
}
export function graphNotice(selection) {
  return `${selection.sequence.length} selected operations · ${Math.min(selection.sequence.length, NODE_LIMIT)} per page${selection.hidden ? ` · ${selection.hidden} on other pages` : ''}${selection.untimed ? ` · ${selection.untimed} lack start times; inspect them in the text list below` : ''}${selection.processHidden ? ` · ${selection.processHidden} process overlays omitted at display limit` : ''}${selection.edgeHidden ? ` · ${selection.edgeHidden} additional edges available in inspection` : ''}`;
}
export function renderProcessTable(page) {
  return `<div class="cp-process-table"><table><thead><tr><th>Recorded process generation</th><th>Agent / ownership evidence</th><th>CPU time</th><th>Peak RSS</th><th>Read / write</th></tr></thead><tbody>${page.rows.map(p => {
    const m = p.metrics || {}, cpu = known(m, ['cpu_time_s','cpu_time_seconds','cpu_seconds']), rss = known(m, ['rss_peak_mb','peak_rss_mb','rss_mb']), read = known(m, ['io_read_mb']), write = known(m, ['io_write_mb']);
    return `<tr><td>${pick(p)}<code>${esc(p.identity || 'Identity unavailable')}</code></td><td>${esc(p.agent || 'Unattributed')}<small>${esc(p.attribution || 'Unknown ownership')}</small></td><td>${seconds(cpu)}</td><td>${rss == null ? 'Unavailable' : `${number(rss)} MiB`}</td><td>${read == null && write == null ? 'Unavailable' : `${number(read)} / ${number(write)} MiB`}</td></tr>`;
  }).join('') || '<tr><td colspan="5">No matching recorded process summaries.</td></tr>'}</tbody></table></div>`;
}
