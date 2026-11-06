// Profiler-only derivations over canonical observations. No subscriptions or trace copies.
export const terminal = a => ['success', 'failed', 'timeout', 'cancelled', 'completed'].includes(a.outcome);
export const toolCall = a => !['model', 'queue', 'dependency', 'unknown', 'setup', 'syscall', 'control'].includes(a.kind);
export const timed = a => terminal(a) && !a.missing_start && !a.missing_end && Number.isFinite(a.start) && Number.isFinite(a.duration) && a.duration >= 0 && !String(a.source).includes('timestamp_missing');
export const failure = a => ['failed', 'timeout'].includes(a.outcome);
export const hasStart = a => !a.missing_start && Number.isFinite(a.start) && !String(a.source).includes('timestamp_missing');
export function quantile(values, p) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length ? sorted[Math.max(0, Math.ceil(sorted.length * p) - 1)] : null;
}
export function operationState(a, run, {transport = null, frozen = false} = {}) {
  if (a.outcome !== 'running') return terminal(a) ? a.outcome : 'incomplete';
  const owner = (run.agents || []).find(agent => agent.id === a.agent);
  const gaps = run.coverage?.gaps?.length || run.coverage?.catching_up;
  if (a.missing_start || gaps || run.status !== 'running' || ['completed', 'failed', 'cancelled', 'stopped', 'unknown'].includes(owner?.status)) return 'incomplete';
  if (transport && transport !== 'connected' && !frozen) return 'incomplete';
  // No per-call heartbeat exists in this projection. Old unmatched starts
  // remain inspectable as unresolved, rather than inflating active counts forever.
  if (!Number.isFinite(a.start) || (run.duration || 0) - a.start > 60) return 'incomplete';
  return 'running'; // Observed open call at this display boundary, never a promise of liveness.
}
export function executionRange(run) {
  const start = Number.isFinite(run.execution_started_at) && Number.isFinite(run.time_origin) ? Math.max(0, run.execution_started_at - run.time_origin) : 0;
  return [start, Math.max(start, Math.min(run.duration || 0, run.measurement?.end_offset ?? run.duration ?? 0))];
}
export function inRange(a, range, end) {
  if (!range) return true;
  if (a.missing_start || !Number.isFinite(a.start)) return false;
  const stop = a.outcome === 'running' || !terminal(a) ? end : a.start + (a.duration || 0);
  return a.start <= range[1] && stop >= range[0];
}
export function callStats(calls) {
  const done = calls.filter(timed), durations = done.map(a => a.duration);
  const usage = key => {
    const known = calls.filter(a => Number.isFinite(a[key]));
    return {value: known.length ? known.reduce((n, a) => n + a[key], 0) : null, samples: known.length};
  };
  const output = done.filter(a => a.duration > 0 && Number.isFinite(a.output_tokens));
  return {calls: calls.length, completed: calls.filter(terminal).length, samples: done.length,
    p50: quantile(durations, .5), p95: quantile(durations, .95), total: durations.reduce((n, d) => n + d, 0),
    slowest: done.reduce((best, a) => !best || a.duration > best.duration ? a : best, null),
    failures: calls.filter(failure).length, input: usage('tokens'), output: usage('output_tokens'),
    throughput: output.length ? output.reduce((n, a) => n + a.output_tokens, 0) / output.reduce((n, a) => n + a.duration, 0) : null};
}
export function breakdown(calls, dimension) {
  const groups = new Map();
  for (const a of calls) {const key = a[dimension] || 'Unknown'; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(a);}
  return [...groups].map(([key, members]) => ({key, ...callStats(members)}));
}
// Sweep interval boundaries. Different simultaneous categories are ambiguous;
// same-category overlaps are a union. Unobserved time stays unknown, never idle.
export function attributeTime(calls, intervals, bounds, run, options = {}) {
  const [start, end] = bounds, events = [], seen = new Set();
  const add = (a, kind) => {
    if (!Number.isFinite(a.start) || a.missing_start || String(a.source).includes('timestamp_missing')) return;
    let stop;
    if (timed(a)) stop = a.start + a.duration;
    else if (a.outcome === 'running' && operationState(a, run, options) === 'running') stop = end;
    else if (!a.outcome && Number.isFinite(a.duration) && a.duration >= 0 && a.source !== 'coverage gap') stop = a.start + a.duration;
    else return;
    const left = Math.max(start, a.start), right = Math.min(end, stop);
    if (right > left) events.push([left, kind, 1], [right, kind, -1]);
  };
  for (const a of calls) {
    seen.add(a.id);
    if (a.kind === 'model') add(a, 'llm');
    else if (toolCall(a)) add(a, 'tools');
    else if (['queue', 'dependency'].includes(a.kind)) add(a, 'waiting');
  }
  for (const i of intervals) if (['queue', 'dependency'].includes(i.kind) && !seen.has(i.action || i.id)) add(i, 'waiting');
  events.sort((a, b) => a[0] - b[0]);
  const active = {llm: 0, tools: 0, waiting: 0}, segments = [], totals = {llm: 0, tools: 0, waiting: 0, unknown: 0};
  let cursor = start, index = 0;
  const append = right => {
    if (right <= cursor) return;
    const kinds = Object.keys(active).filter(k => active[k] > 0), kind = kinds.length === 1 ? kinds[0] : 'unknown';
    totals[kind] += right - cursor;
    const last = segments.at(-1);
    if (last?.kind === kind) last.end = right;
    else segments.push({start: cursor, end: right, kind});
    cursor = right;
  };
  while (index < events.length) {
    const at = events[index][0]; append(at);
    while (index < events.length && events[index][0] === at) {const [, kind, delta] = events[index++]; active[kind] += delta;}
  }
  append(end);
  return {totals, segments, measured: totals.llm + totals.tools + totals.waiting, elapsed: end - start};
}
export function profilerData(run, {agent = '', range = null, model = '', tool = '', status = '', ...options} = {}) {
  const bounds = range || executionRange(run), actions = [], byAgent = new Map(), waits = new Map();
  for (const a of new Map((run.actions || []).map(a => [a.id, a])).values()) {
    if (agent && a.agent !== agent || !inRange(a, range, run.duration)) continue;
    actions.push(a); if (!byAgent.has(a.agent)) byAgent.set(a.agent, []); byAgent.get(a.agent).push(a);
  }
  for (const i of run.intervals || []) if (['queue', 'dependency'].includes(i.kind)) {if (!waits.has(i.agent)) waits.set(i.agent, []); waits.get(i.agent).push(i);}
  const models = actions.filter(a => a.kind === 'model' && (!model || (a.model || 'Unknown') === model));
  const tools = actions.filter(a => toolCall(a) && (!tool || a.name === tool) && (!status || operationState(a, run, options) === status));
  const agents = (run.agents || []).filter(a => a.id !== 'workflow' && (!agent || a.id === agent)).map(a => {
    const calls = byAgent.get(a.id) || [], llm = callStats(calls.filter(c => c.kind === 'model')), tool = callStats(calls.filter(toolCall));
  const reliable = calls.filter(timed), first = reliable.reduce((n, c) => Math.min(n, c.start), Infinity);
    const last = reliable.reduce((n, c) => Math.max(n, c.start + c.duration), -Infinity);
    return {...a, calls, llm, tool, errors: calls.filter(c => (toolCall(c) || c.kind === 'model') && failure(c)).length,
      elapsed: Number.isFinite(first) ? Math.max(0, Math.min(bounds[1], last) - Math.max(bounds[0], first)) : null,
      attribution: attributeTime(calls, waits.get(a.id) || [], bounds, run, options)};
  });
  return {actions, models, tools, agents, bounds, llm: callStats(models), tool: callStats(tools),
    incomplete: actions.filter(a => (toolCall(a) || a.kind === 'model') && operationState(a, run, options) === 'incomplete'),
    active: tools.filter(a => operationState(a, run, options) === 'running'), options};
}
// Keep first/last and bucket extrema, preserving latency spikes with bounded SVGs.
export function sampleExtrema(points, limit = 480) {
  if (points.length <= limit) return points;
  const result = [points[0]], size = Math.ceil((points.length - 2) / Math.floor((limit - 2) / 2));
  for (let i = 1; i < points.length - 1; i += size) {
    let low = points[i], high = points[i];
    for (let j = i; j < Math.min(points.length - 1, i + size); j++) {
      if (points[j][1] < low[1]) low = points[j]; if (points[j][1] > high[1]) high = points[j];
    }
    result.push(...(low === high ? [low] : [low, high].sort((a, b) => a[0] - b[0])));
  }
  result.push(points.at(-1)); return result;
}
export function performanceFindings(data, run, metrics = []) {
  const findings = [], add = (title, evidence, tab, extra = {}) => findings.push({title, evidence, tab, ...extra});
  const slowModel = data.llm.slowest;
  if (data.llm.samples >= 5 && slowModel?.duration > Math.max(5, data.llm.p50 * 3)) add('Model latency spike', `${slowModel.duration.toFixed(2)}s · ${(slowModel.duration / data.llm.p50).toFixed(1)}× median E2E latency`, 'llm', {action: slowModel.id, agent: slowModel.agent, range: [slowModel.start, slowModel.start + slowModel.duration]});
  const groups = breakdown(data.tools, 'name').sort((a, b) => b.total - a.total), top = groups[0];
  if (top && data.tool.samples >= 3 && top.total >= 5 && top.total / data.tool.total >= .5) add(`${top.key} dominates measured tool duration`, `${top.total.toFixed(1)}s · ${(100 * top.total / data.tool.total).toFixed(0)}% of summed completed durations (calls may overlap)`, 'tools', {tool: top.key});
  const errors = groups.find(g => g.failures >= 2);
  if (errors) add(`Repeated ${errors.key} failures`, `${errors.failures} failures / ${errors.calls} observed calls`, 'tools', {tool: errors.key});
  if (data.incomplete.length) add('Completion or timing evidence is incomplete', `${data.incomplete.length} calls have unknown terminal state; they are excluded from active counts and latency estimates`, 'overview');
  const memory = metrics.find(m => m.scope === 'Workload' && m.label === 'RAM');
  if (memory?.points.length > 2) {const first = memory.points[0][1], last = memory.points.at(-1)[1]; if (last - first >= 128 && last > first * 1.5) add('Workload memory grew', `${first.toFixed(0)} → ${last.toFixed(0)} MiB · named workload memory, not process RSS`, 'systems', {range: [memory.points[0][0], memory.points.at(-1)[0]]});}
  if (run.coverage?.client_window) add('Retained history limits attribution', 'Counters may cover the full execution; durations and findings use retained observations only.', 'overview');
  if (data.actions.some(a => terminal(a) && !timed(a))) add('Some completed calls cannot be timed', `${data.actions.filter(a => terminal(a) && !timed(a)).length} completed calls lack reliable start/end boundaries`, 'overview');
  return findings.slice(0, 5);
}
export function processResources(run, metrics, range = null) {
  const groups = new Map(), metadata = new Map();
  for (const process of run.summary?.process_metrics || []) {
    if (!process?.identity) continue;
    const owners = new Set(run.sandbox_owners?.[process.sandbox] || []);
    for (const agent of run.agents || []) if (process.sandbox && (agent.sandbox === process.sandbox || agent.sandboxes?.includes(process.sandbox))) owners.add(agent.id);
    metadata.set(process.identity, {owners, command:process.cmdline || process.display_name || '', pid:process.pid});
  }
  for (const a of run.actions || []) {
    const meta = a.metadata || {}, identity = meta['agency.process_identity'] || meta.process_identity;
    if (!identity) continue;
    const entry = metadata.get(identity) || {owners: new Set(), command: ''};
    if (a.agent && a.agent !== 'workflow') entry.owners.add(a.agent);
    entry.command ||= meta['process.command'] || meta.program || ''; metadata.set(identity, entry);
  }
  for (const m of metrics) {
    const match = m.name.match(/^(?:proc|process):([^:]+):/); if (!match) continue;
    const identity = match[1]; if (!groups.has(identity)) groups.set(identity, []); groups.get(identity).push(m);
  }
  return [...groups].map(([identity, series]) => {
    const read = label => {
      const observations = series.filter(s => s.label === label).flatMap(s => s.points.filter(p => !range || p[0] >= range[0] && p[0] <= range[1]));
      return observations.length ? observations.reduce((last, p) => p[0] > last[0] ? p : last)[1] : null;
    };
    const rss = series.filter(s => s.label === 'RSS').flatMap(s => s.points.filter(p => !range || p[0] >= range[0] && p[0] <= range[1])).map(p => p[1]);
    const last = series.reduce((n, s) => Math.max(n, s.points.at(-1)?.[0] ?? -Infinity), -Infinity), meta = metadata.get(identity);
    return {key: identity, agent: meta?.owners.size === 1 ? [...meta.owners][0] : '', command: meta?.command || '', series,
      cpu: read('CPU'), rss: read('RSS'), peak: rss.length ? rss.reduce((a, b) => Math.max(a, b)) : null,
      read: read('Disk read'), write: read('Disk write'), last,
      status: run.duration - last > 10 ? 'No recent sample · exit unknown' : 'Sample observed · state unknown'};
  }).filter(p => !range || p.series.some(s => s.points.some(point => point[0] >= range[0] && point[0] <= range[1])));
}
