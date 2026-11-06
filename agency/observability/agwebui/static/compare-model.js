// Neutral display calculations. Missing values never become zero.
export const available = value => typeof value === 'number' && Number.isFinite(value);
export const UNAVAILABLE = 'Not available';
export function difference(a, b, mode = 'delta') {
  if (!available(a) || !available(b)) return null;
  if (mode === 'percent' && a === 0) return null;
  const value = mode === 'percent' ? (b - a) / a * 100 : b - a;
  return available(value) ? value : null;
}
export function format(value, unit = '', signed = false) {
  if (value == null || value === '' || typeof value === 'number' && !available(value)) return UNAVAILABLE;
  if (typeof value !== 'number') return typeof value === 'object' ? JSON.stringify(value) : String(value);
  return `${signed && value > 0 ? '+' : ''}${new Intl.NumberFormat('en', {maximumFractionDigits: 3}).format(Object.is(value, -0) ? 0 : value)}${unit ? ' ' + unit : ''}`;
}
export const views = ['Overview', 'Tool Trajectory', 'Timeline', 'Agents', 'LLM & Tools', 'Systems', 'Critical Path', 'Metadata'];
export const metrics = [
  ['makespan', 'Execution duration / makespan', 's'], ['recording_duration', 'Profiler recording duration', 's'],
  ['status', 'Completion status', ''], ['agents', 'Agent count', ''], ['requests', 'Recorded model requests', ''],
  ['input', 'Input tokens', ''], ['output', 'Output tokens', ''], ['tools', 'Recorded tool calls', ''],
  ['tool_errors', 'Tool errors', ''], ['cpu', 'CPU time · workload summary', 's'], ['rss', 'Peak RSS · workload', 'MB'],
  ['disk_read', 'Disk read · workload summary', 'MB'], ['disk_write', 'Disk write · workload summary', 'MB'],
  ['network_rx', 'Network RX · workload summary', 'MB'], ['network_tx', 'Network TX · workload summary', 'MB'],
];
export function pairAgents(a, b) {
  const right = new Map(b.map(agent => [agent.id, agent]));
  const rows = a.map(agent => {const other = right.get(agent.id);right.delete(agent.id);return {id: agent.id, a: agent, b: other ?? null};});
  for (const agent of right.values()) rows.push({id: agent.id, a: null, b: agent});
  return rows;
}
export function matchable(a, b) {
  return Array.isArray(a?.match) && Array.isArray(b?.match) && a.match.length === 2
    && a.match.every((id, i) => typeof id === 'string' && id && id === b.match[i]);
}
export function compatibleSeries(a, b) {
  return !!a?.comparable && !!b?.comparable && a.scope === b.scope && a.metric === b.metric && a.unit === b.unit;
}
export function timeWindow(a, b, from = 0, span = null) {
  const extent = Math.max(available(a) ? a : 0, available(b) ? b : 0);
  const width = Math.min(extent, available(span) && span > 0 ? span : extent);
  const start = Math.max(0, Math.min(available(from) ? from : 0, extent - width));
  return {extent, start, end: start + width, span: width};
}
export function choosePair(runs, a, b) {
  const ids = new Set(runs.map(run => run.id));
  if (ids.has(a) || ids.has(b)) return [ids.has(a) ? a : '', ids.has(b) ? b : ''];
  // A pair has no unambiguous baseline direction without an explicit request.
  return ['', ''];
}
export function historyURL(run, agent = '', event = '') {
  const params = new URLSearchParams({run, view: 'trajectory'});
  if (agent) params.set('agent', agent);
  if (event) params.set('event', event);
  return `/profiler?${params}`;
}
