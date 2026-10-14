// Compact, historical backend analysis is the sole source of causality here.
// The browser indexes spans; it never reconstructs a graph from raw events.
export const NODE_LIMIT = 240;
export const EDGE_LIMIT = 400;
export const finite = value => Number.isFinite(value);
export const kindOf = op => op.band || (op.kind === 'model' ? 'LLM' : op.kind === 'process' ? 'System' : ['dependency', 'queue', 'wait'].includes(op.kind) ? 'Wait' : 'Tool');
export const elapsed = op => finite(op.start) && finite(op.end) && op.end >= op.start ? op.end - op.start : null;
export function profileIndex(profile) {
  const operations = new Map((profile.operations || []).map(op => [op.id, op]));
  const processes = new Map((profile.processes || []).map(p => [p.id, p]));
  const agents = new Map((profile.agents || []).map(a => [a.id, a]));
  const incoming = new Map(), outgoing = new Map(), agentCounts = new Map();
  for (const op of operations.values()) agentCounts.set(op.agent || 'Unattributed', (agentCounts.get(op.agent || 'Unattributed') || 0) + 1);
  for (const edge of profile.edges || []) {
    if (!operations.has(edge.source) || !operations.has(edge.target)) continue;
    if (!incoming.has(edge.target)) incoming.set(edge.target, []);
    if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
    incoming.get(edge.target).push(edge); outgoing.get(edge.source).push(edge);
  }
  return {profile, operations, processes, agents, incoming, outgoing, agentCounts};
}
export function profileBounds(profile) {
  const bounds = profile.bounds || {};
  let start = finite(bounds.start) ? bounds.start : Infinity;
  let end = finite(bounds.end) ? bounds.end : -Infinity;
  for (const op of profile.operations || []) {
    if (finite(op.start)) {start = Math.min(start, op.start); end = Math.max(end, op.start);}
    if (finite(op.end)) end = Math.max(end, op.end);
  }
  if (!finite(start)) start = 0;
  if (!finite(end)) end = start + (profile.summary?.wall_time || profile.run?.duration || 0);
  return {start, end: Math.max(start + .001, end)};
}
const comparison = (a, b) => (a.start ?? Infinity) - (b.start ?? Infinity) || String(a.id).localeCompare(String(b.id));
export function graphSelection(index, options = {}) {
  const {profile, operations, incoming, outgoing, processes} = index;
  const relevant = new Set();
  const mode = options.mode || 'critical', focus = options.agent || '', query = (options.query || '').toLowerCase();
  const critical = new Set(profile.critical_path?.operation_ids || []);
  const joinPairs = new Set((profile.joins || []).flatMap(join => (join.prerequisites || []).map(p => `${p.operation_id}>${join.operation_id}`)));
  const reliable = edge => edge.blocking && ['recorded', 'reconstructed', 'verified'].includes(edge.confidence);
  const joinInput = edge => reliable(edge) && (joinPairs.has(`${edge.source}>${edge.target}`) || kindOf(operations.get(edge.target) || {}) === 'Wait' && ['wait_result', 'join', 'prerequisite', 'request_prerequisite'].includes(edge.kind));
  let candidates = [...operations.values()].filter(op => !focus || op.agent === focus);
  if (mode === 'critical' && !query) {
    const gateIds = new Set((profile.joins || []).flatMap(join => [join.operation_id, ...(join.latest_prerequisite_ids || [])]));
    const main = candidates.filter(op => op.critical || critical.has(op.id) || op.gating || gateIds.has(op.id));
    // A partial trace can have no established path. Keep its observations visible.
    if (main.length) candidates = main;
  }
  if (query) candidates = candidates.filter(op => [op.label, op.id, op.agent, op.model].some(v => String(v || '').toLowerCase().includes(query)));
  for (const op of candidates) relevant.add(op.id);
  // Focus retains directly connected agents. Their appearance is context, never
  // evidence of a new dependency. Critical mode keeps all recorded join inputs.
  for (const op of candidates) for (const edge of [...(incoming.get(op.id) || []), ...(outgoing.get(op.id) || [])]) {
    if (!edge.blocking || !['recorded', 'reconstructed', 'verified'].includes(edge.confidence)) continue;
    const other = edge.source === op.id ? edge.target : edge.source;
    if (operations.get(other)?.agent !== op.agent || joinInput(edge)) relevant.add(other);
  }
  if (options.selected && operations.has(options.selected)) relevant.add(options.selected);
  const sequence = [...relevant].map(id => operations.get(id)).sort(comparison);
  const offset = Math.max(0, Math.min(options.offset || 0, Math.max(0, sequence.length - 1)));
  const nodes = sequence.slice(offset, offset + NODE_LIMIT);
  if (options.selected && operations.has(options.selected) && !nodes.some(n => n.id === options.selected)) nodes.push(operations.get(options.selected));
  const ids = new Set(nodes.map(n => n.id));
  const expanded = options.expanded || new Set();
  let processHidden = 0;
  for (const op of [...nodes]) if (expanded.has(op.id)) {
    const children = (op.process_ids || []).map(id => processes.get(id)).filter(Boolean);
    for (const p of children) {
      if (ids.has(p.id)) continue;
      if (nodes.length >= NODE_LIMIT + 1) {processHidden++; continue;}
      nodes.push({...p, agent:p.agent || op.agent, kind:'process', band:'System', parent_id:op.id, critical:false, overlay:true}); ids.add(p.id);
    }
  }
  if (options.selected && processes.has(options.selected) && !ids.has(options.selected)) {
    const p = processes.get(options.selected); nodes.push({...p, kind:'process', band:'System', critical:false, overlay:true}); ids.add(p.id);
  }
  const dependencyEdges = (profile.edges || []).filter(e => e.source !== e.target && ids.has(e.source) && ids.has(e.target) && (e.blocking || options.relationships) && (!options.crossOnly || operations.get(e.source)?.agent !== operations.get(e.target)?.agent));
  // The simplified path shows only original critical edges, recorded fan-in
  // inputs, and reliable cross-agent context. Container admission/completion
  // fan-out remains in inspection and explicit full-relationship views.
  const displayedEdges = mode === 'critical' && !options.relationships ? dependencyEdges.filter(e => e.critical || joinInput(e) || reliable(e) && operations.get(e.source)?.agent !== operations.get(e.target)?.agent) : dependencyEdges;
  const important = [], local = new Map();
  // Preserve critical and cross-agent links. A prompt can record consumption
  // of many older responses; draw the latest recorded local prerequisite of
  // each kind and keep the other evidence in inspection instead of a hairball.
  for (const edge of displayedEdges) {
    if (edge.critical || edge.gating || joinInput(edge) || operations.get(edge.source)?.agent !== operations.get(edge.target)?.agent || !edge.blocking) important.push(edge);
    else {
      const key = `${edge.target}:${edge.kind}:${edge.confidence}`, prior = local.get(key);
      if (!prior || (edge.source_time ?? -Infinity) > (prior.source_time ?? -Infinity)) local.set(key, edge);
    }
  }
  const edges = [...important, ...local.values()].slice(0, EDGE_LIMIT);
  for (const node of nodes) if (node.overlay && ids.has(node.parent_id)) edges.push({id:`owner:${node.id}`, source:node.parent_id, target:node.id, kind:'process attribution', confidence:'provenance', blocking:false});
  return {nodes, edges, sequence, offset, untimed:nodes.filter(n => !finite(n.start)).length, hidden:Math.max(0, sequence.length - Math.min(NODE_LIMIT, sequence.length - offset)), processHidden, edgeHidden:Math.max(0, dependencyEdges.length - edges.filter(e => e.confidence !== 'provenance').length)};
}
// Pack overlapping spans inside a named agent lane. Process overlays receive
// their own rows; their runtime never becomes additional wall-clock work.
export function swimlaneLayout(index, selection, scale = 10, selected = '') {
  const bounds = profileBounds(index.profile), positions = new Map(), lanes = [], displayNodes = [];
  const groups = new Map();
  for (const node of selection.nodes) {
    const agent = node.agent || 'Unattributed';
    if (!groups.has(agent)) groups.set(agent, []);
    groups.get(agent).push(node);
  }
  // Agents with no selected span remain visible, so independent parallel work
  // is never erased by the simplified path view.
  for (const a of index.agents.values()) if (!groups.has(a.id)) groups.set(a.id, []);
  let y = 42;
  for (const [id, nodes] of groups) {
    const rows = [], top = y;
    const buckets = new Map(), packed = [];
    for (const node of nodes) {
      const width = elapsed(node) == null ? 0 : elapsed(node) * scale;
      if (finite(node.start) && width < 12 && node.id !== selected && !node.overlay) {
        const key = `${kindOf(node)}:${Math.floor((node.start - bounds.start) * scale / 32)}`;
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(node);
      } else packed.push(node);
    }
    for (const members of buckets.values()) {
      members.sort(comparison);
      if (members.length === 1) packed.push(members[0]);
      else {
        const first = members[0], end = Math.max(...members.map(n => finite(n.end) ? n.end : n.start));
        packed.push({id:`display:${first.id}`, agent:id, band:kindOf(first), kind:first.kind, label:`${members.length} ${kindOf(first)} observations`, start:first.start, end, duration:null, critical:members.some(n => n.critical), gating:members.some(n => n.gating), aggregate:true, members:members.map(n => n.id)});
      }
    }
    for (const node of packed.sort(comparison)) {
      // An absent start has no position on a temporal axis. The bounded text
      // list retains access without placing it at a fabricated zero boundary.
      if (!finite(node.start)) continue;
      const at = node.start;
      const stop = finite(node.end) && node.end >= at ? node.end : at;
      const left = (at - bounds.start) * scale, width = Math.max(node.aggregate ? 24 : 5, (stop - at) * scale);
      let row = rows.findIndex(right => right + 5 <= left);
      if (row < 0) row = rows.length;
      rows[row] = left + width;
      positions.set(node.id, {x:left + 16, y:top + row * 48 + 9, width, height:30, node, untimed:!finite(node.start)});
      displayNodes.push(node);
      if (node.aggregate) for (const member of node.members) {
        const actual = index.operations.get(member);
        positions.set(member, {x:16+(actual.start-bounds.start)*scale, y:top+row*48+9, width:Math.max(5,(elapsed(actual)||0)*scale), height:30, node:actual, display_group:node.id});
      }
    }
    const height = Math.max(62, rows.length * 48 + 14);
    lanes.push({id, agent:index.agents.get(id), top, height, count:nodes.length, context:selection.nodes.some(n => n.agent === id && !n.critical)});
    y += height;
  }
  return {bounds, positions, lanes, displayNodes, width:Math.max(700, (bounds.end - bounds.start) * scale + 48), height:Math.max(160, y + 16), scale};
}
export function visibleMarks(selection, layout, viewport = {}) {
  const left = viewport.left || 0, right = viewport.right ?? Infinity, top = viewport.top || 0, bottom = viewport.bottom ?? Infinity;
  const inside = p => p && p.x + Math.max(p.width, 12) >= left && p.x <= right && p.y + p.height >= top && p.y <= bottom;
  return {nodes:(layout.displayNodes || selection.nodes).filter(n => inside(layout.positions.get(n.id))), edges:selection.edges.filter(e => {
    const a = layout.positions.get(e.source), b = layout.positions.get(e.target);
    if (a?.display_group && a.display_group === b?.display_group) return false;
    return a && b && Math.max(a.x + a.width, b.x) >= left && Math.min(a.x, b.x) <= right && Math.max(a.y, b.y) + 40 >= top && Math.min(a.y, b.y) <= bottom;
  })};
}
export function dependencyRoute(edge, layout) {
  const a = layout.positions.get(edge.source), b = layout.positions.get(edge.target);
  if (!a || !b) return null;
  const source = finite(edge.source_time) ? edge.source_time : edge.confidence === 'provenance' ? a.node.start : a.node.end ?? a.node.start;
  const target = finite(edge.target_time) ? edge.target_time : b.node.start;
  // Untimed relationships remain inspectable; inventing an anchor would imply
  // a temporal constraint the recording did not establish.
  if (!finite(source) || !finite(target)) return null;
  const sx = 16 + (source - layout.bounds.start) * layout.scale, tx = 16 + (target - layout.bounds.start) * layout.scale;
  const sy = a.y + a.height, ty = b.y + b.height;
  const channel = Math.max(sy, ty) + 8;
  return {sx, tx, path:`M ${sx} ${sy} V ${channel} H ${tx} V ${ty}`};
}
export function criticalContribution(index, id) {
  const segments = (index.profile.critical_path?.segments || []).filter(s => s.operation_id === id);
  return segments.length ? segments.reduce((total, s) => total + (finite(s.duration) ? s.duration : Math.max(0, (s.end || 0) - (s.start || 0))), 0) : null;
}
export function historyURL(profile, operation, category = '') {
  const query = new URLSearchParams({run:profile.run?.id || '', view:category ? 'profiler' : 'trajectory'});
  if (operation?.action_id) query.set('event', operation.action_id);
  if (operation?.agent) query.set('agent', operation.agent);
  if (category) query.set('profiler', category);
  return `/profiler?${query}`;
}
export const safeLink = value => typeof value === 'string' && (/^https?:\/\//.test(value) || /^\/(?!\/)/.test(value)) ? value : '';
export function processSelection(index, {agent = '', query = '', offset = 0} = {}) {
  const search = query.toLowerCase();
  const matches = [...index.processes.values()].filter(p => (!agent || p.agent === agent) && (!search || [p.id, p.identity, p.label, p.agent, p.attribution, p.metrics?.command_line].some(value => String(value || '').toLowerCase().includes(search))));
  const begin = Math.max(0, Math.min(offset, Math.max(0, matches.length - 1)));
  return {rows:matches.slice(begin, begin + 50), offset:begin, count:matches.length};
}
