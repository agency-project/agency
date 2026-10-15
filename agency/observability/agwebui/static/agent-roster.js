// Deterministic roster projection. Activity observations describe calls; only
// recorded agent metadata supplies agent state and parent/child relationships.
export const DEFAULT_AGENT_COLUMNS = 3;
export const MAX_AGENT_COLUMNS = 6;

export function rosterAgents(run) {
  const agents = new Map();
  for (const agent of run?.agents || []) {
    if (agent.id && agent.id !== 'workflow') agents.set(agent.id, agent);
  }
  // Older recordings may contain action actors without an agent declaration.
  for (const action of run?.actions || []) {
    if (action.agent && action.agent !== 'workflow' && !agents.has(action.agent)) {
      agents.set(action.agent, {id: action.agent});
    }
  }
  return [...agents.values()];
}

export function reconcilePinnedAgents(run, pinnedIds = [], {customized = false, defaultCount = DEFAULT_AGENT_COLUMNS, limit = MAX_AGENT_COLUMNS} = {}) {
  const known = new Set(rosterAgents(run).map(agent => agent.id));
  const ids = [...new Set(pinnedIds)].filter(id => known.has(id)).slice(0, limit);
  if (!customized) {
    for (const id of known) {
      if (ids.length >= Math.min(defaultCount, limit)) break;
      if (!ids.includes(id)) ids.push(id);
    }
  }
  return ids;
}

export function pinAgent(run, pinnedIds, id, {limit = MAX_AGENT_COLUMNS} = {}) {
  const ids = reconcilePinnedAgents(run, pinnedIds, {customized: true, limit});
  if (!rosterAgents(run).some(agent => agent.id === id)) return {ids, changed: false, reason: 'Agent is not recorded in this execution.'};
  if (ids.includes(id)) return {ids, changed: false, reason: ''};
  if (ids.length >= limit) return {ids, changed: false, reason: `Hide a column before opening another; ${limit} columns are already pinned.`};
  return {ids: [...ids, id], changed: true, reason: ''};
}

export function longRunningCalls(run, {agent = '', clock = run?.duration, threshold = 30} = {}) {
  if (!Number.isFinite(clock)) return [];
  return (run?.actions || []).filter(action => action.outcome === 'running'
    && action.kind !== 'syscall'
    && (!agent || action.agent === agent)
    && Number.isFinite(action.start) && !action.missing_start
    && !String(action.source || '').includes('timestamp_missing')
    && clock >= action.start && clock - action.start >= threshold)
    .map(action => ({action, elapsed: clock - action.start, provenance: 'calculated from recorded start and displayed execution clock'}))
    .sort((a, b) => b.elapsed - a.elapsed || compareIds(a.action.id, b.action.id));
}

const compareIds = (a, b) => String(a) < String(b) ? -1 : String(a) > String(b) ? 1 : 0;

export function agentRoster(run, {pinnedIds = [], query = '', status = '', longRunningSeconds = 30, clock = run?.duration} = {}) {
  const agents = rosterAgents(run), known = new Set(agents.map(agent => agent.id));
  const pinned = new Set(pinnedIds), calls = new Map(), longCalls = new Map();
  for (const action of run?.actions || []) {
    if (action.kind === 'syscall') continue;
    if (!calls.has(action.agent)) calls.set(action.agent, []);
    calls.get(action.agent).push(action);
  }
  for (const item of longRunningCalls(run, {clock, threshold: longRunningSeconds})) {
    if (!longCalls.has(item.action.agent)) longCalls.set(item.action.agent, []);
    longCalls.get(item.action.agent).push(item);
  }
  const rows = agents.map(agent => {
    const actions = calls.get(agent.id) || [];
    const failures = actions.filter(action => action.outcome === 'failed')
      .sort((a, b) => (Number.isFinite(b.start) ? b.start : -Infinity) - (Number.isFinite(a.start) ? a.start : -Infinity) || compareIds(a.id, b.id));
    // The canonical parent field is explicit metadata. An unresolved parent is
    // retained separately, without inventing a visible node or a relationship.
    const recordedParent = typeof agent.parent === 'string' && agent.parent !== agent.id ? agent.parent : null;
    return {
      id: agent.id, label: agent.label || agent.id, status: agent.status || 'unknown',
      parentId: recordedParent && known.has(recordedParent) ? recordedParent : null,
      recordedParentId: recordedParent, childIds: [], pinned: pinned.has(agent.id),
      activeCount: actions.filter(action => action.outcome === 'running').length,
      failedActionCount: failures.length, latestFailedAction: failures[0] || null,
      longCalls: longCalls.get(agent.id) || [], agent,
    };
  });
  const byId = new Map(rows.map(row => [row.id, row]));
  const cyclicIds = new Set();
  for (const row of rows) {
    // Malformed cycles remain explicit recorded metadata, but cannot make the
    // roster traversal recursively loop or hide every root agent.
    let parent = row.parentId, cyclic = false;
    const visited = new Set([row.id]);
    while (parent) {
      if (visited.has(parent)) {cyclic = true; break;}
      visited.add(parent); parent = byId.get(parent)?.parentId;
    }
    if (cyclic) cyclicIds.add(row.id);
  }
  for (const row of rows) if (cyclicIds.has(row.id)) row.parentId = null;
  for (const row of rows) if (row.parentId) byId.get(row.parentId).childIds.push(row.id);
  const needle = query.trim().toLowerCase();
  return rows.filter(row => (!needle || `${row.label} ${row.id}`.toLowerCase().includes(needle))
    && (!status || (status === 'hidden' ? !row.pinned : status === 'failed-actions' ? row.failedActionCount > 0 : row.status === status)));
}
