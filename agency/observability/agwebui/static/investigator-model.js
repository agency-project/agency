// Shared, deterministic projections used by all six views.
export function alignSequences(left, right, key = item => item.kind) {
  const grid = Array.from({length: left.length + 1}, () => new Uint16Array(right.length + 1));
  for (let i = left.length - 1; i >= 0; i--) {
    for (let j = right.length - 1; j >= 0; j--) {
      grid[i][j] = key(left[i]) === key(right[j]) ? 1 + grid[i + 1][j + 1] : Math.max(grid[i + 1][j], grid[i][j + 1]);
    }
  }
  const rows = [];
  let i = 0, j = 0, diverged = false;
  while (i < left.length || j < right.length) {
    if (i < left.length && j < right.length && key(left[i]) === key(right[j])) {
      rows.push({left: left[i++], right: right[j++], match: true, reconverged: diverged});
      diverged = false;
    } else if (i < left.length && (j >= right.length || grid[i + 1][j] >= grid[i][j + 1])) {
      rows.push({left: left[i++], right: null, match: false});
      diverged = true;
    } else {
      rows.push({left: null, right: right[j++], match: false});
      diverged = true;
    }
  }
  return rows;
}

export function matchesAction(action, agent, query) {
  if (agent && action.agent !== agent) return false;
  return !query || `${action.name} ${action.intent} ${action.command} ${action.result} ${(action.files || []).join(' ')}`.toLowerCase().includes(query.toLowerCase());
}

export function overlapDuration(intervals, start, end) {
  // Union intervals: parallel and nested waits must not inflate wall time.
  const sorted = intervals.map(i => [Math.max(start, i.start), Math.min(end, i.start + i.duration)])
    .filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
  let total = 0, cursor = start;
  for (const [a, b] of sorted) {
    total += Math.max(0, b - Math.max(a, cursor));
    cursor = Math.max(cursor, b);
  }
  return total;
}

export function packActionTracks(actions, minimumDuration = 0) {
  // Separate overlapping actions, including the visible width of instant events.
  const tracks = [], ends = [];
  for (const action of [...actions].sort((a, b) => a.start - b.start || b.duration - a.duration)) {
    let track = ends.findIndex(end => end <= action.start);
    if (track < 0) {track = tracks.length; tracks.push([]);}
    tracks[track].push(action);
    ends[track] = action.start + Math.max(action.duration, minimumDuration);
  }
  return tracks;
}

export function groupCallStack(actions) {
  const groups=[],running=new Map();
  for(const action of actions) {
    if(action.outcome!=='running') {
      groups.push({id:action.id,agent:action.agent,actions:[action],running:false,
        latest:action.start+Math.max(0,action.duration||0)});
      continue;
    }
    // Only still-running calls share a box. All open intervals for this
    // actor overlap now; model calls remain separate from tool calls.
    const lane=`running:${action.agent}:${action.kind==='model'?'model':'tool'}`;
    let group=running.get(lane);
    if(!group){group={id:lane,agent:action.agent,actions:[],running:true,latest:action.start};running.set(lane,group);groups.push(group);}
    group.actions.push(action);group.latest=Math.max(group.latest,action.start);
  }
  return groups.sort((a,b)=>Number(b.running)-Number(a.running)||b.latest-a.latest||a.id.localeCompare(b.id))
    .map(group=>({...group,actions:group.actions.sort((a,b)=>b.start-a.start||a.id.localeCompare(b.id))}));
}
