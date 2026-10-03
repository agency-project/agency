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
  // Group connected overlapping intervals for one actor. Touching endpoints
  // are sequential, and model calls do not imply concurrent tool execution.
  const groups=[], lanes=new Map();
  for(const action of [...actions].sort((a,b)=>a.start-b.start||a.id.localeCompare(b.id))) {
    const lane=`${action.agent}:${action.kind==='model'?'model':'tool'}`;
    const end=action.outcome==='running'?Infinity:action.start+Math.max(0,action.duration||0);
    let group=lanes.get(lane);
    if(!group||action.start>=group.end||end<=action.start) {
      group={id:action.id,agent:action.agent,actions:[],end,latest:action.start,running:false};
      groups.push(group);lanes.set(lane,group);
    }
    group.actions.push(action);group.end=Math.max(group.end,end);
    group.latest=Math.max(group.latest,action.start);
    group.running ||= action.outcome==='running';
  }
  return groups.sort((a,b)=>Number(b.running)-Number(a.running)||b.latest-a.latest)
    .map(group=>({...group,actions:group.actions.sort((a,b)=>Number(b.outcome==='running')-Number(a.outcome==='running')||b.start-a.start)}));
}
