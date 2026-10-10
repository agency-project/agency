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
        latest:action.start});
      continue;
    }
    // Only still-running calls share a box. All open intervals for this
    // actor overlap now; model calls remain separate from tool calls.
    const lane=`running:${action.agent}:${action.kind==='model'?'model':'tool'}`;
    let group=running.get(lane);
    if(!group){group={id:lane,agent:action.agent,actions:[],running:true,latest:action.start};running.set(lane,group);groups.push(group);}
    group.actions.push(action);group.latest=Math.max(group.latest,action.start);
  }
  return groups.sort((a,b)=>Number(b.running)-Number(a.running)||b.latest-a.latest||(a.id>b.id?1:a.id<b.id?-1:0))
    .map(group=>({...group,actions:group.actions.sort((a,b)=>b.start-a.start||(a.id>b.id?1:a.id<b.id?-1:0))}));
}
export function executionIdentity(run) {
  return run?.time_origin==null?run?.id:JSON.stringify([run.id,run.time_origin]);
}
export function readableExecutionTitle(run) {
  const title=typeof run.title==='string'?run.title.trim():'';
  const task=typeof run.task==='string'?run.task.trim().split('\n').find(line=>line.trim()):'';
  const generic=!title||title===run.id||/^(Current (live )?execution|Current run|Recorded execution)(?:\s*·|$)/i.test(title);
  return (generic&&task?task:title)||task||'Execution';
}

// A value is displayed only when its observation exists. Zero remains valid
// reported usage; absent usage and empty resource series are not zero samples.
export function executionMeasurements(run) {
  const actions=run.actions||[],agents=(run.agents||[]).filter(a=>a.id!=='workflow');
  const models=actions.filter(a=>a.kind==='model');
  const input=models.filter(a=>Number.isFinite(a.tokens)),output=models.filter(a=>Number.isFinite(a.output_tokens));
  const sum=(items,key)=>items.reduce((total,a)=>total+a[key],0);
  const primary=[],secondary=[];
  if(Number.isFinite(run.duration)&&(run.duration>0||actions.length||run.time_origin!=null))primary.push(['Wall time',run.duration,'time','Observed execution wall interval']);
  if(actions.length)primary.push(['Calls',actions.length,'count','Observed model and tool/activity calls']);
  if(agents.length)primary.push(['Agents',agents.length,'count','Recorded actors']);
  if(input.length)primary.push(['Input tokens',sum(input,'tokens'),'count',`${input.length}/${models.length} model calls report input usage`]);
  if(output.length)primary.push(['Output tokens',sum(output,'output_tokens'),'count',`${output.length}/${models.length} model calls report output usage`]);
  if(models.length&&Number.isFinite(run.duration)) {
    const intervals=(run.intervals||[]).filter(i=>i.kind==='model');
    const measured=(intervals.length?intervals:models).filter(i=>Number.isFinite(i.start)&&Number.isFinite(i.duration));
    if(measured.length)secondary.push(['Model intervals',overlapDuration(measured,0,run.duration),'time','Union of observed model intervals']);
  }
  if(input.length&&output.length)secondary.push(['Total tokens',sum(input,'tokens')+sum(output,'output_tokens'),'count',`Reported input (${input.length}/${models.length}) + output (${output.length}/${models.length}); missing usage excluded`]);
  if(output.length&&run.duration>0)secondary.push(['Output rate',sum(output,'output_tokens')/run.duration,'rate',`Run average · reported output / wall time (${output.length}/${models.length} calls)`]);
  const knownStates=agents.filter(a=>a.status&&a.status!=='unknown');
  if(knownStates.length===agents.length&&agents.length)secondary.push(['Active agents',agents.filter(a=>['running','running_skill','running_tool','waiting_llm','executing','busy'].includes(a.status)).length,'count','Recorded states; paused agents excluded']);
  const samples=Object.entries(run.counters||{}).map(([name,points])=>[name,points.filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1])).at(-1)]).filter(([,point])=>point);
  for(const [label,pattern,unit] of [['CPU',/workload.*cpu_pct|cpu.*%/i,'%'],['RAM',/workload.*memory|memory_mb|rss_mb/i,'MB'],['GPU',/^gpu\d+:util_pct/i,'%'],['VRAM',/^gpu\d+:mem_mb/i,'MB']]) {
    const metric=samples.find(([name])=>pattern.test(name));
    if(metric)secondary.push([label,metric[1][1],unit,`${metric[0]} · sample at ${metric[1][0]}s`]);
  }
  const usage=models.length?`Usage reported for ${input.length}/${models.length} input and ${output.length}/${models.length} output calls.`:'No model calls recorded.';
  const resources=samples.length?`${samples.length} sampled resource series; scope retained in details.`:'No system samples recorded.';
  return {primary,secondary,coverage:`${usage} ${resources}`};
}
