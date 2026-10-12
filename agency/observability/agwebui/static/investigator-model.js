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

// Live events contain cumulative CPU/I/O counters; saved traces contain rates.
// Keep their recorded scope rather than adding overlapping workload/PID totals.
const gpuStates=['valid','no_new_samples','unsupported','collection_failure','stale','unavailable'];
const gpuStateLabels={no_new_samples:'No new samples',unsupported:'Unsupported metric',collection_failure:'Collection failure',stale:'Stale measurement',unavailable:'Unavailable'};
const decodeIdentity=value=>{try{return decodeURIComponent(value);}catch{return value;}};
const systemMeasurementCache=new WeakMap(),latestMeasurementCache=new WeakMap();
export function systemMeasurements(run,{latestOnly=false}={}) {
  const counters=run.counters||{},metrics=[];
  // Resource samples are append/insert-only in the trajectory stream. Reuse
  // the projection between renders of the same recording/window.
  const future=Object.values(counters).some(v=>Array.isArray(v)&&v.at(-1)?.[0]>run.duration);
  const window=Number.isFinite(run.duration)?(latestOnly&&!future?'all':run.duration):'all';
  const signature=JSON.stringify([window,Object.keys(counters).some(name=>/^gpu\d+:/.test(name))?run.duration:null,Object.entries(counters).map(([key,v])=>[key,Array.isArray(v)?v.length:0,Array.isArray(v)?v.at(-1):null])]);
  const cache=latestOnly?latestMeasurementCache:systemMeasurementCache;
  const cached=cache.get(run);
  if(cached?.counters===counters&&cached.signature===signature)return cached.metrics;
  const names=new Set(Object.keys(counters).map(name=>/^gpu\d+:.*:state$/.test(name)?name.slice(0,-6):name));
  for(const name of names) {
    const samples=counters[name]||[];
    if(!Array.isArray(samples))continue;
    const match=name.match(/(?:^|[: ])(cpu_pct|cpu_percent|cpu %|cpu_us|cpu_s|memory_mb|rss_mb|vms_mb|mem_mb|process_mem_mb|util_pct|power_w_est|power_w|io_r|io_w|net_rx|net_tx|io_read_mb_s|io_write_mb_s|net_receive_mb_s|net_transmit_mb_s|io read MB\/s|io write MB\/s|net rx MB\/s|net tx MB\/s)$/i);
    if(!match)continue;
    const kind=match[1],prefix=name.slice(0,name.length-kind.length);
    const cpu=/^cpu/.test(kind),rawCPU=/^cpu_(us|s)$/.test(kind);
    if(rawCPU&&Array.isArray(counters[prefix+'cpu_pct'])&&counters[prefix+'cpu_pct'].some(p=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1])&&(!Number.isFinite(run.duration)||p[0]<=run.duration)))continue;
    const bytes=['io_r','io_w','net_rx','net_tx'].includes(kind);
    const label=cpu?'CPU':kind==='rss_mb'?'RSS':/^(memory|mem)_mb$/.test(kind)?'RAM':kind==='vms_mb'?'Virtual memory':kind==='util_pct'?'GPU':kind==='power_w'?'Power':kind==='power_w_est'?'Power estimate':/^(io_r|io_read|io read)/.test(kind)?'Disk read':/^(io_w|io_write|io write)/.test(kind)?'Disk write':/^net_(rx|receive)|^net rx/.test(kind)?'Network RX':'Network TX';
    const unit=cpu||kind==='util_pct'?'%':kind.startsWith('power')?'W':/(_mb|MB)$/.test(kind)?'MB':'MB/s';
    const usable=p=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1])&&p[0]>=0&&(!Number.isFinite(run.duration)||p[0]<=run.duration);
    let source;
    if(latestOnly){
      // Live counter arrays are sorted by timestamp. Summary cards need only
      // the latest gauge or the most recent valid counter delta, not history.
      source=[];
      for(let i=samples.length-1;i>=0;i--){
        const point=samples[i];if(!usable(point))continue;
        if(!(rawCPU||bytes)){source=[point];break;}
        const next=source[0];
        if(next&&next[0]>point[0]&&next[1]>=point[1]){source.unshift(point);break;}
        source=[point];
      }
    }else source=samples.filter(usable).sort((a,b)=>a[0]-b[0]);
    const points=[];
    let previous=null;
    for(const point of source) {
      if(rawCPU||bytes) {
        if(previous&&point[0]>previous[0]&&point[1]>=previous[1])points.push([point[0],(point[1]-previous[1])/(point[0]-previous[0])*(rawCPU?(kind==='cpu_us'?0.0001:100):1/2**20)]);
        previous=point;
      }else points.push(point);
    }
    const parts=name.split(':');
    const gpu=/^gpu\d+:/.test(name);
    const state=gpu?(counters[name+':state']||[]).findLast(usable):null;
    if(!points.length&&!state)continue;
    const last=points.at(-1);
    let status=state?gpuStates[state[1]]||'unavailable':'valid';
    if(gpu&&['valid','no_new_samples'].includes(status)&&last&&Number.isFinite(run.duration)&&run.duration-last[0]>10)status='stale';
    if(!last&&status==='valid')status='unavailable';
    const latest=status==='valid'?last:[state?.[0]??last?.[0]??0,null];
    const workload=/^workload(?:_total)?[: ]/.test(name),host=/^(host|system):/.test(name);
    const device=/^gpu\d+:/.test(name)&&parts.length===2;
    const process=/^(proc|process):/.test(name),sandbox=/^(cg|sandbox):/.test(name);
    const gpuProcess=gpu&&parts[1]==='proc'&&parts[3]==='sandbox';
    const processMemory=gpu&&(parts[1]==='processes'&&kind==='mem_mb'||kind==='process_mem_mb');
    const gpuScope=gpuProcess?`${parts[0]} · Process ${parts[2]}${parts[4]!=='~'?` · Sandbox ${decodeIdentity(parts[4])}`:''}`:gpu&&parts[1]==='sandbox'?`${parts[0]} · Sandbox ${decodeIdentity(parts[2])}`:gpu&&parts[1]==='processes'?`${parts[0]} · Compute processes`:parts.length>2&&gpu?`${parts[0]} · ${parts[1]}`:'Recorded scope';
    const scope=workload?'Workload':host?'Host':device?parts[0]:process?`Process ${parts[1]}`:sandbox?`Sandbox ${parts[1]}`:gpuScope;
    const metricLabel=processMemory?'Process VRAM':gpu&&kind==='mem_mb'?'VRAM':gpuProcess&&kind==='util_pct'?'Process SM activity':label;
    metrics.push({name,label:metricLabel,unit,scope,aggregate:workload||host||device,points,latest,derived:rawCPU||bytes,status,statusLabel:gpuStateLabels[status]||'',processIdentity:gpuProcess?parts[2]:null,sandbox:gpuProcess&&parts[4]!=='~'?decodeIdentity(parts[4]):null});
  }
  cache.set(run,{counters,signature,metrics});
  return metrics;
}

// A value is displayed only when its observation exists. Zero remains valid
// reported usage; absent usage and empty resource series are not zero samples.
export function executionMeasurements(run,{includeResources=true}={}) {
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
  const samples=Object.values(run.counters||{}).filter(points=>Array.isArray(points)&&points.some(p=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1])));
  const resourcesMeasured=includeResources?systemMeasurements(run,{latestOnly:true}):[];
  for(const label of ['CPU','RAM','GPU','VRAM']) {
    const metric=resourcesMeasured.find(m=>m.label===label&&m.aggregate)||resourcesMeasured.find(m=>m.label===label);
    if(metric)secondary.push([label,metric.latest[1],metric.unit,`${metric.name} · ${metric.scope} · sample at ${metric.latest[0]}s${metric.statusLabel?` · ${metric.statusLabel}`:''}`]);
  }
  const usage=models.length?`Usage reported for ${input.length}/${models.length} input and ${output.length}/${models.length} output calls.`:'No model calls recorded.';
  const resources=samples.length?`${samples.length} sampled resource series; scope retained in details.`:'No system samples recorded.';
  return {primary,secondary,coverage:`${usage} ${resources}`};
}

// Container (cgroup) measurements of the sandboxes the engine recorded for
// these agents: CPU is every process in the container, 100% = one core. A
// sandbox also used by an agent outside *ids* is shared and is not attributed.
// Host, workload, orchestrator and runtime-daemon series are never included.
export function agentResources(run,ids,stale=10) {
  const wanted=new Set(ids),owners=new Map();
  const remember=(sandbox,id)=>{
    if(typeof sandbox!=='string'||!sandbox||typeof id!=='string'||!id)return;
    const agents=owners.get(sandbox)||[];
    if(!agents.includes(id))agents.push(id);
    owners.set(sandbox,agents);
  };
  for(const a of run.agents||[])for(const sandbox of new Set([a.sandbox,...(Array.isArray(a.sandboxes)?a.sandboxes:[])]))remember(sandbox,a.id);
  // Saved presentation metadata can name a shared owner with no retained
  // spans. Keep it in the ownership set instead of treating the box as sole-owned.
  for(const [sandbox,agents] of Object.entries(run.sandbox_owners||{}))if(Array.isArray(agents))for(const id of agents)remember(sandbox,id);
  const linked=[...owners].filter(([,agents])=>agents.some(id=>wanted.has(id)));
  const own=linked.filter(([,agents])=>agents.every(id=>wanted.has(id))).map(([sandbox])=>sandbox);
  const metrics=own.length?systemMeasurements(run,{latestOnly:true}):[];
  const read=(label,prefixes)=>{
    let value=0,sandboxes=0;
    for(const sandbox of own) {
      // cg:* is the live series; sandbox:* is the saved canonical series.
      // Pick one latest observation, never add aliases of the same cgroup.
      const metric=metrics.filter(m=>m.label===label&&prefixes.some(prefix=>m.name.startsWith(`${prefix}:${sandbox}:`))&&Number.isFinite(m.latest[1])).sort((a,b)=>b.latest[0]-a.latest[0])[0];
      // A container that stopped reporting has no current utilization.
      if(!metric||Number.isFinite(run.duration)&&run.duration-metric.latest[0]>stale)continue;
      value+=metric.latest[1];sandboxes++;
    }
    return sandboxes?{value,sandboxes}:null;
  };
  return {cpu:read('CPU',['cg','sandbox']),memory:read('RAM',['sandbox']),sandboxes:own.length,shared:linked.length-own.length};
}
