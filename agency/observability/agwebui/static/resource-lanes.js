// Per-agent and host resource series for the Critical Path lanes. Pure
// projections over recorded counters: no interpolation, no attribution beyond
// an explicit, sole-owner sandbox cgroup, and no host value assigned to an agent.
import {systemMeasurements} from './investigator-model.js';

// Each metric names its unit honestly. Agent CPU is cgroup CPU time per wall
// second: 100% is one core fully busy, so values above 100% are valid. Host
// CPU is normalized to all CPUs and never exceeds 100%.
export const AGENT_METRICS=[
 {key:'cpu',label:'CPU',unit:'core%',measure:'CPU',combine:'sum',note:'% of one core (100% = 1 core busy)'},
 {key:'memory',label:'Memory',unit:'MiB',measure:'RAM',combine:'sum',note:'cgroup memory.current'},
 {key:'cpu_time',label:'CPU time',unit:'s',derived:'cpu',note:'cumulative CPU seconds integrated from samples'},
 {key:'disk_read',label:'Disk read',unit:'MiB/s',measure:'Disk read',combine:'sum',note:'cgroup io.stat'},
 {key:'disk_write',label:'Disk write',unit:'MiB/s',measure:'Disk write',combine:'sum',note:'cgroup io.stat'},
 {key:'net_rx',label:'Network RX',unit:'MiB/s',measure:'Network RX',combine:'sum',note:'sandbox network namespace'},
 {key:'net_tx',label:'Network TX',unit:'MiB/s',measure:'Network TX',combine:'sum',note:'sandbox network namespace'},
 {key:'cpu_psi',label:'CPU pressure',unit:'psi%',counter:'cpu_psi_some_pct',combine:'max',note:'cgroup cpu.pressure some'},
 {key:'memory_psi',label:'Memory pressure',unit:'psi%',counter:'memory_psi_some_pct',combine:'max',note:'cgroup memory.pressure some'},
 {key:'io_psi',label:'I/O pressure',unit:'psi%',counter:'io_psi_some_pct',combine:'max',note:'cgroup io.pressure some'},
];
export const AGENT_DEFAULT=['cpu','memory'];
export const HOST_METRICS=[
 {key:'cpu',label:'CPU total',unit:'host%',counter:'host:cpu_util_pct',note:'% of all host CPUs'},
 {key:'memory',label:'Memory',unit:'MiB',counter:'host:mem_used_mb',limit:'host:mem_total_mb',note:'MemTotal − MemAvailable'},
 {key:'cpu_psi',label:'CPU PSI',unit:'psi%',counter:'host:cpu_psi_some_pct',note:'/proc/pressure/cpu some'},
 {key:'memory_psi',label:'Memory PSI',unit:'psi%',counter:'host:memory_psi_some_pct',note:'/proc/pressure/memory some'},
 {key:'io_psi',label:'I/O PSI',unit:'psi%',counter:'host:io_psi_some_pct',note:'/proc/pressure/io some'},
 {key:'disk_read',label:'Disk read',unit:'MiB/s',counter:'host:disk_read_mb_s',note:'physical disks'},
 {key:'disk_write',label:'Disk write',unit:'MiB/s',counter:'host:disk_write_mb_s',note:'physical disks'},
 {key:'net_rx',label:'Network RX',unit:'MiB/s',counter:'host:net_rx_mb_s',note:'physical interfaces'},
 {key:'net_tx',label:'Network TX',unit:'MiB/s',counter:'host:net_tx_mb_s',note:'physical interfaces'},
];
export const HOST_DEFAULT=['cpu','memory','cpu_psi','memory_psi','io_psi'];

export function formatValue(value,unit){
 if(!Number.isFinite(value))return '—';
 if(unit==='MiB')return value>=1024?`${(value/1024).toFixed(value>=10240?1:2)} GiB`:`${Math.round(value)} MiB`;
 if(unit==='MiB/s')return value>=1024?`${(value/1024).toFixed(2)} GiB/s`:`${value.toFixed(value<10?2:1)} MiB/s`;
 if(unit==='s')return `${value.toFixed(value<10?2:1)} s`;
 const pct=`${value<10?value.toFixed(1):Math.round(value)}%`;
 if(unit==='core%')return `${pct} (${(value/100).toFixed(2)} cores)`;
 return pct;
}
export const shortValue=(value,unit)=>unit==='core%'?formatValue(value,'%'):formatValue(value,unit);

const usable=(p,end)=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1])&&p[0]>=0&&(!Number.isFinite(end)||p[0]<=end);
export function sampleInterval(points){
 const gaps=[];for(let i=1;i<points.length;i++){const d=points[i][0]-points[i-1][0];if(d>0)gaps.push(d);}
 if(!gaps.length)return null;gaps.sort((a,b)=>a-b);return gaps[Math.floor(gaps.length/2)];
}
function series(points,meta){
 points=points.slice().sort((a,b)=>a[0]-b[0]);
 const interval=sampleInterval(points);
 // A missing sample is a gap, not a straight line: break after 3 typical intervals.
 return {...meta,points,interval,gap:Math.max(3*(interval||.1),.25),tolerance:Math.max(1.5*(interval||.1),.05),max:points.reduce((m,p)=>Math.max(m,p[1]),0)};
}
// Concurrent sandboxes sampled on the same sampler tick share timestamps;
// combine only exact ticks, so sequential sandboxes simply concatenate.
function combine(lists,mode){
 if(lists.length===1)return lists[0];
 const at=new Map();
 for(const points of lists)for(const [t,v] of points)at.set(t,at.has(t)?(mode==='max'?Math.max(at.get(t),v):at.get(t)+v):v);
 return [...at].sort((a,b)=>a[0]-b[0]);
}
function integrate(points,gap){
 let total=0;const out=[];
 for(let i=0;i<points.length;i++){
  if(i){const dt=points[i][0]-points[i-1][0];if(dt>0&&dt<=gap)total+=(points[i][1]+points[i-1][1])/2/100*dt;}
  out.push([points[i][0],total]);
 }
 return out;
}

// Ownership: an agent owns a sandbox when the engine/profiler recorded it and
// no other recorded agent uses it. Shared sandboxes stay unattributed.
export function sandboxOwnership(run){
 const owners=new Map();
 const remember=(sandbox,id)=>{if(typeof sandbox!=='string'||!sandbox||typeof id!=='string'||!id)return;const list=owners.get(sandbox)||[];if(!list.includes(id))list.push(id);owners.set(sandbox,list);};
 for(const a of run.agents||[])for(const sandbox of new Set([a.sandbox,...(Array.isArray(a.sandboxes)?a.sandboxes:[])]))remember(sandbox,a.id);
 for(const [sandbox,agents] of Object.entries(run.sandbox_owners||{}))if(Array.isArray(agents))for(const id of agents)remember(sandbox,id);
 return owners;
}

const modelCache=new WeakMap();
export function resourceModel(run){
 const counters=run.counters||{},end=run.duration;
 const signature=JSON.stringify([end,(run.agents||[]).map(a=>[a.id,a.sandbox,a.sandboxes]),Object.entries(counters).map(([k,v])=>[k,Array.isArray(v)?v.length:0,Array.isArray(v)?v.at(-1):null])]);
 const cached=modelCache.get(run);if(cached?.signature===signature&&cached.counters===counters)return cached.model;
 const clock=run.clock||null,aligned=!clock||clock.shared!==false;
 const owners=sandboxOwnership(run),measured=aligned?systemMeasurements(run):[];
 const agents=new Map();
 for(const agent of run.agents||[]) {
  const linked=[...owners].filter(([,ids])=>ids.includes(agent.id));
  const own=linked.filter(([,ids])=>ids.length===1).map(([s])=>s),shared=linked.filter(([,ids])=>ids.length>1);
  const entry={id:agent.id,sandboxes:own,shared:shared.map(([s,ids])=>({sandbox:s,agents:ids})),series:{},missing:[],status:'available',reason:''};
  if(!aligned){entry.status='unavailable';entry.reason=`Resource samples use a different clock (${clock.resources}) than events (${clock.events}); they are not aligned.`;}
  else if(!own.length&&shared.length){entry.status='shared';entry.reason=`Sandbox ${shared.map(s=>s.sandbox).join(', ')} is shared with ${[...new Set(shared.flatMap(s=>s.agents))].filter(id=>id!==agent.id).join(', ')}; its cgroup is not attributed to one agent.`;}
  else if(!own.length){entry.status='unavailable';entry.reason=agent.id==='workflow'?'Host workflow: no agent sandbox. Host-side work is reflected only in Host Resources.':'No sandbox cgroup was recorded for this agent (non-container sandbox, or profiling disabled).';}
  else {
   for(const metric of AGENT_METRICS) {
    if(metric.derived)continue;
    const lists=[];
    for(const sandbox of own) {
     if(metric.counter){const pts=(counters[`sandbox:${sandbox}:${metric.counter}`]||[]).filter(p=>usable(p,end));if(pts.length)lists.push(pts);continue;}
     // sandbox:* is the saved series and cg:* the live one: aliases of the same cgroup.
     const found=['sandbox','cg'].map(prefix=>measured.find(m=>m.label===metric.measure&&m.name.startsWith(`${prefix}:${sandbox}:`)&&m.points.length)).find(Boolean);
     if(found)lists.push(found.points);
    }
    if(lists.length)entry.series[metric.key]=series(combine(lists,metric.combine),{...metric,scope:own.length>1?`${own.length} sandboxes (${metric.combine})`:`sandbox ${own[0]}`});
    else entry.missing.push(metric.key);
   }
   if(entry.series.cpu)entry.series.cpu_time=series(integrate(entry.series.cpu.points,entry.series.cpu.gap),{...AGENT_METRICS.find(m=>m.key==='cpu_time'),scope:entry.series.cpu.scope});
   else entry.missing.push('cpu_time');
   if(!Object.keys(entry.series).length){entry.status='unavailable';entry.reason=`Sandbox ${own.join(', ')} was recorded, but no cgroup samples exist for it.`;}
  }
  agents.set(agent.id,entry);
 }
 const host={series:{},missing:[]};
 for(const metric of HOST_METRICS) {
  const pts=aligned?(counters[metric.counter]||[]).filter(p=>usable(p,end)):[];
  if(!pts.length){host.missing.push(metric.key);continue;}
  const limit=metric.limit?(counters[metric.limit]||[]).filter(p=>usable(p,end)).at(-1)?.[1]:null;
  host.series[metric.key]=series(pts,{...metric,scope:'host',limit:Number.isFinite(limit)?limit:null});
 }
 const cpus=(counters['host:cpu_count']||[]).filter(p=>usable(p,end)).at(-1)?.[1];
 host.cpus=Number.isFinite(cpus)?cpus:null;
 host.status=!aligned?'unavailable':Object.keys(host.series).length?'available':'unavailable';
 host.reason=!aligned?'Host samples are on a different clock than events.':host.status==='unavailable'?'Host-wide CPU, memory and pressure were not recorded in this trace (recorded by agprof host sampling).':'';
 const model={agents,host,clock,aligned};
 modelCache.set(run,{signature,counters,model});
 return model;
}

// The recorded sample nearest t, only if it lies within the series' own
// sampling tolerance. Never interpolated.
export function sampleAt(s,t){
 const pts=s?.points;if(!pts?.length||!Number.isFinite(t))return null;
 let lo=0,hi=pts.length;while(lo<hi){const mid=(lo+hi)>>1;if(pts[mid][0]<t)lo=mid+1;else hi=mid;}
 const candidates=[pts[lo-1],pts[lo]].filter(Boolean).sort((a,b)=>Math.abs(a[0]-t)-Math.abs(b[0]-t));
 return candidates[0]&&Math.abs(candidates[0][0]-t)<=s.tolerance?candidates[0]:null;
}
export function latestSample(s,clock){
 const pts=s?.points;if(!pts?.length)return null;
 for(let i=pts.length-1;i>=0;i--)if(!Number.isFinite(clock)||pts[i][0]<=clock+1e-9)return pts[i];
 return null;
}

// Path data for the visible window: at most min+max per pixel column so short
// peaks survive, and a new subpath after every gap so nothing is interpolated.
export function chartPaths(s,{scale,pad,left,right,top,height,max}){
 const pts=s?.points||[];if(!pts.length||!(max>0))return {line:'',area:'',count:0};
 const from=(left-pad)/scale,to=(right-pad)/scale;
 let lo=0,hi=pts.length;while(lo<hi){const mid=(lo+hi)>>1;if(pts[mid][0]<from)lo=mid+1;else hi=mid;}
 const start=Math.max(0,lo-1),y=v=>top+height-Math.max(0,Math.min(1,v/max))*height,x=t=>pad+t*scale;
 const runs=[];let current=null,previous=null;
 for(let i=start;i<pts.length;i++){
  const p=pts[i];if(previous&&p[0]-previous[0]>s.gap)current=null;
  if(!current){current=[];runs.push(current);}
  const column=Math.floor(x(p[0])),last=current.at(-1);
  if(last&&last.column===column){last.min=Math.min(last.min,p[1]);last.max=Math.max(last.max,p[1]);last.end=p[1];}
  else current.push({column,t:p[0],min:p[1],max:p[1],start:p[1],end:p[1]});
  previous=p;if(p[0]>to)break;
 }
 let line='',area='',count=0;
 for(const run of runs){
  const coords=[];
  for(const c of run){const px=x(c.t);if(c.min===c.max)coords.push([px,y(c.start)]);else coords.push([px,y(c.start)],[px,y(c.max)],[px,y(c.min)],[px,y(c.end)]);count++;}
  if(coords.length===1)coords.push([coords[0][0]+1.5,coords[0][1]]);
  line+='M'+coords.map(([a,b])=>`${a.toFixed(1)} ${b.toFixed(1)}`).join('L');
  area+=`M${coords[0][0].toFixed(1)} ${top+height}`+coords.map(([a,b])=>`L${a.toFixed(1)} ${b.toFixed(1)}`).join('')+`L${coords.at(-1)[0].toFixed(1)} ${top+height}Z`;
 }
 return {line,area,count};
}
export function niceMax(value,floor=0){
 const v=Math.max(value||0,floor);if(!(v>0))return 1;
 const p=10**Math.floor(Math.log10(v)),m=v/p;return [1,1.25,1.5,2,2.5,3,4,5,6,8,10].find(n=>n>=m-1e-9)*p;
}
// Agent CPU, memory and throughput share one scale across lanes so agents
// can be compared; host CPU and PSI are percentages with fixed meaning.
export function chartScale(metric,values,{limit=null}={}){
 if(metric.unit==='host%')return 100;
 if(metric.unit==='psi%')return niceMax(Math.max(...values,0),10);
 if(metric.unit==='core%')return niceMax(Math.max(...values,0),100);
 if(limit)return limit;
 return niceMax(Math.max(...values,0));
}
