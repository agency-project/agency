// Pure calculations over the existing projection, never a second subscription.
export const defaults=['wall','agents','tools','models','tokens','errors'];
export const categories=['Activity','LLM','Tools','Systems'];
export const isTool=a=>!['model','queue','dependency','unknown','setup','syscall'].includes(a.kind);
export const completed=a=>['success','failed','timeout','cancelled'].includes(a.outcome);
export function percentile(values,p){const s=values.filter(Number.isFinite).sort((a,b)=>a-b);return s.length?s[Math.max(0,Math.ceil(s.length*p)-1)]:null;}
export function peak(calls,end){const events=calls.filter(a=>!a.missing_start&&(a.outcome==='running'?end>a.start:a.duration>0)).flatMap(a=>[[a.start,1],[a.outcome==='running'?end:a.start+a.duration,-1]]).filter(([t])=>Number.isFinite(t)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);let n=0,max=0;for(const [,d] of events){n+=d;max=Math.max(max,n);}return max;}
export function union(calls,end){let cursor=0,total=0;for(const a of [...calls].filter(a=>!a.missing_start).sort((a,b)=>a.start-b.start)){const right=Math.min(end,a.outcome==='running'?end:a.start+a.duration);total+=Math.max(0,right-Math.max(cursor,a.start));cursor=Math.max(cursor,right);}return total;}
// Agents a metrics scope can name. The workflow actor is shared infrastructure.
export const scopeAgents=run=>(run.agents||[]).filter(a=>a.id!=='workflow');
// A scope is null (all agents: execution-wide totals, new agents included) or
// a Set of agent ids. The set may be empty: no agents selected is a valid
// scope with nothing to measure. Ids no longer recorded are ignored.
export function effectiveScope(scope,run){
 return scope?new Set(scopeAgents(run).filter(a=>scope.has(a.id)).map(a=>a.id)):null;
}
export function projectMetrics(run,scope=null){
 scope=effectiveScope(scope,run);
 const recorded=[...new Map((run.actions||[]).map(a=>[a.id,a])).values()],actions=scope?recorded.filter(a=>scope.has(a.agent)):recorded,end=run.duration||0,tools=actions.filter(isTool),models=actions.filter(a=>a.kind==='model');
 const done=xs=>xs.filter(a=>completed(a)&&!a.missing_start&&Number.isFinite(a.duration)&&!String(a.source).includes('timestamp_missing'));
 const dt=done(tools),dm=done(models),window=Math.min(60,end),recent=xs=>xs.filter(a=>!a.missing_start&&(end<=60?a.start>=0:a.start>end-60)&&a.start<=end).length;
 const usage=key=>{const xs=models.filter(a=>Number.isFinite(a[key]));return xs.length?xs.reduce((n,a)=>n+a[key],0):null;};
 const retained={tools:tools.length,models:models.length,completed_tools:tools.filter(completed).length,tool_errors:tools.filter(a=>a.outcome==='failed').length,input:usage('tokens'),output:usage('output_tokens'),input_samples:models.filter(a=>Number.isFinite(a.tokens)).length,output_samples:models.filter(a=>Number.isFinite(a.output_tokens)).length};
 // Additive counters are summed per selected agent; latency and rates below
 // are recomputed from the selected agents' own observations.
 const totals=!scope?run.metrics_totals||retained:run.metrics_by_agent?Object.fromEntries(Object.keys(retained).map(key=>[key,[...scope].reduce((n,id)=>n+(run.metrics_by_agent[id]?.[key]||0),0)])):retained;
 const roster=scopeAgents(run),agents=scope?roster.filter(a=>scope.has(a.id)):roster,states={};for(const a of agents)states[a.status||'unknown']=(states[a.status||'unknown']||0)+1;
 const perAgent=agents.map(agent=>{const calls=actions.filter(a=>a.agent===agent.id),finished=calls.filter(a=>isTool(a)&&completed(a)).sort((a,b)=>(b.start+b.duration)-(a.start+a.duration));let streak=0;for(const a of finished){if(a.outcome!=='failed')break;streak++;}const last=calls.reduce((n,a)=>Math.max(n,a.start+(completed(a)?a.duration:0)),null);return {...agent,calls,streak,gap:last==null?null:Math.max(0,end-last)};});
 return {actions,tools,models,dt,dm,totals,agents,roster,scoped:Boolean(scope),states,perAgent,end,window,toolRate:window?recent(tools)/window:null,modelRate:window?recent(models)*60/window:null,oldest:actions.filter(a=>(isTool(a)||a.kind==='model')&&a.outcome==='running'&&!a.missing_start).sort((a,b)=>a.start-b.start)[0]};
}
// The backend owns the wall-clock boundary and duration; this only words them
// and decides which reversible measurement actions fit the current state.
export function measurementView(run){
 const m=run?.measurement;if(!m||!Number.isFinite(run.execution_wall_time))return null;
 const started=run.execution_started_at!=null,manual=m.policy==='manual',ended=m.state==='ended';
 const note=!started?'Starts at the first observed agent execution; startup and setup are excluded'
  :m.held?'Kept running by you · counts continuously from the first agent execution until you end it or reset to automatic'
  :m.end_source==='hold_expired'?'Kept running by you until the dashboard server stopped'
  :manual?`Ended by you${m.automatic_end_at==null?'':' · the automatic boundary is kept and can be restored'}`
  :!ended?'First agent execution to workload completion; includes pauses, excludes startup/setup'
  :m.automatic_end_source==='workload_completed'?'Ended automatically when the workload returned; profiler finalization excluded'
  :m.automatic_end_source==='replay_boundary'?'Ended at the end of the saved trace; no lifecycle boundary was recorded'
  :'Ended automatically at the final workload event';
 return {state:!started?'not started':ended?'ended':'running',policy:manual?'manual':'automatic',held:Boolean(m.held),note,
  actions:{end:started&&!ended,hold:!m.held,reset:manual,endAt:started}};
}
export function readPreferences(storage){try{const v=JSON.parse(storage.getItem('agency.metrics.v1'));return {pins:Array.isArray(v?.pins)?[...new Set(v.pins.filter(x=>typeof x==='string'))].slice(0,6):defaults.slice(),category:categories.includes(v?.category)?v.category:'Activity',expanded:!!v?.expanded};}catch{return {pins:defaults.slice(),category:'Activity',expanded:false};}}
