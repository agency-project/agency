// Pure calculations over the existing projection, never a second subscription.
export const defaults=['wall','agents','tools','models','tokens','errors'];
export const categories=['Activity','LLM','Tools','Systems'];
export const isTool=a=>!['model','queue','dependency','unknown','setup'].includes(a.kind);
export const completed=a=>['success','failed','timeout','cancelled'].includes(a.outcome);
export function percentile(values,p){const s=values.filter(Number.isFinite).sort((a,b)=>a-b);return s.length?s[Math.max(0,Math.ceil(s.length*p)-1)]:null;}
export function peak(calls,end){const events=calls.filter(a=>!a.missing_start&&(a.outcome==='running'?end>a.start:a.duration>0)).flatMap(a=>[[a.start,1],[a.outcome==='running'?end:a.start+a.duration,-1]]).filter(([t])=>Number.isFinite(t)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);let n=0,max=0;for(const [,d] of events){n+=d;max=Math.max(max,n);}return max;}
export function union(calls,end){let cursor=0,total=0;for(const a of [...calls].filter(a=>!a.missing_start).sort((a,b)=>a.start-b.start)){const right=Math.min(end,a.outcome==='running'?end:a.start+a.duration);total+=Math.max(0,right-Math.max(cursor,a.start));cursor=Math.max(cursor,right);}return total;}
export function projectMetrics(run){
 const actions=[...new Map((run.actions||[]).map(a=>[a.id,a])).values()],end=run.duration||0,tools=actions.filter(isTool),models=actions.filter(a=>a.kind==='model');
 const done=xs=>xs.filter(a=>completed(a)&&!a.missing_start&&Number.isFinite(a.duration)&&!String(a.source).includes('timestamp_missing'));
 const dt=done(tools),dm=done(models),window=Math.min(60,end),recent=xs=>xs.filter(a=>!a.missing_start&&(end<=60?a.start>=0:a.start>end-60)&&a.start<=end).length;
 const usage=key=>{const xs=models.filter(a=>Number.isFinite(a[key]));return xs.length?xs.reduce((n,a)=>n+a[key],0):null;};
 const totals=run.metrics_totals||{tools:tools.length,models:models.length,completed_tools:tools.filter(completed).length,tool_errors:tools.filter(a=>a.outcome==='failed').length,input:usage('tokens'),output:usage('output_tokens'),input_samples:models.filter(a=>Number.isFinite(a.tokens)).length,output_samples:models.filter(a=>Number.isFinite(a.output_tokens)).length};
 const agents=(run.agents||[]).filter(a=>a.id!=='workflow'),states={};for(const a of agents)states[a.status||'unknown']=(states[a.status||'unknown']||0)+1;
 const perAgent=agents.map(agent=>{const calls=actions.filter(a=>a.agent===agent.id),finished=calls.filter(a=>isTool(a)&&completed(a)).sort((a,b)=>(b.start+b.duration)-(a.start+a.duration));let streak=0;for(const a of finished){if(a.outcome!=='failed')break;streak++;}const last=calls.reduce((n,a)=>Math.max(n,a.start+(completed(a)?a.duration:0)),null);return {...agent,calls,streak,gap:last==null?null:Math.max(0,end-last)};});
 return {actions,tools,models,dt,dm,totals,agents,states,perAgent,end,window,toolRate:window?recent(tools)/window:null,modelRate:window?recent(models)*60/window:null,oldest:actions.filter(a=>(isTool(a)||a.kind==='model')&&a.outcome==='running'&&!a.missing_start).sort((a,b)=>a.start-b.start)[0]};
}
export function readPreferences(storage){try{const v=JSON.parse(storage.getItem('agency.metrics.v1'));return {pins:Array.isArray(v?.pins)?[...new Set(v.pins.filter(x=>typeof x==='string'))].slice(0,6):defaults.slice(),category:categories.includes(v?.category)?v.category:'Activity',expanded:!!v?.expanded};}catch{return {pins:defaults.slice(),category:'Activity',expanded:false};}}
