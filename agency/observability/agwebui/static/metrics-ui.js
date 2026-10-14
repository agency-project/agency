import {defaults,categories,projectMetrics,percentile,peak,union,readPreferences,measurementView,scopeAgents,effectiveScope} from './metrics-model.js';
import {systemMeasurements,agentResources,systemActivity} from './investigator-model.js';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v==null?'Unavailable':new Intl.NumberFormat('en',{maximumFractionDigits:1,notation:v>=10000?'compact':'standard'}).format(v);
const latency=v=>v==null?'Unavailable':`${new Intl.NumberFormat('en',{maximumFractionDigits:2}).format(v*1000)} ms / ${new Intl.NumberFormat('en',{maximumFractionDigits:3}).format(v)} s`;
const sec=v=>v==null?'Unavailable':`${num(v)}s`;
const rate=(n,d)=>d?`${n} · ${num(100*n/d)}%`:`${n} · no completed calls`;
let prefs=readPreferences({getItem:key=>localStorage.getItem(key)}),filter='',sort='start',direction=-1,page=0,pinPicker=null;
// Wall-clock boundary controls exist only for a connected live execution.
let measure={enabled:false,message:''};
// Agents the summary covers: null is all agents, otherwise the selected ids,
// which may be none. Per execution, so never persisted with the pin preferences.
let scope=null,roster=[];
export const metricScope=()=>scope;
export function setMetricScope(ids){scope=ids==null?null:new Set(ids);}
const scopeEmpty=run=>effectiveScope(scope,run)?.size===0;
// Shown in place of the cards: an empty scope has nothing to measure, and the way back is one click.
const emptyScope='<div class="prof-scope-empty" role="status"><strong>No agents selected</strong><span>Summary metrics are hidden. Choose agents in the Metrics scope menu, or restore every agent.</span><button type="button" data-scope-all>Select all agents</button></div>';
const emptyCards='<p class="prof-scope-empty-note">No agents selected · choose agents in the Metrics scope menu.</p>';
const scopeLabel=(selected,agents)=>!selected?`All agents (${agents.length})`:!selected.size?'No agents selected':selected.size===1?agents.find(a=>selected.has(a.id))?.label||[...selected][0]:`${selected.size} of ${agents.length} agents`;
const metricLabels=new Map();
const save=()=>{try{localStorage.setItem('agency.metrics.v1',JSON.stringify(prefs));}catch{/* Private storage may be disabled. */}};
function descriptors(m,run,includeSystems=false,latestSystems=false){const wall=measurementView(run),t=m.totals,span=m.scoped?'selected agents':'execution total',whole=m.scoped?' · entire execution':'',streak=[...m.perAgent].sort((a,b)=>b.streak-a.streak)[0],gap=[...m.perAgent].filter(a=>a.gap!=null).sort((a,b)=>b.gap-a.gap)[0];
 const rows={wall:wall?[`Wall clock · ${wall.policy}${whole}`,`${sec(run.execution_wall_time)} · ${wall.state}`,wall.note,'activity',undefined,'Wall clock',m.scoped]:['Wall clock'+whole,Number.isFinite(run.execution_wall_time)?`${sec(run.execution_wall_time)} · ${run.execution_started_at==null?'not started':run.status||'unknown'}`:run.time_origin==null&&!run.actions?.length?'Unavailable':`${sec(run.duration)} · ${run.status||'unknown'}`,'First agent execution to final workload boundary; includes pauses, excludes startup/setup','activity',undefined,'Wall clock',m.scoped],agents:['Agents',m.scoped?`${num(m.agents.length)} of ${num(m.roster.length)}`:num(m.agents.length),Object.entries(m.states).map(([k,v])=>`${k}: ${v}`).join(' · '),'activity'],tools:['Tool calls',num(t.tools),`Started + completed · ${span}`,'tools'],models:['LLM requests',num(t.models),`Started + completed · ${span}`,'llm'],tokens:['Input / output',`${num(t.input_samples?t.input:null)} / ${num(t.output_samples?t.output:null)}`,`Reported usage · ${span} · ${t.input_samples}/${t.models} input; ${t.output_samples}/${t.models} output`,'llm'],errors:['Tool errors',rate(t.tool_errors,t.completed_tools),'Recorded failures (including nonzero exits); not infrastructure-only','tools'],oldest:['Oldest active',m.oldest?sec(m.end-m.oldest.start):'None',m.oldest?.agent||'No outstanding observed call','activity',m.oldest?.id],streak:['Failure streak',num(streak?.streak??0),streak?.streak?streak.label:'No consecutive recorded tool failures','activity'],toolRate:['Tool calls/sec',num(m.toolRate),`Starts · trailing ${num(m.window)}s`,'activity'],modelRate:['LLM requests/min',num(m.modelRate),`Starts · trailing ${num(m.window)}s`,'activity'],gap:['Last call activity',sec(gap?.gap),`${gap?.label||'No observed calls'} · does not imply stuck`,'activity'],llm50:['E2E P50',latency(percentile(m.dm.map(a=>a.duration),.5)),`${m.dm.length} completed timed requests`,'llm'],llm95:['E2E P95',latency(percentile(m.dm.map(a=>a.duration),.95)),`${m.dm.length} samples${m.dm.length<20?' · sparse tail estimate':''}`,'llm'],ttft50:['TTFT P50','Unavailable','First-token boundary is not projected','llm'],ttft95:['TTFT P95','Unavailable','First-chunk timing is not substituted','llm'],outputRate:['Output tokens/sec',num((()=>{const a=m.dm.filter(a=>Number.isFinite(a.output_tokens)&&a.duration>0);return a.length?a.reduce((n,a)=>n+a.output_tokens,0)/a.reduce((n,a)=>n+a.duration,0):null;})()),'Client estimate: reported output / summed E2E intervals; not decode speed','llm'],llmErrors:['LLM errors',rate(m.models.filter(a=>a.outcome==='failed').length,m.models.filter(a=>['success','failed','cancelled','timeout'].includes(a.outcome)).length),'Recorded request failures','llm'],tool50:['Tool P50',latency(percentile(m.dt.map(a=>a.duration),.5)),`${m.dt.length} completed timed calls`,'tools'],tool95:['Tool P95',latency(percentile(m.dt.map(a=>a.duration),.95)),`${m.dt.length} samples${m.dt.length<20?' · sparse tail estimate':''}`,'tools'],timeouts:['Timeouts','Unavailable','No dedicated timeout classification in current projection','tools'],slowest:['Slowest completed',latency(m.dt.length?Math.max(...m.dt.map(a=>a.duration)):null),'Completed start/end boundaries','tools'],active:['Active tools',num(m.tools.filter(a=>a.outcome==='running').length),'Outstanding observed tool calls','tools']};
 if(includeSystems){
  // Recorded series keep their own scope: Host and Workload cover the entire system, never the agent selection.
  for(const metric of systemMeasurements(run,{latestOnly:latestSystems})){rows['system:'+metric.name]=[`${metric.label} · ${metric.scope}`,`${num(metric.latest[1])} ${metric.unit==='MB'?'MiB':metric.unit==='MB/s'?'MiB/s':metric.unit}`,`${m.scoped?'Not filtered by the agent selection · ':''}${metric.scope} · ${metric.name} · at ${sec(metric.latest[0])}${metric.statusLabel?` · ${metric.statusLabel}`:''}${metric.label==='CPU'?' · 100% = one core':''}`,'systems',undefined,undefined,m.scoped];}
  // Agent-level resources are the selected agents' own sandbox containers and nothing else.
  const resources=agentResources(run,m.agents.map(a=>a.id)),who=m.scoped?'selected agents':'all agents';
  if(m.scoped||resources.sandboxes||resources.shared)for(const [key,label,measured,unit,extra] of [['cpu','CPU',resources.cpu,'%',' · 100% = one core'],['mem','RAM',resources.memory,'MiB','']]){
   const missing=!resources.sandboxes?(resources.shared?'Sandbox is shared with agents outside the selection':'No sandbox recorded for these agents'):'No current container sample';
   rows['scope:'+key]=[`${label} · agent sandboxes`,measured?`${num(measured.value)} ${unit}`:'Unavailable',`${measured?`Sum of ${measured.sandboxes} of ${resources.sandboxes} sandbox containers of ${who}`:missing}${extra}${resources.shared&&resources.sandboxes?` · ${resources.shared} shared sandbox${resources.shared===1?'':'es'} excluded`:''} · excludes orchestrator, host services, runtime daemons and model inference`,'systems'];
  }
 }
 return rows;
}
function summarySystemKeys(run,rows){
 const metrics=systemMeasurements(run,{latestOnly:true}),chosen=['scope:cpu','scope:mem'].filter(key=>rows[key]);
 // Prefer aggregate resource telemetry, then one explicitly named process.
 // Multiple RSS series are independent processes, never interchangeable RAM.
 const rss=metrics.find(m=>m.label==='RSS');
 for(const label of ['CPU','RAM','RSS','Disk read','Disk write','GPU','VRAM','Process VRAM']){
  const candidates=metrics.filter(m=>m.label===label);
  const metric=label==='RSS'?rss:candidates.find(m=>m.aggregate)||candidates.find(m=>rss&&m.scope===rss.scope)||candidates[0];
  if(metric)chosen.push('system:'+metric.name);
 }
 return chosen.slice(0,6);
}
const groups={Activity:['oldest','streak','toolRate','modelRate','agents','gap'],LLM:['llm50','llm95','ttft50','ttft95','outputRate','llmErrors'],Tools:['tool50','tool95','errors','timeouts','slowest','active']};
function metricRow(key,rows){const r=rows[key]||[key,'Unavailable','No samples in this execution','systems'];metricLabels.set(key,r[5]||r[0]);return r;}
// Stopping the measurement is not stopping the agents; the menu says so and needs no confirmation because every choice is reversible.
const measureMenu=`<details class="measure-menu" hidden><summary aria-label="Wall clock measurement options" title="Wall clock measurement options">⋯</summary><div><p class="measure-status"></p><button type="button" data-measure="end">End now</button><button type="button" data-measure="hold">Keep running</button><button type="button" data-measure="reset">Reset to automatic</button><label class="measure-end-at">End at <input type="number" min="0" step="0.1" inputmode="decimal" aria-label="Wall clock end in seconds since the measurement started"> s <button type="button" data-measure="end-at">Set</button></label><p class="measure-hint">Changes the measurement only. Agents and the workload keep running.</p><p class="measure-message" role="status"></p></div></details>`;
function syncMeasure(container,run){
 for(const element of container.querySelectorAll('[data-metric-card]')){
  if(element.dataset.metricCard!=='wall')continue;
  const menu=element.querySelector('.measure-menu'),view=measurementView(run);if(!menu)continue;
  menu.hidden=!(view&&measure.enabled);if(menu.hidden){menu.open=false;continue;}
  menu.querySelector('.measure-status').textContent=view.note;
  for(const [action,selector] of [['end','[data-measure="end"]'],['hold','[data-measure="hold"]'],['reset','[data-measure="reset"]'],['endAt','.measure-end-at']])menu.querySelector(selector).hidden=!view.actions[action];
  menu.querySelector('input').placeholder=run.execution_wall_time.toFixed(1);
  menu.querySelector('.measure-message').textContent=measure.message;
 }
}
function card(key,rows,pin=true){const r=metricRow(key,rows);return `<div class="prof-metric ${key==='agents'?'prof-agent-summary':''}" data-metric-card="${esc(key)}" data-unscoped="${Boolean(r[6])}"><button class="prof-value" ${r[4]?`data-metric-event="${esc(r[4])}"`:`data-metric-view="${r[3]}" data-metric-key="${esc(key)}"`}><small>${esc(r[0])}</small><strong>${esc(r[1])}</strong></button>${pin?`<button type="button" class="prof-pin" data-metric-pin="${esc(key)}" aria-label="${prefs.pins.includes(key)?'Unpin':'Pin'} ${esc(r[0])}" aria-pressed="${prefs.pins.includes(key)}">${prefs.pins.includes(key)?'◆':'◇'}</button>`:''}${key==='wall'?measureMenu:''}<span class="prof-note">${esc(r[2])}</span></div>`;}
function chart(title,points,unit,indexed=false,description='') {
 const data=points.filter(p=>p.every(Number.isFinite)).sort((a,b)=>a[0]-b[0]);if(!data.length)return '';
 const min=data.length===1&&!indexed?0:data[0][0],max=Math.max(min+1,data.at(-1)[0]),low=Math.min(0,...data.map(p=>p[1])),high=Math.max(1,...data.map(p=>p[1]));
 const left=100,right=610,top=20,bottom=155,x=v=>left+(v-min)/(max-min||1)*(right-left),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
 const isLatency=unit==='s',tick=v=>isLatency?latency(v):num(v),yLabel=isLatency?'Latency (ms / s)':unit;
 const xLabel=indexed?'Sorted observation index':'Time since recording start (s)';
 const step=Math.max(1,Math.ceil(data.length/300)),shown=data.filter((_,i)=>i%step===0||i===data.length-1);
 const ticks=[0,.25,.5,.75,1];
 const axes=ticks.map(f=>{const v=low+f*(high-low),at=y(v),label=isLatency?tick(v).split(' / '):[tick(v)];return `<line x1="${left}" x2="${right}" y1="${at}" y2="${at}" class="prof-grid"/><text x="${left-8}" y="${at+4}" text-anchor="end">${label.map((part,i)=>`<tspan x="${left-8}" dy="${i?10:0}">${esc(part)}</tspan>`).join('')}</text>`;}).join('')+ticks.map(f=>{const v=indexed?Math.round(min+f*(max-min)):min+f*(max-min),at=x(v);return `<line x1="${at}" x2="${at}" y1="${bottom}" y2="${bottom+5}" class="prof-axis"/><text x="${at}" y="${bottom+20}" text-anchor="middle">${num(v)}</text>`;}).join('');
 const explanation=description||(isLatency?(indexed?'Each point is one completed call sorted from fastest to slowest. A steep tail reveals unusually slow calls.':'Each point is a completed call at its completion time. Spikes show when request latency increased.'):`Values over recording time; rising values show increased ${unit}. Gaps reflect available observations.`);
 return `<figure class="prof-chart"><figcaption>${esc(title)} · ${data.length} observations</figcaption><svg viewBox="0 0 640 220" role="img" aria-label="${esc(title+'; X: '+xLabel+'; Y: '+yLabel+'; '+explanation)}">${axes}<path d="M ${left} ${top} V ${bottom} H ${right}" class="prof-axis" fill="none"/><polyline fill="none" stroke="currentColor" stroke-width="1.5" points="${shown.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}"/>${shown.length===1?`<circle cx="${x(shown[0][0])}" cy="${y(shown[0][1])}" r="3" fill="currentColor"/>`:''}<text x="${(left+right)/2}" y="205" text-anchor="middle">${esc(xLabel)}</text><text transform="translate(14 90) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text></svg><p class="prof-chart-description">${esc(explanation)}</p></figure>`;
}
function table(calls){const matches=calls.filter(a=>!filter||`${a.agent} ${a.name} ${a.outcome} ${a.error}`.toLowerCase().includes(filter.toLowerCase())).sort((a,b)=>direction*(typeof a[sort]==='number'?a[sort]-b[sort]:String(a[sort]||'').localeCompare(String(b[sort]||''))));page=Math.min(page,Math.max(0,Math.ceil(matches.length/100)-1));return `<div class="prof-table-controls"><label>Filter calls <input id="prof-filter" value="${esc(filter)}" placeholder="Agent, tool, status, error"></label><span>${matches.length} calls · page ${page+1}</span><button data-metric-page="-1" ${page===0?'disabled':''}>Previous</button><button data-metric-page="1" ${(page+1)*100>=matches.length?'disabled':''}>Next</button></div><table class="prof-table"><thead><tr>${[['agent','Agent'],['name','Operation'],['outcome','Status'],['start','Start'],['duration','Duration'],['error','Error']].map(([key,label])=>`<th><button data-metric-sort="${key}">${label}${sort===key?(direction===1?' ↑':' ↓'):''}</button></th>`).join('')}</tr></thead><tbody>${matches.slice(page*100,(page+1)*100).map(a=>`<tr><td>${esc(a.agent)}</td><td><button data-metric-event="${esc(a.id)}">${esc(a.name)} ↗</button></td><td>${esc(a.outcome)}</td><td>${a.missing_start?'Unknown':sec(a.start)}</td><td>${a.outcome==='incomplete'?'End missing':latency(a.duration)}</td><td>${esc(String(a.error||'').slice(0,180))}</td></tr>`).join('')}</tbody></table>`;}
function summaryData(run){
 const includeSystems=(prefs.expanded&&prefs.category==='Systems')||prefs.pins.some(k=>k.startsWith('system:')||k.startsWith('scope:'));
 const rows=descriptors(projectMetrics(run,scope),run,includeSystems,true);
 const keys=prefs.expanded?(prefs.category==='Systems'?summarySystemKeys(run,rows):groups[prefs.category]):[];
 return {rows,keys,slots:prefs.pins.map(k=>card(k,rows,false)).join(''),cards:keys.map(k=>card(k,rows)).join('')||'<p>No scoped resource samples recorded. Enable AGENCY_PROFILE=1 before launching the workload.</p>'};
}
function updateCards(container,keys,rows,pin,html,run){
 const signature=JSON.stringify(keys);
 if(container.metricKeys!==signature){container.innerHTML=html;container.metricKeys=signature;syncMeasure(container,run);return;}
 for(const element of container.querySelectorAll('[data-metric-card]')){
  const key=element.dataset.metricCard,r=metricRow(key,rows),value=element.querySelector('.prof-value');
  element.dataset.unscoped=String(Boolean(r[6]));
  value.querySelector('small').textContent=r[0];value.querySelector('strong').textContent=r[1];
  for(const attr of ['data-metric-event','data-metric-view','data-metric-key'])value.removeAttribute(attr);
  if(r[4])value.dataset.metricEvent=r[4];else{value.dataset.metricView=r[3];value.dataset.metricKey=key;}
  element.querySelector('.prof-note').textContent=r[2];
  if(pin){const button=element.querySelector('.prof-pin'),pinned=prefs.pins.includes(key);button.textContent=pinned?'◆':'◇';button.setAttribute('aria-pressed',String(pinned));button.setAttribute('aria-label',`${pinned?'Unpin':'Pin'} ${r[0]}`);}
 }
 syncMeasure(container,run);
}
function choosePinSlot(key,origin,render){
 if(pinPicker)return;
 const dialog=document.createElement('dialog');pinPicker=dialog;
 dialog.className='metric-pin-picker';dialog.setAttribute('aria-labelledby','metric-pin-title');
 dialog.innerHTML=`<h2 id="metric-pin-title">Pin ${esc(metricLabels.get(key)||key)}</h2><p>Choose the summary metric to replace.</p><div class="metric-pin-slots">${prefs.pins.map((slot,i)=>`<button type="button" data-pin-slot="${i}">${i+1}. ${esc(metricLabels.get(slot)||slot)}</button>`).join('')}</div><button type="button" data-pin-cancel>Cancel</button>`;
 let resolved=false;
 const finish=slot=>{
  if(resolved)return;resolved=true;pinPicker=null;
  dialog.close();dialog.remove();
  if(slot!=null){prefs.pins[slot]=key;save();render();}
  const target=origin?.isConnected?origin:[...document.querySelectorAll('[data-metric-pin]')].find(button=>button.dataset.metricPin===key);
  target?.focus({preventScroll:true});
 };
 for(const button of dialog.querySelectorAll('[data-pin-slot]'))button.addEventListener('click',()=>finish(Number(button.dataset.pinSlot)));
 dialog.querySelector('[data-pin-cancel]').addEventListener('click',()=>finish(null));
 dialog.addEventListener('cancel',event=>{event.preventDefault();finish(null);});
 document.body.append(dialog);dialog.showModal();dialog.querySelector('[data-pin-slot]').focus({preventScroll:true});
}
function summaryShell(){return `<div class="prof-summary"><div class="prof-summary-slots"></div><div class="prof-summary-actions"><details class="metric-scope"><summary title="Agents included in the summary metrics"><small>Metrics scope</small><span class="metric-scope-label">All agents</span></summary><div class="metric-scope-menu" role="group" aria-label="Agents included in the summary metrics"><label><input type="checkbox" data-scope-all checked> All agents</label><div class="metric-scope-agents"></div><p class="metric-scope-hint">Host and Workload resources always cover the entire system.</p></div></details><button data-metric-toggle aria-controls="prof-expanded" aria-expanded="false">Metrics +</button></div></div><section id="prof-expanded" class="prof-expanded" hidden><div class="prof-tabs" role="tablist" aria-label="Summary metric categories">${categories.map(c=>`<button role="tab" aria-selected="${prefs.category===c}" data-metric-category="${c}">${c}</button>`).join('')}<button data-metric-reset>Reset to defaults</button></div><div class="prof-cards"></div><button class="prof-detail-link" data-metric-view="${prefs.category.toLowerCase()}">View detailed ${prefs.category} profiler →</button><span class="prof-note prof-summary-note"></span></section>`;}
const summaryNote=run=>scopeEmpty(run)?'Scope: no agents selected.':`${effectiveScope(scope,run)?`Scope: ${scopeLabel(effectiveScope(scope,run),scopeAgents(run))} · call metrics cover only these agents; wall clock, Host and Workload resources stay execution-wide. `:''}${run.coverage?.client_window?'Statistics: retained calls; counters: execution-wide.':'Statistics: execution to displayed time.'} Pin ◆ to remove; ◇ to add or choose a slot to replace.`;
export function metricsSummary(run){
 const data=scopeEmpty(run)?{slots:emptyScope,cards:emptyCards}:summaryData(run);
 return `<div class="prof-summary"><div class="prof-summary-slots">${data.slots}</div><button data-metric-toggle aria-expanded="${prefs.expanded}">Metrics ${prefs.expanded?'−':'+'}</button></div>${prefs.expanded?`<section class="prof-expanded"><div class="prof-tabs" role="tablist">${categories.map(c=>`<button role="tab" aria-selected="${prefs.category===c}" data-metric-category="${c}">${c}</button>`).join('')}<button data-metric-reset>Reset to defaults</button></div><div class="prof-cards">${data.cards}</div><button class="prof-detail-link" data-metric-view="${prefs.category.toLowerCase()}">View detailed ${prefs.category} profiler →</button><span class="prof-note prof-summary-note">${summaryNote(run)}</span></section>`:''}`;
}
// The agent list is rebuilt only when the roster changes, so a checkbox is
// never replaced under the pointer by a telemetry update.
function syncScope(root,run){
 const menu=root.querySelector('.metric-scope');if(!menu)return;
 roster=scopeAgents(run);
 const selected=effectiveScope(scope,run),list=menu.querySelector('.metric-scope-agents'),signature=JSON.stringify(roster.map(a=>[a.id,a.label]));
 if(list.rosterSignature!==signature){list.innerHTML=roster.map(a=>`<label title="${esc(a.id)}"><input type="checkbox" data-scope-agent="${esc(a.id)}"> <span>${esc(a.label||a.id)}</span><small></small></label>`).join('')||'<p>No agents recorded yet.</p>';list.rosterSignature=signature;}
 menu.querySelector('.metric-scope-label').textContent=scopeLabel(selected,roster);
 menu.dataset.scoped=String(Boolean(selected));
 menu.querySelector('[data-scope-all]').checked=!selected;
 for(const box of list.querySelectorAll('[data-scope-agent]')){const agent=roster.find(a=>a.id===box.dataset.scopeAgent);box.checked=!selected||selected.has(box.dataset.scopeAgent);box.parentElement.querySelector('small').textContent=agent?.status||'unknown';}
}
// Keep interactive controls mounted across telemetry updates. Replacing a
// pressed button between pointer-down and pointer-up causes lost clicks.
export function renderMetricsSummary(root,run,{preferencesOnly=false,measurement=null}={}){
 measure=measurement||{enabled:false,message:''};
 let toggle=root.querySelector('[data-metric-toggle]');
 if(!toggle){root.innerHTML=summaryShell();toggle=root.querySelector('[data-metric-toggle]');}
 toggle.textContent=`Metrics ${prefs.expanded?'−':'+'}`;
 toggle.setAttribute('aria-expanded',String(prefs.expanded));
 const expanded=root.querySelector('.prof-expanded');expanded.hidden=!prefs.expanded;
 for(const tab of root.querySelectorAll('[data-metric-category]'))tab.setAttribute('aria-selected',String(tab.dataset.metricCategory===prefs.category));
 const link=expanded.querySelector('.prof-detail-link');link.dataset.metricView=prefs.category.toLowerCase();link.textContent=`View detailed ${prefs.category} profiler →`;
 // Collapsing needs no projections or trajectory work.
 if(preferencesOnly&&!prefs.expanded)return;
 syncScope(root,run);
 if(scopeEmpty(run)){
  // Nothing is projected for an empty scope, and no earlier values stay on screen.
  for(const [container,html] of [[root.querySelector('.prof-summary-slots'),emptyScope],[expanded.querySelector('.prof-cards'),emptyCards]])if(container.metricKeys!=='empty'){container.innerHTML=html;container.metricKeys='empty';}
  expanded.querySelector('.prof-summary-note').textContent=summaryNote(run);
  return;
 }
 const data=summaryData(run);
 updateCards(root.querySelector('.prof-summary-slots'),prefs.pins,data.rows,false,data.slots,run);
 if(prefs.expanded){updateCards(expanded.querySelector('.prof-cards'),data.keys,data.rows,true,data.cards,run);expanded.querySelector('.prof-summary-note').textContent=summaryNote(run);}
}

export function profilerView(run,category){const m=projectMetrics(run),rows=descriptors(m,run,category==='Systems'),keys=category==='Systems'?Object.keys(rows).filter(k=>k.startsWith('system:')):groups[category];let html=`<h2>${category} profiler</h2><p>${esc(run.title||run.id)} · ${esc(run.id)} · ${esc(run.status)} · displayed 0–${sec(m.end)}. ${run.coverage?.client_window?'Statistics cover retained calls only.':''}</p><div class="prof-cards">${keys.map(k=>card(k,rows)).join('')}</div>`;
 if(category==='Systems'){const activity=systemActivity(run);return html+`<h3>Syscall instrumentation · aggregated</h3><p>${activity.count} events · ${activity.unattributed} without recorded ownership · ${sec(activity.duration)} summed measured time (may overlap) · ${activity.unmeasured} without measured duration. Individual syscalls stay outside the causal graph.</p><table class="prof-table"><tr><th>Syscall</th><th>Count</th><th>Measured time</th><th>Untimed</th></tr>${activity.groups.map(g=>`<tr><td>${esc(g.name)}</td><td>${g.count}</td><td>${sec(g.duration)}</td><td>${g.unmeasured}</td></tr>`).join('')}</table><p>Recorded scopes are independent; do not add overlapping host, workload and process measurements. CPU: 100% = one core. RAM counters represent their named scope; rss_mb is process RSS. Rates use counter deltas and binary MiB. Missing samples are not zero.</p><div class="prof-charts">${systemMeasurements(run).map(v=>chart(v.scope+' · '+v.name,v.points,v.unit.replace('MB','MiB'),false,`${v.label} measured in ${v.scope}. Follow the line to identify resource growth or spikes during execution; samples show this scope only.`)).join('')}</div><p>Coverage: only recorded series are shown. Per-core CPU, pressure, scheduling delay, context switches, page faults, I/O latency and network error counts are not exposed by this projection. GPU series require supported hardware and collector access.</p>`;}
 if(category==='Activity'){
 html+=`<p>Observed peaks: tools ${peak(m.tools,m.end)} · model requests ${peak(m.models,m.end)}. Aggregate completed duration: tools ${sec(m.dt.reduce((n,a)=>n+a.duration,0))} · LLM ${sec(m.dm.reduce((n,a)=>n+a.duration,0))}. These concurrent durations are not elapsed time.</p><p>Wall coverage: tools ${sec(union(m.tools,m.end))} · LLM ${sec(union(m.models,m.end))}; these unions can overlap. Unclassified wall interval ${sec(Math.max(0,m.end-union([...m.tools,...m.models],m.end)))} is not inferred idle time. Agent lifecycle concurrency, restarts and complete waiting durations are unavailable.</p><table class="prof-table"><tr><th>Agent</th><th>State</th><th>Calls</th><th>Failure streak</th><th>Tool / LLM duration</th><th>Since last call activity</th></tr>${m.perAgent.map(a=>`<tr><td>${esc(a.label)}</td><td>${esc(a.status||'unknown')}</td><td>${a.calls.length}</td><td>${a.streak}</td><td>${sec(m.dt.filter(c=>c.agent===a.id).reduce((n,c)=>n+c.duration,0))} / ${sec(m.dm.filter(c=>c.agent===a.id).reduce((n,c)=>n+c.duration,0))}</td><td>${sec(a.gap)}</td></tr>`).join('')}</table>`;
 const width=Math.max(1,m.end/120);const bins=xs=>{const b=new Map(Array.from({length:Math.floor(m.end/width)+1},(_,i)=>[i*width,0]));for(const a of xs){if(a.missing_start)continue;const t=Math.floor(a.start/width)*width;b.set(t,(b.get(t)||0)+1);}return [...b].sort((a,b)=>a[0]-b[0]);};html+=`<div class="prof-charts">${chart('Tool starts over time',bins(m.tools),'calls/bin',false,`Tool calls started in each ${sec(width)} bin. Higher values mean more tool activity; zero means no observed starts.`)}${chart('LLM starts over time',bins(m.models),'requests/bin',false,`Model requests started in each ${sec(width)} bin. Higher values mean more model activity; zero means no observed starts.`)}</div>`;
 return html+table(m.actions.filter(a=>a.outcome==='running'||a.outcome==='failed'));
 }
 const calls=category==='LLM'?m.models:m.tools,done=category==='LLM'?m.dm:m.dt;
 html+=`<div class="prof-charts">${chart('Completed latency over time',done.map(a=>[a.start+a.duration,a.duration]),'s')}${chart('Duration distribution (sorted observations)',done.map(a=>a.duration).sort((a,b)=>a-b).map((v,i)=>[i,v]),'s',true)}</div>`;
 if(category==='LLM')html+=`<div class="prof-charts">${m.agents.map(agent=>chart('Reported input/context · '+(agent.label||agent.id),m.models.filter(a=>a.agent===agent.id).map(a=>[a.start,a.tokens]),'tokens',false,'Reported input tokens per request at request start. Growth shows larger prompts/context; missing usage is excluded.')).join('')}${chart('Output tokens per request',m.models.map(a=>[a.start,a.output_tokens]),'tokens',false,'Reported output tokens per request at request start. Compare response sizes over time; missing usage is excluded.')}${chart('Client E2E output throughput',m.dm.filter(a=>a.duration>0).map(a=>[a.start+a.duration,Number.isFinite(a.output_tokens)?a.output_tokens/a.duration:null]),'tokens/s',false,'Output tokens divided by completed request latency. Higher values mean greater client-observed throughput, including request waiting time.')}</div><p>Input/output usage follows provider reporting; missing usage is excluded. Cached token details, retries, prices, prefill, decode and queue timings are not reliably projected. No cost or server timing is inferred.</p>`;
 for(const dimension of category==='LLM'?['agent','model']:['name','agent']){const grouped=new Map();for(const a of calls){const key=a[dimension]||'Unknown';if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(a);}html+=`<h3>By ${dimension}</h3><table class="prof-table"><tr><th>${dimension}</th><th>Calls</th><th>P50 / P95</th><th>Failed / cancelled</th><th>${category==='LLM'?'Input / output tokens':'Completed duration'}</th></tr>${[...grouped].map(([k,v])=>{const members=new Set(v),d=done.filter(a=>members.has(a)),sum=key=>{const known=v.filter(a=>Number.isFinite(a[key]));return known.length?num(known.reduce((n,a)=>n+a[key],0)):'Unavailable';};return `<tr><td>${esc(k)}</td><td>${v.length}</td><td>${latency(percentile(d.map(a=>a.duration),.5))}<br>${latency(percentile(d.map(a=>a.duration),.95))} · n=${d.length}</td><td>${v.filter(a=>a.outcome==='failed').length} / ${v.filter(a=>a.outcome==='cancelled').length}</td><td>${category==='LLM'?sum('tokens')+' / '+sum('output_tokens'):sec(d.reduce((n,a)=>n+a.duration,0))}</td></tr>`;}).join('')}</table>`;}
 return html+table(calls);
}
export function handleMetricEvent(event,{render,navigate,reveal,measure:sendMeasure,rescope}){const el=event.target.closest('[data-scope-all],[data-scope-agent],[data-measure],[data-metric-toggle],[data-metric-category],[data-metric-reset],[data-metric-pin],[data-metric-view],[data-metric-event],[data-metric-sort],[data-metric-page]');if(!el)return false;
 if(el.hasAttribute('data-scope-all')||el.dataset.scopeAgent!=null){
  // "All agents" switches between every agent and none. Unticking one agent
  // from "All agents" leaves the others selected; completing the set returns
  // to all agents, and clearing the last agent leaves a valid empty scope.
  if(el.dataset.scopeAgent==null)scope=scope?null:new Set();
  else{const ids=roster.map(a=>a.id),next=new Set(scope?ids.filter(id=>scope.has(id)):ids),id=el.dataset.scopeAgent;next.has(id)?next.delete(id):next.add(id);scope=next.size<ids.length?next:null;}
  (rescope||render)();return true;
 }
 if(el.dataset.measure){
  // The backend validates and timestamps every boundary; only the user's chosen elapsed seconds travel.
  if(el.dataset.measure!=='end-at')sendMeasure?.(el.dataset.measure);
  else{const value=el.closest?.('.measure-menu')?.querySelector('input')?.value??'',elapsed=Number(value);sendMeasure?.('end',value.trim()===''||!Number.isFinite(elapsed)||elapsed<0?{error:'Enter the seconds since the measurement started.'}:{elapsed});}
  return true;
 }
 if(el.hasAttribute('data-metric-toggle'))prefs.expanded=!prefs.expanded;
 else if(el.dataset.metricCategory)prefs.category=el.dataset.metricCategory;
 else if(el.hasAttribute('data-metric-reset'))prefs.pins=defaults.slice();
 else if(el.dataset.metricPin){const key=el.dataset.metricPin;if(prefs.pins.includes(key))prefs.pins=prefs.pins.filter(k=>k!==key);else if(prefs.pins.length<6)prefs.pins.push(key);else{choosePinSlot(key,el,render);return true;}}
 else if(el.dataset.metricEvent){reveal(el.dataset.metricEvent);return true;}
 else if(el.dataset.metricView){filter=el.dataset.metricKey==='errors'?'failed':'';sort=el.dataset.metricKey==='llm95'?'duration':'start';direction=-1;page=0;navigate(el.dataset.metricView);return true;}
 else if(el.dataset.metricSort){direction=sort===el.dataset.metricSort?-direction:-1;sort=el.dataset.metricSort;}
 else if(el.dataset.metricPage)page+=Number(el.dataset.metricPage);
 save();render();return true;
}
export function filterMetrics(value){filter=value;page=0;}
