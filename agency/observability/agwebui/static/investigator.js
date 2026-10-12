import {renderMetricsSummary, profilerView, handleMetricEvent, filterMetrics} from './metrics-ui.js';
import {agentRoster, reconcilePinnedAgents, pinAgent} from './agent-roster.js';
import {requestControlConfirmation, refreshControlConfirmation} from './control-confirmation.js';
import {eventCard, actionState, eventRelationships, eventMeasurements, payloadWindow, payloadValue} from './event-details.js';
import {ExecutionTimeline} from './execution-timeline.js';
import {alignSequences, matchesAction, overlapDuration, packActionTracks, groupCallStack, executionIdentity, executionMeasurements, readableExecutionTitle, systemMeasurements} from './investigator-model.js';
import {executionControlState, mergeTrajectory, mergeTrajectorySnapshot, boundTrajectory, TrajectoryStream} from './trajectory-stream.js';

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const formatTime = value => value < 1 ? `${Math.round(value * 1000)}ms` : value >= 60 ? `${Math.floor(Math.round(value) / 60)}m ${Math.round(value) % 60}s` : `${value.toFixed(1)}s`;
const count = value => value == null ? 'n/a' : new Intl.NumberFormat('en', {notation: value >= 10000 ? 'compact' : 'standard', maximumFractionDigits: 1}).format(value);
const colors = {search:'#70b4ed', read:'#89c7da', edit:'#72d6b1', test:'#9fbb84', review:'#a3bdcc', tool:'#70b4ed', model:'#e8bb72', setup:'#758ba2', queue:'#b39ee5', dependency:'#9b88c9', unknown:'#394858', cpu:'#72d6b1', io:'#83c5d5'};
const sources = {'System instructions':'#b39ee5','Original task':'#70b4ed','Supervisor instructions':'#70b4ed','Tool results':'#72d6b1','Files':'#8fc9d8','Conversation':'#e8bb72','Summary':'#d396b0'};
const views = [
  ['trajectory','Trajectory','What did the agent do?'], ['timeline','Timeline','When did agents and resources overlap?'], ['resources','Resources & waits','Where did the time go?'],
  ['context','Context & information','What did the model know?'], ['agents','Tandem trace board','How did agents work together?'],
  ['compare','Compare runs','Where did behavior change?'],
  ['profiler','Profiler','Detailed activity, inference, tools and resource measurements']
];
const params = new URLSearchParams(location.search);
const profilerCategories={activity:'Activity',llm:'LLM',tools:'Tools',systems:'Systems'};
const requestedView=params.get('view');
const initialProfiler=profilerCategories[params.get('profiler')]?params.get('profiler'):profilerCategories[requestedView]?requestedView:'activity';
const state = {profilerCategory:initialProfiler,catalog:[], run:null, comparison:null, view:profilerCategories[requestedView]?'profiler':views.some(v=>v[0]===requestedView)?requestedView:'trajectory', trajectoryLayout:params.get('layout')==='overlay'?'overlay':'episodes', agent:'', query:'', cursor:0, selected:null, inspectorOpen:false, expanded:new Set(), model:null, contextBlock:null, zoom:null, loading:false, compareId:null};
let timelineViewer=null;
let timelineNavigation={scale:null,left:0,top:0,lock:false};
const fullPayloads=new WeakMap();
let commandSequence=0;
const openToolDetails=new Set();
const trajectoryColumns=new Map();
const columnReading=new Map();
const pendingLimit=256;
function retainNotice(set,id){if(set.has(id))return;if(set.size<pendingLimit)set.add(id);else set.overflow=true;}
const noticeCount=set=>`${set.size}${set.overflow?'+':''}`;
function clearNotice(set){set.clear();set.overflow=false;}
let pendingSeek=null;
let loadGeneration = 0;
let compareGeneration = 0;
let trajectoryUpdateTimer=null;
const pendingTrajectoryChanges=new Set();
let pendingResync=false;

function scheduleTrajectoryUpdate(changed,resynced=false) {
  for(const id of changed){retainNotice(pendingTrajectoryChanges,id);if(id===state.selected?.id)pendingTrajectoryChanges.add(id);}
  pendingResync ||= resynced;
  // A bounded batch rather than a debounce: a busy stream still paints.
  if(trajectoryUpdateTimer!==null)return;
  trajectoryUpdateTimer=setTimeout(()=>{
    trajectoryUpdateTimer=null;
    const changes=[...pendingTrajectoryChanges],resync=pendingResync;
    pendingTrajectoryChanges.clear();pendingResync=false;
    updateTrajectory(changes,resync);
  },120);
}
Object.assign(state, {executionControls:null, executionPending:null, executionMessage:'', executionOutcomes:[], interaction:null, rosterQuery:'', rosterStatus:'', rosterOpen:false, rosterExpanded:[], stream:null, replay:params.get('mode')==='replay', transport:'offline', follow:true, unread:new Set(), windowEnd:null, replayState:null, agentColumns:[], columnsCustomized:false, activeColumn:null, displayPaused:false, liveRun:null, displayChanges:new Set(),pendingNew:new Set()});
const currentExecution = () => state.liveRun || state.run;
const executionState = (agentId=null) => executionControlState({...state, executionControls:state.executionControls?.relay_id&&state.executionControls?.execution?state.executionControls:{...state.executionControls,available:false}, run:currentExecution(),
  replay:state.replay||(state.displayPaused&&executionIdentity(state.run)!==executionIdentity(currentExecution())),
  ...(agentId?{agent:agentId,selected:{type:'agent',id:agentId}}:{})});
const displayDuration = action => action.execution_duration ?? action.duration;
const pausedBadge = id => state.run?.agents.find(a=>a.id===id)?.status==='paused' ? pill('paused') : '';
const actionMap = run => new Map((run?.actions || []).map(a=>[a.id,a]));
const visibleActions = run => (run?.actions || []).filter(a=>matchesAction(a,state.agent,state.query));
const agentName = (id, run=state.run) => run?.agents.find(a=>a.id===id)?.label || id;
const pill = (label, kind=label) => `<span class="pill ${esc(kind.replace(/ /g,'-'))}">${esc(label)}</span>`;
const field = (name, value) => `<div class="detail-field"><span>${esc(name)}</span><span>${esc(value)}</span></div>`;
const heading = (title, description, extra='') => `<div class="section-heading"><div><h2>${esc(title)}</h2><p>${esc(description)}</p></div>${extra}</div>`;
const empty = (title, text) => `<div class="empty-state"><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
const legend = entries => `<div class="legend">${entries.map(([key,label])=>`<span><i style="background:${colors[key]||sources[key]}"></i>${esc(label)}</span>`).join('')}</div>`;
const insight = text => `<div class="insight"><span class="insight-icon">↳</span><div>${text}</div></div>`;

function timelineView(){return heading('Execution timeline','Agent calls and measured system resources on a shared wall-clock axis.')+'<div id="execution-timeline" class="panel"></div>';}

function measurementHTML([label,value,unit,note],detailed=false) {
  const formatted=unit==='time'?formatTime(value):unit==='rate'?`${value.toFixed(2)}/s`:unit==='count'?count(value):`${count(value)} ${unit}`;
  return `<div class="metric" title="${esc(note)}"><span class="metric-label">${esc(label)}</span><strong class="metric-value">${esc(formatted)}</strong>${detailed?`<div class="metric-note">${esc(note)}</div>`:''}</div>`;
}
function renderSystemMetrics() {
  const panel=$('system-metrics'),run=state.run;
  if(!panel)return;
  if(!run){panel.innerHTML='<div class="system-metrics-heading"><strong>System metrics</strong><span>Waiting for recorded samples…</span></div>';return;}
  const metrics=systemMeasurements(run),totals=metrics.filter(m=>m.aggregate);
  const shown=(totals.length?totals:metrics.filter(m=>m.label!=='Virtual memory')).slice(0,8);
  const mode=state.displayPaused?'Frozen view':state.replay?'Replay samples':run.source==='synthetic'?'Synthetic samples':run.id==='live'&&run.status==='running'?(state.transport==='connected'?'Live samples':'Disconnected · last received samples'):'Recorded samples';
  const heading=`<div class="system-metrics-heading"><strong>System metrics</strong><span>${esc(mode)}</span><button data-view="timeline">All resource timelines ↗</button></div>`;
  const cards=shown.map(metric=>{
    const [at,value]=metric.latest,start=Math.max(0,at-60),points=metric.points.filter(p=>p[0]>=start);
    const high=Math.max(1,...points.map(p=>p[1])),x=t=>2+(t-start)/(at-start||1)*116,y=v=>30-v/high*26;
    const line=points.map(p=>`${x(p[0])},${y(p[1])}`).join(' '),last=points.at(-1);
    const note=`${metric.name} · ${metric.statusLabel|| (metric.derived?'Rate calculated from counter deltas':'Recorded sample')} at ${formatTime(at)}${metric.label==='CPU'?' · 100% = one CPU core; multiple cores can exceed 100%':''}`;
    return `<div class="system-metric" title="${esc(note)}"><div class="system-metric-value"><span>${esc(metric.label)}</span><strong>${value==null?'Unavailable':count(value)} ${metric.unit}</strong></div><svg viewBox="0 0 120 34" role="img" aria-label="${esc(metric.scope+' '+metric.label)} · recent recorded samples"><polyline points="${line}" fill="none" stroke="currentColor" stroke-width="1.5"/>${last?`<circle cx="${x(last[0])}" cy="${y(last[1])}" r="2" fill="currentColor"/>`:''}</svg><small>${esc(metric.scope)} · at ${formatTime(at)}</small></div>`;
  }).join('');
  panel.innerHTML=heading+(cards?`<div class="system-metrics-cards">${cards}</div>`:'<p class="system-metrics-empty">No usable system samples recorded yet. Profiling must be enabled to collect CPU, memory, I/O and available GPU counters.</p>');
}
function readablePayload(value,limit=4000){
  let text=typeof value==='string'?value:JSON.stringify(value,null,2),json=false;
  if(text==null)text='';
  // Avoid parsing/formatting very large payloads until explicitly requested.
  if(text.length<=200000){try{text=JSON.stringify(JSON.parse(text),null,2);json=true;}catch{}}
  const truncated=text.length>limit; text=text.slice(0,limit);
  const html=json?text.replace(/("(?:\\.|[^"\\])*"\s*:?)|\b(true|false|null)\b|(-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi,token=>`<span class="json-${token.startsWith('"')?(token.endsWith(':')?'key':'string'):'literal'}">${esc(token)}</span>`):esc(text);
  return `<pre class="payload-text">${html}${truncated?'\n… Preview truncated':''}</pre>`;
}
function nearestMetrics(action,run){
  const measured=eventMeasurements(action,run);
  return measured.samples.map(sample=>field(sample.label,`${count(sample.value)} ${sample.unit} · ${sample.scope} · ${sample.provenance} sample at ${formatTime(sample.at)}, Δ ${formatTime(sample.delta)} from call start`)).join('')||'<p class="muted">No recorded resource samples available for this call interval. Current host measurements are not substituted.</p>';
}
function detailBody(a,run){
  const agent=run.agents.find(agent=>agent.id===a.agent),measurements=eventMeasurements(a,run);
  const rows=measurements.rows.map(row=>field(`${row.label} · ${row.provenance}`,`${row.unit==='s'?formatTime(row.value):count(row.value)+' '+row.unit} · ${row.scope}${row.interval?' · '+formatTime(row.interval[0])+' → '+formatTime(row.interval[1]):''}`)).join('');
  return `<div class="detail-label">Captured input / output · ${run.source==='synthetic'?'synthetic example':'recorded evidence'}</div>${['input','output'].map(kind=>`<section class="payload-section"><b>${kind==='input'?'Input':'Output'}</b>${readablePayload(payloadValue(a,kind))}<button data-full-payload="${esc(a.id)}" data-payload-kind="${kind}" data-payload-run="${esc(run.id)}">Open captured ${kind}${a[kind+'_truncated']?' from source':''}</button><button data-copy-payload="${esc(a.id)}" data-payload-kind="${kind}" data-payload-run="${esc(run.id)}">Copy original ${kind}</button><div class="full-payload"></div></section>`).join('')}<details class="event-measurements"><summary>Timing &amp; recorded measurements</summary>${rows}<p class="muted">${esc(measurements.coverage)} Nearest resource samples are contextual observations, not operation attribution.</p>${nearestMetrics(a,run)}</details>${field('Agent ID',a.agent)}${field('Start · projected boundary',a.missing_start||String(a.source||'').includes('timestamp_missing')?'Unavailable':formatTime(a.start))}${field('End · projected boundary',a.outcome==='running'||a.outcome==='incomplete'||String(a.source||'').includes('timestamp_missing')?'Not recorded':formatTime(a.start+a.duration))}${a.kind==='model'?field('Model identity',eventCard(a,run).primary):''}${field('Harness',agent?.harness||run.harness||'Unavailable')}`;
}

function toolDetails(a,run,open=false){return `<details class="tool-details" data-tool-details="${esc(a.id)}" ${open?'open':''}><summary><span class="when-closed">Show details</span><span class="when-open">Hide details</span></summary><div class="tool-detail-body">${open?detailBody(a,run):''}</div></details>`;}
document.addEventListener('toggle',event=>{
  const details=event.target;if(!details.matches?.('[data-tool-details]'))return;
  const id=details.dataset.toolDetails;
  if(details.open){openToolDetails.add(id);const a=actionMap(state.run).get(id);const body=details.querySelector('.tool-detail-body');if(a&&(!body.innerHTML||details.dataset.stale)){body.innerHTML=detailBody(a,state.run);delete details.dataset.stale;const card=details.closest('[data-call-card]');if(card){card.dataset.evidenceVersion=callEvidenceVersion(a);card.querySelector('[data-refresh-details]')?.remove();}}}
  else openToolDetails.delete(id);
},true);
function callEvidenceVersion(a){return JSON.stringify([a.outcome,a.event_ids,a.result,a.error]);}
function refreshCallDetails(button){
  const card=button.closest('[data-call-card]'),a=actionMap(state.run).get(card?.dataset.callCard);
  if(!a)return;
  const details=card.querySelector('.tool-details');
  details.querySelector('.tool-detail-body').innerHTML=detailBody(a,state.run);
  card.dataset.evidenceVersion=callEvidenceVersion(a);delete details.dataset.stale;
  button.remove();details.querySelector('summary').focus({preventScroll:true});
}
async function originalPayload(a,kind,run){
  if(!a[kind+'_truncated']&&!(kind==='input'&&a.kind==='model'&&a.context?.truncated)){const value=payloadValue(a,kind);return typeof value==='string'?value:JSON.stringify(value,null,2)||'';}
  const ids=kind==='output'?[...(a.event_ids||[])].reverse():(a.event_ids||[]);
  for(const id of ids){
    const url=run.id==='live'?`/api/trajectory/live/events/${encodeURIComponent(id)}`:`/api/trajectory/replay/${encodeURIComponent(run.id)}/events/${encodeURIComponent(id)}`;
    const raw=await fetchJson(url),p=raw.payload||{};
    const captured=kind==='input'?(a.kind==='model'?Array.isArray(p.context?.blocks):p.arguments!=null||p.input!=null||p.command!=null):p.error!=null||p.result!=null||p.output!=null||p.model_response!=null;
    if(!captured)continue;
    const value=payloadValue({kind:a.kind,arguments:p.arguments,command:p.command,input:p.input,context:p.context,result:p.result,output:p.output,model_response:p.model_response,error:p.error},kind);
    if(value!=null)return typeof value==='string'?value:JSON.stringify(value,null,2);
  }
  throw new Error('Full payload unavailable in retained source evidence.');
}
document.addEventListener('click',async event=>{
  const button=event.target.closest('[data-full-payload],[data-copy-payload]');if(!button)return;
  const run=button.dataset.payloadRun===state.comparison?.id?state.comparison:state.run;
  const a=actionMap(run).get(button.dataset.fullPayload||button.dataset.copyPayload);if(!a)return;
  const output=button.parentElement.querySelector('.full-payload');button.disabled=true;
  try{
    const raw=await originalPayload(a,button.dataset.payloadKind,run);
    if(button.dataset.copyPayload){await navigator.clipboard.writeText(raw);button.textContent='Copied';}
    else {
      // Explicitly requested, bounded scrolling output; preserve original bytes
      // in a text node for huge strings instead of generating syntax spans.
      renderFullPayload(output,raw);
      button.hidden=true;
    }
  }catch(error){output.textContent=error.message;}finally{button.disabled=false;}
});

async function fetchJson(url) {
  const response = await fetch(url, {cache:'no-store'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function catalogOptions() {
  const model = $('model-filter').value, harness = $('harness-filter').value;
  const runs = state.catalog.filter(r=>(!model || r.model===model) && (!harness || r.harness===harness));
  const selected = state.run?.id || params.get('run');
  $('run-select').innerHTML = ['recorded','synthetic'].map(source=>`<optgroup label="${source==='recorded'?'Live & recorded executions':'Synthetic scenarios'}">${runs.filter(r=>r.source===source).map(r=>`<option value="${esc(r.id)}" ${r.id===selected?'selected':''}>${esc(r.title)} · ${esc(r.subtitle)}</option>`).join('')}</optgroup>`).join('');
  if (!runs.length) $('run-select').innerHTML = '<option value="">No matching executions</option>';
  return runs;
}

async function refresh() {
  const loadAtStart=loadGeneration;
  $('refresh').disabled = true;
  try {
    const data = await fetchJson('/api/investigator/runs');
    state.catalog = data.runs;
    for (const [id,key,label] of [['model-filter','model','All models'],['harness-filter','harness','All harnesses']]) {
      const old = $(id).value;
      $(id).innerHTML = `<option value="">${label}</option>` + [...new Set(data.runs.map(r=>r[key]))].filter(Boolean).sort().map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');
      if ([...$(id).options].some(o=>o.value===old)) $(id).value=old;
    }
    catalogOptions();
    // A user-selected run takes precedence over an in-flight catalog refresh.
    if (loadGeneration!==loadAtStart) return;
    if (!state.run) await loadRun(params.get('run') || data.runs[0]?.id || 'live',null,state.replay);
    else await loadRun(state.run.id,null,state.replay);
  } catch(error) { showError(error); }
  finally { $('refresh').disabled = false; }
}

async function loadRun(id, selected=null, replay=false) {
  if (!id) return;
  clearTimeout(trajectoryUpdateTimer);trajectoryUpdateTimer=null;
  pendingTrajectoryChanges.clear();pendingResync=false;
  const generation = ++loadGeneration;
  clearExecutionPending();state.lastExecutionCommand=null;state.executionReceiptTargets=[];state.executionControls=null;state.executionMessage='';state.executionOutcomes=[];if(state.interaction){state.interaction.invalid='Execution changed. Reopen interaction for the intended agent.';renderInteractionStatus();}
  timelineNavigation={scale:null,left:0,top:0,lock:false};openToolDetails.clear();trajectoryColumns.clear();columnReading.clear();clearNotice(state.pendingNew);
  state.stream?.close();state.stream=null;state.replay=replay;state.replayState=null;
  state.follow=true;state.unread.clear();state.windowEnd=null;
  state.displayPaused=false;state.liveRun=null;clearNotice(state.displayChanges);clearNotice(state.unread);
  state.agentColumns=[];state.columnsCustomized=false;state.activeColumn=null;state.rosterQuery='';state.rosterStatus='';
  state.run=null;state.agent='';state.cursor=0;state.selected=selected?{...selected,execution:id}:null;state.inspectorOpen=Boolean(selected);state.inspectorSelection=null;state.expanded=new Set();state.contextBlock=null;state.model=null;state.zoom=null;
  $('inspector').innerHTML='';state.inspectorMarkup=null;state.inspectorMarkupSelection=null;syncInspector();
  state.comparison=null;state.compareId=null;state.compareError=null;compareGeneration++;
  $('agent-select').innerHTML='<option value="">All agents</option>';
  state.loading = true;
  $('content').innerHTML = '<div class="loading-state">Reading spans, prompt provenance and resource samples…</div>';
  $('status').textContent='Loading execution…';
  if(id==='live'||replay) {
    state.transport='connecting';renderSession();
    state.stream=new TrajectoryStream({runId:id,replay,onTransport:transport=>{if(generation===loadGeneration){state.transport=transport;if(transport!=='connected'&&state.interaction?.pending){state.interaction.status='Connection lost after submission · outcome unknown; no automatic retry.';renderInteractionStatus();}if(transport!=='connected'&&state.executionPending)finishExecutionCommand('Connection lost after submission; outcome unknown. Check recorded state after reconnect; no automatic retry was sent.');renderSession();}},onMessage:message=>{
      if(generation!==loadGeneration)return;
      if(message.type==='execution_command_result'){
        receiveCommandResult(message);
        return;
      }
      state.executionControls=message.execution_controls||null;
      for(const receipt of state.executionControls?.receipts||[])receiveCommandResult(receipt);
      renderInteractionStatus();
      const first=!currentExecution();
      let changed=[];
      const before=new Set((currentExecution()?.actions||[]).map(a=>a.id));
      if(message.type==='snapshot') {
        const previous=currentExecution();
        const same=!state.replay&&previous&&executionIdentity(previous)===executionIdentity(message.run);
        if(same)changed=mergeTrajectorySnapshot(previous,message.run);
        else {
          changed=message.run.actions.map(a=>a.id);
          const incoming={...message.run,actions:[],agents:[],episodes:[],edges:[],intervals:[],signals:[],counters:{}};mergeTrajectorySnapshot(incoming,message.run);
          if(state.displayPaused)state.liveRun=incoming;else state.run=incoming;
          if(previous&&executionIdentity(previous)!==executionIdentity(message.run)&&!state.displayPaused){setSelection(null);trajectoryColumns.clear();columnReading.clear();}
        }
        state.loading=false;
      }
      else for(const patch of message.patches||[]) changed.push(...mergeTrajectory(currentExecution(),patch));
      for(const id of new Set(changed)) {
        if(!state.follow)retainNotice(state.unread,id);
        if(state.displayPaused)retainNotice(state.displayChanges,id);
        if(!before.has(id)&&(state.displayPaused||!state.follow))retainNotice(state.pendingNew,id);
        const actor=currentExecution().actions.find(a=>a.id===id)?.agent,reading=columnReading.get(actor);
        if(reading&&!reading.follow)retainNotice(reading.pending,id);
      }
      if(!state.replay){boundTrajectory(currentExecution(),readingPins());const retained=new Set(currentExecution().actions.map(a=>a.id));for(const id of openToolDetails)if(!retained.has(id))openToolDetails.delete(id);}
      reconcileExecutionCommand();
      state.replayState=message.replay||null;
      if(state.replay&&pendingSeek!=null){const clock=pendingSeek;pendingSeek=null;state.stream.sendReplay('seek',{clock});}
      if(state.replay&&!state.displayPaused&&message.type==='snapshot'&&message.replay)state.cursor=message.replay.clock;
      if(state.follow&&!state.displayPaused) state.cursor=message.replay?.clock??state.run.duration;
      if(first){if(state.selected)state.selected.execution=executionIdentity(state.run);catalogOptions();updateUrl();render();}
      else if(state.displayPaused)renderSession();
      else scheduleTrajectoryUpdate(changed,message.resynced);
    }});
    return;
  }
  state.transport='offline';
  try {
    const run = await fetchJson(`/api/investigator/runs/${encodeURIComponent(id)}`);
    if (generation!==loadGeneration) return;
    state.run=run;if(state.selected)state.selected.execution=executionIdentity(run); state.agent=''; state.cursor=0; state.expanded=new Set(); state.contextBlock=null; state.model=null; state.zoom=null; state.comparison=null;
    if (selected?.type==='action') {
      const action = actionMap(run).get(selected.id);
      if (action) {state.cursor=action.start;state.expanded.add(action.episode);}
    }
    state.compareId = null; state.compareError = null; compareGeneration++;
    state.loading=false;
    $('agent-select').innerHTML='<option value="">All agents</option>'+run.agents.map(a=>`<option value="${esc(a.id)}">${esc(a.label)}</option>`).join('');
    catalogOptions();
    updateUrl(); render();
    if (state.view==='compare') await loadComparison();
  } catch(error) { if (generation===loadGeneration) {state.loading=false;showError(error);} }
}

function showError(error) {
  $('content').innerHTML=`<div class="empty-state"><h3>Could not load execution</h3><p>${esc(error.message)}</p><button data-retry>Retry loading</button><button data-demo>Open synthetic demo</button></div>`;
  $('status').textContent=error.message;
}

function updateUrl() {
  const url = new URL(location.href);
  if (state.run) url.searchParams.set('run',state.run.id);
  url.searchParams.set('view',state.view);
  if(state.view==='profiler')url.searchParams.set('profiler',state.profilerCategory);else url.searchParams.delete('profiler');
  if(state.replay) url.searchParams.set('mode','replay');else url.searchParams.delete('mode');
  if(state.trajectoryLayout==='overlay') url.searchParams.set('layout','overlay');
  else url.searchParams.delete('layout');
  history.replaceState(null,'',url);
}

function render() {
  const run=state.run;
  if (!run || state.loading) return;
  const content=$('content');
  const focusedContent=content.contains(document.activeElement)?document.activeElement:null;
  for(const column of trajectoryColumns.values())if(column.parentElement===document.getElementById('call-stack'))captureColumnReading(column);
  const existingRoster=content.querySelector('#agent-roster');
  if(existingRoster){state.rosterOpen=existingRoster.open;state.rosterExpanded=[...existingRoster.querySelectorAll('[data-roster-parent]')].filter(el=>el.open).map(el=>el.dataset.rosterParent);}
  const heldScroll=content.dataset.run===run.id&&content.dataset.view===state.view
    ? ['#call-stack','.trajectory-scroll','#tandem-columns'].map(selector=>{
      const region=content.querySelector(selector);return [selector,region?.scrollLeft||0,region?.scrollTop||0];
    }):[];
  $('views').innerHTML=views.slice(2).map(([id,label],index)=>`<button data-view="${id}" class="${state.view===id?'active':''}" aria-current="${state.view===id?'page':'false'}"><span class="view-number">${index+3}</span>${label}</button>`).join('');
  $('execution-views').innerHTML=views.slice(0,2).map(([id,label])=>`<button data-view="${id}" class="${state.view===id?'active':''}" aria-pressed="${state.view===id}">${label}</button>`).join('');
  $('view-label').textContent=views.find(v=>v[0]===state.view)[1];
  renderRunChrome();
  const board=$('content').querySelector('#tandem-columns');
  const held=state.view==='agents'&&board?.dataset.run===run.id?{
    episodes:[...board.querySelectorAll('[data-tandem-episode]')].filter(el=>el.open).map(el=>el.dataset.tandemEpisode),
    scroll:[...board.children].map(el=>[el.dataset.tandemAgent,el.querySelector('.col-body').scrollTop])
  }:null;
  timelineViewer?.destroy();timelineViewer=null;
  $('content').innerHTML=({profiler:detailedProfilerView,timeline:timelineView,trajectory:trajectoryView,resources:resourcesView,context:contextView,agents:agentsView,compare:compareView})[state.view]();
  if(state.view==='timeline'){if(state.displayPaused)timelineNavigation.lock=false;timelineViewer=new ExecutionTimeline($('execution-timeline'),timelineNavigation,id=>selectAction(id),()=>holdHistory());timelineViewer.update(run,visibleActions(run),state.agent,state.replayState?.duration);timelineViewer.setSelection?.(state.selected?.run!=='b'&&state.selected?.execution===executionIdentity(run)?state.selected.id:null);timelineViewer.scroll.scrollLeft=timelineNavigation.left;timelineViewer.scroll.scrollTop=timelineNavigation.top||0;timelineViewer.setFrozen(state.displayPaused);}
  if(state.view==='trajectory'&&state.trajectoryLayout==='episodes'){reconcileActivities(true);$('agent-roster').open=state.rosterOpen;for(const el of content.querySelectorAll('[data-roster-parent]'))el.open=state.rosterExpanded.includes(el.dataset.rosterParent);}
  if(state.view==='agents'){
    renderTandemBoard();
    if(held){
      for(const el of $('tandem-columns').querySelectorAll('[data-tandem-episode]'))el.open=held.episodes.includes(el.dataset.tandemEpisode);
      for(const [id,top] of held.scroll)$('tandem-columns').querySelector(`[data-tandem-agent="${CSS.escape(id)}"] .col-body`)?.scrollTo({top});
    }
  }
  for(const [selector,left,top] of heldScroll){const region=content.querySelector(selector);if(region){region.scrollLeft=left;region.scrollTop=top;}}
  content.dataset.run=run.id;content.dataset.view=state.view;
  if(focusedContent&&content.contains(focusedContent))focusedContent.focus({preventScroll:true});
  renderInspector();
}

function renderMetricsDashboard(preferencesOnly=false){
 const visible=state.view!=='profiler';
 $('summary-strip').hidden=!visible;
 $('metrics').hidden=!visible;
 if(visible&&state.run)renderMetricsSummary($('metrics'),state.run,{preferencesOnly});
}
function renderRunChrome() {
  const run=state.run;
  if(!run)return;
  const catalogRun=state.catalog.find(item=>item.id===run.id);
  if(catalogRun)for(const key of ['model','harness'])catalogRun[key]=run[key];
  for(const [id,key] of [['model-filter','model'],['harness-filter','harness']]) {
    if(run[key]&&![...$(id).options].some(option=>option.value===run[key]))$(id).add(new Option(run[key],run[key]));
  }
  $('run-title').textContent=readableExecutionTitle(run);
  $('run-title').title=readableExecutionTitle(run);
  if($('execution-id').value!==run.id){$('execution-id').value=run.id;$('copy-id-status').textContent='';}
  $('execution-metadata').innerHTML=field('Full execution title',run.title||'Unavailable')+(run.task?field('Task',typeof run.task==='string'?run.task:JSON.stringify(run.task)):'')+field('Source events',Number.isFinite(run.coverage.events)?count(run.coverage.events):'Unavailable')+field('Timed spans',['live','replay'].includes(run.mode)?'Not collected in this event projection':Number.isFinite(run.coverage.spans)?count(run.coverage.spans):'Unavailable')+[run.coverage.context,run.coverage.relationships,run.coverage.client_window].filter(Boolean).map(note=>`<p>${esc(note)}</p>`).join('');
  $('run-subtitle').textContent=`${run.condition?.replace('_',' ') || (run.id==='live'?'current execution':state.replay?'recorded replay':'saved execution')} / ${run.harness} / ${run.model}`;
  const agentCount=run.agents.filter(a=>a.id!=='workflow').length;
  $('run-description').textContent=`${agentCount} recorded actor${agentCount===1?'':'s'} · ${run.actions.length} observed calls · ${run.episodes.length} activities`;
  $('provenance').innerHTML=pill(run.source==='synthetic'?'Recorded example · synthetic':state.replay?'Replay · recorded':run.id==='live'?'Live evidence':'Recorded evidence',run.source);
  $('provenance').title=run.source==='synthetic'?'All timings, context and relationships are synthetic.':'Activity grouping is derived; returned results remain observations.';
  renderMetricsDashboard();
  const detailsOpen=$('execution-details').open;
  const measurements=executionMeasurements(run,{includeResources:!!detailsOpen});
  if(detailsOpen)$('secondary-metrics').innerHTML=measurements.primary.concat(measurements.secondary).map(item=>measurementHTML(item,true)).join('');
  $('telemetry-coverage').textContent=measurements.coverage+(run.coverage.client_window?' Live view is windowed: page older calls in each column; full evidence remains in recorded review/export.':'');
  $('cursor').max=Math.ceil((state.replayState?.duration??run.duration)*10)/10;
  $('cursor').value=state.cursor;
  $('cursor-value').textContent=formatTime(state.cursor);
  $('duration-label').textContent=formatTime(state.replayState?.duration??run.duration);
  const known=new Set([...$('agent-select').options].map(option=>option.value));
  for(const agent of run.agents)if(!known.has(agent.id)){$('agent-select').add(new Option(agent.label,agent.id));}
  renderSession();
  $('status').textContent=`${run.source==='synthetic'?'SYNTHETIC':'RECORDED'} · ${count(run.coverage.events)} source events · ${visibleActions(run).length} actions in current filter${state.selected?' · selection linked across views':''}`;
}

function actionButton(action) {
  const card=eventCard(action,state.run);
  return `<button class="action-row ${state.selected?.run!=='b'&&state.selected?.id===action.id?'selected':''}" data-action="${esc(action.id)}"><span class="action-time">${action.missing_start||String(action.source||'').includes('timestamp_missing')?'Unknown':formatTime(action.start)}</span><span class="action-kind">${esc(card.secondary)}</span><span class="action-label">${esc(card.primary)}</span><span class="action-duration" aria-label="Calculated wall interval">${action.outcome==='incomplete'?'End missing':card.elapsed==null?'Unknown':formatTime(card.elapsed)}</span><span class="status-icon ${esc(action.outcome)}" aria-label="${esc(card.state.label)}">${esc(card.state.icon)}</span></button>`;
}

function trajectoryView() {
  const overlay=state.trajectoryLayout==='overlay';
  const controls=`<label class="trajectory-layout">Layout<select id="trajectory-layout" aria-label="Trajectory layout"><option value="episodes" ${overlay?'':'selected'}>Episode list</option><option value="overlay" ${overlay?'selected':''}>Agent overlay</option></select></label>`;
  const header=heading('Execution trajectory',overlay?'Compare agent actions on a shared execution clock.':'Each column is an agent. Select a call to inspect captured evidence; card positions are not a time scale.',controls);
  const key=legend([['search','Search'],['read','Read'],['edit','Edit'],['test','Test'],['model','Model call']]);
  if(overlay) return header+trajectoryOverlay(visibleActions(state.run),key)+trajectoryWorkflow();
  return header+`<details id="trajectory-attention" class="trajectory-attention"><summary>Execution signals <span id="signal-count"></span></summary><div id="trajectory-signals"></div></details><div class="agent-column-controls"><div id="agent-tabs" role="tablist" aria-label="Visible agents"></div><label>Add agent<select id="add-agent-column" aria-label="Add agent column"></select></label><span id="agent-column-limit" class="muted"></span><details id="agent-roster" class="agent-roster"><summary>Agent roster <span id="roster-count"></span></summary><div class="roster-filters"><label>Find agent<input id="roster-search" placeholder="Name or identifier" aria-label="Search agent roster" value="${esc(state.rosterQuery)}"></label><label>Show<select id="roster-status" aria-label="Roster filter"><option value="">All agents</option><option value="hidden">Hidden agents</option><option value="failed-actions">Retained call failures</option><option value="running">Running agents</option><option value="paused">Paused agents</option><option value="completed">Completed agents</option><option value="unknown">Unknown state</option></select></label></div><div id="roster-list"></div></details></div><div id="hidden-agents" aria-label="Hidden agent failures"></div><div id="call-stack" class="agent-columns"></div><details class="activity-history"><summary>Activities grouped by intent</summary><div id="trajectory-glance"></div><div id="trajectory-workflow"></div><div class="activity-window"><button data-older-activities>Earlier activities</button><span id="activity-window-label" class="mono muted"></span><button data-latest-activities>Latest activities</button></div><div id="activity-list"></div><div class="notice">Activity labels use declared purpose when available, otherwise deterministic tool categories. Returned evidence is quoted below each activity. Tool success does not establish task success. Only 50 activities are rendered at once.</div></details>`;
}

function clearExecutionPending() {
  clearTimeout(state.executionPending?.timer);
  state.executionPending=null;
}
function pauseDisplay() {
  if(state.displayPaused||!state.stream||!state.run||state.loading)return;
  // Finish the pending paint before holding a separate, immutable display model.
  if(trajectoryUpdateTimer!==null){
    clearTimeout(trajectoryUpdateTimer);trajectoryUpdateTimer=null;
    updateTrajectory([...pendingTrajectoryChanges],pendingResync);
    pendingTrajectoryChanges.clear();pendingResync=false;
  }
  state.liveRun=state.run;
  state.run=structuredClone(state.liveRun);
  timelineNavigation.lock=false;timelineViewer?.setFrozen?.(true);
  state.displayPaused=true;clearNotice(state.displayChanges);clearNotice(state.pendingNew);
  // Timeline scrolling and resize paints must also read the held snapshot.
  timelineViewer?.update(state.run,visibleActions(state.run),state.agent,state.replayState?.duration);
  renderSession();
}
function resumeDisplay() {
  if(!state.displayPaused)return;
  state.run=state.liveRun;state.liveRun=null;state.displayPaused=false;timelineViewer?.setFrozen?.(false);
  if(state.selected&&state.selected.execution!==executionIdentity(state.run))setSelection(null);
  clearNotice(state.displayChanges);clearNotice(state.pendingNew);
  state.cursor=state.replayState?.clock??state.run.duration;
  if(['trajectory','timeline','agents'].includes(state.view))updateTrajectory(state.run.actions.map(a=>a.id));
  else render();
}
function finishExecutionCommand(message) {
  clearExecutionPending();state.executionMessage=message;renderSession();
}
function commandTargetSnapshot() {
  return {relay_id:state.executionControls?.relay_id,execution:state.executionControls?.execution};
}
function applicationSummary(target){
  if(target.state==='failed')return 'request failed';
  const a=target.application;if(!a)return 'application unconfirmed';
  if(a.state==='confirmed'){
    const locals=(a.observations||[]).map(item=>item.local_state),known={suspended:'local suspension observed',continued:'local suspension released',exited:'local task-tree exit observed'};
    if(locals.length&&locals.every(value=>value===locals[0])&&known[locals[0]])return known[locals[0]];
  }
  return `local application ${a.state}`;
}
function targetReceiptDescription(target){
  if(target.application){const a=target.application;
    const facts=(a.observations||[]).map(item=>`${item.local_state==='suspended'?'Local suspension':item.local_state==='continued'?'Local suspension released':item.local_state==='exited'?'Local task-tree exit':'Local state'}: ${item.state}${item.observed_at?` at ${new Date(item.observed_at*1000).toISOString()}`:''}${item.request_id?` · invocation ${item.request_id}`:''}${item.detail&&item.state!=='confirmed'?' · '+item.detail.replace(/\.$/,''):''}`).join('; ');
    return `API accepted · local application ${a.state}${a.detail?' · '+a.detail:''}${facts?' · '+facts:''}. Remote operations and cleanup are not established.`;
  }
  if(target.scope==='future_retained_context'&&target.context_state==='committed')return 'Retained context committed; model consumption unconfirmed'+(target.detail?' · '+target.detail:'');
  if(target.scope==='future_retained_context'&&target.context_state==='failed')return 'Retained context failed to commit'+(target.detail?' · '+target.detail:'');
  if(target.detail)return target.detail;
  if(target.scope==='future_retained_context')return target.context_state==='committed'?'Retained context committed; model consumption unconfirmed':'Message published at the retained-context tail; consumption unconfirmed';
  if(target.scope==='native_prompt_acknowledgement')return 'Native prompt accepted for this invocation; model response/compliance unconfirmed';
  if(target.scope==='scheduler_submission_not_completion')return `Invocation ${target.invocation||'ID unavailable'} submitted; completion unconfirmed`;
  if(target.scope==='stop_requested_not_process_exit')return 'Stop requested through agent API; process exit unconfirmed';
  if(target.requested==='pause'||target.requested==='resume')return `Agent intent recorded; ${target.rpc_acknowledged?'daemon RPC acknowledged':'daemon RPC not acknowledged'}; actual process state not independently measured`;
  return 'API acknowledgement; application unconfirmed';
}
function receiveCommandResult(receipt) {
  const pending=state.executionPending;
  if(pending?.id===receipt.id){
    if(receipt.error){finishExecutionCommand(receipt.error);return;}
    pending.accepted=receipt.queued||receipt.state==='queued'||receipt.state==='accepted'||receipt.state==='partial';
    if(receipt.targets?.some(t=>typeof t==='object')){pending.receipts=receipt.targets;state.executionReceiptTargets=receipt.targets;state.executionOutcomes=receipt.targets.map(t=>`${agentName(t.agname,currentExecution())}: ${t.state} · ${targetReceiptDescription(t)}`);}
    if(receipt.state==='failed'){finishExecutionCommand('Control failed. '+(state.executionOutcomes.join(' · ')||receipt.detail||'Dispatch was not acknowledged.'));return;}
    state.executionMessage=receipt.state==='partial'?'Some targets failed; waiting for recorded state of accepted targets.':pending.accepted?'Backend accepted the request; waiting for newer recorded agent state.':'Request sent; backend acceptance is unknown.';
    renderSession();
  }
  if(!pending&&state.lastExecutionCommand?.id===receipt.id&&state.lastExecutionCommand.execution===executionIdentity(currentExecution())&&state.lastExecutionCommand.relay===state.executionControls?.relay_id&&receipt.targets?.some(t=>typeof t==='object')){
    state.executionReceiptTargets=receipt.targets;state.executionOutcomes=receipt.targets.map(t=>`${agentName(t.agname,currentExecution())}: ${targetReceiptDescription(t)}`);
    state.executionMessage='Latest recorded control outcome · '+state.executionOutcomes.join(' · ');renderSession();
  }
  const draft=state.interaction;
  if(draft?.requestId===receipt.id){
    if(receipt.error){draft.status='Failed · '+receipt.error;draft.pending=false;}
    else if(receipt.targets?.some(t=>typeof t==='object')){
      const target=receipt.targets.find(t=>t.agname===draft.agent)||receipt.targets[0];
      draft.status=`${target.context_state==='committed'?'Context committed':target.context_state==='failed'?'Context failed':target.state} · ${targetReceiptDescription(target)}`;
      draft.pending=false;
    }else if(receipt.queued){draft.status='Requested · relay file queued; agent dispatch not confirmed.';}
    renderInteractionStatus();
  }
}
function reconcileExecutionCommand() {
  const pending=state.executionPending;if(!pending)return;
  if(pending.relay&&pending.relay!==state.executionControls?.relay_id||pending.execution!==executionIdentity(currentExecution())){finishExecutionCommand('Target execution changed; request outcome unknown. No retry was sent.');return;}
  const run=currentExecution();
  const results=pending.agents.map(id=>{
    const a=run.agents.find(a=>a.id===id),receipt=pending.receipts?.find(t=>t.agname===id);
    if(receipt?.state==='failed')return {id,done:true,text:`${agentName(id,run)}: failed · ${receipt.detail||'dispatch failed'}`};
    if(receipt?.application)return {id,done:receipt.application.state!=='pending',text:`${agentName(id,run)}: ${targetReceiptDescription(receipt)}`};
    if(state.executionControls?.control_observations)return {id,done:false,text:`${agentName(id,run)}: awaiting correlated control receipt; local application unconfirmed`};
    const ended=a&&['completed','failed','cancelled'].includes(a.execution_state||a.status);
    if(pending.stopping&&ended)return {id,done:true,text:`${agentName(id,run)}: terminal state recorded (${a.execution_state||a.status}); process exit not independently confirmed`};
    if(pending.stopping&&a?.stop_ts>pending.observed[id]&&a.stop_force===pending.force)return {id,done:true,text:`${agentName(id,run)}: ${pending.force?'SIGKILL':'SIGTERM'} request recorded; process exit unconfirmed`};
    if(!pending.stopping&&a&&(a.status==='paused')===pending.paused&&a.control_ts>pending.observed[id])return {id,done:true,text:`${agentName(id,run)}: ${pending.paused?'paused':'resumed'} state recorded; process application not independently measured`};
    if(!a||!state.executionControls?.agents.includes(id))return {id,done:true,text:`${agentName(id,run)}: no longer active; application outcome unknown`};
    return {id,done:false,text:`${agentName(id,run)}: ${receipt?.state==='accepted'?'API accepted; awaiting recorded state':'awaiting acknowledgement/state'}`};
  });
  state.executionOutcomes=results.map(r=>r.text);
  if(results.every(r=>r.done))finishExecutionCommand('Control result · '+state.executionOutcomes.join(' · '));
}
function pauseResumeExecution(all,agentId=null) {
  const controls=executionState(agentId);
  if(state.executionPending||!(all?controls.allEnabled:controls.agentEnabled))return;
  sendExecutionCommand(all?controls.allCommand:controls.agentCommand,all,agentId);
}
function stopExecution(all,force=false,agentId=null) {
  const controls=executionState(agentId);
  if(!(all?controls.allEnabled:controls.agentEnabled)||(!force&&state.executionPending)||force&&state.executionPending?.force)return;
  const targets=all?controls.active:[controls.target],fence=commandTargetSnapshot(),execution=executionIdentity(currentExecution()),ids=targets.map(a=>a.id);
  const validate=()=>{
    const now=executionState(agentId),current=all?now.active:[now.target];
    return !(all?now.allEnabled:now.agentEnabled)||execution!==executionIdentity(currentExecution())||fence.relay_id!==state.executionControls?.relay_id||JSON.stringify(fence.execution)!==JSON.stringify(state.executionControls?.execution)||JSON.stringify(ids)!==JSON.stringify(current.map(a=>a?.id))?'Target execution or agent scope changed. Cancel and review the current live targets.':(!force&&state.executionPending)||force&&state.executionPending?.force?'A control request is already pending.':'';
  };
  requestControlConfirmation({force,execution:readableExecutionTitle(currentExecution()),targets:targets.map(a=>`${agentName(a.id,currentExecution())} (${a.id})`),frozen:state.displayPaused,validate,onConfirm:()=>{
    if(validate())return;clearExecutionPending();sendExecutionCommand(`${force?'kill':'stop'}${all?'_all':''}`,all,agentId);
  }});
}
function sendExecutionCommand(command,all,agentId=null) {
  closeControlMenus();
  const controls=executionState(agentId),targets=all?controls.active:[controls.target];
  const id=`ui-${Date.now()}-${++commandSequence}`,fence=commandTargetSnapshot();
  if(!state.stream?.sendExecution(command,{id,...fence,targets:targets.map(a=>a.id),...(all?{}:{agname:controls.target.id})})) {
    finishExecutionCommand('Request not sent; no automatic retry. Reconnect before trying again.');return;
  }
  const stopping=command.startsWith('stop')||command.startsWith('kill');
  const pending={id,command,execution:executionIdentity(currentExecution()),relay:fence.relay_id,agents:targets.map(a=>a.id),paused:command.startsWith('pause'),stopping,force:command.startsWith('kill'),
    observed:Object.fromEntries(targets.map(a=>[a.id,(stopping?a.stop_ts:a.control_ts)||0]))};
  pending.timer=setTimeout(()=>{if(state.executionPending===pending)finishExecutionCommand('No execution confirmation received; outcome unknown. Inspect current state; no automatic retry was sent.');},30000);
  state.executionPending=pending;state.executionReceiptTargets=[];state.executionOutcomes=[];state.executionMessage='Requested · waiting for backend acceptance and recorded agent state.';renderSession();
  state.lastExecutionCommand={id,execution:pending.execution,relay:pending.relay};
}
function agentControlState(id) {
  const controls=executionState(id),pending=state.executionPending;
  const disabled=!controls.agentEnabled,target=esc(id),name=agentName(id,currentExecution());
  const own=pending?.agents.includes(id);
  const label=own?(pending.stopping?(pending.force?'Force stop requested':'Stopping…'):(pending.paused?'Pausing…':'Resuming…')):(controls.target?.status==='paused'?'Pause intent recorded':controls.target?.status||'unknown');
  return {target,name,label,resume:controls.agentCommand==='resume',disabled:Boolean(disabled||pending),killDisabled:Boolean(disabled||pending?.force)};
}
function agentControlButtons(id) {
  const {target,name,label,resume,disabled,killDisabled}=agentControlState(id);
  return `<span class="control-section-label">Agent controls · ${esc(label)}</span><button class="execution-control" data-execution-agent="${target}" aria-label="${resume?'Resume':'Pause'} ${esc(name)}" aria-describedby="control-scope" ${disabled?'disabled':''}>${resume?'▶ Resume':'⏸ Pause'}</button><button class="execution-control" data-execution-stop="${target}" aria-label="Stop ${esc(name)}" ${disabled?'disabled':''}>■ Stop</button><details class="agent-danger-menu"><summary>More</summary><button class="execution-control danger-control" data-execution-kill="${target}" ${killDisabled?'disabled':''}>Force stop · SIGKILL</button></details><button data-interact="${target}">Interact</button>`;
}

function agentExecutionControls(id) {
  return executionState(id).live&&executionState(id).target?`<div class="agent-execution-controls" data-agent-controls="${esc(id)}" role="group" aria-label="Controls for ${esc(agentName(id))}">${agentControlButtons(id)}</div>`:'';
}

function updateAgentControls(group,html) {
  const layout=html.includes('data-execution-agent=')?'live':'recorded';
  if(group.controlLayout!==layout){group.innerHTML=html;group.controlLayout=layout;return;}
  if(layout==='recorded'){group.querySelector('.control-section-label').textContent=`Recorded agent state · ${state.run.agents.find(a=>a.id===group.dataset.agentControls)?.status||'unknown'}`;return;}
  // Keep the pointer/focus target mounted as pause receipts and telemetry arrive.
  const {name,label,resume,disabled,killDisabled}=agentControlState(group.dataset.agentControls);
  group.querySelector('.control-section-label').textContent=`Agent controls · ${label}`;
  const pause=group.querySelector('[data-execution-agent]');
  pause.textContent=resume?'▶ Resume':'⏸ Pause';pause.setAttribute('aria-label',`${resume?'Resume':'Pause'} ${name}`);pause.disabled=disabled;
  group.querySelector('[data-execution-stop]').disabled=disabled;
  group.querySelector('[data-execution-kill]').disabled=killDisabled;
}
function closeControlMenus() {
  for(const menu of document.querySelectorAll('.execution-menu,.agent-danger-menu,.session-menu')) {
    if(!menu.open)continue;
    const focused=menu.contains(document.activeElement);menu.open=false;
    if(focused)menu.querySelector('summary')?.focus({preventScroll:true});
  }
}

function renderSession() {
  // Scoped systems samples are available in the expandable panel and profiler.
  $('system-metrics').hidden=true;
  refreshControlConfirmation();
  const run=currentExecution(), replay=state.replay;
  const mode=replay?'Replay':run?.id==='live'?'Live':'Review';
  const status=run?.status||'unknown';
  const debuggerState=executionState();
  const paused=run?.agents.filter(a=>a.status==='paused').length||0;
  $('execution-all').hidden=!debuggerState.live;
  $('execution-all').textContent=state.executionPending&&!state.executionPending.stopping?(state.executionPending.paused?'Pausing agents…':'Resuming agents…'):debuggerState.allCommand==='resume_all'?'▶ Resume all active agents':'⏸ Pause all active agents';
  $('execution-all').setAttribute('aria-describedby','control-scope');
  $('execution-all').disabled=!debuggerState.allEnabled||Boolean(state.executionPending);
  $('all-agent-controls').hidden=!debuggerState.live;
  for(const force of [false,true]) {
    const button=$(`execution-${force?'kill':'stop'}-all`);
    button.disabled=!debuggerState.allEnabled||(!force&&Boolean(state.executionPending))||force&&Boolean(state.executionPending?.force);
    button.title=`${force?'Force termination with SIGKILL':'Gracefully stop with SIGTERM'} for all active agents`;
  }
  for(const group of document.querySelectorAll('[data-agent-controls]')) {
    group.hidden=!debuggerState.live;
    const html=agentControlButtons(group.dataset.agentControls);
    updateAgentControls(group,html);
  }
  if(state.run)for(const column of trajectoryColumns.values())renderColumnFollow(column);
  renderInteractionStatus();
  $('display-toggle').hidden=!state.stream;
  $('display-toggle').disabled=!state.run||state.loading;
  $('display-toggle').textContent=state.displayPaused?'Unfreeze view':'Freeze view';
  $('display-toggle').title=state.displayPaused?'Apply buffered event updates and preserve reading positions':'Freeze event display; agent execution and ingestion continue';
  $('display-toggle').setAttribute('aria-pressed',String(state.displayPaused));
  $('display-state').textContent=state.displayPaused?`View frozen · ${noticeCount(state.pendingNew)} newly buffered · ${noticeCount(state.displayChanges)} calls changed · agents may still be running`:state.stream?(state.follow?'View following live':`History held · ${noticeCount(state.pendingNew)} newly buffered · ${noticeCount(state.unread)} calls changed`):'';
  $('control-outcomes').textContent=state.executionOutcomes.join(' · ');
  $('control-evidence').hidden=!state.executionOutcomes.length;
  $('control-scope').textContent=debuggerState.live?`${debuggerState.active.length} active agents · bulk actions include hidden agents. Pause holds harness processes; already-issued remote operations may continue.${state.displayPaused?' Controls target CURRENT LIVE execution while event display is frozen.':''}`:'';
  const receiptTargets=state.executionReceiptTargets||[],summaries=receiptTargets.map(applicationSummary);
  const evidenceSummary=receiptTargets.length>3?[...new Set(summaries)].map(label=>`${summaries.filter(value=>value===label).length} ${label}`).join(' · '):receiptTargets.map((target,index)=>`${agentName(target.agname,run)}: ${summaries[index]}`).join(' · ');
  const message=state.executionMessage.startsWith('Control result · ')||state.executionMessage.startsWith('Latest recorded control outcome · ')?'Control evidence updated · '+(evidenceSummary||'Review per-agent evidence'):state.executionMessage;
  $('execution-message').textContent=debuggerState.live?(message||(!state.executionControls?.available&&state.transport==='connected'?'Execution controls unavailable':'')):'';
  $('execution-state').textContent=`${mode} · ${status==='unknown'?'execution state unrecorded':status}${paused?` · ${paused} agent${paused===1?'':'s'} with pause intent recorded`:''}`;
  $('transport-state').textContent=state.stream?`${state.transport==='connected'?'Connected to updates':state.transport==='disconnected'?'Disconnected from updates · execution state retained':state.transport==='reconnecting'?'Reconnecting to updates':'Connecting to updates'}${replay&&state.replayState?` · ${state.replayState.finished?'replay exhausted':state.replayState.playing?'playing':'paused'} at ${formatTime(state.replayState.clock)}`:''}`:'Saved evidence · no live subscription';
  $('execution-state').dataset.mode=mode.toLowerCase();
  $('open-live').hidden=run?.id==='live'&&!replay;
  $('start-replay').hidden=!run||replay;
  for(const id of ['replay-play','replay-step','replay-restart','replay-speed'])$(id).hidden=!replay;
  $('replay-play').textContent=state.replayState?.playing?'Pause replay':'Play replay';
  $('replay-play').disabled=Boolean(state.replayState?.finished)||state.transport!=='connected';
  $('replay-step').disabled=Boolean(state.replayState?.finished)||state.transport!=='connected';
  $('follow-live').hidden=!state.stream;
  $('follow-live').textContent='Jump to live';
  $('follow-live').title='Unfreeze and return every agent column and timeline to live';
  $('reconnect-updates').hidden=!state.stream;
}

function updateTrajectory(changed,resynced=false) {
  if(state.displayPaused){renderSession();return;}
  renderRunChrome();
  if(state.view==='trajectory') {
    if(state.trajectoryLayout==='episodes')reconcileActivities();
    else {
      const focused=document.activeElement?.getAttribute('data-action');
      const previous=$('content').querySelector('.trajectory-scroll');
      const left=previous?.scrollLeft||0,top=previous?.scrollTop||0;
      $('content').innerHTML=trajectoryView();
      const region=$('content').querySelector('.trajectory-scroll');if(region){region.scrollLeft=left;region.scrollTop=top;}
      if(focused)$('content').querySelector(`[data-action="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
    }
  }
  if(state.view==='profiler'&&!$('content').contains(document.activeElement))$('content').innerHTML=detailedProfilerView();
  if(state.view==='agents')renderTandemBoard();
  if(state.view==='timeline')timelineViewer?.update(state.run,visibleActions(state.run),state.agent,state.replayState?.duration);
  const target=executionState().target;
  const controlSignature=`${target?.id}:${target?.status}:${target?.control_ts}`;
  const controlChanged=controlSignature!==state.inspectorControlSignature;
  if(controlChanged||!state.selected||changed.includes(state.selected?.id)){
    if($('inspector').contains(document.activeElement)){state.inspectorDirty=true;const note=$('inspector').querySelector('.inspector-update');if(note)note.textContent='Selected evidence updated. Details refresh when you leave the inspector.';}
    else renderInspector(true);
  }
  state.inspectorControlSignature=controlSignature;
  if(resynced)$('transport-state').textContent+=' · recovered from durable snapshot';
}

function trajectoryAgents() {
  return [...new Set([...state.run.agents.map(agent=>agent.id),...state.run.actions.map(action=>action.agent)])].filter(id=>id&&id!=='workflow');
}

function visibleAgentColumns() {
  const agents=trajectoryAgents();
  state.agentColumns=reconcilePinnedAgents(state.run,state.agentColumns,{customized:state.columnsCustomized});
  const columns=state.agent?[state.agent]:state.agentColumns;
  if(!columns.includes(state.activeColumn))state.activeColumn=columns[0]||null;
  return columns;
}

function addAgentColumn(id) {
  if(!trajectoryAgents().includes(id)||state.agentColumns.includes(id))return;
  const pinned=pinAgent(state.run,state.agentColumns,id);
  if(!pinned.changed){$('agent-column-limit').textContent=pinned.reason;return;}
  state.columnsCustomized=true;state.agentColumns=pinned.ids;state.activeColumn=id;
  state.agent='';$('agent-select').value='';
  if(state.view==='profiler'&&!$('content').contains(document.activeElement))$('content').innerHTML=detailedProfilerView();
  if(state.view==='agents')renderTandemBoard();else renderCallStack();
  scrollAgentColumn(id);
}

function removeAgentColumn(id) {
  if(!state.agentColumns.includes(id))return;
  state.columnsCustomized=true;state.agentColumns=state.agentColumns.filter(agent=>agent!==id);if(state.agent===id){state.agent='';$('agent-select').value='';}
  if(state.view==='profiler'&&!$('content').contains(document.activeElement))$('content').innerHTML=detailedProfilerView();
  if(state.view==='agents')renderTandemBoard();else renderCallStack();
}

function selectAgentColumn(id) {
  if(!state.agentColumns.includes(id))return;
  state.agent='';$('agent-select').value='';state.activeColumn=id;if(state.view==='agents')renderTandemBoard();else renderCallStack();
  scrollAgentColumn(id);
}

function scrollAgentColumn(id){
  const board=$(state.view==='agents'?'tandem-columns':'call-stack');
  const column=[...board.children].find(el=>(el.dataset.agentColumn||el.dataset.tandemAgent)===id);
  if(column)board.scrollTo?.({left:board.scrollLeft+column.getBoundingClientRect().left-board.getBoundingClientRect().left,behavior:'smooth'});
}

function agentActivityCounts(id) {
  const actions=state.run.actions.filter(action=>action.agent===id);
  return `${actions.filter(action=>action.outcome==='running').length} running · ${actions.filter(action=>action.outcome==='failed').length} failed`;
}

function renderAgentColumnControls(columns) {
  const agents=trajectoryAgents(),hidden=agents.filter(id=>!state.agentColumns.includes(id));
  const full=state.agentColumns.length>=6;
  const picker=$('add-agent-column');
  const options='<option value="">Add agent…</option>'+hidden.map(id=>`<option value="${esc(id)}">${esc(agentName(id))}</option>`).join('');
  if(picker.innerHTML!==options)picker.innerHTML=options;
  picker.disabled=full||!hidden.length;
  $('agent-column-limit').textContent=`${state.agentColumns.length} of ${agents.length} pinned · up to 6 readable columns`;
  const tabs=columns.map(id=>`<button role="tab" aria-selected="${id===state.activeColumn}" tabindex="${id===state.activeColumn?0:-1}" data-agent-tab="${esc(id)}">${esc(agentName(id))}<small>${agentActivityCounts(id)}</small></button>`).join('');
  if($('agent-tabs').innerHTML!==tabs)$('agent-tabs').innerHTML=tabs;
  renderRoster(state.agentColumns);
  const failures=agentRoster(state.run,{pinnedIds:state.agentColumns}).filter(a=>!a.pinned&&a.failedActionCount>0);
  const badges=failures.slice(0,5).map(a=>`<button data-add-column="${esc(a.id)}" ${full?'disabled':''}>${esc(a.label)} · ${a.failedActionCount} retained call failure${a.failedActionCount===1?'':'s'}</button>`).join('');
  if($('hidden-agents').innerHTML!==badges)$('hidden-agents').innerHTML=badges;
}
function renderRoster(columns=state.agentColumns) {
  const roster=$('roster-list');if(!roster)return;
  const rows=agentRoster(state.run,{pinnedIds:columns,query:state.rosterQuery,status:state.rosterStatus});
  $('roster-count').textContent=`· ${trajectoryAgents().length} agents · ${columns.length} pinned`;
  $('roster-status').value=state.rosterStatus;
  const row=a=>`<div class="roster-row"><button data-agent="${esc(a.id)}" title="${esc(a.id)}">${esc(a.label)}</button>${pill(a.status||'unknown')}<small>${a.activeCount} in flight${a.failedActionCount?` · ${a.failedActionCount} retained call failures`:''}</small><div class="roster-row-actions">${a.pinned?`<button data-agent-tab="${esc(a.id)}">Return</button><button data-remove-column="${esc(a.id)}">Unpin</button>`:`<button data-add-column="${esc(a.id)}" ${columns.length>=6?'disabled':''}>Pin column</button>`}${a.latestFailedAction?`<button data-reveal-action="${esc(a.latestFailedAction.id)}">Inspect failed call</button>`:''}${a.longCalls?.length?`<button data-reveal-action="${esc(a.longCalls[0].action.id)}">Open for ${formatTime(a.longCalls[0].elapsed)} · calculated</button>`:''}</div></div>`;
  const byId=new Map(rows.map(a=>[a.id,a]));
  const rendered=new Set();
  const branch=(a,ancestors=new Set())=>{if(rendered.has(a.id))return '';rendered.add(a.id);const children=(a.childIds||[]).map(id=>byId.get(id)).filter(c=>c&&!ancestors.has(c.id));return row(a)+(children.length?`<details class="roster-children" data-roster-parent="${esc(a.id)}"><summary>${children.length} explicitly recorded child agent${children.length===1?'':'s'}</summary>${children.map(c=>branch(c,new Set([...ancestors,a.id]))).join('')}</details>`:a.parentId?`<small class="muted">Recorded parent: ${esc(agentName(a.parentId))}</small>`:'');};
  const html=rows.filter(a=>!a.parentId||!byId.has(a.parentId)).map(a=>branch(a)).join('')+rows.filter(a=>!rendered.has(a.id)).map(a=>branch(a)).join('');
  if(!roster.contains(document.activeElement)&&roster.innerHTML!==html){const expanded=new Set([...roster.querySelectorAll('[data-roster-parent]')].filter(el=>el.open).map(el=>el.dataset.rosterParent));roster.innerHTML=html||'<p class="muted">No matching agents. Clear the roster filters.</p>';for(const el of roster.querySelectorAll('[data-roster-parent]'))el.open=expanded.has(el.dataset.rosterParent);}

}

function readingState(agent) {
  if(!columnReading.has(agent))columnReading.set(agent,{follow:state.follow,pending:new Set(),ids:null,activeIds:null,scrollTop:0});
  return columnReading.get(agent);
}
function readingPins() {
  const pins=[];
  if(state.selected?.run!=='b'&&state.selected){
    pins.push(state.selected.id);
    if(state.selected.type==='episode')pins.push(...(state.run?.episodes.find(e=>e.id===state.selected.id)?.actions||[]));
    if(state.selected.type==='interval'){const interval=state.run?.intervals.find(i=>i.id===state.selected.id);if(interval?.action)pins.push(interval.action);}
  }
  const focused=document.activeElement?.closest?.('[data-call-card]')?.dataset.callCard;
  if(focused)pins.push(focused);
  const live=currentExecution();
  const agents=state.agentColumns.length?state.agentColumns:(live?.agents||[]).map(a=>a.id).filter(id=>id!=='workflow').slice(0,3);
  for(const agent of agents){
    const r=columnReading.get(agent);if(r)pins.push(...(r.ids||[]),...(r.activeIds||[]));
    // Retain each visible column's live history page as well as its held page,
    // so Jump to live cannot mistakenly relabel an old page as current.
    pins.push(...(live?.actions||[]).filter(a=>a.agent===agent&&a.outcome!=='running').sort((a,b)=>b.start-a.start||(a.id>b.id?1:a.id<b.id?-1:0)).slice(0,50).map(a=>a.id));
  }
  pins.push(...openToolDetails);
  return [...new Set(pins)].slice(0,2000);
}
function renderColumnFollow(column) {
  const r=readingState(column.dataset.agentColumn),heading=column.querySelector('.agent-column-heading');
  const count=heading.querySelector('.column-call-count');
  if(count.dataset.running!=null)count.textContent=`${count.dataset.running} in flight${state.displayPaused?' · frozen':''}`;
  const follow=heading.querySelector('.column-follow');follow.hidden=!state.stream&&r.follow;
  follow.querySelector('span').textContent=state.displayPaused?`View frozen${!r.follow?' · history held':''}`:r.error||(!state.stream?'Recorded history':!r.follow?`${noticeCount(r.pending)} changed · history held`:state.replay?'Following replay':'Following live');
  follow.querySelector('button').textContent=!state.stream?'Newest in column':state.replay?'Latest replay activity':'Live in this column';
  follow.querySelector('button').hidden=r.follow;follow.querySelector('button').disabled=state.displayPaused;
  if(state.displayPaused)for(const selector of ['[data-column-older]','[data-active-older]'])column.querySelector(selector).disabled=true;
}
function captureColumnReading(column) {
  const feed=column.querySelector('.agent-call-stack'),r=readingState(column.dataset.agentColumn);
  r.scrollTop=feed.scrollTop||0;
  const top=feed.getBoundingClientRect().top;
  const visible=[...feed.querySelectorAll('[data-call-card]')].find(card=>card.getBoundingClientRect().bottom>top);
  if(visible)r.anchor={id:visible.dataset.callCard,offset:visible.getBoundingClientRect().top-top};
}
function holdColumn(agent) {
  const r=readingState(agent);r.follow=false;state.follow=false;
  const column=trajectoryColumns.get(agent);if(column)captureColumnReading(column);
  renderSession();
}
function liveColumn(agent) {
  const r=readingState(agent);Object.assign(r,{follow:true,ids:null,activeIds:null,anchor:null,scrollTop:0,more:null,activeMore:null});clearNotice(r.pending);
  const column=trajectoryColumns.get(agent);if(column)column.querySelector('.agent-call-stack').scrollTop=0;
  state.follow=[...columnReading.values()].every(r=>r.follow);
  if(state.follow){clearNotice(state.unread);clearNotice(state.pendingNew);}
  renderCallStack();renderSession();
}
async function olderCalls(agent,active=false) {
  if(state.displayPaused)return;
  const r=readingState(agent),key=active?'activeIds':'ids';
  const map=actionMap(state.run),last=(r[key]||[]).map(id=>map.get(id)).filter(Boolean).at(-1);
  if(!last&&!state.run.coverage.client_window)return;
  holdColumn(agent);r.loading=true;renderCallStack();
  const generation=loadGeneration,execution=executionIdentity(state.run),search=state.query;
  let pageChanged=false;
  try {
    let page;
    if(state.stream&&!state.replay){
      const query=new URLSearchParams({agent,active,query:state.query});
      if(last){query.set('before_start',last.start);query.set('before_id',last.id);}
      page=await fetchJson(`/api/trajectory/live/history?${query}`);
      if(!page.execution||executionIdentity(page.execution)!==execution)throw new Error("History page belongs to a different or unidentified execution. Reconnect updates before paging.");
    } else {
      const calls=visibleActions(state.run).filter(a=>a.agent===agent&&(a.outcome==='running')===active&&(a.start<last.start||a.start===last.start&&a.id>last.id)).sort((a,b)=>b.start-a.start||(a.id>b.id?1:a.id<b.id?-1:0));
      page={actions:calls.slice(0,50),has_more:calls.length>50};
    }
    if(generation!==loadGeneration||state.displayPaused||executionIdentity(state.run)!==execution||state.query!==search)return;
    pageChanged=page.actions.length>0;
    if(page.actions.length){r[key]=page.actions.map(a=>a.id);r.anchor=null;r.scrollTop=0;r.resetPosition=true;mergeTrajectory(state.run,page);}
    r[active?'activeMore':'more']=page.has_more;
    if(!state.replay&&state.stream)boundTrajectory(state.run,readingPins());
  }catch(error){if(generation===loadGeneration)r.error=error.message;}
  finally{r.loading=false;if(generation===loadGeneration&&executionIdentity(state.run)===execution){renderCallStack();const c=trajectoryColumns.get(agent);
    if(pageChanged&&c){const feed=c.querySelector('.agent-call-stack'),section=c.querySelector(active?'.agent-current':'.agent-history');feed.scrollTop+=section.getBoundingClientRect().top-feed.getBoundingClientRect().top;captureColumnReading(c);section.querySelector('[data-action]')?.focus({preventScroll:true});}
    else c?.querySelector('[data-column-live]')?.focus({preventScroll:true});
  }}
}
function renderCallStack() {
  const stack=$('call-stack');if(!stack)return;
  const columns=visibleAgentColumns();renderAgentColumnControls(columns);
  stack.style.setProperty('--agent-column-count',Math.max(1,columns.length));
  const actions=visibleActions(state.run);
  for(const [columnIndex,agent] of columns.entries()) {
    let column=trajectoryColumns.get(agent);
    const r=readingState(agent);
    if(column&&column.parentElement===stack&&!r.resetPosition)captureColumnReading(column);r.resetPosition=false;
    if(!column){
      column=document.createElement('section');column.dataset.agentColumn=agent;
      column.innerHTML=`<header class="agent-column-heading"><strong></strong><span class="column-call-count"></span><button data-column-order="${esc(agent)}" data-direction="-1" aria-label="Move ${esc(agentName(agent))} column left">←</button><button data-column-order="${esc(agent)}" data-direction="1" aria-label="Move ${esc(agentName(agent))} column right">→</button><button data-remove-column="${esc(agent)}" title="Hide column only; agent keeps executing" aria-label="Hide ${esc(agentName(agent))}">Hide</button><div class="agent-execution-controls" data-agent-controls="${esc(agent)}" role="group" aria-label="Authoritative controls for ${esc(agentName(agent))}"></div><div class="column-follow"><span></span><button data-column-live="${esc(agent)}">Live in this column</button></div></header><div class="agent-call-stack" tabindex="0" aria-label="Activity for ${esc(agentName(agent))}"><section class="agent-current"><h3>Current activity</h3><p class="current-coverage"></p><div class="agent-current-calls"></div><button data-active-older="${esc(agent)}">More in-flight calls</button></section><section class="agent-history"><h3>History · newest starts first</h3><div class="agent-history-calls"></div><button data-column-older="${esc(agent)}">Older calls</button></section></div>`;
      trajectoryColumns.set(agent,column);
      const feed=column.querySelector('.agent-call-stack');
      feed.addEventListener('scroll',()=>{captureColumnReading(column);if(feed.scrollTop>2&&r.follow)holdColumn(agent);});
    }
    column.className=`agent-column ${agent===state.activeColumn?'active-agent-column':''}`;
    const heading=column.querySelector('.agent-column-heading');
    heading.querySelector('strong').textContent=agentName(agent);
    const running=actions.filter(a=>a.agent===agent&&a.outcome==='running').length;
    heading.querySelector('.column-call-count').dataset.running=running;
    heading.querySelector('[data-remove-column]').disabled=Boolean(state.agent);
    const controls=heading.querySelector('[data-agent-controls]');controls.hidden=false;
    const controlHtml=executionState().live?agentControlButtons(agent):`<span class="control-section-label">Recorded agent state · ${esc(state.run.agents.find(a=>a.id===agent)?.status||'unknown')}</span><button data-interact="${esc(agent)}">Interaction info</button>`;
    updateAgentControls(controls,controlHtml);
    renderColumnFollow(column);
    if(stack.children[columnIndex]!==column)stack.insertBefore(column,stack.children[columnIndex]||null);
    const feed=column.querySelector('.agent-call-stack');
    const existing=new Map([...feed.querySelectorAll('[data-call-group]')].map(el=>[el.dataset.callGroup,el]));
    const cards=new Map([...feed.querySelectorAll('[data-call-card]')].map(el=>[el.dataset.callCard,el]));
    const focused=document.activeElement;
    const groups=groupCallStack(actions.filter(a=>a.agent===agent));
    const completed=groups.filter(g=>!g.running).map(g=>g.actions[0]);
    const active=groups.filter(g=>g.running).flatMap(g=>g.actions).sort((a,b)=>b.start-a.start||(a.id>b.id?1:a.id<b.id?-1:0));
    if(r.follow||!r.ids)r.ids=completed.slice(0,50).map(a=>a.id);
    if(!r.activeIds||r.follow)r.activeIds=active.slice(0,50).map(a=>a.id);
    const selected=state.selected?.run!=='b'?state.selected?.id:null;
    const historyIds=new Set(r.ids),activeIds=new Set(r.activeIds);
    // Completion migrates an inspected or already-visible active card into
    // its start-ordered history position, preserving its DOM and selection.
    for(const a of completed)if(activeIds.has(a.id))historyIds.add(a.id);
    const focusedId=focused?.closest?.('[data-call-card]')?.dataset.callCard;
    if(selected)historyIds.add(selected);if(focusedId)historyIds.add(focusedId);for(const id of openToolDetails)historyIds.add(id);
    const historyPool=completed.filter(a=>historyIds.has(a.id));
    const visibleHistory=historyPool.slice(0,50);
    for(const a of historyPool)if((a.id===selected||a.id===focusedId||openToolDetails.has(a.id))&&!visibleHistory.includes(a)&&visibleHistory.length<102)visibleHistory.push(a);
    visibleHistory.sort((a,b)=>b.start-a.start||(a.id>b.id?1:a.id<b.id?-1:0));
    r.ids=visibleHistory.map(a=>a.id);
    const visibleActive=active.filter(a=>activeIds.has(a.id)).slice(0,50);
    for(const a of active)if((a.id===selected||a.id===focusedId||openToolDetails.has(a.id))&&!visibleActive.includes(a)&&visibleActive.length<102)visibleActive.push(a);
    r.activeIds=visibleActive.map(a=>a.id);
    const visible=groupCallStack(visibleActive).concat(groupCallStack(visibleHistory));
    const visibleGroups=new Set(),visibleCards=new Set();let currentPosition=0,historyPosition=0;
    for(const group of visible) {
      visibleGroups.add(group.id);let section=existing.get(group.id);
      if(!section){section=document.createElement('section');section.dataset.callGroup=group.id;section.innerHTML='<div class="call-group-heading"></div><div class="call-group-grid"></div>';}
      section.className=`call-stack-group ${group.running?'running-group':''}`;
      section.querySelector('.call-group-heading').hidden=group.actions.length===1&&group.actions[0].kind!=='control';
      section.querySelector('.call-group-heading').textContent=`${group.running?'In flight':group.actions.every(a=>a.kind==='control')?'Recorded intervention':'Finished'}${group.actions.length>1?' · Concurrent calls':''}`;
      const grid=section.querySelector('.call-group-grid');
      for(const [index,action] of group.actions.entries()) {
        visibleCards.add(action.id);let card=cards.get(action.id);
        if(!card){card=document.createElement('div');card.className='call-card';card.dataset.callCard=action.id;card.innerHTML=actionButton(action)+'<div class="call-card-status"></div>'+toolDetails(action,state.run,openToolDetails.has(action.id));cards.set(action.id,card);}
        const template=document.createElement('template');template.innerHTML=actionButton(action);const next=template.content.firstElementChild;
        const button=card.querySelector('[data-action]');button.className=next.className;
        if(button.innerHTML!==next.innerHTML)button.innerHTML=next.innerHTML;
        const status=card.querySelector('.call-card-status');
        const details=card.querySelector('.tool-details'),version=callEvidenceVersion(action);
        if(!card.dataset.evidenceVersion||!details?.querySelector('.tool-detail-body')?.innerHTML)card.dataset.evidenceVersion=version;
        if(card.dataset.evidenceVersion!==version&&details)details.dataset.stale='true';
        const cardSummary=eventCard(action,state.run);
        const html=pill(cardSummary.state.label,action.outcome)+(details?.dataset.stale?`<button data-refresh-details="${esc(action.id)}">New evidence · Refresh details</button>`:'')+(cardSummary.preview?`<p class="event-preview">${esc(cardSummary.preview)}</p>`:'')+(cardSummary.usage.length?`<small class="muted">${esc(cardSummary.usage.join(' · '))}</small>`:'');
        if(status.innerHTML!==html)status.innerHTML=html;
        if(grid.children[index]!==card)grid.insertBefore(card,grid.children[index]||null);
      }
      const target=column.querySelector(group.running?'.agent-current-calls':'.agent-history-calls'),position=group.running?currentPosition++:historyPosition++;
      if(target.children[position]!==section)target.insertBefore(section,target.children[position]||null);
    }
    for(const [id,el] of existing)if(!visibleGroups.has(id))el.remove();
    for(const [id,card] of cards)if(!visibleCards.has(id))card.remove();
    column.querySelector('.current-coverage').textContent=active.length?`${visibleActive.length} retained in-flight calls shown${state.displayPaused?' · frozen observation':''}`:'No in-flight call is recorded in this view.';
    const older=column.querySelector('[data-column-older]');older.disabled=state.displayPaused||r.loading||r.more===false||(!state.run.coverage.client_window&&completed.length<=visibleHistory.length&&!r.more);
    older.hidden=!state.run.coverage.client_window&&completed.length<=visibleHistory.length&&!r.more;
    const more=column.querySelector('[data-active-older]');more.hidden=r.activeMore===false||(!state.run.coverage.client_inflight_windowed&&active.length<=50&&!r.activeMore);more.disabled=state.displayPaused||r.loading;
    if(r.follow&&!state.displayPaused)feed.scrollTop=0;
    else {
      // Browsers may reset a detached scroll container before it is reinserted.
      // Restore its offset before measuring the retained reading anchor.
      feed.scrollTop=r.scrollTop;
      const anchor=[...cards.values()].find(card=>card.dataset.callCard===r.anchor?.id&&visibleCards.has(card.dataset.callCard));
      feed.scrollTop=r.scrollTop+(anchor?anchor.getBoundingClientRect().top-feed.getBoundingClientRect().top-r.anchor.offset:0);
    }
    if(focused&&feed.contains(focused)&&document.activeElement!==focused)focused.focus({preventScroll:true});
  }
  for(const [id,column] of trajectoryColumns)if(!columns.includes(id))column.remove();
  // Keep detached columns for view switches, with a bounded cache for hidden agents.
  for(const [id] of trajectoryColumns)if(trajectoryColumns.size>Math.max(10,columns.length)&&!columns.includes(id)){trajectoryColumns.delete(id);columnReading.delete(id);}
  if(!columns.length)stack.innerHTML=firstExperience();
  else {stack.querySelector('.empty-state')?.remove();stack.querySelector('.first-experience')?.remove();}
}

function relevantEpisodes() {
  const visible=new Set(visibleActions(state.run).map(a=>a.id));
  return state.run.episodes.filter(e=>e.actions.some(id=>visible.has(id)));
}

function reconcileActivities(initial=false) {
  const list=$('activity-list');if(!list)return;
  renderCallStack();
  const run=state.run,map=actionMap(run),episodes=relevantEpisodes();
  const end=state.windowEnd==null?episodes.length:Math.min(state.windowEnd,episodes.length);
  const start=Math.max(0,end-50),window=episodes.slice(start,end).reverse();
  const existing=new Map([...list.querySelectorAll('[data-episode-card]')].map(el=>[el.dataset.episodeCard,el]));
  const ids=new Set(window.map(e=>e.id));
  for(const [id,el] of existing)if(!ids.has(id))el.remove();
  if(episodes.length)list.querySelector('.empty-state')?.remove();
  else if(!list.querySelector('.empty-state'))list.innerHTML=empty(state.agent||state.query?'No matching activities':'Waiting for recorded activity',run.status==='running'?'Execution started. Long-running calls will appear before their results.':'The event stream does not yet establish what the agent is doing.');
  for(const episode of window) {
    let card=existing.get(episode.id);
    if(!card){card=document.createElement('details');card.className='episode activity';card.dataset.episodeCard=episode.id;card.innerHTML=`<span class="episode-number"></span><summary data-episode="${esc(episode.id)}"></summary><div class="activity-evidence"></div><div class="episode-actions"></div>`;card.open=state.expanded.has(episode.id);}
    const position=window.indexOf(episode);
    if(list.children[position]!==card)list.insertBefore(card,list.children[position]||null);
    const actions=episode.actions.map(id=>map.get(id)).filter(Boolean).reverse();
    card.classList.toggle('selected',state.selected?.run!=='b'&&(state.selected?.id===episode.id||actions.some(a=>a.id===state.selected?.id)));
    card.classList.toggle('new-evidence',actions.some(a=>state.unread.has(a.id)));
    card.querySelector('.episode-number').textContent=run.episodes.indexOf(episode)+1;
    const summary=`<span class="chevron">›</span><div class="episode-title"><h3>${esc(episode.title)}</h3><p>${esc(agentName(episode.agent))} · ${episode.label_source==='declared'?'Declared purpose':'Inferred activity'}${episode.late?' · late telemetry':''}</p><div class="activity-result"><span>${episode.kind==='model'?'Model reply':'Observed'}</span> ${esc(episode.latest_result||'No result recorded yet.')}</div></div>${pill(episode.status)}${pausedBadge(episode.agent)}`;
    const summaryNode=card.querySelector('summary');if(summaryNode.innerHTML!==summary)summaryNode.innerHTML=summary;
    const annotation=episode.annotation?`<p class="notice">Suggested label: ${esc(episode.annotation.label)} · ${esc(episode.annotation.source)}. ${esc(episode.annotation.summary||'')}</p>`:'';
    const evidence=annotation+actions.flatMap(a=>a.artifacts||[]).map(a=>`<span class="file-chip">${esc(a.path||a.name||'Artifact')} · ${esc(a.change||'recorded reference')}</span>`).join('');
    const evidenceNode=card.querySelector('.activity-evidence');if(evidenceNode.innerHTML!==evidence)evidenceNode.innerHTML=evidence;
    const calls=card.querySelector('.episode-actions');
    const oldCalls=new Map([...calls.children].map(el=>[el.dataset.action,el]));
    for(const [position,action] of actions.entries()) {
      let button=oldCalls.get(action.id);
      const template=document.createElement('template');template.innerHTML=actionButton(action);
      const next=template.content.firstElementChild;
      if(!button)button=next;
      else{button.className=next.className;if(button.innerHTML!==next.innerHTML)button.innerHTML=next.innerHTML;}
      if(calls.children[position]!==button)calls.insertBefore(button,calls.children[position]||null);
    }
  }
  $('activity-window-label').textContent=episodes.length?`${start+1}–${end} of ${episodes.length} activities · newest first`:'No activities yet';
  $('content').querySelector('[data-older-activities]').disabled=start===0;
  const active=run.actions.filter(a=>a.outcome==='running'&&(!state.agent||a.agent===state.agent));
  const input=(run.signals||[]).find(signal=>signal.active!==false&&signal.severity==='required');
  const latest=run.actions.filter(a=>a.result_preview||a.result).reduce((last,a)=>!last||a.end_ts>=last.end_ts?a:last,null);
  const current=active.length?active.slice(-3).map(a=>`<div><b>${esc(agentName(a.agent))}</b><span>${esc(a.intent||run.episodes.find(e=>e.id===a.episode)?.title||a.name)}</span><small>${esc(a.kind)} · ${state.run.agents.find(agent=>agent.id===a.agent)?.status==='paused'?'paused at':'running for'} ${formatTime(displayDuration(a))}</small></div>`).join(''):`<p>${input?'Waiting for requested human input. '+esc(input.explanation):run.status==='completed'?'Execution finished. Review the returned evidence; task correctness may remain unverified.':run.status==='cancelled'?'Execution cancelled. Unfinished spans retain incomplete telemetry.':run.status==='failed'?'Execution reported a failure. Inspect the evidence below.':state.replayState?.finished?'Recording exhausted. The final execution state was not recorded.':'No active call is recorded. A quiet stream alone does not establish a stall.'}</p>`;
  const glance=`<section class="trajectory-glance">${run.task?`<div class="detail-label">Requested task</div><p class="requested-task">${esc(String(run.task).slice(0,400))}</p>`:''}<div class="detail-label activity-overview-label">${!state.follow&&state.stream?'Overview held at selection':active.length?'Currently recorded':'Current situation'}</div><div class="current-activities">${current}</div>${latest?`<div class="latest-evidence"><span>${latest.kind==='model'?'Latest visible model reply':'Latest returned evidence'}</span><button data-action="${esc(latest.id)}">${esc((latest.result_preview||latest.result).slice(0,180))} ↗</button></div>`:''}</section>`;
  if((state.follow||initial)&&$('trajectory-glance').innerHTML!==glance)$('trajectory-glance').innerHTML=glance;
  const activeSignals=(run.signals||[]).filter(s=>s.active!==false);
  const repeatedEvidence=new Set(activeSignals.filter(s=>s.title==='Repeated matching failures').flatMap(s=>s.evidence));
  const signals=activeSignals.filter(s=>s.title!=='Recorded call failure'||!s.evidence.every(id=>repeatedEvidence.has(id)));
  const gap=(run.coverage.catching_up?'<div class="notice">Recovering retained events. This is a partial history until catch-up completes.</div>':'')+(run.coverage.gaps?.length?`<div class="attention-signal"><strong>Telemetry coverage is incomplete</strong><p>${esc(run.coverage.gaps.join(' · '))}. Execution state may be unknown.</p></div>`:'');
  const signalHtml=gap+signals.slice(-5).map(signal=>`<div class="attention-signal ${signal.severity==='required'?'required':''}"><strong>${signal.severity==='required'?'Action requested':'Worth inspecting'} · ${esc(signal.title)}</strong><p>${esc(signal.explanation)}</p><div>${signal.evidence.map((id,i)=>`<button ${map.has(id)?'data-action':'data-raw-event'}="${esc(id)}">Evidence ${i+1} ↗</button>`).join('')}</div></div>`).join('');
  if((state.follow||initial)&&$('trajectory-signals').innerHTML!==signalHtml&&!$('trajectory-signals').contains(document.activeElement))$('trajectory-signals').innerHTML=signalHtml;
  if(state.follow||initial){
    $('trajectory-attention').hidden=!signalHtml;
    $('signal-count').textContent=`· ${signals.length}${gap?' · incomplete coverage':''}${input?' · action requested':''}`;
    const requiredKey=input?.id||input?.title;
    if(requiredKey&&requiredKey!==$('trajectory-attention').dataset.required)$('trajectory-attention').open=true;
    $('trajectory-attention').dataset.required=requiredKey||'';
  }
  const workflow=trajectoryWorkflow();
  if((state.follow||initial)&&$('trajectory-workflow').innerHTML!==workflow)$('trajectory-workflow').innerHTML=workflow;
}

function trajectoryWorkflow() {
  return state.run.edges.filter(e=>!state.agent||e.from===state.agent||e.to===state.agent).slice(-3).map(edge=>`<div class="workflow-note"><span>${esc(agentName(edge.from))} → ${esc(agentName(edge.to))}</span><p>${esc(edge.label)}</p><button data-view="agents">${state.run.source==='synthetic'?'Fixture':'Recorded'} ${esc(edge.kind)} ↗</button></div>`).join('');
}

function trajectoryOverlay(actions, key) {
  if(!actions.length) return empty('No matching actions','Clear the search or select a different agent.');
  const width=900,left=156,right=14,plot=width-left-right;
  const duration=Math.max(state.run.duration,1),x=t=>left+t/duration*plot;
  const grouped=new Map();
  for(const action of actions) {
    if(!grouped.has(action.agent)) grouped.set(action.agent,[]);
    grouped.get(action.agent).push(action);
  }
  const lanes=[...grouped].map(([agent,items])=>({agent,items,tracks:packActionTracks(items,8/plot*duration)}));
  let height=36;
  for(const lane of lanes) {lane.y=height;lane.height=lane.tracks.length*30+36;height+=lane.height;}
  height+=14;
  const selectedEpisode=state.selected?.run!=='b'&&state.selected?.type==='episode'?state.run.episodes.find(e=>e.id===state.selected.id):null;
  let svg=`<svg class="timeline-svg trajectory-overlay" viewBox="0 0 ${width} ${height}" role="group" aria-label="Agent trajectories on a shared time axis">`;
  svg+=Array.from({length:6},(_,index)=>{
    const time=state.run.duration*index/5;
    return `<line class="gridline" x1="${x(time)}" x2="${x(time)}" y1="25" y2="${height-14}"/><text x="${x(time)}" y="14" text-anchor="${index===5?'end':'middle'}">${formatTime(time)}</text>`;
  }).join('');
  for(const lane of lanes) {
    const name=agentName(lane.agent),label=name.length>21?name.slice(0,20)+'…':name;
    svg+=`<g class="trajectory-agent"><title>${esc(name)}</title><text class="lane-label" x="0" y="${lane.y+18}">${esc(label)}</text><text x="0" y="${lane.y+35}">${lane.items.length} action${lane.items.length===1?'':'s'}</text></g>`;
    lane.tracks.forEach((track,index)=>{
      const y=lane.y+index*30;
      svg+=`<line class="trajectory-path" x1="${x(track[0].start)}" x2="${x(track.at(-1).start+track.at(-1).duration)}" y1="${y+12}" y2="${y+12}"/>`;
      for(const action of track) {
        const w=Math.max(8,x(action.start+action.duration)-x(action.start));
        const selected=state.selected?.run!=='b'&&(state.selected?.id===action.id||selectedEpisode?.actions.includes(action.id));
        const description=`${name} · ${action.intent||action.name} · ${formatTime(action.start)} → ${formatTime(action.start+action.duration)} · ${action.outcome}`;
        svg+=`<g class="trajectory-action ${selected?'selected':''} ${action.outcome==='failed'?'failed':''}" data-action="${esc(action.id)}" tabindex="0" role="button" aria-label="${esc(description)}"><title>${esc(description)}</title><rect x="${x(action.start)}" y="${y}" width="${w}" height="24" rx="4" fill="${colors[action.kind]||colors.tool}"/>${w>action.kind.length*6+14?`<text x="${x(action.start)+7}" y="${y+16}">${esc(action.kind)}</text>`:''}</g>`;
      }
    });
    svg+=`<line class="trajectory-divider" x1="0" x2="${width}" y1="${lane.y+lane.height-15}" y2="${lane.y+lane.height-15}"/>`;
  }
  svg+=`<line class="cursor-line" x1="${x(state.cursor)}" x2="${x(state.cursor)}" y1="24" y2="${height-14}"/></svg>`;
  return insight('<strong>See who worked in parallel.</strong> Every agent shares the same time axis. Select an action to inspect its command, result and episode.')+
    `<div class="panel"><div class="panel-heading"><h3>Agent trajectories</h3><span class="mono muted">${lanes.length} AGENTS · ${actions.length} ACTIONS</span></div>${key}<div class="trajectory-scroll" tabindex="0" role="region" aria-label="Scrollable agent trajectory chart">${svg}</div><div class="trajectory-caption">Bar width = wall interval · Duration labels exclude recorded execution pauses · Dashed line = time between actions · Light vertical line = shared time cursor</div></div>`+
    '<div class="notice">Overlapping actions within an agent use separate tracks. Timing and agent ownership come from the selected execution; concurrency alone does not establish delegation.</div>';
}

function timeline(intervals, agents, compact=false) {
  const [start,end]=state.zoom || [0,state.run.duration];
  const width=760,left=130,right=12,plot=width-left-right;
  const x=t=>left+(t-start)/(end-start||1)*plot;
  const kinds=compact?['all']:['model','tool','cpu','queue','dependency','io','unknown'];
  const lanes=[];
  for (const agent of agents) {
    for (const kind of kinds) {
      const items=intervals.filter(i=>i.agent===agent.id && (kind==='all'||i.kind===kind) && i.start+i.duration>=start && i.start<=end);
      if(items.length){
        const tracks=compact?packActionTracks(items,(end-start)/plot*3):[items];
        tracks.forEach((track,n)=>lanes.push({label:kind==='all'?`${agent.label}${n?' / '+(n+1):''}`:`${agent.label.slice(0,16)} / ${kind}`,items:track}));
      }
    }
  }
  const height=32+lanes.length*32+20;
  const ticks=Array.from({length:6},(_,i)=>start+(end-start)*i/5);
  let svg=`<svg class="timeline-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Execution intervals on a shared time axis">`;
  svg+=ticks.map(t=>`<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="22" y2="${height-14}"/><text x="${x(t)}" y="12" text-anchor="${t===end?'end':'middle'}">${formatTime(t)}</text>`).join('');
  lanes.forEach((lane,index)=>{
    const y=30+index*32;
    svg+=`<text class="lane-label" x="0" y="${y+13}">${esc(lane.label)}</text>`;
    svg+=lane.items.map(i=>{
      const a=Math.max(start,i.start),b=Math.min(end,i.start+i.duration);
      const selected=state.selected?.run!=='b'&&(state.selected?.id===i.id || state.selected?.interval===i.id || state.selected?.type==='episode' && state.run.episodes.find(e=>e.id===state.selected.id)?.actions.includes(i.action));
      return `<rect class="event ${selected?'selected':''}" ${i.direct_action?`data-action="${esc(i.action)}"`:`data-interval="${esc(i.id)}"`} x="${x(a)}" y="${y}" width="${Math.max(2,x(b)-x(a))}" height="20" rx="3" fill="${colors[i.kind]||colors.tool}" opacity="${i.kind==='unknown'?.4:.85}" tabindex="0" role="button" aria-label="${esc(i.label)}, ${formatTime(i.duration)}"><title>${esc(i.label)} · ${formatTime(i.duration)} · ${esc(i.source)}</title></rect>`;
    }).join('');
  });
  if (state.cursor>=start&&state.cursor<=end) svg+=`<line class="cursor-line" x1="${x(state.cursor)}" x2="${x(state.cursor)}" y1="20" y2="${height-10}"/>`;
  return svg+'</svg>';
}

function sampleChart(name, samples, color) {
  const [start,end]=state.zoom || [0,state.run.duration];
  const data=samples.filter(p=>p[0]>=start&&p[0]<=end);
  if (!data.length) return empty('No samples in this interval','Reset the timeline zoom to see available observations.');
  const max=Math.max(1,...data.map(p=>p[1])), width=400,height=115;
  const x=t=>35+(t-start)/(end-start||1)*350, y=v=>90-v/max*70;
  const points=data.map(p=>`${x(p[0])},${y(p[1])}`).join(' ');
  return `<div class="chart-title">${esc(name)} · peak ${count(max)}</div><svg class="timeline-svg sample-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(name)} sampled utilization" data-chart><text x="1" y="22">${count(max)}</text><text x="13" y="93">0</text><line class="gridline" x1="35" x2="385" y1="90" y2="90"/><polygon points="35,90 ${points} ${x(data.at(-1)[0])},90" fill="${color}" opacity=".1"/><polyline points="${points}" stroke="${color}" fill="none" stroke-width="1.7"/><line class="cursor-line" x1="${x(state.cursor)}" x2="${x(state.cursor)}" y1="10" y2="94"/><text x="35" y="109">${formatTime(start)}</text><text x="385" y="109" text-anchor="end">${formatTime(end)}</text></svg>`;
}

function resourcesView() {
  const run=state.run, agents=run.agents.filter(a=>!state.agent||a.id===state.agent);
  const ids=new Set(visibleActions(run).map(a=>a.id));
  const intervals=run.intervals.filter(i=>(!state.agent||i.agent===state.agent)&&(!state.query||ids.has(i.action)));
  const waits=intervals.filter(i=>['model','queue','dependency','unknown'].includes(i.kind)).sort((a,b)=>b.duration-a.duration).slice(0,6);
  const counters=Object.entries(run.counters);
  const cpu=counters.find(([key])=>/workload.*cpu|cpu.*%/i.test(key));
  const memory=counters.find(([key])=>/workload.*memory|memory_mb|rss_mb/i.test(key));
  return heading('Resources & waits','Action-aligned intervals and sampled machine observations.',`<span class="mono">${state.zoom?'FOCUSED INTERVAL':'FULL EXECUTION'}</span>`)+
    insight('<strong>Correlate a wait with its cause.</strong> Select a span to inspect the model, tool or dependency. Unattributed intervals are coverage gaps.')+
    `<div class="panel">${legend([['model','Model / API'],['tool','Tool execution'],['cpu','CPU work'],['queue','Scheduler queue'],['dependency','Dependency'],['unknown','Unattributed']])}${intervals.length?timeline(intervals,agents):empty('No intervals match','Try clearing filters.')}<div class="chart-controls"><button data-zoom>Zoom to selection</button><button data-reset-zoom>Full execution</button><span class="muted mono">${state.zoom?`${formatTime(state.zoom[0])} → ${formatTime(state.zoom[1])}`:'Select a span, then focus its interval'}</span></div></div>`+
    `<div class="two-columns">${cpu?`<div class="panel">${sampleChart(cpu[0],cpu[1],colors.cpu)}</div>`:empty('CPU samples unavailable','No utilization counter was recorded in this trace.')}${memory?`<div class="panel">${sampleChart(memory[0],memory[1],colors.tool)}</div>`:empty('Memory samples unavailable','No memory counter was recorded in this trace.')}</div>`+
    heading('Longest waits & gaps','Ranked intervals; nested or parallel spans may overlap.')+
    waits.map(i=>`<button class="wait-item" ${i.direct_action?`data-action="${esc(i.action)}"`:`data-interval="${esc(i.id)}"`}><span style="color:${colors[i.kind]}">▰</span><span class="wait-text"><strong>${esc(i.label)}</strong><small>${esc(agentName(i.agent))} · ${formatTime(i.start)} → ${formatTime(i.start+i.duration)}</small></span><span class="mono">${formatTime(i.duration)}</span></button>`).join('')+
    `<div class="notice">Model spans include request processing and transport. Thread CPU counters are inclusive and can overlap. Samples may miss brief peaks; CPU percentages can exceed 100% across multiple cores.</div>`;
}

function activeModel() {
  const calls=visibleActions(state.run).filter(a=>a.kind==='model');
  return calls.find(a=>a.id===state.model) || calls.find(a=>state.selected?.run!=='b'&&a.id===state.selected?.id) || calls.filter(a=>a.start<=state.cursor).at(-1) || calls[0];
}

function contextView() {
  const calls=visibleActions(state.run).filter(a=>a.kind==='model'), call=activeModel();
  if (!call) return heading('Context & information','The information boundary at each model call.')+empty('No model calls match','Choose another agent or clear the event search.');
  const context=call.context;
  let detail='';
  if (!context) detail=empty('Prompt composition was not recorded','Timing and token usage may be available. Capture llm.messages or exchange prompt chains to inspect provenance.');
  else {
    const totals={};
    context.blocks.forEach(b=>totals[b.source]=(totals[b.source]||0)+b.chars);
    const added=context.blocks.filter(b=>b.change==='added').reduce((sum,b)=>sum+b.chars,0);
    detail=`<div class="panel"><div class="panel-heading"><h3>Information composition</h3>${pill(context.source==='synthetic'?'Synthetic':'Recorded',context.source)}</div><p class="muted" style="font-size:11px">${count(context.chars)} characters · ${call.tokens==null?'token usage unavailable':`${count(call.tokens)} reported input tokens`}</p><div class="composition-bar">${Object.entries(totals).map(([source,total])=>`<button style="flex:${total};background:${sources[source]||colors.setup}" data-source="${esc(source)}" aria-label="${esc(source)}: ${count(total)} characters" title="${esc(source)} · ${count(total)} characters"></button>`).join('')}</div><div class="source-breakdown">${Object.entries(totals).map(([source,total])=>`<div><span style="color:${sources[source]||colors.setup}">${esc(source)}</span><span class="mono">${count(total)} chars</span></div>`).join('')}</div>${field('Added since previous call',context.comparison_available===false?'Unknown (comparison not recorded)':`${count(added)} chars`)}${field('Repeated content',context.comparison_available===false?'Unknown (comparison not recorded)':`${count(context.chars-added)} chars`)}${field('Dropped blocks',context.comparison_available===false?'Unknown (comparison not recorded)':context.truncated?'Unknown (truncated prompt)':context.dropped.length)}${context.truncated?'<p class="notice">The recorded prompt was truncated; this composition is incomplete.</p>':''}</div>`;
    detail+=heading('Prompt provenance','Select a block to inspect its source and content.');
    detail+=context.blocks.concat(context.dropped).map((b,index)=>`<button class="context-block" data-block="${index}" data-model-id="${esc(call.id)}"><div><span>${esc(b.source)} <small>· ${esc(b.role)}</small></span><span class="change ${b.change}">${b.change==='added'?'+':b.change==='dropped'?'−':b.change==='recorded'?'':'↻'} ${b.change} · ${count(b.chars)} chars</span></div><p>${esc(b.preview)}</p></button>`).join('');
  }
  const handoff=state.run.handoff;
  const boundary=handoff?`<div class="panel" style="margin-top:22px">${heading('Supervisor → worker boundary',`${agentName(handoff.from)} → ${agentName(handoff.to)} · synthetic handoff example`)}<div class="two-columns"><div class="handoff-box"><h3 style="color:var(--green)">Transferred into worker prompt</h3>${handoff.transferred.map(t=>`<p>↳ ${esc(t)}</p>`).join('')}</div><div class="handoff-box retained"><h3 style="color:var(--amber)">Retained by supervisor</h3>${handoff.retained.map(t=>`<p>− ${esc(t)}</p>`).join('')}</div></div></div>`:`<div class="notice">Supervisor/worker transfer manifests are unavailable for this execution. Recorded parent relationships alone do not establish which information crossed the boundary.</div>`;
  return heading('Context & information','Prompt composition and changes across observed model calls.',`<span class="mono">${calls.length} MODEL CALLS</span>`)+
    `<div class="context-layout"><div class="call-list">${calls.map((a,i)=>`<button class="call-button ${call.id===a.id?'selected':''}" data-model="${esc(a.id)}">Call ${i+1} · ${esc(agentName(a.agent))}<small>${formatTime(a.start)} · ${count(a.tokens)} tokens</small></button>`).join('')}</div><div>${detail}</div></div>`+boundary+`<div class="notice">Composition uses recorded character counts, not estimated token percentages. Added, repeated and dropped labels compare exact content hashes against the preceding recorded prompt for that agent. A summary label is inferred from text.</div>`;
}

function graphConnectors(agents, edges) {
  const positions=new Map(agents.map((a,index)=>[a.id,index===0?[330,50]:[index%2?165:495,200+Math.floor((index-1)/2)*150]]));
  const paths=edges.filter(e=>e.kind==='delegation'&&positions.has(e.from)&&positions.has(e.to)).map(e=>{
    const [sx,sy]=positions.get(e.from),[tx,ty]=positions.get(e.to),mid=(sy+ty)/2;
    return `<path d="M${sx} ${sy+35} C${sx} ${mid},${tx} ${mid},${tx} ${ty-40}" stroke="#3c596b" fill="none" stroke-width="1.5" stroke-dasharray="4 5"/>`;
  }).join('');
  return paths?`<svg viewBox="0 0 660 ${100+Math.ceil((agents.length-1)/2)*150}" preserveAspectRatio="none">${paths}</svg>`:'';
}

// Adapted from the supplied tandem_trace_board.html / board.css. Recorded
// ownership and results drive the board; concurrency never implies delegation.
function tandemEpisodeHTML(episode, actions) {
  return `<summary><span>${esc(episode.title)}</span>${pill(episode.status)}${pausedBadge(episode.agent)}</summary><div class="seg-items">${actions.map(a=>`<article class="item ${a.kind==='model'?'text':'tool_call'} ${a.outcome==='failed'?'trace-failed':''}"><div class="lbl"><button data-action="${esc(a.id)}">${esc(a.kind)} · ${esc(a.name)}</button><span>${formatTime(a.start)} · ${formatTime(displayDuration(a))}</span></div>${a.intent?`<p>${esc(a.intent)}</p>`:''}${a.command?readablePayload(a.arguments||a.command,400):''}${a.result?`<div class="tool-result"><span class="lbl">Observed result</span>${readablePayload(a.result,400)}</div>`:''}<div class="trace-metrics">${pill(a.outcome)}${a.tokens!=null?`<span>${count(a.tokens)} input · ${count(a.output_tokens)} output tokens</span>`:''}<button data-action="${esc(a.id)}">Inspect evidence ↗</button></div>${toolDetails(a,state.run,openToolDetails.has(a.id))}</article>`).join('')}</div>`;
}

function tandemMetricTable(agents) {
  return `<div class="mtable-wrap"><table class="mtable"><caption>Token & time metrics · observed calls</caption><thead><tr><th>Agent</th><th>Calls</th><th>Running</th><th>Model time¹</th><th>Input tokens</th><th>Output tokens</th><th>Usage coverage</th></tr></thead><tbody>${agents.map(id=>{
    const actions=visibleActions(state.run).filter(a=>a.agent===id),models=actions.filter(a=>a.kind==='model'),known=models.filter(a=>a.tokens!=null);
    const duration=overlapDuration(models,0,state.run.duration);
    return `<tr><th scope="row">${esc(agentName(id))}</th><td>${actions.length}</td><td>${actions.filter(a=>a.outcome==='running').length}</td><td>${formatTime(duration)}</td><td>${known.length?count(known.reduce((n,a)=>n+a.tokens,0)):'n/a'}</td><td>${models.some(a=>a.output_tokens!=null)?count(models.reduce((n,a)=>n+(a.output_tokens||0),0)):'n/a'}</td><td>${known.length}/${models.length}</td></tr>`;
  }).join('')}</tbody></table></div><p class="notice">¹ Union of model-call intervals for each agent. Token counts sum reported usage; they do not estimate unique context or cache savings.</p>`;
}

function systemMetricTimelines() {
  const [start,end]=state.zoom||[0,state.run.duration],width=760,left=130,right=12,plot=width-left-right;
  const x=t=>left+(t-start)/(end-start||1)*plot;
  const series=Object.entries(state.run.counters||{}).filter(([,samples])=>Array.isArray(samples));
  if(!series.length)return empty('System metrics unavailable','This execution has no recorded resource samples. Enable profiling for future runs; tool-call timestamps alone cannot measure CPU or memory.');
  return series.map(([name,samples],index)=>{
    const data=samples.filter(p=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1])&&p[0]>=start&&p[0]<=end).sort((a,b)=>a[0]-b[0]);
    if(!data.length)return `<div class="metric-track-empty">${esc(name)} · no samples in the selected interval</div>`;
    const low=Math.min(0,...data.map(p=>p[1])),high=Math.max(1,...data.map(p=>p[1])),y=v=>70-(v-low)/(high-low)*44;
    const color=[colors.cpu,colors.io,colors.queue][index%3],points=data.map(p=>`${x(p[0])},${y(p[1])}`).join(' ');
    const ticks=Array.from({length:6},(_,i)=>start+(end-start)*i/5);
    return `<div class="system-metric-track"><div class="chart-title">${esc(name)} <span class="muted">· ${data.length} samples · peak ${count(Math.max(...data.map(p=>p[1])))}</span></div><svg class="timeline-svg" viewBox="0 0 ${width} 96" role="img" aria-label="${esc(name)} on the shared execution clock">${ticks.map(t=>`<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="20" y2="72"/><text x="${x(t)}" y="90" text-anchor="${t===end?'end':'middle'}">${formatTime(t)}</text>`).join('')}<text x="0" y="30">${count(high)}</text><text x="0" y="70">${count(low)}</text><polygon points="${x(data[0][0])},70 ${points} ${x(data.at(-1)[0])},70" fill="${color}" opacity=".12"/><polyline points="${points}" stroke="${color}" stroke-width="1.7" fill="none"/>${data.map(p=>`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="2" fill="${color}"><title>${esc(name)}: ${p[1]} at ${formatTime(p[0])}</title></circle>`).join('')}<line class="cursor-line" x1="${x(Math.max(start,Math.min(end,state.cursor)))}" x2="${x(Math.max(start,Math.min(end,state.cursor)))}" y1="18" y2="74"/></svg></div>`;
  }).join('');
}

function tandemConcurrentWork(agents) {
  // Canonical live actions carry timing even when profiler spans are absent.
  const actions=visibleActions(state.run).filter(a=>agents.some(agent=>agent.id===a.agent));
  const intervals=actions.map(a=>({id:a.id,action:a.id,agent:a.agent,start:a.start,duration:a.duration,kind:a.kind==='model'?'model':'tool',label:a.intent||a.name,direct_action:true,source:a.source}));
  return timeline(intervals,agents,true);
}

function agentsView() {
  return heading('Tandem trace board','Side-by-side agent traces, recorded handoffs and a shared execution clock.')+
    `<div class="tandem-board"><div class="task-box"><span class="k">Requested task</span><p>${esc(state.run.task||state.run.title)}</p></div><div class="cat-legend"><span class="supervisor-key">● Supervisor / parent</span><span class="worker-key">● Worker</span><span>Roles follow recorded ownership</span></div><div class="agent-column-controls"><div id="agent-tabs" role="tablist" aria-label="Visible agents"></div><label>Add agent<select id="add-agent-column" aria-label="Add agent column"></select></label><span id="agent-column-limit" class="muted"></span></div><div id="hidden-agents"></div><div id="tandem-metrics"></div><div class="section-heading"><h3>Agent transcripts</h3><span class="muted">Newest activity first · expand to inspect calls</span></div><div id="tandem-columns" class="board agent-columns"></div><div class="panel"><div class="panel-heading"><h3>Concurrent work</h3><span class="mono muted">SHARED TIME AXIS</span></div>${legend([['model','Model'],['tool','Tool']])}<div id="tandem-concurrent"></div><div class="panel-heading"><h3>System metrics</h3><span class="mono muted">SAME TIME AXIS</span></div><div id="tandem-system-metrics"></div><p class="notice">Sampled counters retain their recorded scope and units. Overlap shows concurrency; it does not establish causality.</p></div><div id="tandem-handoffs"></div></div>`;
}

function renderTandemBoard() {
  const host=$('tandem-columns');if(!host)return;
  const focused=document.activeElement?.getAttribute('data-action');
  const columns=visibleAgentColumns(),run=state.run;
  renderAgentColumnControls(columns);
  host.dataset.run=run.id;
  host.style.setProperty('--agent-column-count',columns.length||1);
  const agents=columns.map(id=>run.agents.find(a=>a.id===id)||{id,label:id});
  const existing=new Map([...host.children].map(el=>[el.dataset.tandemAgent,el]));
  for(const [id,el] of existing)if(!columns.includes(id))el.remove();
  agents.forEach((agent,index)=>{
    let col=existing.get(agent.id);
    if(!col){col=document.createElement('section');col.dataset.tandemAgent=agent.id;col.className='col';col.innerHTML='<header class="col-head"></header><div class="col-body"></div>';}
    col.classList.toggle('active-agent-column',agent.id===state.activeColumn);
    if(host.children[index]!==col)host.insertBefore(col,host.children[index]||null);
    const parent=run.agents.some(a=>a.parent===agent.id),role=parent?'Supervisor':agent.parent?'Worker':'Agent';
    col.classList.toggle('supervisor-column',parent);
    const header=`<div class="agent"><button data-agent="${esc(agent.id)}">${esc(agent.label)}</button><button data-remove-column="${esc(agent.id)}" ${state.agent||columns.length===1?'disabled':''}>Hide</button></div><div class="meta"><span class="badge ${parent?'sup':'wrk'}">${role}</span><span class="badge">${esc(agent.model||run.model)}</span>${pausedBadge(agent.id)}</div><p class="muted">${agentActivityCounts(agent.id)}${agent.parent?` · parent ${esc(agentName(agent.parent))}`:''}</p>`;
    const head=col.querySelector('.col-head');
    if(!head.controlsMounted){head.innerHTML=`<div class="tandem-agent-heading"></div><div class="agent-execution-controls" data-agent-controls="${esc(agent.id)}" role="group" aria-label="Controls for ${esc(agent.label)}"></div>`;head.controlsMounted=true;}
    if(head.agentMarkup!==header){head.querySelector('.tandem-agent-heading').innerHTML=header;head.agentMarkup=header;}
    const controls=head.querySelector('[data-agent-controls]');controls.hidden=!executionState(agent.id).live;
    if(!controls.hidden)updateAgentControls(controls,agentControlButtons(agent.id));
    const body=col.querySelector('.col-body'),actions=visibleActions(run).filter(a=>a.agent===agent.id),map=new Map(actions.map(a=>[a.id,a])),ids=new Set(map.keys());
    const episodes=run.episodes.filter(e=>e.agent===agent.id&&e.actions.some(id=>ids.has(id))).slice().reverse();
    const old=new Map([...body.children].map(el=>[el.dataset.tandemEpisode,el]));
    for(const [id,el] of old)if(!episodes.some(e=>e.id===id))el.remove();
    episodes.forEach((episode,n)=>{
      let seg=old.get(episode.id);if(!seg){seg=document.createElement('details');seg.className='seg';seg.dataset.tandemEpisode=episode.id;seg.open=episode.status==='running';}
      const html=tandemEpisodeHTML(episode,episode.actions.filter(id=>ids.has(id)).map(id=>map.get(id)).reverse());
      if(seg.innerHTML!==html){
        const scroll=[...seg.querySelectorAll('pre')].map(el=>[el.scrollTop,el.scrollLeft]);
        seg.innerHTML=html;
        [...seg.querySelectorAll('pre')].forEach((el,i)=>{if(scroll[i])[el.scrollTop,el.scrollLeft]=scroll[i];});
      }
      if(body.children[n]!==seg)body.insertBefore(seg,body.children[n]||null);
    });
  });
  for(const [id,html] of [['tandem-metrics',tandemMetricTable(columns)],['tandem-concurrent',tandemConcurrentWork(agents)],['tandem-system-metrics',systemMetricTimelines()],['tandem-handoffs',heading('Communication & handoffs','Explicitly recorded relationships.')+(run.edges.length?run.edges.map(edge=>`<button class="communication-item" ${edge.action?`data-action="${esc(edge.action)}"`:`data-agent="${esc(edge.to)}"`}><span>${edge.time==null?'—':formatTime(edge.time)}</span><div><strong>${esc(agentName(edge.from))} → ${esc(agentName(edge.to))}</strong><p>${esc(edge.label)} · ${esc(edge.source)}</p></div></button>`).join(''):empty('No handoff events','Parallel execution does not imply delegation.'))]]) {
    const node=$(id);if(node.innerHTML!==html)node.innerHTML=html;
  }
  if(focused)host.querySelector(`[data-action="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
}

function chooseComparison() {
  const run=state.run;
  const counterpart=state.catalog.find(r=>r.id!==run.id&&r.task&&r.task===run.task&&r.condition!==run.condition&&r.repetition===run.repetition);
  return counterpart?.id || state.catalog.find(r=>r.id!==run.id&&r.source===run.source)?.id;
}

async function loadComparison(id=state.compareId||chooseComparison()) {
  if (!id) return;
  const generation=++compareGeneration;
  state.compareId=id; state.comparison=null;
  if(state.view==='compare') render();
  try {
    const run=await fetchJson(`/api/investigator/runs/${encodeURIComponent(id)}`);
    if(generation!==compareGeneration||state.compareId!==id) return;
    state.comparison=run;
    if(state.view==='compare') render();
  } catch(error) {
    if(generation===compareGeneration) {state.compareError=error.message;if(state.view==='compare') render();}
  }
}

function compareView() {
  const a=state.run,b=state.comparison;
  const controls=`<div class="comparison-controls"><span class="muted">Compare against</span><select id="comparison-select" aria-label="Comparison execution">${state.catalog.filter(r=>r.id!==a.id).map(r=>`<option value="${esc(r.id)}" ${r.id===state.compareId?'selected':''}>${esc(r.title)} · ${esc(r.subtitle)}</option>`).join('')}</select></div>`;
  if(!b) return heading('Compare executions','Align trajectories to find the first divergence.')+controls+(state.compareError?empty('Comparison could not load',state.compareError):'<div class="loading-state">Aligning the second execution…</div>');
  const left=visibleActions(a).filter(x=>x.kind!=='model'&&x.kind!=='dependency'&&x.kind!=='queue'&&x.kind!=='setup');
  // Agent identities differ between runs; compare all agents on B unless an equivalent label exists.
  const selectedLabel=state.agent?agentName(state.agent):null;
  const right=b.actions.filter(x=>x.kind!=='model'&&x.kind!=='dependency'&&x.kind!=='queue'&&x.kind!=='setup'&&(!selectedLabel||agentName(x.agent,b)===selectedLabel)&&matchesAction(x,'',state.query));
  const rows=alignSequences(left,right);
  const matches=rows.filter(r=>r.match).length;
  const first=rows.findIndex(r=>!r.match);
  const similarity=left.length+right.length?2*matches/(left.length+right.length):null;
  const filesA=new Set(a.actions.flatMap(x=>x.files)),filesB=new Set(b.actions.flatMap(x=>x.files));
  const fileDelta=[...filesA].filter(f=>!filesB.has(f)).concat([...filesB].filter(f=>!filesA.has(f)));
  const delta=b.duration-a.duration;
  const resourceStats=(run,pattern)=>{
    const series=Object.entries(run.counters).find(([name])=>pattern.test(name));
    if(!series?.[1]?.length)return null;
    const values=series[1].map(p=>p[1]);
    return {mean:values.reduce((sum,v)=>sum+v,0)/values.length,peak:Math.max(...values)};
  };
  const cpuA=resourceStats(a,/workload.*cpu|cpu.*%/i),cpuB=resourceStats(b,/workload.*cpu|cpu.*%/i);
  const memA=resourceStats(a,/workload.*memory|memory_mb|rss_mb/i),memB=resourceStats(b,/workload.*memory|memory_mb|rss_mb/i);
  const resourceField=(label,left,right,unit)=>field(label,`${left?count(left.peak):'n/a'} → ${right?count(right.peak):'n/a'} ${unit} (sampled peak)`);
  const metadata=run=>`<h3>${esc(run.title)}</h3><p>${esc(run.subtitle)}<br>${esc(run.model)} / ${esc(run.harness)}<br>${formatTime(run.duration)} · ${run.actions.length} actions · outcome ${run.resolved==null?'unvalidated':run.resolved?'resolved':'unresolved'}<br>${count(Object.keys(run.counters).length)} resource series · ${new Set(run.actions.flatMap(x=>x.files)).size} referenced files</p>`;
  const cell=(action,side)=>action?`<button class="align-cell" data-compare-action="${esc(action.id)}" data-side="${side}"><strong><span style="color:${colors[action.kind]}">${esc(action.kind)}</span> · ${esc(action.intent||action.command||action.name)}</strong><small>${formatTime(action.start)} · ${formatTime(displayDuration(action))} · ${esc(action.outcome)}</small></button>`:'<div class="align-cell"><small>— No matching action</small></div>';
  return heading('Compare executions','Aligned action categories reveal insertion, deletion and reconvergence.')+controls+
    insight(first<0?`<strong>The observed action categories remain aligned.</strong> Run B takes ${formatTime(Math.abs(delta))} ${delta>=0?'longer':'less time'}. Commands, results and timings can still differ.`:`<strong>First divergence at aligned step ${first+1}.</strong> ${rows.filter(r=>!r.match).length} inserted / deleted actions. Select either side to inspect the underlying command.`)+
    `<div class="compare-metadata"><div>${metadata(a)}</div><div>${metadata(b)}</div></div>`+
    `<div class="panel">${field('Category similarity',similarity==null?'n/a':`${Math.round(similarity*100)}% · 2 × matched / total actions`)}${field('Wall-time change',`${delta>=0?'+':'−'}${formatTime(Math.abs(delta))}`)}${field('Different referenced paths',fileDelta.length?fileDelta.slice(0,4).join(', '):'Same observed path set')}${resourceField('CPU utilization A → B',cpuA,cpuB,'%')}${resourceField('Memory A → B',memA,memB,'MB')}</div>`+
    `<div class="alignment"><div class="alignment-heading"><span>A · current execution</span><span></span><span>B · comparison</span></div>${rows.length?rows.map((row,index)=>`<div class="alignment-row ${row.match?'':'divergent'} ${row.reconverged?'reconverged':''}" title="${row.reconverged?'Categories reconverge':row.match?'Matching action category':'Trajectory divergence'}">${cell(row.left,'a')}<div class="alignment-connector"><span>${row.reconverged?'↔':row.match?'=':'≠'}</span>${row.left&&row.right?`<small>${row.right.duration>=row.left.duration?'+':'−'}${formatTime(Math.abs(row.right.duration-row.left.duration))}</small>`:''}</div>${cell(row.right,'b')}</div>`).join(''):empty('No comparable actions','Clear search and agent filters.')}</div>`+
    `<div class="notice">Alignment uses the longest common subsequence of tool action categories, following the saved experiment’s category comparison. Matching categories do not imply identical commands, behavior or correctness. Different tasks can be compared; interpretation requires care. Referenced file paths are extracted from command text, not a verified artifact inventory.</div>`;
}

function moveColumn(id,direction) {
  const index=state.agentColumns.indexOf(id),next=index+direction;
  if(index<0||next<0||next>=state.agentColumns.length)return;
  const order=[...state.agentColumns];[order[index],order[next]]=[order[next],order[index]];
  state.columnsCustomized=true;state.agentColumns=order;renderCallStack();
  trajectoryColumns.get(id)?.querySelector(`[data-column-order][data-direction="${direction}"]`)?.focus({preventScroll:true});
}
function firstExperience() {
  return `<section class="first-experience"><h3>${trajectoryAgents().length?'No pinned columns':'Ready to see agent activity'}</h3><p>${trajectoryAgents().length?'Choose agents in the roster to return them to your workspace.':'Each column is an agent. Select a call to inspect captured input and output. Execution controls affect agents; Freeze view holds only the display.'}</p><button data-demo>Open recorded example</button><button data-connect>Run or connect your agents</button><p class="muted">The example is synthetic, read-only evidence. It does not run agents or measure a benchmark.</p></section>`;
}
function showConnectionGuide() {
  $('connection-guide').hidden=false;$('connection-guide').querySelector('button').focus();
}
function revealEvent(view) {
  const a=actionMap(state.run).get(state.selected?.id);
  if(!a||state.selected?.run==='b')return;
  if(!matchesAction(a,state.agent,state.query)){
    if(!window.confirm('This event is excluded by the agent/search filter. Clear those filters to reveal this recorded event?'))return;
    state.agent='';state.query='';$('agent-select').value='';$('search').value='';
  }
  if(view==='timeline'){
    setView('timeline');timelineViewer?.setSelection(a.id);
    const result=timelineViewer?.revealEvent(a.id);
    if(result&&!result.revealed)$('status').textContent=result.reason;
  }else {
    if(!state.agentColumns.includes(a.agent)){
      const result=pinAgent(state.run,state.agentColumns,a.agent);
      if(!result.changed){$('status').textContent=result.reason+' Selection remains inspectable.';return;}
      state.columnsCustomized=true;state.agentColumns=result.ids;
    }
    state.trajectoryLayout='episodes';state.activeColumn=a.agent;
    const r=readingState(a.agent);r.follow=false;
    const calls=visibleActions(state.run).filter(c=>c.agent===a.agent&&(c.outcome==='running')===(a.outcome==='running')).sort((x,y)=>y.start-x.start||(x.id>y.id?1:-1));
    const index=calls.findIndex(c=>c.id===a.id),start=Math.max(0,index-12);
    r[a.outcome==='running'?'activeIds':'ids']=calls.slice(start,start+50).map(c=>c.id);
    r.anchor=null;r.scrollTop=0;r.resetPosition=true;state.follow=false;
    setView('trajectory');scrollAgentColumn(a.agent);
    const column=trajectoryColumns.get(a.agent),feed=column?.querySelector('.agent-call-stack'),card=[...(feed?.querySelectorAll('[data-call-card]')||[])].find(c=>c.dataset.callCard===a.id);
    if(card){feed.scrollTop+=card.getBoundingClientRect().top-feed.getBoundingClientRect().top;captureColumnReading(column);card.querySelector('[data-action]').focus({preventScroll:true});}
  }
}
function interactionCapabilities(agent) {
  if(!executionState(agent).live||state.transport!=='connected'||!state.executionControls?.available)return null;
  return state.executionControls.capabilities?.[agent]||null;
}
function openInteraction(agent) {
  if(!currentExecution()?.agents.some(a=>a.id===agent))return;
  const caps=interactionCapabilities(agent);
  const selected=state.selected?.execution===executionIdentity(currentExecution())?actionMap(currentExecution()).get(state.selected.id):null;
  state.interaction={agent,execution:executionIdentity(currentExecution()),relay:state.executionControls?.relay_id,
    invocation:caps?.redirect?.includes(selected?.request_id)?selected.request_id:caps?.redirect?.[0]||'',pending:false,submitted:false,status:''};
  state.inspectorOpen=true;syncInspector();renderComposer();
  $('interaction-message')?.focus({preventScroll:true});
}
function closeInteraction(){state.interaction=null;state.inspectorMarkup=null;state.inspectorMarkupSelection=null;state.inspectorOpen=Boolean(state.selected);renderInspector(true);$('inspector-toggle').focus({preventScroll:true});}
function composerDescription(operation) {
  return operation==='queue_message'?'Queue retained context after all work already submitted to this agent. Already-submitted invocations do not receive it; later retained contexts do. This does not interrupt current work.':operation==='redirect'?'Try to interrupt the selected active invocation and submit a native prompt. A target gone before dispatch is rejected. If native delivery fails after dispatch, the SDK queues retained context at the then-current tail. Prompt acknowledgement does not prove the model followed it.':'Submit one registered skill with validated input through this agent’s existing run API. Submission creates a pending invocation; it does not prove task completion.';
}
function renderComposer() {
  const draft=state.interaction;if(!draft)return;
  // The inspector survives view switches and live paints. Keep its form nodes,
  // multiline draft, selection and focus until the user chooses a new target.
  if(draft.rendered&&$('inspector').querySelector('[data-interaction-close]')){renderInteractionStatus();return;}
  const caps=interactionCapabilities(draft.agent),operations=[];
  if(caps?.queue_message)operations.push(['queue_message','Queue retained message']);
  if(caps?.redirect?.length)operations.push(['redirect','Redirect active invocation']);
  if(caps?.run?.length)operations.push(['run','Run task']);
  $('inspector').innerHTML=`<div class="inspector-top"><span>AGENT INTERACTION</span><button data-interaction-close>Close</button></div><h2>${esc(agentName(draft.agent,currentExecution()))}</h2>${field('Agent identifier',draft.agent)}${field('Execution',readableExecutionTitle(currentExecution()))}<p id="interaction-target-status" role="status"></p>${operations.length?`<label>Operation<select id="interaction-operation">${operations.map(([id,label])=>`<option value="${id}">${label}</option>`).join('')}</select></label><p id="interaction-description"></p><label id="interaction-invocation-label">Invocation<select id="interaction-invocation">${(caps.redirect||[]).map(id=>`<option value="${esc(id)}" ${id===draft.invocation?'selected':''}>${esc(id)}</option>`).join('')}</select></label><label id="interaction-task-label">Registered task<select id="interaction-task">${(caps.run||[]).map(task=>`<option value="${esc(task.id)}">${esc(task.label||task.id)}</option>`).join('')}</select></label><pre id="interaction-schema" class="payload-text"></pre><label id="interaction-message-label">Message<textarea id="interaction-message" rows="7" maxlength="32000" placeholder="Multiline instructions…"></textarea></label><label id="interaction-input-label">Task input (JSON object)<textarea id="interaction-input" rows="7" maxlength="32000" spellcheck="false">{}</textarea></label><button data-interaction-submit>Submit explicitly</button>`:`<p>Interaction unavailable. ${!executionState(draft.agent).live?'Recorded examples, saved executions and replay are read-only.':state.transport!=='connected'?'Reconnect before interacting with this execution.':'This execution has no enabled interaction relay for this agent. Queue messages need a live SDK agent; redirect needs an owned active invocation and interactive harness; Run task requires an explicitly registered skill.'}</p>`}<p id="interaction-status" role="status" aria-live="polite"></p><p class="muted">No work starts by opening this panel or selecting a task.</p>`;
  draft.rendered=true;renderInteractionOperation();renderInteractionStatus();
}
function renderInteractionOperation() {
  const draft=state.interaction,operation=$('interaction-operation')?.value;if(!draft||!operation)return;
  $('interaction-description').textContent=composerDescription(operation);
  for(const [id,visible] of [['interaction-invocation-label',operation==='redirect'],['interaction-task-label',operation==='run'],['interaction-message-label',operation!=='run'],['interaction-input-label',operation==='run'],['interaction-schema',operation==='run']])$(id).hidden=!visible;
  const task=interactionCapabilities(draft.agent)?.run?.find(task=>task.id===$('interaction-task').value);
  $('interaction-schema').textContent=task?(typeof task.input_schema==='string'?task.input_schema:JSON.stringify(task.input_schema,null,2)):'';
}
function renderInteractionStatus() {
  const draft=state.interaction;if(!draft)return;
  const run=currentExecution(),caps=interactionCapabilities(draft.agent);
  if(draft.execution!==executionIdentity(run)||draft.relay&&draft.relay!==state.executionControls?.relay_id)draft.invalid='Target execution changed. Close and reopen for the intended live agent.';
  else if(['completed','failed','cancelled'].includes(run?.status))draft.invalid='Execution finished. This interaction target is read-only.';
  const invocation=$('interaction-invocation');if(invocation&&!invocation.hidden&&$('interaction-operation')?.value==='redirect'&&!caps?.redirect?.includes(invocation.value)){draft.invalid='Selected invocation finished or is no longer redirectable. Close and reopen before redirecting.';}
  const target=$('interaction-target-status');if(target)target.textContent=draft.invalid||`${state.displayPaused?'View frozen · ':''}Controls target CURRENT LIVE execution, agent ${agentName(draft.agent,run)}.${state.transport!=='connected'?' Disconnected; request outcome may be unknown.':''}`;
  const status=$('interaction-status');if(status)status.textContent=draft.status||'';
  const submit=$('inspector').querySelector('[data-interaction-submit]');
  if(submit)submit.disabled=Boolean(draft.invalid)||draft.submitted||draft.pending||!caps;
}
function submitInteraction() {
  const draft=state.interaction,caps=draft&&interactionCapabilities(draft.agent);if(!draft||!caps||draft.invalid||draft.submitted||draft.pending)return;
  if(draft.execution!==executionIdentity(currentExecution())||draft.relay!==state.executionControls?.relay_id){draft.invalid='Execution target changed.';renderInteractionStatus();return;}
  const command=$('interaction-operation').value,extra={};
  if(command==='queue_message'&&!caps.queue_message)return;
  if(command==='run'){
    extra.task=$('interaction-task').value;if(!caps.run?.some(t=>t.id===extra.task))return;
    try{extra.input=JSON.parse($('interaction-input').value);if(!extra.input||typeof extra.input!=='object'||Array.isArray(extra.input))throw Error('Task input must be a JSON object.');}
    catch(error){draft.status='Invalid input · '+error.message;renderInteractionStatus();return;}
  }else {
    extra.message=$('interaction-message').value;
    if(extra.message.length>32000){draft.status='Message exceeds 32,000 characters.';renderInteractionStatus();return;}
    if(!extra.message.trim()){draft.status='Enter a nonblank message.';renderInteractionStatus();return;}
    if(command==='redirect'){
      extra.invocation=$('interaction-invocation').value;
      if(!caps.redirect?.includes(extra.invocation)){draft.status='Invocation is no longer redirectable. Queue retained context explicitly or reopen the panel.';renderInteractionStatus();return;}
    }
  }
  const id=`ui-${Date.now()}-${++commandSequence}`;
  draft.pending=true;draft.submitted=true;draft.requestId=id;draft.status='Requested · waiting for relay acceptance.';
  if(!state.stream?.sendExecution(command,{id,...commandTargetSnapshot(),agname:draft.agent,...extra})){draft.pending=false;draft.status='Not sent · connection unavailable. No retry was sent.';}
  else setTimeout(()=>{if(state.interaction===draft&&draft.pending){draft.pending=false;draft.status='Outcome unknown · no acknowledgement received. Inspect execution history; no retry was sent.';renderInteractionStatus();}},30000);
  renderInteractionStatus();
}
function renderFullPayload(viewer,text,offset=0) {
  const page=payloadWindow(text,{offset,limit:20000});
  fullPayloads.set(viewer,{text,offset:page.offset});
  viewer.innerHTML=`<p class="muted">${page.total?`Captured text · characters ${page.offset+1}–${page.offset+page.text.length} of ${page.total}`:'No captured text available in this evidence.'}</p><pre class="payload-text" tabindex="0"></pre><div class="payload-paging"><button data-payload-page="previous" ${page.offset===0?'disabled':''}>Previous</button><button data-payload-page="next" ${page.offset+page.text.length>=page.total?'disabled':''}>Next</button></div>`;
  viewer.querySelector('pre').textContent=page.text;
}
function pagePayload(button) {
  const viewer=button.closest('.full-payload'),data=fullPayloads.get(viewer);if(!data)return;
  renderFullPayload(viewer,data.text,data.offset+(button.dataset.payloadPage==='next'?20000:-20000));viewer.querySelector('pre').focus({preventScroll:true});
}

function selectedInterval() {
  if(state.selected?.run==='b')return null;
  if(state.selected?.interval) {const i=state.run.intervals.find(i=>i.id===state.selected.interval);if(i)return [i.start,i.start+i.duration];}
  if(state.selected?.type==='episode') {
    const e=state.run.episodes.find(e=>e.id===state.selected.id);
    return e?[e.start,e.end]:null;
  }
  const action=(state.selected?.run==='b'?actionMap(state.comparison):actionMap(state.run)).get(state.selected?.id);
  if(action) return [action.start,action.start+action.duration];
  const interval=state.run.intervals.find(i=>i.id===state.selected?.id);
  return interval?[interval.start,interval.start+interval.duration]:null;
}

function setSelection(selection,run=state.run) {
  state.interaction=null;state.selected=selection?{...selection,execution:executionIdentity(run)}:null;
  state.inspectorOpen=Boolean(selection)||Boolean(state.interaction);
}

function syncInspector() {
  const selection=state.selected;
  const signature=selection?`${selection.execution||state.run?.id}:${selection.run||'a'}:${selection.type}:${selection.id}`:null;
  if(signature&&signature!==state.inspectorSelection)state.inspectorOpen=true;
  if(!selection&&!state.interaction)state.inspectorOpen=false;
  state.inspectorSelection=signature;
  $('inspector').hidden=!state.inspectorOpen;
  $('investigation').classList.toggle('inspector-open',state.inspectorOpen);
  $('inspector-toggle').disabled=!selection&&!state.interaction;
  $('inspector-toggle').setAttribute('aria-expanded',String(state.inspectorOpen));
}

function toggleInspector(open=!state.inspectorOpen) {
  if(!state.selected&&!state.interaction)return;
  state.inspectorOpen=open;
  syncInspector();
  if(open){renderInspector(true);$('inspector').querySelector('[data-inspector-close]')?.focus({preventScroll:true});}
  else $('inspector-toggle').focus({preventScroll:true});
}

function renderInspector(preserveEvidence=false) {
  syncInspector();
  if(state.interaction){renderComposer();return;}
  if(!state.selected){$('inspector').innerHTML='';state.inspectorMarkup=null;state.inspectorMarkupSelection=null;return;}
  state.inspectorDirty=false;
  const top=`<p class="inspector-update" role="status"></p><div class="inspector-top"><span>SELECTION INSPECTOR${state.selected.run==='b'?' · COMPARISON B':''}</span><button data-inspector-close aria-label="Close inspector">Close</button></div>`;
  const run=state.selected?.run==='b'?state.comparison:state.run;
  const map=actionMap(run);
  let content='';
  const selection=state.selected;
  if(selection.type==='action') {
    const a=map.get(selection.id);
    if(!a) content='<p>Selection is no longer available.</p>';
    else {
      const card=eventCard(a,run);
      content=`${pill(a.kind,'neutral')} ${pill(card.state.label,a.outcome)}${run===state.run?pausedBadge(a.agent):''}<h2>${esc(card.primary)}</h2><p>${esc(card.secondary)}</p><p>${esc(agentName(a.agent,run))}</p>${field('Evidence',run.source==='synthetic'?'Synthetic example':a.source||'Recorded source unspecified')}${field('Timing provenance',a.timing||'Not recorded')}${a.tokens!=null?field('Input / output tokens',`${count(a.tokens)} / ${count(a.output_tokens)}`):''}${toolDetails(a,run,true)}`;
      content+=field('Lifecycle',a.missing_start?'End recorded; start missing':a.outcome==='running'?'Started; end not recorded yet':a.outcome==='incomplete'?'End missing at final execution boundary':a.outcome);
      if(a.intent)content+=`<div class="detail-label">Declared purpose</div><p>${esc(a.intent)}</p>`;
      const relationships=eventRelationships(a,run);
      content+=`<div class="detail-label">Explicit event relationships</div>`+(relationships.links.length?relationships.links.map(link=>`<button class="jump" data-reveal-action="${esc(link.id)}">${esc(link.label)} →</button>`).join(''):field('Correlation',relationships.unavailable));
      if(a.artifacts?.length)content+=`<div class="detail-label">Recorded artifacts</div><pre>${esc(JSON.stringify(a.artifacts,null,2))}</pre>`;
      if(a.event_ids?.length)content+=`<div class="detail-label">Immutable source evidence</div>${a.event_ids.slice(-10).map((id,i)=>`<button class="jump" data-raw-event="${esc(id)}">Source event ${i+1} · ${esc(id.slice(-16))} ↗</button>`).join('')}`;
      if(selection.interval&&selection.interval!==selection.id){const interval=run.intervals.find(i=>i.id===selection.interval);if(interval)content+=`<div class="detail-label">Linked wait</div><p>${esc(interval.label)}</p>${field('Wait interval',`${formatTime(interval.start)} → ${formatTime(interval.start+interval.duration)}`)}`;}
      if(state.contextBlock) content+=`<div class="detail-label">${esc(state.contextBlock.source)} · ${esc(state.contextBlock.change)}</div><pre>${esc(state.contextBlock.preview)}</pre>`;
      if(a.files?.length) content+=`<div class="detail-label">Referenced files</div>${a.files.map(f=>`<span class="file-chip">${esc(f)}</span>`).join('')}`;
      if(selection.run==='b') content+='<button class="jump" data-open-comparison>Open B in trajectory →</button>';
      else content+=`<button class="jump" data-reveal-event="trajectory">Reveal in trajectory →</button><button class="jump" data-reveal-event="timeline">Reveal on timeline →</button><button class="jump" data-view="resources">Correlate resource interval →</button>${a.kind==='model'?'<button class="jump" data-view="context">Inspect this model’s context →</button>':''}`;
      content+=`<details><summary class="detail-label">Event metadata</summary><div id="event-metadata-payload" class="full-payload"></div></details>`;
    }
  } else if(selection.type==='episode') {
    const episode=run.episodes.find(e=>e.id===selection.id);
    content=`${pill(episode.status)}${pausedBadge(episode.agent)}<h2>${esc(episode.title)}</h2><p>${esc(agentName(episode.agent))}</p>${field('Interval',`${formatTime(episode.start)} → ${formatTime(episode.end)}`)}${field('Grouping',episode.label_source==='declared'?'Declared purpose + structural boundaries':'Deterministic tool categories + actor boundaries')}${field('Actions',episode.actions.length)}<div class="detail-label">Observed evidence</div><p>${esc(episode.latest_result||'No returned result recorded.')}</p><div class="detail-label">Calls</div>${episode.actions.map(id=>map.get(id)).filter(Boolean).map(a=>`<button class="jump" data-action="${esc(a.id)}">${esc(a.kind)} · ${esc(a.intent||a.name)} →</button>`).join('')}<button class="jump" data-view="resources">Correlate this episode →</button>`;
  } else if(selection.type==='obligation') {
    const o=run.obligations.find(o=>o.id===selection.id);
    content=`${pill(o.status)}<h2>${esc(o.title)}</h2><p>${esc(o.note)}</p>${field('Obligation source',o.inferred?'Inferred from actions':'Explicit / evaluator')}<div class="detail-label">Supporting evidence</div>${o.evidence.map(id=>map.get(id)).filter(Boolean).map(a=>`<button class="jump" data-action="${esc(a.id)}">${esc(a.kind)} · ${formatTime(a.start)} · ${esc(a.outcome)} →</button>`).join('')||'<p>No supporting action has been recorded.</p>'}<button class="jump" data-view="trajectory">Follow the trajectory →</button>`;
  } else if(selection.type==='agent') {
    const agent=run.agents.find(a=>a.id===selection.id);
    content=agent?`${pill('Agent')} ${pill(agent.status||'unknown')}<h2>${esc(agent.label)}</h2><p>${esc(agent.role)}</p>${field('Model',agent.model||'unavailable')}${field('Harness',agent.harness||'unavailable')}${field('Parent',agent.parent?agentName(agent.parent):'No recorded parent')}${field('State',agent.status||'unknown')}${field('Actions',run.actions.filter(a=>a.agent===agent.id).length)}<button class="jump" data-view="trajectory">Inspect agent trajectory →</button><button class="jump" data-view="context">Inspect agent context →</button>`:'<p>The selected agent is no longer available.</p>';
  } else if(selection.type==='interval') {
    const i=run.intervals.find(i=>i.id===selection.id);
    content=`${pill(i.kind)}<h2>${esc(i.label)}</h2>${field('Start',formatTime(i.start))}${field('Duration',formatTime(i.duration))}${field('Source',i.source)}<p>This interval has no observed action attribution. It may contain uninstrumented work, transport, waiting or idle time.</p>`;
  } else if(selection.type==='artifact') {
    content=`${pill('Referenced artifact')}<h2 class="mono">${esc(selection.id)}</h2><p>Extracted from recorded command text.</p>${run.actions.filter(a=>a.files.includes(selection.id)).map(a=>`<button class="jump" data-action="${esc(a.id)}">${esc(a.kind)} · ${formatTime(a.start)} →</button>`).join('')}`;
  }
  const markup=top+content;
  if(state.inspectorMarkup===markup&&state.inspectorMarkupSelection===state.inspectorSelection)return;
  state.inspectorMarkup=markup;state.inspectorMarkupSelection=state.inspectorSelection;
  const raw=preserveEvidence?$('raw-evidence'):null,inspectorTop=$('inspector').scrollTop;
  $('inspector').innerHTML=markup;$('inspector').scrollTop=inspectorTop;
  if(raw)$('inspector').append(raw);
  if(selection.type==='action'){const action=map.get(selection.id);if(action){loadInspectorPayloads(action,run);const metadata=$('event-metadata-payload');if(metadata)renderFullPayload(metadata,JSON.stringify(action.metadata,null,2));}}
}

async function loadInspectorPayloads(action,run){
  const selectionKey=state.inspectorSelection;
  for(const button of $('inspector').querySelectorAll('[data-full-payload]')){
    const viewer=button.parentElement.querySelector('.full-payload');
    try{const text=await originalPayload(action,button.dataset.payloadKind,run);if(selectionKey!==state.inspectorSelection||!$('inspector').contains(button))return;renderFullPayload(viewer,text);button.parentElement.querySelector('.payload-text').hidden=true;button.hidden=true;}
    catch(error){if(selectionKey===state.inspectorSelection)viewer.textContent=error.message;}
  }
}

function selectAction(id, comparison=false, intervalId=null) {
  const run=comparison?state.comparison:state.run;
  const action=actionMap(run).get(id);
  if(!action) return;
  holdSelection();
  const episodeIndex=relevantEpisodes().findIndex(e=>e.id===action.episode);
  if(episodeIndex>=0&&state.windowEnd!=null&&(episodeIndex>=state.windowEnd||episodeIndex<state.windowEnd-50))state.windowEnd=episodeIndex+1;
  setSelection({type:'action',id,run:comparison?'b':'a',interval:intervalId},run);
  if(!comparison)state.cursor=intervalId?state.run.intervals.find(i=>i.id===intervalId).start:action.start;
  state.contextBlock=null;
  if(action.kind==='model'&&!comparison) state.model=id;
  if(!comparison) state.expanded.add(action.episode);
  render();
}

function detailedProfilerView(){return `<nav class="prof-tabs prof-detail-tabs" role="tablist" aria-label="Detailed profiler categories">${Object.entries(profilerCategories).map(([key,label])=>`<button role="tab" aria-selected="${state.profilerCategory===key}" data-profiler-category="${key}">${label}</button>`).join('')}</nav><section role="tabpanel" aria-label="${profilerCategories[state.profilerCategory]} profiler">${profilerView(state.run,profilerCategories[state.profilerCategory])}</section>`;}
document.addEventListener('click',event=>{const tab=event.target.closest('[data-profiler-category]');if(tab)setView(tab.dataset.profilerCategory);});

function setView(view) {
  if(profilerCategories[view]){state.profilerCategory=view;view='profiler';}
  if(!views.some(v=>v[0]===view))return;
  // Only explicit "Open B" navigation changes execution. A view switch keeps
  // the selected execution and any comparison evidence in the inspector.
  const focusedView=document.activeElement?.dataset.view;
  state.view=view;updateUrl();render();
  if(focusedView)document.querySelector(`nav [data-view="${view}"]`)?.focus({preventScroll:true});
  if(view==='compare'&&!state.comparison) loadComparison();
}

function handleClick(event) {
  const el=event.target.closest('[data-interact],[data-interaction-close],[data-interaction-submit],[data-payload-page],[data-reveal-event],[data-reveal-action],[data-column-order],[data-connect],[data-refresh-details],[data-column-live],[data-column-older],[data-active-older],[data-inspector-close],[data-execution-agent],[data-execution-stop],[data-execution-kill],[data-add-column],[data-remove-column],[data-agent-tab],button[data-view],[data-action],[data-episode],[data-interval],[data-agent],[data-model],[data-block],[data-source],[data-obligation],[data-artifact],[data-compare-action],[data-open-comparison],[data-zoom],[data-reset-zoom],[data-retry],[data-demo],[data-older-activities],[data-latest-activities],[data-raw-event]');
  if(!el||!state.run&&!(el.hasAttribute?.('data-retry')||el.hasAttribute?.('data-demo'))) return;
  if(el.dataset.interact)openInteraction(el.dataset.interact);
  else if(el.hasAttribute?.('data-interaction-close'))closeInteraction();
  else if(el.hasAttribute?.('data-interaction-submit'))submitInteraction();
  else if(el.hasAttribute?.('data-connect'))showConnectionGuide();
  else if(el.dataset.payloadPage)pagePayload(el);
  else if(el.dataset.revealEvent)revealEvent(el.dataset.revealEvent);
  else if(el.dataset.revealAction){const comparison=$('inspector').contains(el)&&state.selected?.run==='b';selectAction(el.dataset.revealAction,comparison);if(!comparison)revealEvent('trajectory');}
  else if(el.dataset.columnOrder)moveColumn(el.dataset.columnOrder,Number(el.dataset.direction));
  else if(el.dataset.refreshDetails)refreshCallDetails(el);
  else if(el.dataset.columnLive)liveColumn(el.dataset.columnLive);
  else if(el.dataset.columnOlder)olderCalls(el.dataset.columnOlder);
  else if(el.dataset.activeOlder)olderCalls(el.dataset.activeOlder,true);
  else if(el.hasAttribute?.('data-inspector-close')){if(state.interaction)closeInteraction();else toggleInspector(false);}
  else if(el.hasAttribute?.('data-execution-agent'))pauseResumeExecution(false,el.dataset.executionAgent);
  else if(el.hasAttribute?.('data-execution-stop'))stopExecution(false,false,el.dataset.executionStop);
  else if(el.hasAttribute?.('data-execution-kill'))stopExecution(false,true,el.dataset.executionKill);
  else if(el.dataset.addColumn)addAgentColumn(el.dataset.addColumn);
  else if(el.dataset.removeColumn)removeAgentColumn(el.dataset.removeColumn);
  else if(el.dataset.agentTab)selectAgentColumn(el.dataset.agentTab);
  else if(el.dataset.view) setView(el.dataset.view);
  else if(el.dataset.action) selectAction(el.dataset.action,$('inspector').contains(el)&&state.selected?.run==='b');
  else if(el.dataset.episode) {
    event.preventDefault();
    holdSelection();
    const id=el.dataset.episode;
    state.expanded.has(id)?state.expanded.delete(id):state.expanded.add(id);
    setSelection({type:'episode',id});state.cursor=state.run.episodes.find(e=>e.id===id).start;state.contextBlock=null;render();
  } else if(el.dataset.interval) {
    const interval=state.run.intervals.find(i=>i.id===el.dataset.interval);
    if(interval?.action) selectAction(interval.action,false,interval.id);
    else if(interval) {setSelection({type:'interval',id:interval.id});state.cursor=interval.start;render();}
  } else if(el.dataset.agent) {
    state.agent=el.dataset.agent;setSelection({type:'agent',id:state.agent});$('agent-select').value=state.agent;render();
  } else if(el.dataset.model) selectAction(el.dataset.model);
  else if(el.hasAttribute?.('data-block')) {
    const a=actionMap(state.run).get(el.dataset.modelId);
    const block=a.context.blocks.concat(a.context.dropped)[Number(el.dataset.block)];
    state.model=a.id;setSelection({type:'action',id:a.id});state.contextBlock=block;state.cursor=a.start;render();
  } else if(el.dataset.source) {
    const a=activeModel();setSelection({type:'action',id:a.id});state.model=a.id;state.contextBlock=a.context.blocks.find(b=>b.source===el.dataset.source);render();
  } else if(el.dataset.obligation) {setSelection({type:'obligation',id:el.dataset.obligation});render();}
  else if(el.dataset.artifact) {setSelection({type:'artifact',id:el.dataset.artifact});render();}
  else if(el.dataset.compareAction) selectAction(el.dataset.compareAction,el.dataset.side==='b');
  else if(el.hasAttribute?.('data-open-comparison')) {const id=state.selected.id;state.view='trajectory';loadRun(state.comparison.id,{type:'action',id});}
  else if(el.hasAttribute?.('data-zoom')) {const interval=selectedInterval();if(interval){state.zoom=[Math.max(0,interval[0]-1),Math.min(state.run.duration,interval[1]+1)];render();}else{$('status').textContent='Select an episode or span to focus its interval.';}}
  else if(el.hasAttribute?.('data-reset-zoom')) {state.zoom=null;render();}
  else if(el.hasAttribute?.('data-retry')) refresh();
  else if(el.hasAttribute?.('data-demo')){state.view='trajectory';loadRun('demo-baseline');}
  else if(el.hasAttribute?.('data-older-activities')) {holdHistory();state.windowEnd=Math.max(50,(state.windowEnd??relevantEpisodes().length)-50);render();}
  else if(el.hasAttribute?.('data-latest-activities')) {state.windowEnd=null;render();}
  else if(el.dataset.rawEvent) showRawEvidence(el.dataset.rawEvent,$('inspector').contains(el)&&state.selected?.run==='b'?state.comparison:state.run);
}

function holdHistory() {
  if(!state.stream)return;
  state.follow=false;
  const overview=document.querySelector('.activity-overview-label');if(overview)overview.textContent='Overview held at selection';
  if(state.windowEnd==null)state.windowEnd=relevantEpisodes().length;
  renderSession();
}

// Inspecting evidence should not stop a live task from appearing as it runs.
// Replays retain their historical selection behavior.
function holdSelection() {
  if(state.replay)holdHistory();
}

function followLatest() {
  if(state.displayPaused)resumeDisplay();
  state.follow=true;state.windowEnd=null;clearNotice(state.unread);clearNotice(state.pendingNew);
  for(const reading of columnReading.values()){reading.follow=true;reading.more=null;reading.activeMore=null;reading.ids=null;reading.activeIds=null;reading.anchor=null;reading.scrollTop=0;clearNotice(reading.pending);reading.error=null;}
  for(const column of trajectoryColumns.values())column.querySelector('.agent-call-stack').scrollTop=0;
  timelineNavigation.lock=true;
  if(state.run)state.cursor=state.replayState?.clock??state.run.duration;
  renderRunChrome();
  if(state.view==='trajectory'&&state.trajectoryLayout==='episodes')reconcileActivities();
  if(state.view==='profiler'&&!$('content').contains(document.activeElement))$('content').innerHTML=detailedProfilerView();
  if(state.view==='agents')renderTandemBoard();
  if(state.view==='timeline'){timelineViewer?.update(state.run,visibleActions(state.run),state.agent,state.replayState?.duration);timelineViewer?.latest();}
}

async function showRawEvidence(id,run=state.run) {
  holdSelection();
  // Evidence buttons can be used without first selecting a call.
  if(!state.selected){setSelection({type:'source',id},run);renderInspector();}
  state.inspectorOpen=true;syncInspector();
  const runId=run.id;
  const url=runId==='live'?`/api/trajectory/live/events/${encodeURIComponent(id)}`:`/api/trajectory/replay/${encodeURIComponent(runId)}/events/${encodeURIComponent(id)}`;
  let panel=$('raw-evidence');
  if(!panel){panel=document.createElement('section');panel.id='raw-evidence';$('inspector').append(panel);}
  panel.innerHTML='<div class="detail-label">Source event</div><p>Reading immutable evidence…</p>';
  try{const raw=await fetchJson(url);if((state.run?.id===runId||state.comparison?.id===runId)&&panel.isConnected){panel.innerHTML='<div class="detail-label">Immutable source event · recorded evidence</div><div class="full-payload"></div>';renderFullPayload(panel.querySelector('.full-payload'),JSON.stringify(raw,null,2));}}
  catch(error){if(panel.isConnected)panel.innerHTML=`<p>${esc(error.message)}</p>`;}
}

document.addEventListener('focusout',()=>{if(state.inspectorDirty)setTimeout(()=>{if(!$('inspector').contains(document.activeElement))renderInspector(true);},0);});
document.addEventListener('click',handleClick);
document.addEventListener('keydown',event=>{
  if(event.target.dataset.agentTab&&['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
    event.preventDefault();
    const columns=visibleAgentColumns(),index=columns.indexOf(state.activeColumn);
    const next=event.key==='Home'?0:event.key==='End'?columns.length-1:(index+(event.key==='ArrowRight'?1:-1)+columns.length)%columns.length;
    selectAgentColumn(columns[next]);
    $('agent-tabs').querySelector(`[data-agent-tab="${CSS.escape(columns[next])}"]`)?.focus({preventScroll:true});
    return;
  }
  if(event.key==='Escape'&&state.inspectorOpen) {event.preventDefault();toggleInspector(false);return;}
  if(event.target.matches('input,select,textarea')) return;
  if(event.key>='1'&&Number(event.key)<=views.length&&!event.metaKey&&!event.ctrlKey) {event.preventDefault();setView(views[Number(event.key)-1][0]);}
  if(event.key==='Escape') {setSelection(null);state.contextBlock=null;render();}
  if((event.key==='Enter'||event.key===' ')&&event.target.matches('[data-interval],.trajectory-action')) {event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
});
$('run-select').onchange=event=>loadRun(event.target.value);
for(const id of ['model-filter','harness-filter']) $(id).onchange=()=>{const runs=catalogOptions();if(runs.length&&!runs.some(r=>r.id===state.run?.id)) loadRun(runs[0].id);else if(!runs.length){$('status').textContent='No executions match these model and harness filters.';}};
$('agent-select').onchange=event=>{holdSelection();state.agent=event.target.value;setSelection(state.agent?{type:'agent',id:state.agent}:null);render();};
$('search').oninput=event=>{holdSelection();state.query=event.target.value;render();};
$('cursor').oninput=event=>{
  const clock=Number(event.target.value);
  if(state.displayPaused){state.cursor=clock;state.model=null;render();return;}
  holdHistory();state.cursor=clock;state.model=null;
  $('cursor-value').textContent=formatTime(clock);
  if(state.replay)state.stream?.sendReplay('seek',{clock});
  else {pendingSeek=clock;loadRun(state.run.id,null,true);}
};
$('clear-selection').onclick=()=>{setSelection(null);state.contextBlock=null;state.zoom=null;render();};
$('inspector-toggle').onclick=()=>toggleInspector();
$('copy-execution-id').onclick=async()=>{
  const id=state.run?.id;if(!id)return;
  try {await navigator.clipboard.writeText(id);if(state.run?.id===id)$('copy-id-status').textContent='Copied';}
  catch {if(state.run?.id===id){$('execution-id').focus();$('execution-id').select();$('copy-id-status').textContent='Select and copy the identifier.';}}
};
$('demo').onclick=()=>{state.view='trajectory';$('model-filter').value='';$('harness-filter').value='';loadRun('demo-baseline');};
$('refresh').onclick=refresh;
$('export').onclick=()=>{
  if(!state.run) return;
  if(state.run.id==='live'&&!state.replay&&state.run.coverage.client_window){const anchor=document.createElement('a');anchor.href='/api/investigator/runs/live';anchor.download='agency-live-recording.json';anchor.click();return;}
  const blob=new Blob([JSON.stringify(state.run,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);
  const anchor=document.createElement('a');anchor.href=url;anchor.download=`agency-${state.run.id}.json`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.addEventListener('change',event=>{
  if(event.target.id==='interaction-operation'||event.target.id==='interaction-task')renderInteractionOperation();
  else if(event.target.id==='roster-status'){state.rosterStatus=event.target.value;renderRoster();}
  else if(event.target.id==='add-agent-column'){addAgentColumn(event.target.value);event.target.value='';}
  else if(event.target.id==='comparison-select'){state.compareError=null;loadComparison(event.target.value);}
  else if(event.target.id==='trajectory-layout'){state.trajectoryLayout=event.target.value;updateUrl();render();}
});
document.addEventListener('input',event=>{if(event.target.id==='roster-search'){state.rosterQuery=event.target.value;renderRoster();}});
$('connection-close').onclick=()=>{$('connection-guide').hidden=true;};
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&state.follow&&state.stream&&!state.displayPaused)followLatest();});
window.addEventListener('pagehide',()=>state.stream?.close());
$('execution-all').onclick=()=>pauseResumeExecution(true);
for(const force of [false,true])
  $(`execution-${force?'kill':'stop'}-all`).onclick=()=>stopExecution(true,force);
$('display-toggle').onclick=()=>{if(state.displayPaused)resumeDisplay();else pauseDisplay();};
$('open-live').onclick=()=>{state.trajectoryLayout='episodes';state.view='trajectory';loadRun('live');};
$('start-replay').onclick=()=>loadRun(state.run.id,null,true);
$('replay-restart').onclick=()=>loadRun(state.run.id,null,true);
function pauseReplay() {return state.stream?.sendReplay('pause');}
$('replay-play').onclick=()=>{if(state.replayState?.playing)pauseReplay();else{state.follow=true;state.stream?.sendReplay('play',{speed:Number($('replay-speed').value)});}};
$('replay-step').onclick=()=>state.stream?.sendReplay('step');
$('replay-speed').onchange=()=>{if(state.replayState?.playing)state.stream?.sendReplay('play',{speed:Number($('replay-speed').value)});};
$('follow-live').onclick=()=>{followLatest();$('call-stack')?.firstElementChild?.scrollIntoView({block:'start'});};
$('reconnect-updates').onclick=()=>state.stream?.reconnect();
// Transient control menus close after acting, clicking elsewhere, or Escape.
document.addEventListener('click',event=>{
  for(const menu of document.querySelectorAll('.execution-menu,.agent-danger-menu,.session-menu'))if(!menu.contains(event.target))menu.open=false;
});
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeControlMenus();});
refresh();

document.addEventListener('click',event=>handleMetricEvent(event,{render:()=>{if(state.view==='profiler')$('content').innerHTML=detailedProfilerView();else renderMetricsDashboard(true);},navigate:setView,reveal:id=>{selectAction(id);revealEvent('trajectory');}}));
document.addEventListener('change',event=>{if(event.target.id==='prof-filter'){filterMetrics(event.target.value);render();}});

document.addEventListener('toggle',event=>{if(event.target.id==='execution-details'&&event.target.open)renderRunChrome();},true);
