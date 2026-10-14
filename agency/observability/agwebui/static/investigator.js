import {handleProfilerTableEvent, resetProfilerTables} from './profiler-ui.js';
import {profilerData} from './profiler-model.js';
import {renderMetricsSummary, profilerView, handleMetricEvent, filterMetrics, metricScope, setMetricScope} from './metrics-ui.js';
import {agentRoster, reconcilePinnedAgents, pinAgent} from './agent-roster.js';
import {requestControlConfirmation, refreshControlConfirmation} from './control-confirmation.js';
import {eventCard, activitySummary, activityClock, eventRelationships, eventMeasurements, payloadWindow, payloadValue, prettyJson, highlightJson} from './event-details.js';
import {ExecutionTimeline} from './execution-timeline.js';
import {CausalGraph, causalSummary, systemActivity} from './causal-graph.js';
import {matchesAction, trajectoryActions, groupCallStack, executionIdentity, executionMeasurements, readableExecutionTitle, systemMeasurements, agentTokens, agentResources} from './investigator-model.js';
import {executionControlState, mergeTrajectory, mergeTrajectorySnapshot, boundTrajectory, TrajectoryStream} from './trajectory-stream.js';

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const formatTime = value => value < 1 ? `${Math.round(value * 1000)}ms` : value >= 60 ? `${Math.floor(Math.round(value) / 60)}m ${Math.round(value) % 60}s` : `${value.toFixed(1)}s`;
const count = value => value == null ? 'n/a' : new Intl.NumberFormat('en', {notation: value >= 10000 ? 'compact' : 'standard', maximumFractionDigits: 1}).format(value);
const colors = {search:'var(--chart-event)', read:'var(--chart-event)', edit:'var(--chart-event)', test:'var(--chart-event)', review:'var(--chart-event)', tool:'var(--tool-color)', model:'var(--model-color)', setup:'var(--muted)', queue:'var(--amber)', dependency:'var(--amber)', unknown:'var(--muted)', cpu:'var(--chart-cpu)', io:'var(--chart-io)'};
// Primary views in navigation order; the agent-column trajectory is the default.
const views = [['trajectory','Execution Trajectory'], ['profiler','Profiler'], ['critical','Critical Path']];
const params = new URLSearchParams(location.search);
const profilerCategories={overview:'Overview',llm:'LLM',tools:'Tools',systems:'Systems'};
const requestedView=params.get('view')==='timeline'?'critical':params.get('view')==='activity'?'overview':params.get('view');
// Preserve existing links to the former Profiler category.
const initialView=requestedView==='profiler'&&params.get('profiler')==='critical'?'critical':profilerCategories[requestedView]?'profiler':views.some(v=>v[0]===requestedView)?requestedView:'trajectory';
const requestedProfiler=params.get('profiler')==='activity'?'overview':params.get('profiler');
const initialProfiler=profilerCategories[requestedProfiler]?requestedProfiler:profilerCategories[requestedView]?requestedView:'overview';
const state = {profilerCategory:initialProfiler, profiler:{agent:'',range:null,model:'',tool:'',status:'',toolOrder:'total',showZero:false,process:''},catalog:[], run:null, view:initialView, inspectorWidth:360, inspectorExpanded:false, agent:'', query:'', cursor:0, selected:null, inspectorOpen:false, expanded:new Set(), loading:false};
let timelineViewer=null;
let causalViewer=null;
let causalNavigation={scale:null,left:0,top:0,lock:false};
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
// Playback clock. The playhead is the instant on display. Between server
// observations it advances at the wall-clock rate, so it never waits for an
// event to move. Freezing holds the display only; it never pauses agents.
const playback={edge:null,head:null,seeking:false,duration:0,grewAt:0,held:null,scrubbing:false,scrubRange:null,bounds:{id:null,start:null,end:null}};
let seekSequence=0;
// A clock the server reports while it runs at a known rate. Reports arrive
// late by a varying amount, so one that lands a little behind the local
// projection is delivery jitter, not a rewind: the projection keeps its
// course. A report ahead of it, or far from it, is taken as given.
function followClock(previous,value,rate,now) {
  if(previous&&rate>0&&previous.rate===rate){
    const ahead=projectClock(previous,now)-value;
    if(ahead>=0&&ahead<=2*rate){previous.seen=now;return previous;}
  }
  return {value,rate,at:now,seen:now};
}
// Without reports the projection holds where it is instead of running on.
function projectClock(clock,now=Date.now()) {
  return clock.value+(Math.min(now,clock.seen+3000)-clock.at)/1000*clock.rate;
}
function observePlayback(message) {
  const run=currentExecution(),replay=state.replay&&state.replayState,now=Date.now();
  const duration=replay?.duration??run?.duration??0;
  if(playback.duration&&duration>playback.duration)playback.grewAt=now;
  playback.duration=duration;
  const ended=!state.replay&&run?.status&&!['running','unknown'].includes(run.status);
  playback.edge=followClock(playback.edge,duration,!ended&&now-playback.grewAt<2000?1:0,now);
  // Frames already in flight when a seek was sent still carry the old position.
  if(playback.seeking&&message?.type!=='snapshot')return;
  playback.seeking=false;
  playback.head=replay?followClock(playback.head,replay.clock,replay.playing&&replay.speed<=20?replay.speed:0,now):null;
}
// The cursor spans the measured wall clock: first agent execution to the
// selected end. A replay prefix may not have reached a boundary yet, so the
// boundaries already observed for this execution are kept.
function wallClockRange() {
  const run=currentExecution(),known=playback.bounds;
  if(!run)return {start:0,end:0};
  // Replay bounds describe the full recording, including before its first call.
  // A prefix cannot establish the recording's start or final measured end.
  if(state.replay&&state.replayState?.wall_clock)return state.replayState.wall_clock;
  if(known.id!==run.id)Object.assign(known,{id:run.id,start:null,end:null});
  const m=run.measurement;
  const start=run.execution_started_at==null?null:Math.max(0,run.execution_started_at-(run.time_origin??0));
  const end=m&&!m.held&&Number.isFinite(m.end_offset)?m.end_offset:null;
  if(state.replay){known.start??=start;known.end??=end;}else Object.assign(known,{start,end});
  // Setup and waiting precede the wall clock: until the first agent execution
  // is observed the cursor has not started. Traces without a measurement
  // (no recorded start at all) keep their trace extent.
  if(known.start==null&&run.execution_started_at===null)return {start:0,end:0,pending:true};
  const reported=state.replayState?.duration??run.duration??0;
  const duration=playback.edge?Math.max(reported,projectClock(playback.edge)):reported;
  const stop=Math.min(known.end??duration,duration),begin=Math.min(known.start??0,stop);
  return {start:begin,end:stop};
}
function viewFrozen() {
  return state.replay?!state.replayState?.playing:state.stream?state.displayPaused:true;
}
function playheadPosition(range=wallClockRange()) {
  const replay=state.replay&&state.replayState;
  let at=state.cursor;
  if(replay)at=playback.head?projectClock(playback.head):replay.clock;
  else if(state.displayPaused)at=playback.held??state.cursor;
  else if(state.stream)at=range.end;
  return Math.min(range.end,Math.max(range.start,at));
}
// Tenths of a second from zero, so a running readout changes one digit at a time.
const clockLabel=value=>value<60?`${Math.max(0,value).toFixed(1)}s`:formatTime(value);
function paintPlayhead() {
  if(!state.run||state.loading||playback.scrubbing)return;
  const range=wallClockRange(),at=playheadPosition(range),slider=$('cursor');
  // Exact bounds and a continuous step: rounding either one makes the thumb
  // hop between steps, most visibly while the range is still short.
  slider.min=range.start;slider.max=range.end;slider.value=at;
  slider.style.setProperty('--playback-progress',`${range.end>range.start?(at-range.start)/(range.end-range.start)*100:0}%`);
  slider.disabled=Boolean(range.pending);
  slider.title=range.pending?'Starts with the wall clock, at the first agent execution':'Execution clock · elapsed since the first observed agent execution';
  $('cursor-value').title='Elapsed since the first observed agent execution; expanded event timing also shows the trace clock';
  $('duration-label').title='Total wall interval since the first observed agent execution';
  // Labels count from the wall-clock start, like the Wall clock measurement.
  $('cursor-value').textContent=clockLabel(at-range.start);
  $('duration-label').textContent=clockLabel(range.end-range.start);
}
function seekPlayback(clock) {
  const run=currentExecution();if(!run||state.loading)return;
  const range=wallClockRange(),frozen=viewFrozen();
  if(range.pending)return;
  clock=Math.min(range.end,Math.max(range.start,clock));
  // Scrubbing to the end of the current execution rejoins it instead of trailing it.
  if(run.id==='live'&&!frozen&&clock>=range.end-.5){if(state.replay)loadRun('live');else followLatest();return;}
  if(state.replay&&state.replayState){
    const now=Date.now();
    const seek_id=++seekSequence;
    if(!state.stream?.sendReplay('seek',{clock,seek_id}))return;
    state.replayState.clock=clock;state.cursor=clock;playback.seeking=seek_id;
    playback.head={value:clock,rate:playback.head?.rate??0,at:now,seen:now};
    paintPlayhead();return;
  }
  // Leaving the live edge opens the recording at that instant, still playing
  // unless the view is frozen.
  loadRun(run.id,null,true,{clock,play:!frozen,speed:Number($('replay-speed').value)||1});
}
function togglePlayback() {
  const run=currentExecution();if(!run||state.loading)return;
  if(state.replay){
    const replay=state.replayState;if(!replay)return;
    if(replay.playing){pauseReplay();return;}
    if(replay.finished)state.stream?.sendReplay('seek',{clock:wallClockRange().start});
    state.follow=true;state.stream?.sendReplay('play',{speed:Number($('replay-speed').value)||1});
  }
  else if(!state.stream){const range=wallClockRange();loadRun(run.id,null,true,{clock:state.cursor>=range.end-.05?range.start:Math.max(range.start,state.cursor),play:true,speed:Number($('replay-speed').value)||1});}
  else if(state.displayPaused)resumeDisplay();else pauseDisplay();
}
let loadGeneration = 0;
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
Object.assign(state, {measurementMessage:'', executionControls:null, executionPending:null, executionMessage:'', executionOutcomes:[], interaction:null, rosterQuery:'', rosterStatus:'', rosterOpen:false, rosterExpanded:[], stream:null, replay:params.get('mode')==='replay', transport:'offline', follow:true, unread:new Set(), windowEnd:null, replayState:null, agentColumns:[], columnsCustomized:false, activeColumn:null, displayPaused:false, liveRun:null, displayChanges:new Set(),pendingNew:new Set()});
const currentExecution = () => state.liveRun || state.run;
const executionState = (agentId=null) => executionControlState({...state, executionControls:state.executionControls?.relay_id&&state.executionControls?.execution?state.executionControls:{...state.executionControls,available:false}, run:currentExecution(),
  replay:state.replay||(state.displayPaused&&executionIdentity(state.run)!==executionIdentity(currentExecution())),
  ...(agentId?{agent:agentId,selected:{type:'agent',id:agentId}}:{})});
const displayDuration = action => action.execution_duration ?? action.duration;
const pausedBadge = id => state.run?.agents.find(a=>a.id===id)?.status==='paused' ? pill('paused') : '';
const actionMap = run => new Map((run?.actions || []).map(a=>[a.id,a]));
const visibleActions = run => trajectoryActions(run).filter(a=>matchesAction(a,state.agent,state.query));
const agentName = (id, run=state.run) => run?.agents.find(a=>a.id===id)?.label || id;
const pill = (label, kind=label) => `<span class="pill ${esc(kind.replace(/ /g,'-'))}">${esc(label)}</span>`;
// Keep the shared inspector renderer's prose in the interface face and its
// identifiers, commands, measurements, and counters in the technical face.
const technicalFields = new Set(['Agent ID', 'Agent identifier', 'Exchange ID',
  'Request ID', 'Request IDs', 'Model tool ID', 'Tool IDs', 'PID', 'PPID',
  'Executable path', 'Command line', 'CPU time', 'CPU utilization', 'Memory / RSS',
  'Disk read / write bytes', 'Exit status', 'Input / output tokens', 'Interval',
  'Measured duration', 'Measured weight', 'Process start / end', 'TTFT',
  'Timestamp on trace', 'Wall-clock timestamp', 'End on trace',
  'Start · projected boundary', 'End · projected boundary', 'Unexplained elapsed gap',
  'Slack in observed chain', 'Slack in observed operation chain',
  'Summed measured syscall time (may overlap)']);
const field = (name, value) => `<div class="detail-field"><span>${esc(name)}</span><span${technicalFields.has(name)||typeof value==='number'?' class="mono"':''}>${esc(value)}</span></div>`;
const heading = (title, description, extra='') => `<div class="section-heading"><div><h2>${esc(title)}</h2><p>${esc(description)}</p></div>${extra}</div>`;
const empty = (title, text) => `<div class="empty-state"><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
const legend = entries => `<div class="legend">${entries.map(([key,label])=>`<span><i style="background:${colors[key]}"></i>${esc(label)}</span>`).join('')}</div>`;
const insight = text => `<div class="insight"><span class="insight-icon">↳</span><div>${text}</div></div>`;



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
  const heading=`<div class="system-metrics-heading"><strong>System metrics</strong><span>${esc(mode)}</span><button data-view="profiler">Systems profiler ↗</button></div>`;
  const cards=shown.map(metric=>{
    const [at,value]=metric.latest,start=Math.max(0,at-60),points=metric.points.filter(p=>p[0]>=start);
    const high=Math.max(1,...points.map(p=>p[1])),x=t=>2+(t-start)/(at-start||1)*116,y=v=>30-v/high*26;
    const line=points.map(p=>`${x(p[0])},${y(p[1])}`).join(' '),last=points.at(-1);
    const note=`${metric.name} · ${metric.statusLabel|| (metric.derived?'Rate calculated from counter deltas':'Recorded sample')} at ${formatTime(at)}${metric.label==='CPU'?' · 100% = one CPU core; multiple cores can exceed 100%':''}`;
    return `<div class="system-metric" title="${esc(note)}"><div class="system-metric-value"><span>${esc(metric.label)}</span><strong>${value==null?'Unavailable':count(value)} ${metric.unit}</strong></div><svg viewBox="0 0 120 34" role="img" aria-label="${esc(metric.scope+' '+metric.label)} · recent recorded samples"><polyline points="${line}" fill="none" stroke="currentColor" stroke-width="1.5"/>${last?`<circle cx="${x(last[0])}" cy="${y(last[1])}" r="2" fill="currentColor"/>`:''}</svg><small>${esc(metric.scope)} · at ${formatTime(at)}</small></div>`;
  }).join('');
  panel.innerHTML=heading+(cards?`<div class="system-metrics-cards">${cards}</div>`:'<p class="system-metrics-empty">No usable system samples recorded yet. Profiling must be enabled to collect CPU, memory, I/O and available GPU counters.</p>');
}
// A payload is shown as recorded. Valid JSON is also offered pretty-printed,
// which is the default; the Raw view and Copy always use the recorded text.
function payloadPreview(value,limit=4000){
  const raw=typeof value==='string'?value:JSON.stringify(value,null,2)??'',pretty=prettyJson(raw);
  const clip=text=>text.length>limit?'\n… Preview truncated':'';
  const plain=`<pre class="payload-text${pretty==null?'':' payload-raw'}">${esc(raw.slice(0,limit))}${clip(raw)}</pre>`;
  return {json:pretty!=null,html:pretty==null?plain:`<pre class="payload-text payload-pretty">${highlightJson(pretty.slice(0,limit))}${clip(pretty)}</pre>`+plain};
}
// The reader's Pretty/Raw choice for one payload, kept while its markup is rebuilt.
const payloadFormats=new Map();
// Recover original payloads even after the bounded live window evicts a call.
// The graph retains immutable event IDs and small display metadata.
function operationEvidence(op,run) {
  if(!op)return null;
  const retained=actionMap(run).get(op.action_id);
  if(retained)return retained;
  const start=run.causal_graph?.nodes.find(n=>n.id===op.start_node);
  return {id:op.action_id||op.id,name:op.label,agent:op.agent,kind:op.band==='LLM'?'model':'tool',model:op.model,start:start?.time,duration:op.duration,outcome:op.outcome,arguments:op.command_preview||'',metadata:op.metrics||{},event_ids:op.metrics?.event_ids||[],input_truncated:true,output_truncated:true};
}
function payloadSection(a,kind){
  const label=kind==='input'?'Input':'Output',preview=payloadPreview(payloadValue(a,kind)),view=payloadFormats.get(`${a.id}:${kind}`)||'pretty';
  return `<section class="payload-section" data-payload-view="${view}" data-payload-key="${esc(a.id)}:${kind}"><div class="payload-head"><b>${label}</b><span class="payload-format" role="group" aria-label="${label} format" ${preview.json?'':'hidden'}><button data-payload-format="pretty" aria-pressed="${view==='pretty'}">Pretty</button><button data-payload-format="raw" aria-pressed="${view==='raw'}">Raw</button></span><button data-copy-payload="${esc(a.id)}" data-payload-kind="${kind}" title="Copy the complete recorded ${kind}, exactly as captured">Copy</button></div>${preview.html}<button data-full-payload="${esc(a.id)}" data-payload-kind="${kind}">Open captured ${kind}${a[kind+'_truncated']?' from source':''}</button><div class="full-payload"></div></section>`;
}
function nearestMetrics(action,run){
  const measured=eventMeasurements(action,run);
  return measured.samples.map(sample=>field(sample.label,`${count(sample.value)} ${sample.unit} · ${sample.scope} · ${sample.provenance} sample at ${formatTime(sample.at)}, Δ ${formatTime(sample.delta)} from call start`)).join('')||'<p class="muted">No recorded resource samples available for this call interval. Current host measurements are not substituted.</p>';
}
function detailBody(a,run){
  const agent=run.agents.find(agent=>agent.id===a.agent),measurements=eventMeasurements(a,run),clock=activityClock(a,run);
  const rows=measurements.rows.map(row=>field(`${row.label} · ${row.provenance}`,`${row.unit==='s'?formatTime(row.value):count(row.value)+' '+row.unit} · ${row.scope}${row.interval?' · '+formatTime(row.interval[0])+' → '+formatTime(row.interval[1]):''}`)).join('');
  return `<div class="detail-label">Captured input / output · ${run.source==='synthetic'?'synthetic example':'recorded evidence'}</div>${payloadSection(a,'input')}${payloadSection(a,'output')}<details class="event-measurements"><summary>Timing &amp; recorded measurements</summary>${rows}${clock?field('Start on trace clock',formatTime(clock.trace))+field('Start on execution clock',clock.execution==null?'Unavailable':(clock.execution<0?'−':'')+formatTime(Math.abs(clock.execution))):''}<p class="muted">Trace time starts at the recorded trace origin. The toolbar and wall clock start at the first observed agent execution. ${esc(measurements.coverage)} Nearest resource samples are contextual observations, not operation attribution.</p>${nearestMetrics(a,run)}</details>${field('Agent ID',a.agent)}${field('Start · projected boundary',a.missing_start||String(a.source||'').includes('timestamp_missing')?'Unavailable':formatTime(a.start))}${field('End · projected boundary',a.outcome==='running'||a.outcome==='incomplete'||String(a.source||'').includes('timestamp_missing')?'Not recorded':formatTime(a.start+a.duration))}${a.kind==='model'?field('Model identity',eventCard(a,run).primary):''}${field('Harness',agent?.harness||run.harness||'Unavailable')}`;
}

function toolDetails(a,run,open=false){return `<details class="tool-details" data-tool-details="${esc(a.id)}" ${open?'open':''}><summary title="Captured arguments, output and metadata" aria-label="Details for ${esc(a.name||a.id)}"><span aria-hidden="true">⌄</span><span class="sr-only when-closed">Show details</span><span class="sr-only when-open">Hide details</span></summary><div class="tool-detail-body">${open?detailBody(a,run):''}</div></details>`;}
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
async function copyText(text) {
  try{await navigator.clipboard.writeText(text);}
  catch{
    // The clipboard API needs a secure context; a profiler opened by IP address has none.
    const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';
    document.body.append(area);area.select();const copied=document.execCommand('copy');area.remove();
    if(!copied)throw new Error('Copying is unavailable in this browser context.');
  }
}
async function handlePayloadClick(event) {
  const format=event.target.closest('[data-payload-format]');
  if(format){
    const section=format.closest('.payload-section'),viewer=section.querySelector('.full-payload'),loaded=fullPayloads.get(viewer);
    section.dataset.payloadView=format.dataset.payloadFormat;payloadFormats.set(section.dataset.payloadKey,format.dataset.payloadFormat);
    for(const option of section.querySelectorAll('[data-payload-format]'))option.setAttribute('aria-pressed',String(option===format));
    if(loaded)renderFullPayload(viewer,loaded.text);
    return;
  }
  const button=event.target.closest('[data-full-payload],[data-copy-payload]');if(!button)return;
  const payloadId=button.dataset.fullPayload||button.dataset.copyPayload;const a=actionMap(state.run).get(payloadId)||operationEvidence(state.run.causal_graph?.operations.find(o=>o.action_id===payloadId||o.id===payloadId),state.run);if(!a)return;
  const section=button.closest('.payload-section'),output=section.querySelector('.full-payload');button.disabled=true;
  try{
    // Always the complete recorded text, whichever view is showing.
    const raw=fullPayloads.get(output)?.text??await originalPayload(a,button.dataset.payloadKind,state.run);
    if(button.dataset.copyPayload){await copyText(raw);button.textContent='Copied ✓';}
    else {
      // Explicitly requested, bounded scrolling output in place of the preview.
      renderFullPayload(output,raw);
      section.classList.add('payload-loaded');button.hidden=true;
    }
  }catch(error){if(button.dataset.copyPayload){button.textContent='Copy failed';button.title=error.message;}else output.textContent=error.message;}
  finally{button.disabled=false;if(button.dataset.copyPayload)setTimeout(()=>{button.textContent='Copy';},1600);}
}
document.addEventListener('click',handlePayloadClick);
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
    if (!state.run) {
      await loadRun(params.get('run') || data.runs[0]?.id || 'live',null,state.replay);
      // Historical comparison links reveal only recorded identities, without replay.
      if (state.run && params.get('run')===state.run.id && state.run.id!=='live') {
        const linkedEvent=params.get('event'),linkedAgent=params.get('agent');
        if(linkedEvent&&actionMap(state.run).has(linkedEvent)){selectAction(linkedEvent);revealEvent('trajectory');}
        else if(linkedAgent&&state.run.agents.some(a=>a.id===linkedAgent)){
          state.agent=linkedAgent;state.agentColumns=[linkedAgent];state.columnsCustomized=true;state.activeColumn=linkedAgent;render();
        }
      }
    }
    else await loadRun(state.run.id,null,state.replay);
  } catch(error) { showError(error); }
  finally { $('refresh').disabled = false; }
}

async function loadRun(id, selected=null, replay=false, start={}) {
  if (!id) return;
  clearTimeout(trajectoryUpdateTimer);trajectoryUpdateTimer=null;
  pendingTrajectoryChanges.clear();pendingResync=false;
  const generation = ++loadGeneration;
  if(state.run?.id!==id){state.profiler={agent:'',range:null,model:'',tool:'',status:'',toolOrder:'total',showZero:false,process:''};resetProfilerTables();}
  clearExecutionPending();state.lastExecutionCommand=null;state.executionReceiptTargets=[];state.executionControls=null;state.executionMessage='';state.measurementMessage='';state.executionOutcomes=[];if(state.interaction){state.interaction.invalid='Execution changed. Reopen interaction for the intended agent.';renderInteractionStatus();}
  causalNavigation={scale:null,left:0,top:0,lock:false};
  timelineNavigation={scale:null,left:0,top:0,lock:false};openToolDetails.clear();payloadFormats.clear();trajectoryColumns.clear();columnReading.clear();clearNotice(state.pendingNew);setMetricScope(null);
  state.stream?.close();state.stream=null;state.replay=replay;state.replayState=null;
  state.follow=true;state.unread.clear();state.windowEnd=null;
  state.displayPaused=false;state.liveRun=null;clearNotice(state.displayChanges);clearNotice(state.unread);
  Object.assign(playback,{edge:null,head:null,seeking:false,duration:0,grewAt:0,held:null,scrubbing:false,scrubRange:null});
  state.agentColumns=[];state.columnsCustomized=false;state.activeColumn=null;state.rosterQuery='';state.rosterStatus='';
  state.run=null;state.agent='';state.cursor=0;state.selected=selected?{...selected,execution:id}:null;state.inspectorOpen=Boolean(selected);state.inspectorSelection=null;state.expanded=new Set();
  $('inspector').innerHTML='';state.inspectorMarkup=null;state.inspectorMarkupSelection=null;syncInspector();
  state.loading = true;
  setContent('<div class="loading-state">Reading spans, prompt provenance and resource samples…</div>');
  $('status').textContent='Loading execution…';
  if(id==='live'||replay) {
    state.transport='connecting';renderSession();
    state.stream=new TrajectoryStream({runId:id,replay,...start,onTransport:transport=>{if(generation===loadGeneration){state.transport=transport;if(transport!=='connected'&&state.interaction?.pending){state.interaction.status='Connection lost after submission · outcome unknown; no automatic retry.';renderInteractionStatus();}if(transport!=='connected'&&state.executionPending)finishExecutionCommand('Connection lost after submission; outcome unknown. Check recorded state after reconnect; no automatic retry was sent.');renderSession();}},onMessage:message=>{
      if(generation!==loadGeneration)return;
      if(message.type==='execution_command_result'){
        receiveCommandResult(message);
        return;
      }
      if(message.type==='measurement_command_result'){
        // The new boundary itself arrives with the next trajectory patch.
        state.measurementMessage=message.error||'';
        if(!message.error)for(const menu of document.querySelectorAll('.measure-menu'))menu.open=false;
        renderMetricsDashboard();
        return;
      }
      // Old frames, including an earlier seek's snapshot, must not rewind a
      // newer user selection while its acknowledgement is still in flight.
      if(state.replay&&playback.seeking&&message.replay?.seek_id!==playback.seeking)return;
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
          if(previous&&executionIdentity(previous)!==executionIdentity(message.run)&&!state.displayPaused){setSelection(null);trajectoryColumns.clear();columnReading.clear();setMetricScope(null);}
        }
        state.loading=false;
      }
      else for(const patch of message.patches||[]) changed.push(...mergeTrajectory(currentExecution(),patch));
      for(const id of new Set(changed)) {
        if(currentExecution().actions.find(a=>a.id===id)?.kind==='syscall')continue;
        if(!state.follow)retainNotice(state.unread,id);
        if(state.displayPaused)retainNotice(state.displayChanges,id);
        if(!before.has(id)&&(state.displayPaused||!state.follow))retainNotice(state.pendingNew,id);
        const actor=currentExecution().actions.find(a=>a.id===id)?.agent,reading=columnReading.get(actor);
        if(reading&&!reading.follow)retainNotice(reading.pending,id);
      }
      if(!state.replay){boundTrajectory(currentExecution(),readingPins());const retained=new Set(currentExecution().actions.map(a=>a.id));for(const id of openToolDetails)if(!retained.has(id))openToolDetails.delete(id);}
      reconcileExecutionCommand();
      state.replayState=message.replay||null;
      observePlayback(message);
      // A replay of the current execution that catches up is live again.
      if(state.replay&&currentExecution().id==='live'&&message.replay?.playing&&message.replay.position===message.replay.total&&message.replay.clock>=message.replay.duration-.25){loadRun('live');return;}
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
    state.run=run;if(state.selected)state.selected.execution=executionIdentity(run); state.agent=''; state.cursor=0; state.expanded=new Set();
    if (selected?.type==='action') {
      const action = actionMap(run).get(selected.id);
      if (action) {state.cursor=action.start;state.expanded.add(action.episode);}
    }
    state.loading=false;
    catalogOptions();
    updateUrl(); render();
  } catch(error) { if (generation===loadGeneration) {state.loading=false;showError(error);} }
}

// The execution controls are one mounted element. It is parked while the view
// markup is replaced and returned beside the agent management utilities, so
// open menus, pending states and handlers survive every re-render.
function setContent(html) {
  const controls=$('trajectory-controls'),focused=controls.contains(document.activeElement)?document.activeElement:null;
  $('control-park').append(controls);
  $('details-park').append($('execution-details'));
  $('content').innerHTML=html;
  $('trajectory-controls-slot')?.append(controls);
  $('execution-details-slot')?.append($('execution-details'));
  focused?.focus({preventScroll:true});
}

function showError(error) {
  setContent(`<div class="empty-state"><h3>Could not load execution</h3><p>${esc(error.message)}</p><button data-retry>Retry loading</button><button data-demo>Open synthetic demo</button></div>`);
  $('status').textContent=error.message;
}

function updateUrl() {
  const url = new URL(location.href);
  if (state.run) url.searchParams.set('run',state.run.id);
  url.searchParams.set('view',state.view);
  if(state.view==='profiler')url.searchParams.set('profiler',state.profilerCategory);else url.searchParams.delete('profiler');
  if(state.replay) url.searchParams.set('mode','replay');else url.searchParams.delete('mode');
  url.searchParams.delete('layout');
  history.replaceState(null,'',url);
}

function executionNavigation() {
  return views.map(([id,label],index)=>`<button data-view="${id}" class="${state.view===id?'active':''}" aria-current="${state.view===id?'page':'false'}"><span class="view-number">${index+1}</span>${label}</button>`).join('') + '<a href="/compare" class="compare-nav-link"><span class="view-number">4</span>Compare Runs</a>';
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
    ? ['#call-stack','.trajectory-scroll'].map(selector=>{
      const region=content.querySelector(selector);return [selector,region?.scrollLeft||0,region?.scrollTop||0];
    }):[];
  $('views').innerHTML=executionNavigation();
  $('view-label').textContent=views.find(v=>v[0]===state.view)[1];
  $('workspace').dataset.view=state.view;
  renderRunChrome();
  timelineViewer?.destroy();timelineViewer=null;
  causalViewer?.destroy();causalViewer=null;
  setContent(({profiler:detailedProfilerView,critical:criticalPathView,trajectory:trajectoryView})[state.view]());

  if(state.view==='critical')mountCausalGraph();
  if(state.view==='trajectory'){reconcileActivities(true);$('agent-roster').open=state.rosterOpen;for(const el of content.querySelectorAll('[data-roster-parent]'))el.open=state.rosterExpanded.includes(el.dataset.rosterParent);}
  for(const [selector,left,top] of heldScroll){const region=content.querySelector(selector);if(region){region.scrollLeft=left;region.scrollTop=top;}}
  content.dataset.run=run.id;content.dataset.view=state.view;
  if(focusedContent&&content.contains(focusedContent))focusedContent.focus({preventScroll:true});
  renderInspector();
}

function renderMetricsDashboard(preferencesOnly=false){
 const visible=!['profiler','critical'].includes(state.view);
 $('summary-strip').hidden=!visible;
 $('metrics').hidden=!visible;
 if(visible&&state.run)renderMetricsSummary($('metrics'),state.run,{preferencesOnly,measurement:{enabled:measurementEnabled(),message:state.measurementMessage},activeAgents:executionState().live&&state.executionControls?.available?executionState().active.map(a=>a.id):null});
}
// Measuring is separate from execution control: it stays available after the
// workload and its command relay have finished, and never reaches an agent.
const measurementEnabled=()=>state.run?.id==='live'&&!state.replay&&!state.displayPaused&&state.transport==='connected'&&Boolean(state.executionControls?.execution);
function sendMeasurement(command,extra={}) {
  if(!extra.error&&!measurementEnabled())extra={error:'Measurement controls need the connected live execution.'};
  if(!extra.error&&!state.stream?.sendMeasurement(command,{id:`measure-${Date.now()}-${++commandSequence}-${Math.random().toString(36).slice(2,10)}`,execution:state.executionControls.execution,...extra}))extra={error:'Not sent · reconnect and try again.'};
  state.measurementMessage=extra.error||'';
  renderMetricsDashboard();
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
  $('run-select').title=readableExecutionTitle(run)+(run.source==='synthetic'?' · Synthetic example; no agents or commands executed':'');
  renderMetricsDashboard();
  const detailsOpen=state.view==='profiler';
  const measurements=executionMeasurements(run,{includeResources:!!detailsOpen});
  if(detailsOpen)$('secondary-metrics').innerHTML=measurements.primary.concat(measurements.secondary).map(item=>measurementHTML(item,true)).join('');
  $('metrics').title=measurements.coverage+(run.coverage.client_window?' Live view is windowed: page older calls in each column; full evidence remains in recorded review/export.':'');
  paintPlayhead();
  renderSession();
  $('status').textContent=`${run.source==='synthetic'?'Synthetic':'Recorded'} · ${count(run.coverage.events)} source events · ${visibleActions(run).length} actions in current filter${state.selected?' · selection linked across views':''}`;
}

function actionButton(action) {
  const card=eventCard(action,state.run),model=card.type==='model',summary=activitySummary(action,card);
  const kind=model?'Model request':card.secondary;
  const status=card.state.label.replace(/^(Tool|Request) /,'');
  const clock=activityClock(action,state.run);
  const timing=clock?`Start on trace: ${formatTime(clock.trace)}${clock.execution==null?'':` · Execution clock: ${clock.execution<0?'−':''}${formatTime(Math.abs(clock.execution))}`}`:'Start unrecorded';
  return `<button class="action-row type-${card.type} ${state.selected?.id===action.id?'selected':''}" data-action="${esc(action.id)}" title="Inspect ${esc(kind)} · ${esc(card.state.label)} · ${esc(timing)}"><span class="action-primary"><span class="action-kind"><span class="call-type">${model?'✦':card.type==='syscall'?'⌘':'⌁'}<span class="sr-only">${card.typeLabel}</span></span>${esc(kind)}</span><span class="action-label${summary.technical?' technical-label':''}" title="${esc(summary.text)}">${esc(summary.text)}</span></span><span class="action-duration" title="Wall interval, including recorded pauses" aria-label="Calculated wall interval">${action.outcome==='incomplete'?'End missing':card.elapsed==null?'Unknown':formatTime(card.elapsed)}</span><span class="status-icon ${esc(card.state.semantic)}" aria-label="${esc(card.state.label)}">${esc(card.state.icon)} ${esc(status)}</span></button>`;
}

function trajectoryView() {
  const management=`<div class="agent-column-controls"><div id="trajectory-controls-slot"></div><div id="agent-tabs" role="tablist" aria-label="Visible agents"></div><label class="sr-only-label"><span class="sr-only">Add agent</span><select id="add-agent-column" aria-label="Add agent column"></select></label><span id="agent-column-limit" class="sr-only"></span><button id="agent-focus-clear" data-clear-agent hidden></button><details id="agent-roster" class="agent-roster"><summary>Agent roster <span id="roster-count"></span></summary><div class="roster-popup"><div class="roster-filters"><label>Find agent<input id="roster-search" placeholder="Name or identifier" aria-label="Search agent roster" value="${esc(state.rosterQuery)}"></label><label>Show<select id="roster-status" aria-label="Roster filter"><option value="">All agents</option><option value="hidden">Hidden agents</option><option value="failed-actions">Retained call failures</option><option value="running">Running agents</option><option value="paused">Paused agents</option><option value="completed">Completed agents</option><option value="unknown">Unknown state</option></select></label></div><div id="roster-list"></div></div></details></div>`;
  const notifications='<div id="trajectory-attention" class="trajectory-attention" aria-label="Execution notifications" hidden><div id="trajectory-signals"></div><div id="hidden-agents" aria-label="Hidden agent failures"></div><span id="signal-count" class="sr-only"></span></div>';
  return management+notifications+`<div id="call-stack" class="agent-columns"></div><details class="activity-history"><summary>Activities grouped by intent</summary><div id="trajectory-glance"></div><div id="trajectory-workflow"></div><div class="activity-window"><button data-older-activities>Earlier activities</button><span id="activity-window-label" class="mono muted"></span><button data-latest-activities>Latest activities</button></div><div id="activity-list"></div><div class="notice">Activity labels use declared purpose when available, otherwise deterministic tool categories. Returned evidence is quoted below each activity. Tool success does not establish task success. Only 50 activities are rendered at once.</div></details>`;
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
  playback.held=playheadPosition();
  state.liveRun=state.run;
  state.run=structuredClone(state.liveRun);
  timelineNavigation.lock=false;timelineViewer?.setFrozen?.(true);
  state.displayPaused=true;clearNotice(state.displayChanges);clearNotice(state.pendingNew);
  // Timeline scrolling and resize paints must also read the held snapshot.
  timelineViewer?.update(state.run,visibleActions(state.run),state.agent,state.replayState?.duration);
  causalViewer?.update(state.run);causalViewer?.setFrozen(true);
  renderSession();
}
function resumeDisplay() {
  if(!state.displayPaused)return;
  state.run=state.liveRun;state.liveRun=null;state.displayPaused=false;playback.held=null;timelineViewer?.setFrozen?.(false);
  if(state.selected&&state.selected.execution!==executionIdentity(state.run))setSelection(null);
  clearNotice(state.displayChanges);clearNotice(state.pendingNew);
  state.cursor=state.replayState?.clock??state.run.duration;
  if(['trajectory','timeline'].includes(state.view))updateTrajectory(state.run.actions.map(a=>a.id));
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
  return `<span class="control-section-label sr-only">${esc(label)}</span><button class="execution-control" data-execution-agent="${target}" aria-label="${resume?'Resume':'Pause'} ${esc(name)}" aria-describedby="control-scope" ${disabled?'disabled':''}>${resume?'Resume':'Pause'}</button><button class="execution-control" data-execution-stop="${target}" aria-label="Stop ${esc(name)}" ${disabled?'disabled':''}>Stop</button><details class="agent-danger-menu"><summary aria-label="More actions for ${esc(name)}">⋯</summary><div class="agent-overflow"><button data-remove-column="${target}" title="Hide column only; agent keeps executing">Hide column</button><button data-agent="${target}">Inspect agent</button><button class="execution-control danger-control" data-execution-kill="${target}" ${killDisabled?'disabled':''}>Force stop · SIGKILL</button></div></details><button data-interact="${target}">Interact</button>`;
}

function recordedAgentControls(id) {
  return `<span class="control-section-label sr-only">Recorded agent state · ${esc(state.run.agents.find(a=>a.id===id)?.status||'unknown')}</span><button data-interact="${esc(id)}">Interact</button><details class="agent-danger-menu"><summary aria-label="More actions for ${esc(agentName(id))}">⋯</summary><div class="agent-overflow"><button data-remove-column="${esc(id)}" title="Hide column only">Hide column</button><button data-agent="${esc(id)}">Inspect agent</button></div></details>`;
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
  group.querySelector('.control-section-label').textContent=`${label}`;
  const pause=group.querySelector('[data-execution-agent]');
  pause.textContent=resume?'Resume':'Pause';pause.setAttribute('aria-label',`${resume?'Resume':'Pause'} ${name}`);pause.disabled=disabled;
  group.querySelector('[data-execution-stop]').disabled=disabled;
  group.querySelector('[data-execution-kill]').disabled=killDisabled;
}
function closeControlMenus() {
  for(const menu of document.querySelectorAll('.execution-menu,.agent-danger-menu,.session-menu,.measure-menu,.metric-scope')) {
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
  $('execution-all').hidden=false;
  $('execution-all').textContent=state.executionPending&&!state.executionPending.stopping?(state.executionPending.paused?'Pausing…':'Resuming…'):debuggerState.allCommand==='resume_all'?'Resume All':'Pause All';
  $('execution-all').setAttribute('aria-describedby','control-scope');
  $('execution-all').disabled=!debuggerState.allEnabled||Boolean(state.executionPending);
  $('all-agent-controls').hidden=false;
  $('all-agent-controls').title=debuggerState.live?'Controls target every active agent, including hidden agents':'Recorded executions and replay are read-only';
  for(const force of [false,true]) {
    const button=$(`execution-${force?'kill':'stop'}-all`);
    button.disabled=!debuggerState.allEnabled||(!force&&Boolean(state.executionPending))||force&&Boolean(state.executionPending?.force);
    button.title=`${force?'Force termination with SIGKILL':'Gracefully stop with SIGTERM'} for all active agents`;
  }
  for(const group of document.querySelectorAll('[data-agent-controls]')) {
    group.hidden=false;
    const html=debuggerState.live?agentControlButtons(group.dataset.agentControls):recordedAgentControls(group.dataset.agentControls);
    updateAgentControls(group,html);
  }
  if(state.run)for(const column of trajectoryColumns.values())renderColumnFollow(column);
  renderInteractionStatus();
  const frozen=viewFrozen();
  $('display-toggle').hidden=!run;
  $('display-toggle').disabled=!state.run||state.loading||replay&&state.transport!=='connected';
  $('display-toggle').textContent=replay?(frozen?'Play':'Pause'):!state.stream?'Play recording':frozen?'Unfreeze view':'Freeze view';
  $('display-toggle').title=!state.stream?'Play the recording from the time cursor':frozen?(replay?'Continue playing from the time cursor':'Apply buffered event updates and return to live'):'Hold the display at the time cursor; agents keep running';
  $('display-toggle').setAttribute('aria-pressed',String(frozen));
  $('display-state').title=state.displayPaused?`View frozen · ${noticeCount(state.pendingNew)} newly buffered · ${noticeCount(state.displayChanges)} calls changed · agents may still be running`:state.stream?(state.follow?'View following live':`History held · ${noticeCount(state.pendingNew)} newly buffered · ${noticeCount(state.unread)} calls changed`):'';
  $('display-state').textContent=state.displayPaused?'Frozen':state.stream?(state.follow?(replay?'Following replay':'Following live'):'History held'):run?.source==='synthetic'?'Synthetic example':'Recorded history';
  $('display-state').setAttribute('aria-label',$('display-state').title||$('display-state').textContent);
  $('control-outcomes').textContent=state.executionOutcomes.join(' · ');
  $('control-evidence').hidden=!state.executionOutcomes.length;
  $('control-scope').textContent=debuggerState.live?`${debuggerState.active.length} active agents · bulk actions include hidden agents. Pause holds harness processes; already-issued remote operations may continue.${state.displayPaused?' Controls target the current live execution while event display is frozen.':''}`:'';
  const receiptTargets=state.executionReceiptTargets||[],summaries=receiptTargets.map(applicationSummary);
  const evidenceSummary=receiptTargets.length>3?[...new Set(summaries)].map(label=>`${summaries.filter(value=>value===label).length} ${label}`).join(' · '):receiptTargets.map((target,index)=>`${agentName(target.agname,run)}: ${summaries[index]}`).join(' · ');
  const message=state.executionMessage.startsWith('Control result · ')||state.executionMessage.startsWith('Latest recorded control outcome · ')?'Control evidence updated · '+(evidenceSummary||'Review per-agent evidence'):state.executionMessage;
  $('execution-message').textContent=debuggerState.live?(message||(!state.executionControls?.available&&state.transport==='connected'?'Execution controls unavailable':'')):'';
  $('execution-state').textContent=`${mode} · ${status==='unknown'?'execution state unrecorded':status}${paused?` · ${paused} agent${paused===1?'':'s'} with pause intent recorded`:''}`;
  $('transport-state').textContent=state.stream?`${state.transport==='connected'?'Connected to updates':state.transport==='disconnected'?'Disconnected from updates · execution state retained':state.transport==='reconnecting'?'Reconnecting to updates':'Connecting to updates'}${replay&&state.replayState?` · ${state.replayState.finished?'replay exhausted':state.replayState.playing?'playing':'paused'} at ${formatTime(state.replayState.clock)}`:''}`:'Saved evidence · no live subscription';
  $('execution-state').dataset.mode=mode.toLowerCase();
  $('transport-state').title=$('transport-state').textContent;
  $('execution-message').title=$('execution-message').textContent;
  $('open-live').hidden=run?.id==='live'&&!replay;
  $('start-replay').hidden=!run||replay||run.id!=='live';
  $('replay-speed').hidden=!replay;
  $('follow-live').hidden=!state.stream;
  $('follow-live').textContent='Jump to live';
  $('follow-live').title='Unfreeze and return every agent column and the causal graph to live';
  $('reconnect-updates').hidden=!state.stream||state.transport==='connected';
}

function updateTrajectory(changed,resynced=false) {
  if(state.displayPaused){renderSession();return;}
  renderRunChrome();
  if(state.view==='trajectory') {
    reconcileActivities();
  }
  if(state.view==='critical'){causalViewer?.update(state.run);const summary=$('causal-summary');if(summary)summary.innerHTML=causalSummary(state.run.causal_graph,state.run);}
  else if(state.view==='profiler'&&!$('content').contains(document.activeElement))setContent(detailedProfilerView());
  if(state.view==='timeline')timelineViewer?.update(state.run,visibleActions(state.run),state.agent,state.replayState?.duration);
  const target=executionState().target;
  const controlSignature=`${target?.id}:${target?.status}:${target?.control_ts}`;
  const controlChanged=controlSignature!==state.inspectorControlSignature;
  if(controlChanged||!state.selected||state.selected.type.startsWith('causal-')||changed.includes(state.selected?.id)){
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
  state.agent='';
  if(state.view==='critical'){causalViewer?.update(state.run);const summary=$('causal-summary');if(summary)summary.innerHTML=causalSummary(state.run.causal_graph,state.run);}
  else if(state.view==='profiler'&&!$('content').contains(document.activeElement))setContent(detailedProfilerView());
  renderCallStack();
  scrollAgentColumn(id);
}

function removeAgentColumn(id) {
  if(!state.agentColumns.includes(id))return;
  state.columnsCustomized=true;state.agentColumns=state.agentColumns.filter(agent=>agent!==id);if(state.agent===id)state.agent='';
  if(state.view==='critical'){causalViewer?.update(state.run);const summary=$('causal-summary');if(summary)summary.innerHTML=causalSummary(state.run.causal_graph,state.run);}
  else if(state.view==='profiler'&&!$('content').contains(document.activeElement))setContent(detailedProfilerView());
  renderCallStack();
}

function selectAgentColumn(id) {
  if(!state.agentColumns.includes(id))return;
  state.agent='';state.activeColumn=id;renderCallStack();
  scrollAgentColumn(id);
}

function scrollAgentColumn(id){
  const board=$('call-stack');
  const column=[...board.children].find(el=>el.dataset.agentColumn===id);
  if(column)board.scrollTo?.({left:board.scrollLeft+column.getBoundingClientRect().left-board.getBoundingClientRect().left,behavior:'smooth'});
}

function agentActivityCounts(id) {
  const actions=trajectoryActions(state.run).filter(action=>action.agent===id);
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
  // Focusing one agent from the roster hides the other columns; say so and offer the way back.
  $('agent-focus-clear').hidden=!state.agent;
  $('agent-focus-clear').textContent=state.agent?`Showing only ${agentName(state.agent)} · show all pinned`:'';
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
  if(state.selected){
    pins.push(state.selected.id);if(state.selected.action_id)pins.push(state.selected.action_id);
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
    pins.push(...trajectoryActions(live).filter(a=>a.agent===agent&&a.outcome!=='running').sort((a,b)=>b.start-a.start||(a.id>b.id?1:a.id<b.id?-1:0)).slice(0,50).map(a=>a.id));
  }
  pins.push(...openToolDetails);
  return [...new Set(pins)].slice(0,2000);
}
function renderColumnFollow(column) {
  const r=readingState(column.dataset.agentColumn),heading=column.querySelector('.agent-column-heading');
  const count=heading.querySelector('.column-call-count');
  if(count.dataset.running!=null)count.textContent=Number(count.dataset.running)>0?`${count.dataset.running} active calls`:'';
  const follow=heading.querySelector('.column-follow');follow.hidden=r.follow&&!r.error;
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
const compactCount=value=>new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(value);
// One line beside the column status: the agent's own sandbox CPU and its
// reported tokens. Never a host or workload value; unavailable shows as a dash.
function agentUsage(id) {
  const tokens=agentTokens(state.run,id),resources=agentResources(state.run,[id]),cpu=resources.cpu?.value,memory=resources.memory?.value;
  const cpuNote=cpu!=null?'CPU: all processes in this agent\'s sandbox container, latest sample · 100% = one core':resources.shared?'CPU unavailable: sandbox shared with other agents':resources.sandboxes?'CPU unavailable: no current container sample':'CPU unavailable: no sandbox measurement recorded for this agent';
  const tokenNote=tokens?`Tokens: ${count(tokens.input)} input + ${count(tokens.output)} output reported · ${tokens.reported}/${tokens.models} model calls report usage`:'Tokens: no usage reported';
  const cpuText=cpu==null?'—':(cpu<10?cpu.toFixed(1):Math.round(cpu))+'%',ramText=memory==null?'—':compactCount(memory)+' MiB',tokenText=tokens?compactCount(tokens.total):'—';
  return {text:`CPU ${cpuText} · ${memory==null?'RAM —':ramText+' RAM'} · ${tokenText} tok`,html:`<span class="usage-label">CPU</span> <span class="usage-value">${esc(cpuText)}</span> · <span class="usage-label">RAM</span> <span class="usage-value">${esc(ramText)}</span> · <span class="usage-value">${esc(tokenText)}</span> <span class="usage-label">tok</span>`,title:`${cpuNote} · ${tokenNote}`};
}
// A column header selects its agent as the dashboard scope; again returns to
// all agents. Columns, pins and live following are untouched.
function scopeColumn(id) {
  const scope=metricScope();
  setMetricScope(scope?.size===1&&scope.has(id)?null:[id]);
  renderMetricsDashboard();markScopedColumns();
}
function markScopedColumns() {
  const scope=metricScope();
  for(const column of document.querySelectorAll('[data-agent-column]')) {
    const id=column.dataset.agentColumn,scoped=Boolean(scope?.has(id));
    column.classList.toggle('metric-scoped-column',scoped);
    column.querySelector('[data-scope-name]')?.setAttribute('aria-pressed',String(scoped));
  }
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
      column.innerHTML=`<header class="agent-column-heading" data-scope-column="${esc(agent)}" title="Select the header to scope the summary metrics to this agent"><strong data-scope-name role="button" tabindex="0" aria-pressed="false"></strong><span class="column-call-count"></span><span class="column-status"></span><span class="column-usage"></span><div class="agent-execution-controls" data-agent-controls="${esc(agent)}" role="group" aria-label="Authoritative controls for ${esc(agentName(agent))}"></div><div class="column-follow"><span></span><button data-column-live="${esc(agent)}">Live in this column</button></div></header><div class="agent-call-stack" tabindex="0" aria-label="Activity for ${esc(agentName(agent))}"><section class="agent-current"><h3 class="sr-only">Active calls</h3><p class="current-coverage"></p><div class="agent-current-calls"></div><button data-active-older="${esc(agent)}">More in-flight calls</button></section><section class="agent-history"><h3 class="history-label">History</h3><div class="agent-history-calls"></div><button data-column-older="${esc(agent)}">Older calls</button></section></div>`;
      trajectoryColumns.set(agent,column);
      const feed=column.querySelector('.agent-call-stack');
      feed.addEventListener('scroll',()=>{captureColumnReading(column);if(feed.scrollTop>2&&r.follow)holdColumn(agent);});
    }
    const scoped=Boolean(metricScope()?.has(agent));
    column.className=`agent-column ${agent===state.activeColumn?'active-agent-column':''}${scoped?' metric-scoped-column':''}`;
    const heading=column.querySelector('.agent-column-heading');
    heading.querySelector('strong').textContent=agentName(agent);heading.querySelector('strong').title=agentName(agent)===agent?agent:`${agentName(agent)} · ${agent}`;heading.querySelector('strong').setAttribute('aria-pressed',String(scoped));
    const usage=agentUsage(agent),usageNode=heading.querySelector('.column-usage');
    if(usageNode.dataset.usage!==usage.text){usageNode.textContent=usage.text;usageNode.innerHTML=usage.html;usageNode.dataset.usage=usage.text;}
    if(usageNode.title!==usage.title)usageNode.title=usage.title;
    const running=actions.filter(a=>a.agent===agent&&a.outcome==='running').length;
    heading.querySelector('.column-call-count').dataset.running=running;
    const controls=heading.querySelector('[data-agent-controls]');controls.hidden=false;
    const controlHtml=executionState().live?agentControlButtons(agent):recordedAgentControls(agent);
    updateAgentControls(controls,controlHtml);
    const agentStatus=state.run.agents.find(a=>a.id===agent)?.status||'unknown';
    const statusNode=heading.querySelector('.column-status');statusNode.textContent=['waiting_on_llm','waiting_llm'].includes(agentStatus)?'Waiting for LLM':agentStatus.replace(/^agent[_-]/,'').replace(/[_-]/g,' ');statusNode.dataset.status=agentStatus;statusNode.title=statusNode.textContent;
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
    const selected=state.selected?.id;
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
      section.querySelector('.call-group-heading').textContent=group.actions.length>1?`${group.actions.length} concurrent calls`:group.actions[0].kind==='control'?'Recorded intervention':'';
      const grid=section.querySelector('.call-group-grid');
      for(const [index,action] of group.actions.entries()) {
        visibleCards.add(action.id);let card=cards.get(action.id);
        if(!card){card=document.createElement('div');card.className=`call-card type-${eventCard(action,state.run).type}`;card.dataset.callCard=action.id;card.innerHTML=actionButton(action)+'<div class="call-card-status"></div>'+toolDetails(action,state.run,openToolDetails.has(action.id));cards.set(action.id,card);}
        const template=document.createElement('template');template.innerHTML=actionButton(action);const next=template.content.firstElementChild;
        const button=card.querySelector('[data-action]');button.className=next.className;
        if(button.innerHTML!==next.innerHTML)button.innerHTML=next.innerHTML;
        const status=card.querySelector('.call-card-status');
        const details=card.querySelector('.tool-details'),version=callEvidenceVersion(action);
        if(!card.dataset.evidenceVersion||!details?.querySelector('.tool-detail-body')?.innerHTML)card.dataset.evidenceVersion=version;
        if(card.dataset.evidenceVersion!==version&&details)details.dataset.stale='true';
        const cardSummary=eventCard(action,state.run);
        card.dataset.status=cardSummary.state.semantic;
        const html=details?.dataset.stale?`<button data-refresh-details="${esc(action.id)}">New evidence · Refresh details</button>`:'';
        if(status.innerHTML!==html)status.innerHTML=html;
        if(grid.children[index]!==card)grid.insertBefore(card,grid.children[index]||null);
      }
      const target=column.querySelector(group.running?'.agent-current-calls':'.agent-history-calls'),position=group.running?currentPosition++:historyPosition++;
      if(target.children[position]!==section)target.insertBefore(section,target.children[position]||null);
    }
    for(const [id,el] of existing)if(!visibleGroups.has(id))el.remove();
    for(const [id,card] of cards)if(!visibleCards.has(id))card.remove();
    const coverage=column.querySelector('.current-coverage');coverage.textContent=active.length>visibleActive.length?`${visibleActive.length} of ${active.length} retained active calls`:'';coverage.hidden=!coverage.textContent;
    column.querySelector('.agent-current').hidden=!active.length&&!r.activeMore;
    column.querySelector('.history-label').hidden=!active.length;
    column.querySelector('.agent-history').classList.toggle('empty-history',!visibleHistory.length);
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
    const actions=episode.actions.map(id=>map.get(id)).filter(a=>a&&a.kind!=='syscall').reverse();
    card.classList.toggle('selected',state.selected?.id===episode.id||actions.some(a=>a.id===state.selected?.id));
    card.classList.toggle('new-evidence',actions.some(a=>state.unread.has(a.id)));
    card.querySelector('.episode-number').textContent=episodes.indexOf(episode)+1;
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
  const active=trajectoryActions(run).filter(a=>a.outcome==='running'&&(!state.agent||a.agent===state.agent));
  const input=(run.signals||[]).find(signal=>signal.active!==false&&signal.severity==='required');
  const latest=trajectoryActions(run).filter(a=>a.result_preview||a.result).reduce((last,a)=>!last||a.end_ts>=last.end_ts?a:last,null);
  const current=active.length?active.slice(-3).map(a=>`<div><b>${esc(agentName(a.agent))}</b><span>${esc(a.intent||run.episodes.find(e=>e.id===a.episode)?.title||a.name)}</span><small>${esc(a.kind)} · ${state.run.agents.find(agent=>agent.id===a.agent)?.status==='paused'?'paused at':'running for'} ${formatTime(displayDuration(a))}</small></div>`).join(''):`<p>${input?'Waiting for requested human input. '+esc(input.explanation):run.status==='completed'?'Execution finished. Review the returned evidence; task correctness may remain unverified.':run.status==='cancelled'?'Execution cancelled. Unfinished spans retain incomplete telemetry.':run.status==='failed'?'Execution reported a failure. Inspect the evidence below.':state.replayState?.finished?'Recording exhausted. The final execution state was not recorded.':'No active call is recorded. A quiet stream alone does not establish a stall.'}</p>`;
  const glance=`<section class="trajectory-glance">${run.task?`<div class="detail-label">Requested task</div><p class="requested-task">${esc(String(run.task).slice(0,400))}</p>`:''}<div class="detail-label activity-overview-label">${!state.follow&&state.stream?'Overview held at selection':active.length?'Currently recorded':'Current situation'}</div><div class="current-activities">${current}</div>${latest?`<div class="latest-evidence"><span>${latest.kind==='model'?'Latest visible model reply':'Latest returned evidence'}</span><button data-action="${esc(latest.id)}">${esc((latest.result_preview||latest.result).slice(0,180))} ↗</button></div>`:''}</section>`;
  if((state.follow||initial)&&$('trajectory-glance').innerHTML!==glance)$('trajectory-glance').innerHTML=glance;
  const activeSignals=(run.signals||[]).filter(s=>s.active!==false)
    .filter(s=>!s.evidence?.length||s.evidence.some(id=>map.get(id)?.kind!=='syscall'))
    .map(s=>({...s,evidence:(s.evidence||[]).filter(id=>map.get(id)?.kind!=='syscall')}));
  const repeatedEvidence=new Set(activeSignals.filter(s=>s.title==='Repeated matching failures').flatMap(s=>s.evidence));
  const signals=activeSignals.filter(s=>s.title!=='Recorded call failure'||!s.evidence.every(id=>repeatedEvidence.has(id)));
  const gap=(run.coverage.catching_up?'<span class="execution-alert info">Recovering retained events · partial history</span>':'')+(run.coverage.gaps?.length?`<span class="execution-alert warning" title="${esc(run.coverage.gaps.join(' · '))}">Incomplete telemetry · execution state may be unknown</span>`:'');
  const signalHtml=gap+signals.map(signal=>`<details class="execution-alert ${signal.severity==='required'?'warning':/failure|error/i.test(signal.title)?'error':'info'}"><summary title="${esc(signal.explanation)}">${signal.severity==='required'?'⚠':'!'} ${esc(signal.title)}</summary><div><p>${esc(signal.explanation)}</p>${signal.evidence.map((id,i)=>`<button ${map.has(id)?'data-action':'data-raw-event'}="${esc(id)}">Evidence ${i+1} ↗</button>`).join('')}</div></details>`).join('');
  if((state.follow||initial)&&$('trajectory-signals').innerHTML!==signalHtml&&!$('trajectory-signals').contains(document.activeElement))$('trajectory-signals').innerHTML=signalHtml;
  if(state.follow||initial){
    $('trajectory-attention').hidden=!signalHtml&&!$('hidden-agents').innerHTML;
    $('signal-count').textContent=`${signals.length} notifications${gap?' · incomplete coverage':''}${input?' · action requested':''}`;
  }
  const workflow=trajectoryWorkflow();
  if((state.follow||initial)&&$('trajectory-workflow').innerHTML!==workflow)$('trajectory-workflow').innerHTML=workflow;
}

function trajectoryWorkflow() {
  return state.run.edges.filter(e=>!state.agent||e.from===state.agent||e.to===state.agent).slice(-3).map(edge=>`<div class="workflow-note"><span>${esc(agentName(edge.from))} → ${esc(agentName(edge.to))}</span><p>${esc(edge.label)}</p><small class="muted">${state.run.source==='synthetic'?'Fixture':'Recorded'} ${esc(edge.kind)}</small></div>`).join('');
}

function firstExperience() {
  return `<section class="first-experience"><h3>${trajectoryAgents().length?'No pinned columns':'Ready to see agent activity'}</h3><p>${trajectoryAgents().length?'Choose agents in the roster to return them to your workspace.':'Each column is an agent. Select a call to inspect captured input and output. Execution controls affect agents; Freeze view holds only the display.'}</p><button data-demo>Open recorded example</button><button data-connect>Run or connect your agents</button><p class="muted">The example is synthetic, read-only evidence. It does not run agents or measure a benchmark.</p></section>`;
}
function showConnectionGuide() {
  $('connection-guide').hidden=false;$('connection-guide').querySelector('button').focus();
}
function revealEvent(view) {
  const a=actionMap(state.run).get(state.selected?.id);
  if(!a)return;
  if(!matchesAction(a,state.agent,state.query)){
    if(!window.confirm('This event is excluded by the agent/search filter. Clear those filters to reveal this recorded event?'))return;
    state.agent='';state.query='';$('search').value='';
  }
  if(view==='critical'||view==='timeline'){
    setView('critical');causalViewer?.setSelection(a.id);
    const result=causalViewer?.revealEvent(a.id);
    if(result&&!result.revealed)$('status').textContent=result.reason;
  }else {
    if(!state.agentColumns.includes(a.agent)){
      const result=pinAgent(state.run,state.agentColumns,a.agent);
      if(!result.changed){$('status').textContent=result.reason+' Selection remains inspectable.';return;}
      state.columnsCustomized=true;state.agentColumns=result.ids;
    }
    state.activeColumn=a.agent;
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
function closeInteraction(){state.interaction=null;state.inspectorMarkup=null;state.inspectorMarkupSelection=null;state.inspectorOpen=Boolean(state.selected);renderInspector(true);focusInspectionOrigin();}
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
  $('inspector').innerHTML=`<div class="inspector-top"><span>Agent interaction</span>${inspectorSizeButton()}<button data-interaction-close>Close</button></div><h2>${esc(agentName(draft.agent,currentExecution()))}</h2>${field('Agent identifier',draft.agent)}${field('Execution',readableExecutionTitle(currentExecution()))}<p id="interaction-target-status" role="status"></p>${operations.length?`<label>Operation<select id="interaction-operation">${operations.map(([id,label])=>`<option value="${id}">${label}</option>`).join('')}</select></label><p id="interaction-description"></p><label id="interaction-invocation-label">Invocation<select id="interaction-invocation">${(caps.redirect||[]).map(id=>`<option value="${esc(id)}" ${id===draft.invocation?'selected':''}>${esc(id)}</option>`).join('')}</select></label><label id="interaction-task-label">Registered task<select id="interaction-task">${(caps.run||[]).map(task=>`<option value="${esc(task.id)}">${esc(task.label||task.id)}</option>`).join('')}</select></label><pre id="interaction-schema" class="payload-text"></pre><label id="interaction-message-label">Message<textarea id="interaction-message" rows="7" maxlength="32000" placeholder="Multiline instructions…"></textarea></label><label id="interaction-input-label">Task input (JSON object)<textarea id="interaction-input" rows="7" maxlength="32000" spellcheck="false">{}</textarea></label><button data-interaction-submit>Submit explicitly</button>`:`<p>Interaction unavailable. ${!executionState(draft.agent).live?'Recorded examples, saved executions and replay are read-only.':state.transport!=='connected'?'Reconnect before interacting with this execution.':'This execution has no enabled interaction relay for this agent. Queue messages need a live SDK agent; redirect needs an owned active invocation and interactive harness; Run task requires an explicitly registered skill.'}</p>`}<p id="interaction-status" role="status" aria-live="polite"></p><p class="muted">No work starts by opening this panel or selecting a task.</p>`;
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
  const target=$('interaction-target-status');if(target)target.textContent=draft.invalid||`${state.displayPaused?'View frozen · ':''}Controls target the current live execution, agent ${agentName(draft.agent,run)}.${state.transport!=='connected'?' Disconnected; request outcome may be unknown.':''}`;
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
// Pages a complete payload. Valid JSON is shown pretty-printed unless its
// section is switched to Raw; the text itself is kept unmodified for both.
function renderFullPayload(viewer,text,starts=[0]) {
  const section=viewer.closest?.('.payload-section'),pretty=prettyJson(text),formatted=pretty!=null&&section?.dataset.payloadView!=='raw';
  const page=payloadWindow(formatted?pretty:text,{offset:starts.at(-1),limit:20000});
  // A pretty page ends on a line break, so no string is highlighted in halves.
  const body=formatted&&page.next&&page.text.lastIndexOf('\n')>0?page.text.slice(0,page.text.lastIndexOf('\n')+1):page.text,end=page.offset+body.length;
  fullPayloads.set(viewer,{text,starts,next:end});
  if(section&&pretty!=null)section.querySelector('.payload-format').hidden=false;
  viewer.innerHTML=`<p class="muted">${page.total?`${formatted?'Formatted JSON':'Captured text'} · characters ${page.offset+1}–${end} of ${page.total}`:'No captured text available in this evidence.'}</p><pre class="payload-text" tabindex="0" ${page.total?'':'hidden'}></pre>${starts.length>1||end<page.total?`<div class="payload-paging"><button data-payload-page="previous" ${starts.length>1?'':'disabled'}>Previous</button><button data-payload-page="next" ${end<page.total?'':'disabled'}>Next</button></div>`:''}`;
  const pre=viewer.querySelector('pre');
  if(formatted)pre.innerHTML=highlightJson(body);else pre.textContent=body;
}
function pagePayload(button) {
  const viewer=button.closest('.full-payload'),data=fullPayloads.get(viewer);if(!data)return;
  renderFullPayload(viewer,data.text,button.dataset.payloadPage==='next'?[...data.starts,data.next]:data.starts.slice(0,-1));viewer.querySelector('pre').focus({preventScroll:true});
}

function setSelection(selection) {
  state.interaction=null;state.selected=selection?{...selection,execution:executionIdentity(state.run)}:null;
  state.inspectorOpen=Boolean(selection)||Boolean(state.interaction);
}

function inspectorWidthBounds() {
  const available=$('investigation').getBoundingClientRect().width||window.innerWidth||1200;
  return {min:280,max:Math.max(280,available-328)};
}
function setInspectorWidth(width) {
  const bounds=inspectorWidthBounds();
  state.inspectorWidth=Math.round(Math.min(bounds.max,Math.max(bounds.min,width)));
  $('investigation').style.setProperty('--inspector-width',`${state.inspectorWidth}px`);
  const divider=$('inspector-resizer');
  divider.setAttribute('aria-valuemin',bounds.min);divider.setAttribute('aria-valuemax',Math.floor(bounds.max));
  divider.setAttribute('aria-valuenow',state.inspectorWidth);divider.setAttribute('aria-valuetext',`${state.inspectorWidth} pixels wide`);
  const button=$('inspector').querySelector('[data-inspector-size]');
  if(button){button.textContent=state.inspectorExpanded?'Restore':'Expand';button.setAttribute('aria-pressed',String(state.inspectorExpanded));}
}
function toggleInspectorSize() {
  if(!state.inspectorExpanded)state.inspectorRestoreWidth=state.inspectorWidth;
  state.inspectorExpanded=!state.inspectorExpanded;
  setInspectorWidth(state.inspectorExpanded?Math.min(900,inspectorWidthBounds().max):state.inspectorRestoreWidth||360);
}
const inspectorSizeButton=()=>`<button data-inspector-size aria-label="Expand or restore selection inspector" aria-pressed="${Boolean(state.inspectorExpanded)}">${state.inspectorExpanded?'Restore':'Expand'}</button>`;
function syncInspector() {
  const selection=state.selected;
  const signature=selection?`${selection.execution||state.run?.id}:${selection.type}:${selection.id}`:null;
  if(signature&&signature!==state.inspectorSelection)state.inspectorOpen=true;
  if(!selection&&!state.interaction)state.inspectorOpen=false;
  state.inspectorSelection=signature;
  $('inspector').hidden=!state.inspectorOpen;
  $('inspector-resizer').hidden=!state.inspectorOpen;
  $('investigation').classList.toggle('inspector-open',state.inspectorOpen);
  setInspectorWidth(state.inspectorExpanded?Math.min(900,inspectorWidthBounds().max):state.inspectorWidth);
}

function focusInspectionOrigin() {
  const target=[...$('content').querySelectorAll('[data-action]')].find(el=>el.dataset.action===state.selected?.id);
  (target||$('content')).focus({preventScroll:true});
}
function toggleInspector(open=!state.inspectorOpen) {
  if(!state.selected&&!state.interaction)return;
  state.inspectorOpen=open;
  syncInspector();
  if(open){renderInspector(true);$('inspector').querySelector('[data-inspector-close]')?.focus({preventScroll:true});}
  else focusInspectionOrigin();
}

function renderInspector(preserveEvidence=false) {
  syncInspector();
  if(state.interaction){renderComposer();return;}
  if(!state.selected){$('inspector').innerHTML='';state.inspectorMarkup=null;state.inspectorMarkupSelection=null;return;}
  state.inspectorDirty=false;
  const top=`<p class="inspector-update" role="status"></p><div class="inspector-top"><span>Selection inspector</span>${inspectorSizeButton()}<button data-inspector-close aria-label="Close inspector">Close</button></div>`;
  const run=state.run;
  const map=actionMap(run);
  let content='';
  const selection=state.selected;
  if(selection.type==='causal-group') {
    const operations=new Map((run.causal_graph?.operations||[]).map(o=>[o.id,o]));
    content=`<h2>Aggregated recorded operations</h2><p class="notice">Overview groups nearby labels. Each member retains its original timestamps and dependencies; this group is not an execution operation.</p>`+(selection.members||[]).map(id=>{const op=operations.get(id);return op?`<p>${esc(op.label)} · ${esc(op.agent)} · ${op.duration==null?'duration unknown':formatTime(op.duration)} <button data-causal-operation="${esc(op.id)}">Inspect operation${op.metrics?.pid!=null?' · PID '+esc(op.metrics.pid):''} →</button></p>`:'';}).join('');
  } else if(selection.type==='causal-operation'||selection.type==='causal-node'||selection.type==='causal-edge') {
    const graph=run.causal_graph;
    const item=(selection.type==='causal-operation'?graph?.operations:selection.type==='causal-node'?graph?.nodes:graph?.edges)?.find(n=>n.id===selection.id);
    if(!item)content='<p>Selected graph evidence is unavailable in this prefix.</p>';
    else if(selection.type==='causal-operation') {
      const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
      const start=nodes.get(item.start_node),end=nodes.get(item.end_node),open=nodes.get(item.open_node);
      const boundaries=new Set([item.start_node,item.end_node,item.open_node].filter(Boolean));
      const activity=systemActivity(run,item.id);
      const relations=graph.edges.filter(e=>(boundaries.has(e.source)||boundaries.has(e.target))&&e.kind!=='duration'&&e.kind!=='in_flight');
      const analysis=graph.operation_critical_path||graph.critical_path;
      content=`${pill(item.kind||item.band)}<h2>${esc(item.label)}</h2>${field('Agent',item.agent)}${field(item.metrics?.execution_observation?'Execution observed at':'Start on trace',start?start.time.toFixed(6)+'s':'Unknown')}${field('End on trace',end?end.time.toFixed(6)+'s':open?'In progress; end not observed':'Unknown')}${field('Timing evidence',start?.timing||'No observed start')}${field('Measured duration',item.duration==null?'Unknown':formatTime(item.duration))}${field('Outcome',item.outcome)}${item.metrics?.execution_observation?field('Process start / end','Lifecycle boundaries unavailable'):''}${field('Request ID',item.request_id||'Unavailable')}${field('Exchange ID',item.exchange_id||'Unavailable')}${field('Model tool ID',item.tool_id||'Unavailable')}${field('Slack in observed operation chain',item.duration==null||start?.timing!=='observed'||analysis?.slack?.[item.start_node]==null?'Unknown':formatTime(analysis.slack[item.start_node]))}<p class="notice">Only verified blocking dependencies enter the operation path. Ownership and history are provenance. Missing observations may hide constraints.</p>`;
      if(item.band==='LLM')content+=`${field('Model',item.model||'Unavailable')}${field('Input / output tokens',`${item.metrics?.tokens??'unavailable'} / ${item.metrics?.output_tokens??'unavailable'}`)}${field('TTFT',item.metrics?.ttft_ms==null?'Unavailable':item.metrics.ttft_ms+' ms')}${field('Retry evidence',relations.filter(e=>e.kind==='retry').map(e=>e.confidence+': '+e.evidence).join(' · ')||'No recorded retry relationship')}`;
      if(item.kind!=='process'&&item.action_id){const evidence=operationEvidence(item,run);content+=`<div class="detail-label">Original recorded request / arguments and response / output</div>${payloadSection(evidence,'input')}${payloadSection(evidence,'output')}`;}
      if(item.kind==='process') {
        const metrics=item.metrics||{};
        content+=`<div class="detail-label">Linux process · captured metrics</div>${field('PID',metrics.pid??'Unavailable')}${field('PPID',metrics.ppid??'Unavailable')}${field('Executable path',metrics.executable_path||metrics.path||'Unavailable')}${field('Command line',metrics.command_line||(Array.isArray(metrics.argv)?metrics.argv.join(' '):metrics.argv)||'Unavailable')}${field('CPU utilization',metrics.cpu_pct==null?'Unavailable':metrics.cpu_pct+'%')}${field('CPU time',metrics.cpu_time_s==null?(metrics.cpu_ms==null?'Unavailable':metrics.cpu_ms+' ms'):metrics.cpu_time_s+'s')}${field('Memory / RSS',metrics.rss_mb==null?'Unavailable':metrics.rss_mb+' MiB')}${field('Disk read / write bytes',metrics.io_read_bytes==null&&metrics.io_write_bytes==null?'Unavailable':`${metrics.io_read_bytes??'unknown'} / ${metrics.io_write_bytes??'unknown'}`)}${field('Exit status',metrics.exit_status??'Unavailable')}${field('Execution evidence',metrics.execution_observation||'Recorded process lifecycle')}<p class="notice">An execve observation proves a process image executed. Its syscall duration is not process runtime. Missing lifecycle/resource information remains unavailable; namespace PIDs are not matched to unrelated host counters.</p>`;
      }
      if(item.action_id&&map.has(item.action_id))content+=`<button class="jump" data-action="${esc(item.action_id)}">Inspect call and profiling metrics →</button>`;
      content+=`<div class="detail-label">Incoming / outgoing relationships · ${relations.length}</div>`+relations.slice(0,80).map(e=>{
        const other=nodes.get(boundaries.has(e.source)?e.target:e.source);
        return `<p>${esc(other?.label||'Unknown operation')} · ${esc(e.kind)} · ${esc(e.confidence)} · ${e.blocking&&e.confidence==='verified'?'blocking dependency':'nonblocking relationship'}<br>${esc(e.evidence)}${e.gap?` · unexplained gap ${formatTime(e.gap)}`:''}</p>`;
      }).join('')+(relations.length>80?'<p class="notice">First 80 relationships shown; full evidence remains in the recording.</p>':'');
      content+=`<div class="detail-label">Operation metrics</div><pre>${esc(JSON.stringify(item.metrics,null,2))}</pre><div class="detail-label">Attributed system activity</div>`;
      if(activity.count)content+=field('Explicitly attributed syscall events',activity.count)+field('Summed measured syscall time (may overlap)',formatTime(activity.duration))+field('Events without measured duration',activity.unmeasured)+`<pre>${esc(JSON.stringify(activity.groups,null,2))}</pre><p class="notice">Attribution uses captured process_action_id, tool_action_id, or operation_action_id. Timing overlap alone does not establish ownership.</p>`;
      else content+='<p class="notice">No syscall ownership was recorded for this operation. System activity has not been attached by timestamp or PID overlap.</p>';
      content+='<button data-profiler-category="systems">Inspect aggregated Systems profiler →</button>';
      if(item.metrics?.event_ids?.length)content+=item.metrics.event_ids.slice(-5).map(id=>`<button class="jump" data-raw-event="${esc(id)}">Source evidence · ${esc(id)} →</button>`).join('');
    }
    else if(selection.type==='causal-node') {
      const op=graph.operations.find(o=>o.id===item.operation);
      content=`${pill(item.band)}<h2>${esc(item.label)} · ${esc(item.phase)}</h2>${field('Agent',item.agent)}${field('Timestamp on trace',formatTime(item.time))}${field('Wall-clock timestamp',Number.isFinite(run.time_origin)?new Date((run.time_origin+item.time)*1000).toISOString():'Relative recording clock')}${field('Timing evidence',item.timing)}${field('Measured duration',op.duration==null?'Unknown':formatTime(op.duration))}${field('Request ID',item.request_id||'Unavailable')}${field('Exchange ID',op.exchange_id||'Unavailable')}${field('Model tool ID',item.tool_id||'Unavailable')}${field('Slack in observed chain',item.slack==null?'Unknown':formatTime(item.slack))}<p class="notice">Slack uses observed dependency timing; missing dependencies may hide constraints.</p>`;
      if(op.action_id&&map.has(op.action_id))content+=`<button class="jump" data-action="${esc(op.action_id)}">Inspect call and profiling metrics →</button>`;
      content+=`<div class="detail-label">Incoming / outgoing dependency evidence</div>`+graph.edges.filter(e=>e.source===item.id||e.target===item.id).map(e=>`<p>${esc(e.kind)} · ${esc(e.confidence)} · ${esc(e.evidence)}</p>`).join('');
      content+=`<pre>${esc(JSON.stringify(op.metrics,null,2))}</pre>`;
      if(op.metrics.event_ids?.length)content+=op.metrics.event_ids.slice(-5).map(id=>`<button class="jump" data-raw-event="${esc(id)}">Source evidence · ${esc(id)} →</button>`).join('');
    } else {
      const left=graph.nodes.find(n=>n.id===item.source),right=graph.nodes.find(n=>n.id===item.target);
      content=`${field('From',left?`${left.label} · ${left.phase} · ${formatTime(left.time)}`:item.source)}${field('To',right?`${right.label} · ${right.phase} · ${formatTime(right.time)}`:item.target)}${field('Request IDs',[left?.request_id,right?.request_id].filter(Boolean).join(' → ')||'Unavailable')}${field('Tool IDs',[left?.tool_id,right?.tool_id].filter(Boolean).join(' → ')||'Unavailable')}${pill(item.confidence)}<h2>${esc(item.kind)}</h2>${field('Source',item.source)}${field('Target',item.target)}${field('Dependency evidence',item.evidence)}${field('Blocking',item.blocking?'Verified dependency':'Not included in path calculation')}${field('Measured weight',formatTime(item.weight))}${field(item.blocking?'Wait before consumption':'Observation offset',formatTime(item.wait))}${field('Unexplained elapsed gap',item.blocking&&item.confidence==='verified'?formatTime(item.gap):'Not a blocking wait')}${field('On operation critical path',(graph.operation_critical_path?.edge_ids.includes(item.id)??item.critical)?'Yes':'No')}`;
    }
    content+=`<details><summary>Missing observations and gaps</summary>${graph.coverage.unknown.slice(0,50).map(g=>`<p>${esc(g.operation)}: ${esc(g.reason)}</p>`).join('')}</details>`;
  } else if(selection.type==='action') {
    const a=map.get(selection.id);
    if(!a) content='<p>Selection is no longer available.</p>';
    else {
      const card=eventCard(a,run);
      content=`${pill(a.kind,'neutral')} ${pill(card.state.label,a.outcome)}${pausedBadge(a.agent)}<h2${card.json?' class="mono json-label"':['tool','syscall'].includes(card.type)?' class="mono"':''}>${card.json?highlightJson(card.primary):esc(card.primary)}</h2><p>${esc(card.secondary)}</p><p>${esc(agentName(a.agent,run))}</p>${field('Evidence',run.source==='synthetic'?'Synthetic example':a.source||'Recorded source unspecified')}${field('Timing provenance',a.timing||'Not recorded')}${a.tokens!=null?field('Input / output tokens',`${count(a.tokens)} / ${count(a.output_tokens)}`):''}${toolDetails(a,run,true)}`;
      content+=field('Lifecycle',a.missing_start?'End recorded; start missing':a.outcome==='running'?'Started; end not recorded yet':a.outcome==='incomplete'?'End missing at final execution boundary':a.outcome);
      if(a.intent)content+=`<div class="detail-label">Declared purpose</div><p>${esc(a.intent)}</p>`;
      const relationships=eventRelationships(a,run);
      content+=`<div class="detail-label">Explicit event relationships</div>`+(relationships.links.length?relationships.links.map(link=>`<button class="jump" data-reveal-action="${esc(link.id)}">${esc(link.label)} →</button>`).join(''):field('Correlation',relationships.unavailable));
      if(a.artifacts?.length)content+=`<div class="detail-label">Recorded artifacts</div><pre>${highlightJson(JSON.stringify(a.artifacts,null,2))}</pre>`;
      if(a.event_ids?.length)content+=`<div class="detail-label">Immutable source evidence</div>${a.event_ids.slice(-10).map((id,i)=>`<button class="jump" data-raw-event="${esc(id)}">Source event ${i+1} · ${esc(id.slice(-16))} ↗</button>`).join('')}`;
      if(a.files?.length) content+=`<div class="detail-label">Referenced files</div>${a.files.map(f=>`<span class="file-chip">${esc(f)}</span>`).join('')}`;
      content+=`<button class="jump" data-reveal-event="trajectory">Reveal in trajectory →</button><button class="jump" data-reveal-event="critical">Reveal on Critical Path →</button>`;
      content+=`<details><summary class="detail-label">Event metadata</summary><div id="event-metadata-payload" class="full-payload"></div></details>`;
    }
  } else if(selection.type==='episode') {
    const episode=run.episodes.find(e=>e.id===selection.id);
    content=`${pill(episode.status)}${pausedBadge(episode.agent)}<h2>${esc(episode.title)}</h2><p>${esc(agentName(episode.agent))}</p>${field('Interval',`${formatTime(episode.start)} → ${formatTime(episode.end)}`)}${field('Grouping',episode.label_source==='declared'?'Declared purpose + structural boundaries':'Deterministic tool categories + actor boundaries')}${field('Actions',episode.actions.length)}<div class="detail-label">Observed evidence</div><p>${esc(episode.latest_result||'No returned result recorded.')}</p><div class="detail-label">Calls</div>${episode.actions.map(id=>map.get(id)).filter(Boolean).map(a=>`<button class="jump" data-action="${esc(a.id)}">${esc(a.kind)} · ${esc(a.intent||a.name)} →</button>`).join('')}`;
  } else if(selection.type==='agent') {
    const agent=run.agents.find(a=>a.id===selection.id);
    content=agent?`${pill('Agent')} ${pill(agent.status||'unknown')}<h2>${esc(agent.label)}</h2><p>${esc(agent.role)}</p>${field('Model',agent.model||'unavailable')}${field('Harness',agent.harness||'unavailable')}${field('Parent',agent.parent?agentName(agent.parent):'No recorded parent')}${field('State',agent.status||'unknown')}${field('Actions',run.actions.filter(a=>a.agent===agent.id).length)}<button class="jump" data-view="trajectory">Inspect agent trajectory →</button>`:'<p>The selected agent is no longer available.</p>';
  }
  const markup=top+content;
  if(state.inspectorMarkup===markup&&state.inspectorMarkupSelection===state.inspectorSelection)return;
  state.inspectorMarkup=markup;state.inspectorMarkupSelection=state.inspectorSelection;
  const raw=preserveEvidence?$('raw-evidence'):null,inspectorTop=$('inspector').scrollTop;
  $('inspector').innerHTML=markup;$('inspector').scrollTop=inspectorTop;
  if(raw)$('inspector').append(raw);
  if(selection.type==='causal-operation'){const op=run.causal_graph?.operations.find(o=>o.id===selection.id);if(op&&op.kind!=='process'&&op.action_id)loadInspectorPayloads(operationEvidence(op,run),run);}
  if(selection.type==='action'){const action=map.get(selection.id);if(action){loadInspectorPayloads(action,run);const metadata=$('event-metadata-payload');if(metadata)renderFullPayload(metadata,JSON.stringify(action.metadata,null,2));}}
}

async function loadInspectorPayloads(action,run){
  const selectionKey=state.inspectorSelection;
  for(const button of $('inspector').querySelectorAll('[data-full-payload]')){
    const viewer=button.parentElement.querySelector('.full-payload');
    try{const text=await originalPayload(action,button.dataset.payloadKind,run);if(selectionKey!==state.inspectorSelection||!$('inspector').contains(button))return;renderFullPayload(viewer,text);button.parentElement.classList.add('payload-loaded');button.hidden=true;}
    catch(error){if(selectionKey===state.inspectorSelection)viewer.textContent=error.message;}
  }
}

function selectAction(id) {
  const action=actionMap(state.run).get(id);
  if(!action) return;
  holdSelection();
  const episodeIndex=relevantEpisodes().findIndex(e=>e.id===action.episode);
  if(episodeIndex>=0&&state.windowEnd!=null&&(episodeIndex>=state.windowEnd||episodeIndex<state.windowEnd-50))state.windowEnd=episodeIndex+1;
  setSelection({type:'action',id});
  state.cursor=action.start;
  state.expanded.add(action.episode);
  render();
}

function criticalPathView(){return `<section aria-label="Critical Path">${heading('Critical Path','Follow recorded dependencies on a wall-clock axis. Zoom into LLM requests, tools, and Linux processes.')}<div id="causal-summary">${causalSummary(state.run.causal_graph,state.run)}</div><div id="causal-graph" class="panel"></div></section>`;}
function profilerOptions(){return {...state.profiler,model:state.profilerCategory==='llm'?state.profiler.model:'',tool:state.profilerCategory==='tools'?state.profiler.tool:'',status:state.profilerCategory==='tools'?state.profiler.status:'',cursor:playheadPosition(),transport:state.stream&&!state.replay?state.transport:null,frozen:state.displayPaused};}
function profilerToolbar(){
 const p=state.profiler,range=p.range||wallClockRange(),start=Array.isArray(range)?range[0]:range.start,end=Array.isArray(range)?range[1]:range.end;
 return `<div class="prof-investigation-controls"><label>Agent <select data-prof-filter="agent" aria-label="Profiler agent"><option value="">All agents</option>${state.run.agents.filter(a=>a.id!=='workflow').map(a=>`<option value="${esc(a.id)}" ${p.agent===a.id?'selected':''}>${esc(a.label||a.id)}</option>`).join('')}</select></label>${p.agent?'<button data-prof-clear-agent>Clear agent</button>':''}<label>From <input type="number" min="0" step="any" data-prof-range="from" aria-label="Interval start in trace seconds" value="${start}"></label><label>To <input type="number" min="0" step="any" data-prof-range="to" aria-label="Interval end in trace seconds" value="${end}"></label><button data-prof-apply-range>Select / zoom</button><button data-prof-reset-range ${p.range?'':'disabled'}>Full range</button><span role="status">${p.range?`${formatTime(start)}–${formatTime(end)} selected`:'Full execution window'} · trace clock</span></div>`;
}
function detailedProfilerView(){return `<details class="prof-execution-context"><summary>Execution context</summary><div id="execution-details-slot"></div></details><nav class="prof-tabs prof-detail-tabs" role="tablist" aria-label="Detailed profiler categories">${Object.entries(profilerCategories).map(([key,label])=>`<button role="tab" aria-selected="${state.profilerCategory===key}" data-profiler-category="${key}">${label}</button>`).join('')}</nav>${profilerToolbar()}<section role="tabpanel" aria-label="${profilerCategories[state.profilerCategory]} profiler">${profilerView(state.run,profilerCategories[state.profilerCategory],profilerOptions())}</section>`;}
function refreshProfiler(){if(state.view==='profiler'){setContent(detailedProfilerView());paintPlayhead();}}
function selectProfilerRange(from,to){
 const bounds=wallClockRange(),start=Math.max(bounds.start,Math.min(from,to)),end=Math.min(bounds.end,Math.max(from,to));
 if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return;
 if(state.replay){pauseReplay();holdHistory();}if(state.stream)pauseDisplay();
 state.profiler.range=[start,end];state.cursor=end;if(state.displayPaused)playback.held=end;
 refreshProfiler();
}
function inspectProfilerOperation(id){
 const action=actionMap(state.run).get(id);if(!action)return;
 if(state.replay){pauseReplay();holdHistory();}if(state.stream)pauseDisplay();
 setSelection({type:'action',id});state.cursor=action.start;if(state.displayPaused)playback.held=action.start;
 refreshProfiler();renderInspector();
}
function handleProfilerClick(event){
 if(state.view!=='profiler')return;
 if(handleProfilerTableEvent(event,refreshProfiler))return;
 const el=event.target.closest('[data-prof-agent],[data-prof-model],[data-prof-tool],[data-prof-process],[data-prof-resource],[data-prof-operation],[data-prof-finding],[data-prof-time-agent],[data-prof-clear-agent],[data-prof-apply-range],[data-prof-reset-range]');if(!el)return;
 if(el.dataset.profOperation){inspectProfilerOperation(el.dataset.profOperation);return;}
 if(el.dataset.profFinding){
  if(el.dataset.agent)state.profiler.agent=el.dataset.agent;
  if(el.dataset.tool)state.profiler.tool=el.dataset.tool;
  if(el.dataset.from)selectProfilerRange(Number(el.dataset.from),Number(el.dataset.to));
  setView(el.dataset.profFinding);
  if(el.dataset.operation)inspectProfilerOperation(el.dataset.operation);return;
 }
 if(el.dataset.profTimeAgent){
  const agent=profilerData(state.run,profilerOptions()).agents.find(a=>a.id===el.dataset.profTimeAgent);
  const intervals=agent?.attribution.segments.filter(s=>s.kind===el.dataset.kind)||[];
  state.profiler.agent=el.dataset.profTimeAgent;
  if(intervals.length)selectProfilerRange(intervals[0].start,intervals[0].end);
  setView(el.dataset.kind==='llm'?'llm':el.dataset.kind==='tools'?'tools':'overview');return;
 }
 if(el.dataset.profAgent!=null)state.profiler.agent=el.dataset.profAgent;
 if(el.dataset.profModel!=null)state.profiler.model=el.dataset.profModel;
 if(el.dataset.profTool!=null)state.profiler.tool=el.dataset.profTool;
 if(el.dataset.profProcess!=null)state.profiler.process=el.dataset.profProcess;
 if(el.dataset.profResource!=null)state.profiler.resource=el.dataset.profResource;
 if(el.hasAttribute('data-prof-clear-agent'))state.profiler.agent='';
 if(el.hasAttribute('data-prof-reset-range'))state.profiler.range=null;
 if(el.hasAttribute('data-prof-apply-range')){selectProfilerRange(Number(document.querySelector('[data-prof-range="from"]').value),Number(document.querySelector('[data-prof-range="to"]').value));return;}
 refreshProfiler();
}
let profilerDrag=null;
document.addEventListener('pointerdown',event=>{
 const svg=event.target.closest('svg[data-prof-chart]');if(!svg||event.button!==0||event.target.closest('[data-prof-operation]'))return;
 const rect=svg.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*720;
 if(x<64||x>700)return;profilerDrag={svg,x,pointer:event.pointerId};svg.setPointerCapture(event.pointerId);
});
document.addEventListener('pointermove',event=>{
 if(!profilerDrag)return;const {svg,x}=profilerDrag,rect=svg.getBoundingClientRect(),end=Math.max(64,Math.min(700,(event.clientX-rect.left)/rect.width*720));
 const selection=svg.querySelector('.prof-drag-selection');selection.setAttribute('x',Math.min(x,end));selection.setAttribute('width',Math.abs(end-x));
});
document.addEventListener('pointerup',event=>{
 if(!profilerDrag)return;const {svg,x,pointer}=profilerDrag;profilerDrag=null;
 svg.releasePointerCapture(pointer);const rect=svg.getBoundingClientRect(),end=Math.max(64,Math.min(700,(event.clientX-rect.left)/rect.width*720));
 const from=Number(svg.dataset.start),to=Number(svg.dataset.end),time=v=>from+(v-64)/636*(to-from);
 if(Math.abs(end-x)>4)selectProfilerRange(time(x),time(end));
});
document.addEventListener('pointercancel',()=>{profilerDrag=null;});
document.addEventListener('click',handleProfilerClick);
document.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)&&event.target.matches('circle[data-prof-operation]')){event.preventDefault();inspectProfilerOperation(event.target.dataset.profOperation);}});
document.addEventListener('change',event=>{
 if(state.view!=='profiler')return;
 if(event.target.dataset.profSearch){handleProfilerTableEvent(event,refreshProfiler);return;}
 const key=event.target.dataset.profFilter;if(key){state.profiler[key]=event.target.type==='checkbox'?event.target.checked:event.target.value;refreshProfiler();}
});

function mountCausalGraph(){
  causalViewer=new CausalGraph($('causal-graph'),causalNavigation,selection=>{
    holdSelection();const graph=state.run.causal_graph;const node=[...graph.operations,...graph.nodes].find(n=>n.id===selection.id);setSelection({...selection,action_id:node?.action_id});causalNavigation.selected=selection.id;
    renderInspector();causalViewer.setSelection(selection.id);
  },()=>holdHistory(),()=>{if(state.run.id==='live'&&!state.replay){followLatest();return true;}return false;});
  causalViewer.update(state.run);causalViewer.setSelection(state.selected?.id);causalViewer.setFrozen(state.displayPaused);
}
document.addEventListener('click',event=>{const tab=event.target.closest('[data-profiler-category]');if(tab)setView(tab.dataset.profilerCategory);});

function setView(view) {
  if(view==='activity')view='overview';
  if(profilerCategories[view]){state.profilerCategory=view;view='profiler';}
  if(view==='timeline')view='critical';
  if(!views.some(v=>v[0]===view))return;
  // A view switch keeps the selected execution and its inspector evidence.
  const focusedView=document.activeElement?.dataset.view;
  state.view=view;updateUrl();render();
  if(focusedView)document.querySelector(`nav [data-view="${view}"]`)?.focus({preventScroll:true});
}

function handleClick(event) {
  const el=event.target.closest('[data-causal-operation],[data-interact],[data-interaction-close],[data-interaction-submit],[data-payload-page],[data-reveal-event],[data-reveal-action],[data-connect],[data-refresh-details],[data-column-live],[data-column-older],[data-active-older],[data-inspector-size],[data-inspector-close],[data-execution-agent],[data-execution-stop],[data-execution-kill],[data-add-column],[data-remove-column],[data-agent-tab],button[data-view],[data-action],[data-episode],[data-agent],[data-clear-agent],[data-retry],[data-demo],[data-older-activities],[data-latest-activities],[data-raw-event]');
  // Header background and name only: every control inside a header keeps its own action.
  const scopeHeader=state.run&&event.target.closest?.('[data-scope-column]');
  if(scopeHeader&&!event.target.closest('button,summary,details,a,input,select,label,[data-agent-controls]')){scopeColumn(scopeHeader.dataset.scopeColumn);return;}
  if(!el||!state.run&&!(el.hasAttribute?.('data-retry')||el.hasAttribute?.('data-demo'))) return;
  if(el.dataset.causalOperation){setSelection({type:'causal-operation',id:el.dataset.causalOperation});renderInspector();causalViewer?.revealEvent(el.dataset.causalOperation);}
  else if(el.dataset.interact)openInteraction(el.dataset.interact);
  else if(el.hasAttribute?.('data-interaction-close'))closeInteraction();
  else if(el.hasAttribute?.('data-interaction-submit'))submitInteraction();
  else if(el.hasAttribute?.('data-connect'))showConnectionGuide();
  else if(el.dataset.payloadPage)pagePayload(el);
  else if(el.dataset.revealEvent)revealEvent(el.dataset.revealEvent);
  else if(el.dataset.revealAction){selectAction(el.dataset.revealAction);revealEvent('trajectory');}
  else if(el.dataset.refreshDetails)refreshCallDetails(el);
  else if(el.dataset.columnLive)liveColumn(el.dataset.columnLive);
  else if(el.dataset.columnOlder)olderCalls(el.dataset.columnOlder);
  else if(el.dataset.activeOlder)olderCalls(el.dataset.activeOlder,true);
  else if(el.hasAttribute?.('data-inspector-size'))toggleInspectorSize();
  else if(el.hasAttribute?.('data-inspector-close')){if(state.interaction)closeInteraction();else toggleInspector(false);}
  else if(el.hasAttribute?.('data-execution-agent'))pauseResumeExecution(false,el.dataset.executionAgent);
  else if(el.hasAttribute?.('data-execution-stop'))stopExecution(false,false,el.dataset.executionStop);
  else if(el.hasAttribute?.('data-execution-kill'))stopExecution(false,true,el.dataset.executionKill);
  else if(el.dataset.addColumn)addAgentColumn(el.dataset.addColumn);
  else if(el.dataset.removeColumn)removeAgentColumn(el.dataset.removeColumn);
  else if(el.dataset.agentTab)selectAgentColumn(el.dataset.agentTab);
  else if(el.dataset.view) setView(el.dataset.view);
  else if(el.dataset.action) selectAction(el.dataset.action);
  else if(el.dataset.episode) {
    event.preventDefault();
    holdSelection();
    const id=el.dataset.episode;
    state.expanded.has(id)?state.expanded.delete(id):state.expanded.add(id);
    setSelection({type:'episode',id});state.cursor=state.run.episodes.find(e=>e.id===id).start;render();
  } else if(el.dataset.agent) {
    state.agent=el.dataset.agent;setSelection({type:'agent',id:state.agent});render();
  } else if(el.hasAttribute?.('data-clear-agent')) {state.agent='';render();}
  else if(el.hasAttribute?.('data-retry')) refresh();
  else if(el.hasAttribute?.('data-demo')){state.view='trajectory';loadRun('demo-baseline');}
  else if(el.hasAttribute?.('data-older-activities')) {holdHistory();state.windowEnd=Math.max(50,(state.windowEnd??relevantEpisodes().length)-50);render();}
  else if(el.hasAttribute?.('data-latest-activities')) {state.windowEnd=null;render();}
  else if(el.dataset.rawEvent) showRawEvidence(el.dataset.rawEvent);
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
  if(state.view==='trajectory')reconcileActivities();
  if(state.view==='critical'){causalViewer?.update(state.run);const summary=$('causal-summary');if(summary)summary.innerHTML=causalSummary(state.run.causal_graph,state.run);}
  else if(state.view==='profiler'&&!$('content').contains(document.activeElement))setContent(detailedProfilerView());
  if(state.view==='timeline'){timelineViewer?.update(state.run,visibleActions(state.run),state.agent,state.replayState?.duration);timelineViewer?.latest();}
}

async function showRawEvidence(id) {
  holdSelection();
  // Evidence buttons can be used without first selecting a call.
  if(!state.selected){setSelection({type:'source',id});renderInspector();}
  state.inspectorOpen=true;syncInspector();
  const runId=state.run.id;
  const url=runId==='live'?`/api/trajectory/live/events/${encodeURIComponent(id)}`:`/api/trajectory/replay/${encodeURIComponent(runId)}/events/${encodeURIComponent(id)}`;
  let panel=$('raw-evidence');
  if(!panel){panel=document.createElement('section');panel.id='raw-evidence';$('inspector').append(panel);}
  panel.innerHTML='<div class="detail-label">Source event</div><p>Reading immutable evidence…</p>';
  try{const raw=await fetchJson(url);if(state.run?.id===runId&&panel.isConnected){panel.innerHTML='<div class="detail-label">Immutable source event · recorded evidence</div><div class="full-payload"></div>';renderFullPayload(panel.querySelector('.full-payload'),JSON.stringify(raw,null,2));}}
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
  if(['Enter',' '].includes(event.key)&&event.target.hasAttribute?.('data-scope-name')) {event.preventDefault();scopeColumn(event.target.closest('[data-scope-column]').dataset.scopeColumn);return;}
  if(event.key==='Escape'&&state.inspectorOpen) {event.preventDefault();toggleInspector(false);return;}
  if(event.target.matches('input,select,textarea')) return;
  if(event.key>='1'&&Number(event.key)<=views.length&&!event.metaKey&&!event.ctrlKey) {event.preventDefault();setView(views[Number(event.key)-1][0]);}
  if(event.key==='Escape') {setSelection(null);render();}
  if((event.key==='Enter'||event.key===' ')&&event.target.matches('.trajectory-action')) {event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
});
$('run-select').onchange=event=>loadRun(event.target.value);
for(const id of ['model-filter','harness-filter']) $(id).onchange=()=>{const runs=catalogOptions();if(runs.length&&!runs.some(r=>r.id===state.run?.id)) loadRun(runs[0].id);else if(!runs.length){$('status').textContent='No executions match these model and harness filters.';}};
$('search').oninput=event=>{holdSelection();state.query=event.target.value;render();};
// Dragging previews the position; releasing seeks, as on a video scrubber.
$('cursor').addEventListener('pointerdown',()=>{
  playback.scrubRange=wallClockRange();playback.scrubbing=true;
});
$('cursor').oninput=event=>{
  playback.scrubbing=true;
  const range=playback.scrubRange??wallClockRange(),at=Number(event.target.value);
  $('cursor-value').textContent=clockLabel(at-range.start);
  $('cursor').style.setProperty('--playback-progress',`${range.end>range.start?(at-range.start)/(range.end-range.start)*100:0}%`);
};
$('cursor').onchange=event=>{playback.scrubbing=false;playback.scrubRange=null;seekPlayback(Number(event.target.value));};
// A release that changes nothing fires no change event; resume painting anyway.
for(const type of ['pointerup','pointercancel','blur'])$('cursor').addEventListener(type,()=>setTimeout(()=>{playback.scrubbing=false;playback.scrubRange=null;paintPlayhead();}));
// Painted every frame so the playhead glides instead of stepping.
if(typeof requestAnimationFrame==='function')(function frame(){paintPlayhead();requestAnimationFrame(frame);})();
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
});
document.addEventListener('input',event=>{if(event.target.id==='roster-search'){state.rosterQuery=event.target.value;renderRoster();}});
$('connection-close').onclick=()=>{$('connection-guide').hidden=true;};
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&state.follow&&state.stream&&!state.displayPaused)followLatest();});
window.addEventListener('pagehide',()=>state.stream?.close());
$('execution-all').onclick=()=>pauseResumeExecution(true);
for(const force of [false,true])
  $(`execution-${force?'kill':'stop'}-all`).onclick=()=>stopExecution(true,force);
$('display-toggle').onclick=togglePlayback;
$('open-live').onclick=()=>{state.view='trajectory';loadRun('live');};
$('start-replay').onclick=()=>loadRun(state.run.id,null,true);
function pauseReplay() {return state.stream?.sendReplay('pause');}
$('replay-speed').onchange=()=>{if(state.replayState?.playing)state.stream?.sendReplay('play',{speed:Number($('replay-speed').value)});};
$('follow-live').onclick=()=>{if(state.replay&&state.run?.id==='live'){loadRun('live');return;}followLatest();$('call-stack')?.firstElementChild?.scrollIntoView({block:'start'});};
$('reconnect-updates').onclick=()=>state.stream?.reconnect();
// Transient control menus close after acting, clicking elsewhere, or Escape.
document.addEventListener('click',event=>{
  for(const menu of document.querySelectorAll('.execution-menu,.agent-danger-menu,.session-menu,.measure-menu,.metric-scope'))if(!menu.contains(event.target))menu.open=false;
});
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeControlMenus();});

document.addEventListener('click',event=>handleMetricEvent(event,{render:()=>{if(state.view==='profiler')setContent(detailedProfilerView());else renderMetricsDashboard(true);},navigate:setView,reveal:id=>{selectAction(id);revealEvent('trajectory');},measure:sendMeasurement,rescope:()=>{renderMetricsDashboard();markScopedColumns();}}));
document.addEventListener('change',event=>{if(event.target.id==='prof-filter'){filterMetrics(event.target.value);render();}});

// Resizing changes layout only: selection, payload pages and interaction drafts
// keep their DOM nodes, even while the execution stream is painting new data.
let inspectorDrag=null;
$('inspector-resizer').addEventListener('pointerdown',event=>{
  if(event.button!==0)return;
  event.preventDefault();state.inspectorExpanded=false;
  inspectorDrag={x:event.clientX,width:state.inspectorWidth};
  $('inspector-resizer').setPointerCapture(event.pointerId);
  $('investigation').classList.add('resizing-inspector');
});
$('inspector-resizer').addEventListener('pointermove',event=>{
  if(inspectorDrag)setInspectorWidth(inspectorDrag.width+inspectorDrag.x-event.clientX);
});
for(const type of ['pointerup','pointercancel','lostpointercapture'])$('inspector-resizer').addEventListener(type,()=>{
  inspectorDrag=null;$('investigation').classList.remove('resizing-inspector');
});
$('inspector-resizer').addEventListener('keydown',event=>{
  const bounds=inspectorWidthBounds();
  const widths={ArrowLeft:state.inspectorWidth+24,ArrowRight:state.inspectorWidth-24,Home:bounds.min,End:bounds.max};
  if(widths[event.key]==null)return;
  event.preventDefault();state.inspectorExpanded=false;setInspectorWidth(widths[event.key]);
});
window.addEventListener('resize',()=>{if(state.inspectorOpen)setInspectorWidth(state.inspectorExpanded?Math.min(900,inspectorWidthBounds().max):state.inspectorWidth);});

refresh();
