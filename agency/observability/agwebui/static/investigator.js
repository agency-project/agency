import {alignSequences, matchesAction, overlapDuration, packActionTracks, groupCallStack} from './investigator-model.js';
import {changedTrajectoryActions, mergeTrajectory, TrajectoryStream} from './trajectory-stream.js';

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const formatTime = value => value < 1 ? `${Math.round(value * 1000)}ms` : value >= 60 ? `${Math.floor(value / 60)}m ${(value % 60).toFixed(0)}s` : `${value.toFixed(1)}s`;
const count = value => value == null ? 'n/a' : new Intl.NumberFormat('en', {notation: value >= 10000 ? 'compact' : 'standard', maximumFractionDigits: 1}).format(value);
const colors = {search:'#70b4ed', read:'#89c7da', edit:'#72d6b1', test:'#9fbb84', review:'#a3bdcc', tool:'#70b4ed', model:'#e8bb72', setup:'#758ba2', queue:'#b39ee5', dependency:'#9b88c9', unknown:'#394858', cpu:'#72d6b1', io:'#83c5d5'};
const sources = {'System instructions':'#b39ee5','Original task':'#70b4ed','Supervisor instructions':'#70b4ed','Tool results':'#72d6b1','Files':'#8fc9d8','Conversation':'#e8bb72','Summary':'#d396b0'};
const views = [
  ['trajectory','Trajectory','What did the agent do?'], ['resources','Resources & waits','Where did the time go?'],
  ['context','Context & information','What did the model know?'], ['agents','Tandem trace board','How did agents work together?'],
  ['compare','Compare runs','Where did behavior change?']
];
const params = new URLSearchParams(location.search);
const state = {catalog:[], run:null, comparison:null, view:views.some(v=>v[0]===params.get('view')) ? params.get('view') : 'trajectory', trajectoryLayout:params.get('layout')==='overlay'?'overlay':'episodes', agent:'', query:'', cursor:0, selected:null, expanded:new Set(), model:null, contextBlock:null, zoom:null, loading:false, compareId:null};
let loadGeneration = 0;
let compareGeneration = 0;
let trajectoryUpdateTimer=null;
const pendingTrajectoryChanges=new Set();
let pendingResync=false;

function scheduleTrajectoryUpdate(changed,resynced=false) {
  for(const id of changed)pendingTrajectoryChanges.add(id);
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
Object.assign(state, {stream:null, replay:params.get('mode')==='replay', transport:'offline', follow:true, unread:new Set(), windowEnd:null, replayState:null, agentColumns:[], columnsCustomized:false, activeColumn:null});
const actionMap = run => new Map((run?.actions || []).map(a=>[a.id,a]));
const visibleActions = run => (run?.actions || []).filter(a=>matchesAction(a,state.agent,state.query));
const agentName = (id, run=state.run) => run?.agents.find(a=>a.id===id)?.label || id;
const pill = (label, kind=label) => `<span class="pill ${esc(kind.replace(/ /g,'-'))}">${esc(label)}</span>`;
const field = (name, value) => `<div class="detail-field"><span>${esc(name)}</span><span>${esc(value)}</span></div>`;
const heading = (title, description, extra='') => `<div class="section-heading"><div><h2>${esc(title)}</h2><p>${esc(description)}</p></div>${extra}</div>`;
const empty = (title, text) => `<div class="empty-state"><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
const legend = entries => `<div class="legend">${entries.map(([key,label])=>`<span><i style="background:${colors[key]||sources[key]}"></i>${esc(label)}</span>`).join('')}</div>`;
const insight = text => `<div class="insight"><span class="insight-icon">↳</span><div>${text}</div></div>`;

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
  state.stream?.close();state.stream=null;state.replay=replay;state.replayState=null;
  state.follow=true;state.unread.clear();state.windowEnd=null;
  state.agentColumns=[];state.columnsCustomized=false;state.activeColumn=null;
  state.run=null;state.agent='';state.cursor=0;state.selected=selected;state.expanded=new Set();state.contextBlock=null;state.model=null;state.zoom=null;
  state.comparison=null;state.compareId=null;state.compareError=null;compareGeneration++;
  $('agent-select').innerHTML='<option value="">All agents</option>';
  state.loading = true;
  $('content').innerHTML = '<div class="loading-state">Reading spans, prompt provenance and resource samples…</div>';
  $('status').textContent='Loading execution…';
  if(id==='live'||replay) {
    state.transport='connecting';renderSession();
    state.stream=new TrajectoryStream({runId:id,replay,onTransport:transport=>{if(generation===loadGeneration){state.transport=transport;renderSession();}},onMessage:message=>{
      if(generation!==loadGeneration)return;
      const first=!state.run;
      let changed=[];
      if(message.type==='snapshot') {
        if(state.run)changed=changedTrajectoryActions(state.run.actions,message.run.actions);
        state.run=message.run;state.loading=false;
      }
      else for(const patch of message.patches||[]) changed.push(...mergeTrajectory(state.run,patch));
      state.replayState=message.replay||null;
      if(!state.follow) for(const actionId of changed) state.unread.add(actionId);
      if(state.follow) state.cursor=message.replay?.clock??state.run.duration;
      if(first){catalogOptions();updateUrl();render();}
      else scheduleTrajectoryUpdate(changed,message.resynced);
    }});
    return;
  }
  state.transport='offline';
  try {
    const run = await fetchJson(`/api/investigator/runs/${encodeURIComponent(id)}`);
    if (generation!==loadGeneration) return;
    state.run=run; state.agent=''; state.cursor=0; state.selected=selected; state.expanded=new Set(); state.contextBlock=null; state.model=null; state.zoom=null; state.comparison=null;
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
  if(state.replay) url.searchParams.set('mode','replay');else url.searchParams.delete('mode');
  if(state.trajectoryLayout==='overlay') url.searchParams.set('layout','overlay');
  else url.searchParams.delete('layout');
  history.replaceState(null,'',url);
}

function render() {
  const run=state.run;
  if (!run || state.loading) return;
  $('views').innerHTML=views.map(([id,label],index)=>`<button data-view="${id}" class="${state.view===id?'active':''}" aria-current="${state.view===id?'page':'false'}"><span class="view-number">0${index+1}</span>${label}</button>`).join('');
  $('view-label').textContent=views.find(v=>v[0]===state.view)[1];
  renderRunChrome();
  const board=$('content').querySelector('#tandem-columns');
  const held=state.view==='agents'&&board?.dataset.run===run.id?{
    episodes:[...board.querySelectorAll('[data-tandem-episode]')].filter(el=>el.open).map(el=>el.dataset.tandemEpisode),
    scroll:[...board.children].map(el=>[el.dataset.tandemAgent,el.querySelector('.col-body').scrollTop])
  }:null;
  $('content').innerHTML=({trajectory:trajectoryView,resources:resourcesView,context:contextView,agents:agentsView,compare:compareView})[state.view]();
  if(state.view==='trajectory'&&state.trajectoryLayout==='episodes') reconcileActivities(true);
  if(state.view==='agents'){
    renderTandemBoard();
    if(held){
      for(const el of $('tandem-columns').querySelectorAll('[data-tandem-episode]'))el.open=held.episodes.includes(el.dataset.tandemEpisode);
      for(const [id,top] of held.scroll)$('tandem-columns').querySelector(`[data-tandem-agent="${CSS.escape(id)}"] .col-body`)?.scrollTo({top});
    }
  }
  renderInspector();
}

function renderRunChrome() {
  const run=state.run;
  if(!run)return;
  const catalogRun=state.catalog.find(item=>item.id===run.id);
  if(catalogRun)for(const key of ['model','harness'])catalogRun[key]=run[key];
  for(const [id,key] of [['model-filter','model'],['harness-filter','harness']]) {
    if(run[key]&&![...$(id).options].some(option=>option.value===run[key]))$(id).add(new Option(run[key],run[key]));
  }
  $('metrics').hidden=state.view!=='trajectory';
  $('run-title').textContent=run.title;
  $('run-subtitle').textContent=`${run.condition?.replace('_',' ') || (run.id==='live'?'current execution':state.replay?'recorded replay':'saved execution')} / ${run.harness} / ${run.model}`;
  const agentCount=run.agents.filter(a=>a.id!=='workflow').length;
  $('run-description').textContent=`${agentCount} recorded actor${agentCount===1?'':'s'} · ${run.actions.length} observed calls · ${run.episodes.length} activities`;
  $('provenance').innerHTML=pill(run.source==='synthetic'?'Synthetic fixture':'Recorded evidence',run.source)+`<div class="provenance-note">${run.source==='synthetic'?'All timings, context and relationships are synthetic.':'Activity grouping is derived; returned results remain observations.'}</div>`;
  const modelActions=run.actions.filter(a=>a.kind==='model');
  const knownTokens=modelActions.filter(a=>a.tokens!=null);
  const modelSpans=run.intervals.filter(i=>i.kind==='model');
  const unionWait=overlapDuration(modelSpans.length?modelSpans:modelActions,0,run.duration);
  $('metrics').innerHTML=[
    ['Wall time',formatTime(run.duration),run.status==='running'?'From execution start to now':'From execution start to finish'],
    ['Actions',String(run.actions.filter(a=>a.kind!=='model').length),`${modelActions.length} model calls · ${agentCount} agents`],
    ['Model intervals',formatTime(unionWait),`${run.duration ? Math.round(unionWait/run.duration*100) : 0}% wall time · union of observed spans`],
    ['Input tokens',knownTokens.length?count(knownTokens.reduce((sum,a)=>sum+a.tokens,0)):'n/a',knownTokens.length===modelActions.length&&modelActions.length?'Reported across all observed calls':`${knownTokens.length}/${modelActions.length} calls report usage`]
  ].map(([label,value,note])=>`<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${esc(note)}</div></div>`).join('');
  $('cursor').max=run.duration;
  $('cursor').value=state.cursor;
  $('cursor-value').textContent=formatTime(state.cursor);
  $('duration-label').textContent=formatTime(run.duration);
  const known=new Set([...$('agent-select').options].map(option=>option.value));
  for(const agent of run.agents)if(!known.has(agent.id)){$('agent-select').add(new Option(agent.label,agent.id));}
  renderSession();
  $('status').textContent=`${run.source==='synthetic'?'SYNTHETIC':'RECORDED'} · ${count(run.coverage.events)} source events · ${visibleActions(run).length} actions in current filter${state.selected?' · selection linked across views':''}`;
}

function actionButton(action) {
  return `<button class="action-row ${state.selected?.id===action.id?'selected':''}" data-action="${esc(action.id)}"><span class="action-time">${formatTime(action.start)}</span><span class="action-kind" style="color:${colors[action.kind]}">${esc(action.kind)}</span><span class="action-label">${esc(action.intent||action.name)}<small>${esc(action.command || (action.tokens==null?'Token usage not recorded':`${count(action.tokens)} input → ${count(action.output_tokens)} output tokens`))}</small></span><span class="action-duration">${formatTime(action.duration)}</span><span class="status-icon ${action.outcome}">${action.outcome==='success'?'✓':action.outcome==='failed'?'×':'·'}</span></button>`;
}

function trajectoryView() {
  const overlay=state.trajectoryLayout==='overlay';
  const controls=`<label class="trajectory-layout">Layout<select id="trajectory-layout" aria-label="Trajectory layout"><option value="episodes" ${overlay?'':'selected'}>Episode list</option><option value="overlay" ${overlay?'selected':''}>Agent overlay</option></select></label>`;
  const header=heading('Execution trajectory',overlay?'Compare agent actions on a shared execution clock.':'A narrative of the run, grouped by action intent.',controls);
  const key=legend([['search','Search'],['read','Read'],['edit','Edit'],['test','Test'],['model','Model call']]);
  if(overlay) return header+trajectoryOverlay(visibleActions(state.run),key)+trajectoryWorkflow();
  return header+`<div id="trajectory-signals"></div><div class="agent-column-controls"><div id="agent-tabs" role="tablist" aria-label="Visible agents"></div><label>Add agent<select id="add-agent-column" aria-label="Add agent column"></select></label><span id="agent-column-limit" class="muted"></span></div><div id="hidden-agents" aria-label="Hidden agent activity"></div><div id="call-stack" class="agent-columns"></div><details class="activity-history"><summary>Activities grouped by intent</summary><div id="trajectory-glance"></div><div id="trajectory-workflow"></div><div class="activity-window"><button data-older-activities>Earlier activities</button><span id="activity-window-label" class="mono muted"></span><button data-latest-activities>Latest activities</button></div><div id="activity-list"></div><div class="notice">Activity labels use declared purpose when available, otherwise deterministic tool categories. Returned evidence is quoted below each activity. Tool success does not establish task success. Only 50 activities are rendered at once.</div></details>`;
}

function renderSession() {
  const run=state.run, replay=state.replay;
  const mode=replay?'Replay':run?.id==='live'?'Live':'Review';
  const status=run?.status||'unknown';
  $('execution-state').textContent=`${mode} · ${status==='unknown'?'execution state unrecorded':status}`;
  $('transport-state').textContent=state.stream?`${state.transport==='connected'?'Connected to updates':state.transport==='disconnected'?'Disconnected from updates · execution state retained':state.transport==='reconnecting'?'Reconnecting to updates':'Connecting to updates'}${replay&&state.replayState?` · ${state.replayState.finished?'replay exhausted':state.replayState.playing?'playing':'paused'} at ${formatTime(state.replayState.clock)}`:''}`:'Saved evidence · no live subscription';
  $('execution-state').dataset.mode=mode.toLowerCase();
  $('open-live').hidden=run?.id==='live';
  $('start-replay').hidden=!run||run.id==='live'||replay;
  for(const id of ['replay-play','replay-step','replay-restart','replay-speed'])$(id).hidden=!replay;
  $('replay-play').textContent=state.replayState?.playing?'Pause replay':'Play replay';
  $('replay-play').disabled=Boolean(state.replayState?.finished)||state.transport!=='connected';
  $('replay-step').disabled=Boolean(state.replayState?.finished)||state.transport!=='connected';
  $('follow-live').hidden=!state.stream;
  $('follow-live').textContent=state.follow?'Following latest':state.unread.size?`${state.unread.size} calls changed · Follow latest`:'History held · Follow latest';
  $('follow-live').setAttribute('aria-pressed',String(state.follow));
  $('reconnect-updates').hidden=!state.stream;
}

function updateTrajectory(changed,resynced=false) {
  renderRunChrome();
  if(state.view==='trajectory') {
    if(state.trajectoryLayout==='episodes')reconcileActivities();
    else {
      const focused=document.activeElement?.getAttribute('data-action');
      const scroll=$('content').querySelector('.trajectory-scroll')?.scrollLeft||0;
      $('content').innerHTML=trajectoryView();
      const region=$('content').querySelector('.trajectory-scroll');if(region)region.scrollLeft=scroll;
      if(focused)$('content').querySelector(`[data-action="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
    }
  }
  if(state.view==='agents')renderTandemBoard();
  if((!state.selected||changed.includes(state.selected?.id))&&!$('inspector').contains(document.activeElement))renderInspector(true);
  if(resynced)$('transport-state').textContent+=' · recovered from durable snapshot';
}

const MAX_AGENT_COLUMNS=3;
function trajectoryAgents() {
  return [...new Set([...state.run.agents.map(agent=>agent.id),...state.run.actions.map(action=>action.agent)])].filter(id=>id&&id!=='workflow');
}

function visibleAgentColumns() {
  const agents=trajectoryAgents();
  state.agentColumns=state.agentColumns.filter(id=>agents.includes(id));
  if(!state.columnsCustomized)for(const id of agents)if(state.agentColumns.length<2&&!state.agentColumns.includes(id))state.agentColumns.push(id);
  const columns=state.agent?[state.agent]:state.agentColumns;
  if(!columns.includes(state.activeColumn))state.activeColumn=columns[0]||null;
  return columns;
}

function addAgentColumn(id) {
  if(!trajectoryAgents().includes(id)||state.agentColumns.includes(id)||state.agentColumns.length>=MAX_AGENT_COLUMNS)return;
  state.columnsCustomized=true;state.agentColumns.push(id);state.activeColumn=id;
  state.agent='';$('agent-select').value='';
  if(state.view==='agents')renderTandemBoard();else renderCallStack();
}

function removeAgentColumn(id) {
  if(state.agentColumns.length<=1)return;
  state.columnsCustomized=true;state.agentColumns=state.agentColumns.filter(agent=>agent!==id);
  if(state.view==='agents')renderTandemBoard();else renderCallStack();
}

function selectAgentColumn(id) {
  if(!visibleAgentColumns().includes(id))return;
  state.activeColumn=id;if(state.view==='agents')renderTandemBoard();else renderCallStack();
}

function agentActivityCounts(id) {
  const actions=state.run.actions.filter(action=>action.agent===id);
  return `${actions.filter(action=>action.outcome==='running').length} running · ${actions.filter(action=>action.outcome==='failed').length} failed`;
}

function renderAgentColumnControls(columns) {
  const agents=trajectoryAgents(),hidden=agents.filter(id=>!columns.includes(id));
  const full=state.agentColumns.length>=MAX_AGENT_COLUMNS;
  const picker=$('add-agent-column');
  const options='<option value="">Add agent…</option>'+hidden.map(id=>`<option value="${esc(id)}">${esc(agentName(id))}</option>`).join('');
  if(picker.innerHTML!==options)picker.innerHTML=options;
  picker.disabled=full||!hidden.length;
  $('agent-column-limit').textContent=`${columns.length} visible · max ${MAX_AGENT_COLUMNS}`;
  const tabs=columns.map(id=>`<button role="tab" aria-selected="${id===state.activeColumn}" tabindex="${id===state.activeColumn?0:-1}" data-agent-tab="${esc(id)}">${esc(agentName(id))}<small>${agentActivityCounts(id)}</small></button>`).join('');
  if($('agent-tabs').innerHTML!==tabs)$('agent-tabs').innerHTML=tabs;
  const badges=hidden.map(id=>{
    return `<button data-add-column="${esc(id)}" ${full?'disabled':''}>${esc(agentName(id))} · ${agentActivityCounts(id)}</button>`;
  }).join('');
  if($('hidden-agents').innerHTML!==badges)$('hidden-agents').innerHTML=badges;
}

function renderCallStack() {
  const stack=$('call-stack');if(!stack)return;
  const columns=visibleAgentColumns();
  renderAgentColumnControls(columns);
  stack.style.setProperty('--agent-column-count',Math.max(1,columns.length));
  const oldColumns=new Map([...stack.querySelectorAll('[data-agent-column]')].map(el=>[el.dataset.agentColumn,el]));
  const existing=new Map([...stack.querySelectorAll('[data-call-group]')].map(el=>[el.dataset.callGroup,el]));
  const cards=new Map([...stack.querySelectorAll('[data-call-card]')].map(el=>[el.dataset.callCard,el]));
  const positions=new Map([...cards].map(([id,el])=>[id,el.getBoundingClientRect?.()]));
  const focused=stack.contains(document.activeElement)?document.activeElement?.getAttribute('data-action'):null;
  const visibleCards=new Set(),visibleGroups=new Set();
  const actions=visibleActions(state.run);
  for(const [columnIndex,agent] of columns.entries()) {
    let column=oldColumns.get(agent);
    if(!column){column=document.createElement('section');column.dataset.agentColumn=agent;column.innerHTML='<header class="agent-column-heading"></header><div class="agent-call-stack"></div>';}
    column.className=`agent-column ${agent===state.activeColumn?'active-agent-column':''}`;
    const heading=column.querySelector('.agent-column-heading');
    const running=state.run.actions.filter(action=>action.agent===agent&&action.outcome==='running').length;
    const html=`<strong>${esc(agentName(agent))}</strong><span>${running} running</span><button data-remove-column="${esc(agent)}" aria-label="Hide ${esc(agentName(agent))}" ${state.agent||columns.length<=1?'disabled':''}>Hide</button>`;
    if(heading.innerHTML!==html)heading.innerHTML=html;
    if(stack.children[columnIndex]!==column)stack.insertBefore(column,stack.children[columnIndex]||null);
    const feed=column.querySelector('.agent-call-stack');
    const groups=groupCallStack(actions.filter(action=>action.agent===agent));
    const visible=groups.filter(group=>group.running).concat(groups.filter(group=>!group.running).slice(0,50));
    for(const [position,group] of visible.entries()) {
      visibleGroups.add(group.id);
      let section=existing.get(group.id);
      if(!section){section=document.createElement('section');section.dataset.callGroup=group.id;section.innerHTML='<div class="call-group-heading"></div><div class="call-group-grid"></div>';}
      section.className=`call-stack-group ${group.running?'running-group':''}`;
      section.querySelector('.call-group-heading').textContent=`${group.running?'Running':'Finished'}${group.actions.length>1?' · Concurrent calls':''}`;
      const grid=section.querySelector('.call-group-grid');
      for(const [index,action] of group.actions.entries()) {
        visibleCards.add(action.id);
        let card=cards.get(action.id);
        if(!card){card=document.createElement('div');card.className='call-card';card.dataset.callCard=action.id;cards.set(action.id,card);}
        const html=actionButton(action)+`<div class="call-card-status">${pill(action.outcome)}${action.result_preview||action.result?`<p>${esc((action.result_preview||action.result).slice(0,240))}</p>`:''}</div>`;
        if(card.innerHTML!==html)card.innerHTML=html;
        if(grid.children[index]!==card)grid.insertBefore(card,grid.children[index]||null);
      }
      if(feed.children[position]!==section)feed.insertBefore(section,feed.children[position]||null);
    }
    if(!visible.length)feed.innerHTML=empty('No matching calls','Waiting for recorded activity for this agent.');
    else feed.querySelector('.empty-state')?.remove();
  }
  for(const [id,column] of oldColumns)if(!columns.includes(id))column.remove();
  for(const [id,el] of existing)if(!visibleGroups.has(id))el.remove();
  for(const [id,card] of cards)if(!visibleCards.has(id))card.remove();
  if(!columns.length)stack.innerHTML=empty('Waiting for recorded agents','Agent columns will appear when execution begins.');
  else stack.querySelector('.empty-state')?.remove();
  if(focused)stack.querySelector(`[data-action="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
  if(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)return;
  for(const id of visibleCards) {
    const card=cards.get(id),before=positions.get(id);
    if(!card.animate||!card.getBoundingClientRect)continue;
    card.getAnimations?.().forEach(animation=>animation.cancel());
    const after=card.getBoundingClientRect();
    if(before) {
      const x=before.left-after.left,y=before.top-after.top;
      if(Math.abs(x)+Math.abs(y)>1)card.animate([{transform:`translate(${x}px,${y}px)`},{transform:'translate(0,0)'}],{duration:220,easing:'ease-out'});
    } else card.animate([{opacity:0,transform:'translateY(-6px)'},{opacity:1,transform:'translateY(0)'}],{duration:180,easing:'ease-out'});
  }
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
    card.classList.toggle('selected',state.selected?.id===episode.id||actions.some(a=>a.id===state.selected?.id));
    card.classList.toggle('new-evidence',actions.some(a=>state.unread.has(a.id)));
    card.querySelector('.episode-number').textContent=run.episodes.indexOf(episode)+1;
    const summary=`<span class="chevron">›</span><div class="episode-title"><h3>${esc(episode.title)}</h3><p>${esc(agentName(episode.agent))} · ${episode.label_source==='declared'?'Declared purpose':'Inferred activity'}${episode.late?' · late telemetry':''}</p><div class="activity-result"><span>${episode.kind==='model'?'Model reply':'Observed'}</span> ${esc(episode.latest_result||'No result recorded yet.')}</div></div>${pill(episode.status)}`;
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
  const current=active.length?active.slice(-3).map(a=>`<div><b>${esc(agentName(a.agent))}</b><span>${esc(a.intent||run.episodes.find(e=>e.id===a.episode)?.title||a.name)}</span><small>${esc(a.kind)} · running for ${formatTime(a.duration)}</small></div>`).join(''):`<p>${input?'Waiting for requested human input. '+esc(input.explanation):run.status==='completed'?'Execution finished. Review the returned evidence; task correctness may remain unverified.':run.status==='cancelled'?'Execution cancelled. Unfinished spans retain incomplete telemetry.':run.status==='failed'?'Execution reported a failure. Inspect the evidence below.':state.replayState?.finished?'Recording exhausted. The final execution state was not recorded.':'No active call is recorded. A quiet stream alone does not establish a stall.'}</p>`;
  const glance=`<section class="trajectory-glance">${run.task?`<div class="detail-label">Requested task</div><p class="requested-task">${esc(String(run.task).slice(0,400))}</p>`:''}<div class="detail-label activity-overview-label">${!state.follow&&state.stream?'Overview held at selection':active.length?'Currently recorded':'Current situation'}</div><div class="current-activities">${current}</div>${latest?`<div class="latest-evidence"><span>${latest.kind==='model'?'Latest visible model reply':'Latest returned evidence'}</span><button data-action="${esc(latest.id)}">${esc((latest.result_preview||latest.result).slice(0,180))} ↗</button></div>`:''}</section>`;
  if((state.follow||initial)&&$('trajectory-glance').innerHTML!==glance)$('trajectory-glance').innerHTML=glance;
  const activeSignals=(run.signals||[]).filter(s=>s.active!==false);
  const repeatedEvidence=new Set(activeSignals.filter(s=>s.title==='Repeated matching failures').flatMap(s=>s.evidence));
  const signals=activeSignals.filter(s=>s.title!=='Recorded call failure'||!s.evidence.every(id=>repeatedEvidence.has(id)));
  const gap=(run.coverage.catching_up?'<div class="notice">Recovering retained events. This is a partial history until catch-up completes.</div>':'')+(run.coverage.gaps?.length?`<div class="attention-signal"><strong>Telemetry coverage is incomplete</strong><p>${esc(run.coverage.gaps.join(' · '))}. Execution state may be unknown.</p></div>`:'');
  const signalHtml=gap+signals.slice(-5).map(signal=>`<div class="attention-signal ${signal.severity==='required'?'required':''}"><strong>${signal.severity==='required'?'Action requested':'Worth inspecting'} · ${esc(signal.title)}</strong><p>${esc(signal.explanation)}</p><div>${signal.evidence.map((id,i)=>`<button ${map.has(id)?'data-action':'data-raw-event'}="${esc(id)}">Evidence ${i+1} ↗</button>`).join('')}</div></div>`).join('');
  if((state.follow||initial)&&$('trajectory-signals').innerHTML!==signalHtml&&!$('trajectory-signals').contains(document.activeElement))$('trajectory-signals').innerHTML=signalHtml;
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
  const selectedEpisode=state.selected?.type==='episode'?state.run.episodes.find(e=>e.id===state.selected.id):null;
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
        const selected=state.selected?.id===action.id||selectedEpisode?.actions.includes(action.id);
        const description=`${name} · ${action.intent||action.name} · ${formatTime(action.start)} → ${formatTime(action.start+action.duration)} · ${action.outcome}`;
        svg+=`<g class="trajectory-action ${selected?'selected':''} ${action.outcome==='failed'?'failed':''}" data-action="${esc(action.id)}" tabindex="0" role="button" aria-label="${esc(description)}"><title>${esc(description)}</title><rect x="${x(action.start)}" y="${y}" width="${w}" height="24" rx="4" fill="${colors[action.kind]||colors.tool}"/>${w>action.kind.length*6+14?`<text x="${x(action.start)+7}" y="${y+16}">${esc(action.kind)}</text>`:''}</g>`;
      }
    });
    svg+=`<line class="trajectory-divider" x1="0" x2="${width}" y1="${lane.y+lane.height-15}" y2="${lane.y+lane.height-15}"/>`;
  }
  svg+=`<line class="cursor-line" x1="${x(state.cursor)}" x2="${x(state.cursor)}" y1="24" y2="${height-14}"/></svg>`;
  return insight('<strong>See who worked in parallel.</strong> Every agent shares the same time axis. Select an action to inspect its command, result and episode.')+
    `<div class="panel"><div class="panel-heading"><h3>Agent trajectories</h3><span class="mono muted">${lanes.length} AGENTS · ${actions.length} ACTIONS</span></div>${key}<div class="trajectory-scroll" tabindex="0" role="region" aria-label="Scrollable agent trajectory chart">${svg}</div><div class="trajectory-caption">Bar width = action duration · Dashed line = time between actions · Light vertical line = shared time cursor</div></div>`+
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
      const selected=state.selected?.id===i.id || state.selected?.interval===i.id || state.selected?.type==='episode' && state.run.episodes.find(e=>e.id===state.selected.id)?.actions.includes(i.action);
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
  return calls.find(a=>a.id===state.model) || calls.find(a=>a.id===state.selected?.id) || calls.filter(a=>a.start<=state.cursor).at(-1) || calls[0];
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
  return `<summary><span>${esc(episode.title)}</span>${pill(episode.status)}</summary><div class="seg-items">${actions.map(a=>`<article class="item ${a.kind==='model'?'text':'tool_call'} ${a.outcome==='failed'?'trace-failed':''}"><div class="lbl"><button data-action="${esc(a.id)}">${esc(a.kind)} · ${esc(a.name)}</button><span>${formatTime(a.start)} · ${formatTime(a.duration)}</span></div>${a.intent?`<p>${esc(a.intent)}</p>`:''}${a.command?`<pre>${esc(a.command)}</pre>`:''}${a.result?`<div class="tool-result"><span class="lbl">Observed result</span><pre>${esc(a.result)}</pre></div>`:''}<div class="trace-metrics">${pill(a.outcome)}${a.tokens!=null?`<span>${count(a.tokens)} input · ${count(a.output_tokens)} output tokens</span>`:''}<button data-action="${esc(a.id)}">Inspect evidence ↗</button></div></article>`).join('')}</div>`;
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
    const header=`<div class="agent"><button data-agent="${esc(agent.id)}">${esc(agent.label)}</button><button data-remove-column="${esc(agent.id)}" ${state.agent||columns.length===1?'disabled':''}>Hide</button></div><div class="meta"><span class="badge ${parent?'sup':'wrk'}">${role}</span><span class="badge">${esc(agent.model||run.model)}</span></div><p class="muted">${agentActivityCounts(agent.id)}${agent.parent?` · parent ${esc(agentName(agent.parent))}`:''}</p>`;
    const head=col.querySelector('.col-head');if(head.innerHTML!==header)head.innerHTML=header;
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
  const cell=(action,side)=>action?`<button class="align-cell" data-compare-action="${esc(action.id)}" data-side="${side}"><strong><span style="color:${colors[action.kind]}">${esc(action.kind)}</span> · ${esc(action.intent||action.command||action.name)}</strong><small>${formatTime(action.start)} · ${formatTime(action.duration)} · ${esc(action.outcome)}</small></button>`:'<div class="align-cell"><small>— No matching action</small></div>';
  return heading('Compare executions','Aligned action categories reveal insertion, deletion and reconvergence.')+controls+
    insight(first<0?`<strong>The observed action categories remain aligned.</strong> Run B takes ${formatTime(Math.abs(delta))} ${delta>=0?'longer':'less time'}. Commands, results and timings can still differ.`:`<strong>First divergence at aligned step ${first+1}.</strong> ${rows.filter(r=>!r.match).length} inserted / deleted actions. Select either side to inspect the underlying command.`)+
    `<div class="compare-metadata"><div>${metadata(a)}</div><div>${metadata(b)}</div></div>`+
    `<div class="panel">${field('Category similarity',similarity==null?'n/a':`${Math.round(similarity*100)}% · 2 × matched / total actions`)}${field('Wall-time change',`${delta>=0?'+':'−'}${formatTime(Math.abs(delta))}`)}${field('Different referenced paths',fileDelta.length?fileDelta.slice(0,4).join(', '):'Same observed path set')}${resourceField('CPU utilization A → B',cpuA,cpuB,'%')}${resourceField('Memory A → B',memA,memB,'MB')}</div>`+
    `<div class="alignment"><div class="alignment-heading"><span>A · current execution</span><span></span><span>B · comparison</span></div>${rows.length?rows.map((row,index)=>`<div class="alignment-row ${row.match?'':'divergent'} ${row.reconverged?'reconverged':''}" title="${row.reconverged?'Categories reconverge':row.match?'Matching action category':'Trajectory divergence'}">${cell(row.left,'a')}<div class="alignment-connector"><span>${row.reconverged?'↔':row.match?'=':'≠'}</span>${row.left&&row.right?`<small>${row.right.duration>=row.left.duration?'+':'−'}${formatTime(Math.abs(row.right.duration-row.left.duration))}</small>`:''}</div>${cell(row.right,'b')}</div>`).join(''):empty('No comparable actions','Clear search and agent filters.')}</div>`+
    `<div class="notice">Alignment uses the longest common subsequence of tool action categories, following the saved experiment’s category comparison. Matching categories do not imply identical commands, behavior or correctness. Different tasks can be compared; interpretation requires care. Referenced file paths are extracted from command text, not a verified artifact inventory.</div>`;
}

function selectedInterval() {
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

function renderInspector(preserveEvidence=false) {
  const top='<div class="inspector-top"><span>SELECTION INSPECTOR</span><span>↗ LINKED</span></div>';
  const run=state.selected?.run==='b'?state.comparison:state.run;
  const map=actionMap(run);
  let content='';
  const selection=state.selected;
  if(!selection) {
    content=`<div class="empty-inspector-icon">⌖</div><h2>Follow the evidence.</h2><p>Select an episode, action, model call or obligation. Its details stay here as you move between views.</p><ul class="inspector-list"><li><b>01</b> Expand an episode in the trajectory</li><li><b>02</b> Correlate its resource interval</li><li><b>03</b> Inspect the model’s information</li></ul><div class="detail-label">Measurement coverage</div>${field('Source events',count(run.coverage.events))}${field('Timed spans',['live','replay'].includes(run.mode)?'Not collected in this event projection':count(run.coverage.spans))}<p>${esc(run.coverage.context)}</p><button class="jump" data-view="resources">Inspect resources →</button><button class="jump" data-view="context">Inspect context →</button>`;
  } else if(selection.type==='action') {
    const a=map.get(selection.id);
    if(!a) content='<p>Selection is no longer available.</p>';
    else {
      content=`${pill(a.kind,'neutral')} ${pill(a.outcome)}<h2>${esc(a.intent||a.name)}</h2><p>${esc(agentName(a.agent,run))}</p>${field('Start',formatTime(a.start))}${field('Duration',formatTime(a.duration))}${field('Timing',a.timing)}${field('Data source',a.source)}${a.tokens!=null?field('Input / output tokens',`${count(a.tokens)} / ${count(a.output_tokens)}`):''}${a.command?`<div class="detail-label">Command / input</div><pre>${esc(a.command)}</pre>`:''}${a.result?`<div class="detail-label">Observed result</div><pre>${esc(a.result)}</pre>`:''}`;
      content+=field('Lifecycle',a.missing_start?'End recorded; start missing':a.outcome==='running'?'Started; end not recorded yet':a.outcome==='incomplete'?'End missing at final execution boundary':a.outcome);
      if(a.intent)content+=`<div class="detail-label">Declared purpose</div><p>${esc(a.intent)}</p>`;
      if(a.kind!=='model')content+=a.model_id?`<button class="jump" data-action="${esc(a.model_id)}">Related model invocation (explicit call ID) →</button>`:field('Model linkage','Unlinked; timing alone is not attribution');
      if(a.artifacts?.length)content+=`<div class="detail-label">Recorded artifacts</div><pre>${esc(JSON.stringify(a.artifacts,null,2))}</pre>`;
      if(a.event_ids?.length)content+=`<div class="detail-label">Immutable source evidence</div>${a.event_ids.slice(-10).map((id,i)=>`<button class="jump" data-raw-event="${esc(id)}">Source event ${i+1} · ${esc(id.slice(-16))} ↗</button>`).join('')}`;
      if(selection.interval&&selection.interval!==selection.id){const interval=run.intervals.find(i=>i.id===selection.interval);if(interval)content+=`<div class="detail-label">Linked wait</div><p>${esc(interval.label)}</p>${field('Wait interval',`${formatTime(interval.start)} → ${formatTime(interval.start+interval.duration)}`)}`;}
      if(state.contextBlock) content+=`<div class="detail-label">${esc(state.contextBlock.source)} · ${esc(state.contextBlock.change)}</div><pre>${esc(state.contextBlock.preview)}</pre>`;
      if(a.files.length) content+=`<div class="detail-label">Referenced files</div>${a.files.map(f=>`<span class="file-chip">${esc(f)}</span>`).join('')}`;
      if(selection.run==='b') content+='<button class="jump" data-open-comparison>Open B in trajectory →</button>';
      else content+=`<button class="jump" data-view="trajectory">Locate in trajectory →</button><button class="jump" data-view="resources">Correlate resource interval →</button>${a.kind==='model'?'<button class="jump" data-view="context">Inspect this model’s context →</button>':''}`;
      content+=`<details><summary class="detail-label">Event metadata</summary><pre>${esc(JSON.stringify(a.metadata,null,2))}</pre></details>`;
    }
  } else if(selection.type==='episode') {
    const episode=run.episodes.find(e=>e.id===selection.id);
    content=`${pill(episode.status)}<h2>${esc(episode.title)}</h2><p>${esc(agentName(episode.agent))}</p>${field('Interval',`${formatTime(episode.start)} → ${formatTime(episode.end)}`)}${field('Grouping',episode.label_source==='declared'?'Declared purpose + structural boundaries':'Deterministic tool categories + actor boundaries')}${field('Actions',episode.actions.length)}<div class="detail-label">Observed evidence</div><p>${esc(episode.latest_result||'No returned result recorded.')}</p><div class="detail-label">Calls</div>${episode.actions.map(id=>map.get(id)).filter(Boolean).map(a=>`<button class="jump" data-action="${esc(a.id)}">${esc(a.kind)} · ${esc(a.intent||a.name)} →</button>`).join('')}<button class="jump" data-view="resources">Correlate this episode →</button>`;
  } else if(selection.type==='obligation') {
    const o=run.obligations.find(o=>o.id===selection.id);
    content=`${pill(o.status)}<h2>${esc(o.title)}</h2><p>${esc(o.note)}</p>${field('Obligation source',o.inferred?'Inferred from actions':'Explicit / evaluator')}<div class="detail-label">Supporting evidence</div>${o.evidence.map(id=>map.get(id)).filter(Boolean).map(a=>`<button class="jump" data-action="${esc(a.id)}">${esc(a.kind)} · ${formatTime(a.start)} · ${esc(a.outcome)} →</button>`).join('')||'<p>No supporting action has been recorded.</p>'}<button class="jump" data-view="trajectory">Follow the trajectory →</button>`;
  } else if(selection.type==='agent') {
    const agent=run.agents.find(a=>a.id===selection.id);
    content=`${pill('Agent')}<h2>${esc(agent.label)}</h2><p>${esc(agent.role)}</p>${field('Model',agent.model||'unavailable')}${field('Harness',agent.harness||'unavailable')}${field('Parent',agent.parent?agentName(agent.parent):'No recorded parent')}${field('Actions',run.actions.filter(a=>a.agent===agent.id).length)}<button class="jump" data-view="trajectory">Inspect agent trajectory →</button><button class="jump" data-view="context">Inspect agent context →</button>`;
  } else if(selection.type==='interval') {
    const i=run.intervals.find(i=>i.id===selection.id);
    content=`${pill(i.kind)}<h2>${esc(i.label)}</h2>${field('Start',formatTime(i.start))}${field('Duration',formatTime(i.duration))}${field('Source',i.source)}<p>This interval has no observed action attribution. It may contain uninstrumented work, transport, waiting or idle time.</p>`;
  } else if(selection.type==='artifact') {
    content=`${pill('Referenced artifact')}<h2 class="mono">${esc(selection.id)}</h2><p>Extracted from recorded command text.</p>${run.actions.filter(a=>a.files.includes(selection.id)).map(a=>`<button class="jump" data-action="${esc(a.id)}">${esc(a.kind)} · ${formatTime(a.start)} →</button>`).join('')}`;
  }
  const raw=preserveEvidence?$('raw-evidence'):null;
  $('inspector').innerHTML=top+content;
  if(raw)$('inspector').append(raw);
}

function selectAction(id, comparison=false, intervalId=null) {
  const run=comparison?state.comparison:state.run;
  const action=actionMap(run).get(id);
  if(!action) return;
  holdSelection();
  const episodeIndex=relevantEpisodes().findIndex(e=>e.id===action.episode);
  if(episodeIndex>=0&&state.windowEnd!=null&&(episodeIndex>=state.windowEnd||episodeIndex<state.windowEnd-50))state.windowEnd=episodeIndex+1;
  state.selected={type:'action',id,run:comparison?'b':'a',interval:intervalId};state.cursor=intervalId?state.run.intervals.find(i=>i.id===intervalId).start:action.start;state.contextBlock=null;
  if(action.kind==='model') state.model=id;
  if(!comparison) state.expanded.add(action.episode);
  render();
}

function setView(view) {
  if(!views.some(v=>v[0]===view))return;
  if(state.selected?.run==='b'&&view!=='compare') {
    const runId=state.comparison.id,id=state.selected.id;
    state.view=view;loadRun(runId,{type:'action',id});return;
  }
  state.view=view;updateUrl();render();
  if(view==='compare'&&!state.comparison) loadComparison();
}

function handleClick(event) {
  const el=event.target.closest('[data-add-column],[data-remove-column],[data-agent-tab],[data-view],[data-action],[data-episode],[data-interval],[data-agent],[data-model],[data-block],[data-source],[data-obligation],[data-artifact],[data-compare-action],[data-open-comparison],[data-zoom],[data-reset-zoom],[data-retry],[data-demo],[data-older-activities],[data-latest-activities],[data-raw-event]');
  if(!el||!state.run&&!(el.hasAttribute('data-retry')||el.hasAttribute('data-demo'))) return;
  if(el.dataset.addColumn)addAgentColumn(el.dataset.addColumn);
  else if(el.dataset.removeColumn)removeAgentColumn(el.dataset.removeColumn);
  else if(el.dataset.agentTab)selectAgentColumn(el.dataset.agentTab);
  else if(el.dataset.view) setView(el.dataset.view);
  else if(el.dataset.action) selectAction(el.dataset.action);
  else if(el.dataset.episode) {
    event.preventDefault();
    holdSelection();
    const id=el.dataset.episode;
    state.expanded.has(id)?state.expanded.delete(id):state.expanded.add(id);
    state.selected={type:'episode',id};state.cursor=state.run.episodes.find(e=>e.id===id).start;state.contextBlock=null;render();
  } else if(el.dataset.interval) {
    const interval=state.run.intervals.find(i=>i.id===el.dataset.interval);
    if(interval?.action) selectAction(interval.action,false,interval.id);
    else if(interval) {state.selected={type:'interval',id:interval.id};state.cursor=interval.start;render();}
  } else if(el.dataset.agent) {
    state.agent=el.dataset.agent;state.selected={type:'agent',id:state.agent};$('agent-select').value=state.agent;render();
  } else if(el.dataset.model) selectAction(el.dataset.model);
  else if(el.hasAttribute('data-block')) {
    const a=actionMap(state.run).get(el.dataset.modelId);
    const block=a.context.blocks.concat(a.context.dropped)[Number(el.dataset.block)];
    state.model=a.id;state.selected={type:'action',id:a.id};state.contextBlock=block;state.cursor=a.start;render();
  } else if(el.dataset.source) {
    const a=activeModel();state.selected={type:'action',id:a.id};state.model=a.id;state.contextBlock=a.context.blocks.find(b=>b.source===el.dataset.source);render();
  } else if(el.dataset.obligation) {state.selected={type:'obligation',id:el.dataset.obligation};render();}
  else if(el.dataset.artifact) {state.selected={type:'artifact',id:el.dataset.artifact};render();}
  else if(el.dataset.compareAction) selectAction(el.dataset.compareAction,el.dataset.side==='b');
  else if(el.hasAttribute('data-open-comparison')) {const id=state.selected.id;state.view='trajectory';loadRun(state.comparison.id,{type:'action',id});}
  else if(el.hasAttribute('data-zoom')) {const interval=selectedInterval();if(interval){state.zoom=[Math.max(0,interval[0]-1),Math.min(state.run.duration,interval[1]+1)];render();}else{$('status').textContent='Select an episode or span to focus its interval.';}}
  else if(el.hasAttribute('data-reset-zoom')) {state.zoom=null;render();}
  else if(el.hasAttribute('data-retry')) refresh();
  else if(el.hasAttribute('data-demo')) loadRun('demo-baseline');
  else if(el.hasAttribute('data-older-activities')) {holdHistory();state.windowEnd=Math.max(50,(state.windowEnd??relevantEpisodes().length)-50);render();}
  else if(el.hasAttribute('data-latest-activities')) {state.windowEnd=null;render();}
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
  state.follow=true;state.windowEnd=null;state.unread.clear();
  if(state.run)state.cursor=state.replayState?.clock??state.run.duration;
  renderRunChrome();
  if(state.view==='trajectory'&&state.trajectoryLayout==='episodes') {
    reconcileActivities();
  }
  if(state.view==='agents')renderTandemBoard();
}

async function showRawEvidence(id) {
  holdSelection();
  const runId=state.run.id;
  const url=runId==='live'?`/api/trajectory/live/events/${encodeURIComponent(id)}`:`/api/trajectory/replay/${encodeURIComponent(runId)}/events/${encodeURIComponent(id)}`;
  let panel=$('raw-evidence');
  if(!panel){panel=document.createElement('section');panel.id='raw-evidence';$('inspector').append(panel);}
  panel.innerHTML='<div class="detail-label">Source event</div><p>Reading immutable evidence…</p>';
  try{const raw=await fetchJson(url);if(state.run.id===runId&&panel.isConnected)panel.innerHTML=`<div class="detail-label">Immutable source event</div><pre>${esc(JSON.stringify(raw,null,2))}</pre>`;}
  catch(error){if(panel.isConnected)panel.innerHTML=`<p>${esc(error.message)}</p>`;}
}

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
  if(event.target.matches('input,select,textarea')) return;
  if(event.key>='1'&&Number(event.key)<=views.length&&!event.metaKey&&!event.ctrlKey) {event.preventDefault();setView(views[Number(event.key)-1][0]);}
  if(event.key==='Escape') {state.selected=null;state.contextBlock=null;render();}
  if((event.key==='Enter'||event.key===' ')&&event.target.matches('[data-interval],.trajectory-action')) {event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
});
$('run-select').onchange=event=>loadRun(event.target.value);
for(const id of ['model-filter','harness-filter']) $(id).onchange=()=>{const runs=catalogOptions();if(runs.length&&!runs.some(r=>r.id===state.run?.id)) loadRun(runs[0].id);else if(!runs.length){$('status').textContent='No executions match these model and harness filters.';}};
$('agent-select').onchange=event=>{holdSelection();state.agent=event.target.value;state.selected=state.agent?{type:'agent',id:state.agent}:null;render();};
$('search').oninput=event=>{holdSelection();state.query=event.target.value;render();};
$('cursor').oninput=event=>{holdHistory();state.cursor=Number(event.target.value);state.model=null;$('cursor-value').textContent=formatTime(state.cursor);if(state.view==='resources'||state.view==='agents'||state.view==='context'||state.view==='trajectory'&&state.trajectoryLayout==='overlay') render();};
$('clear-selection').onclick=()=>{state.selected=null;state.contextBlock=null;state.zoom=null;render();};
$('demo').onclick=()=>{state.view='trajectory';$('model-filter').value='';$('harness-filter').value='';loadRun('demo-baseline');};
$('refresh').onclick=refresh;
$('export').onclick=()=>{
  if(!state.run) return;
  const blob=new Blob([JSON.stringify(state.run,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);
  const anchor=document.createElement('a');anchor.href=url;anchor.download=`agency-${state.run.id}.json`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.addEventListener('change',event=>{
  if(event.target.id==='add-agent-column'){addAgentColumn(event.target.value);event.target.value='';}
  else if(event.target.id==='comparison-select'){state.compareError=null;loadComparison(event.target.value);}
  else if(event.target.id==='trajectory-layout'){state.trajectoryLayout=event.target.value;updateUrl();render();}
});
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&state.follow&&state.stream)followLatest();});
window.addEventListener('pagehide',()=>state.stream?.close());
$('open-live').onclick=()=>{state.trajectoryLayout='episodes';state.view='trajectory';loadRun('live');};
$('start-replay').onclick=()=>{state.trajectoryLayout='episodes';state.view='trajectory';loadRun(state.run.id,null,true);};
$('replay-restart').onclick=()=>loadRun(state.run.id,null,true);
$('replay-play').onclick=()=>state.stream?.send(state.replayState?.playing?'pause':'play',{speed:Number($('replay-speed').value)});
$('replay-step').onclick=()=>state.stream?.send('step');
$('replay-speed').onchange=()=>{if(state.replayState?.playing)state.stream?.send('play',{speed:Number($('replay-speed').value)});};
$('follow-live').onclick=()=>{followLatest();$('call-stack')?.firstElementChild?.scrollIntoView({block:'start'});};
$('reconnect-updates').onclick=()=>state.stream?.reconnect();
refresh();
