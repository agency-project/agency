// A shared normalized model for snapshot, incremental live updates and replay.
const indexes = new WeakMap();
const windowedRuns = new WeakSet();
const inflightWindowedRuns = new WeakSet();
export function executionControlState({run, replay, transport, executionControls: controls, selected, agent}) {
  const live = run?.id === 'live' && !replay;
  let id = agent || null;
  if (selected && selected.run !== 'b') {
    if (selected.type === 'agent') id = selected.id;
    else if (selected.type === 'action') id = run?.actions.find(a => a.id === selected.id)?.agent || id;
    else if (selected.type === 'episode') id = run?.episodes.find(e => e.id === selected.id)?.agent || id;
    else if (selected.type === 'interval') id = run?.intervals.find(i => i.id === selected.id)?.agent || id;
  } else if (selected?.run === 'b') id = null;
  const target = run?.agents.find(a => a.id === id);
  const active = (controls?.agents || []).map(id => run?.agents.find(a => a.id === id)).filter(Boolean);
  const enabled = live && transport === 'connected' && Boolean(controls?.available) && !['completed','failed','cancelled'].includes(run?.status);
  return {live, target, active, agentEnabled: enabled && active.includes(target), allEnabled: enabled && active.length > 0,
    agentCommand: target?.status === 'paused' ? 'resume' : 'pause',
    allCommand: active.some(a => a.status === 'paused') ? 'resume_all' : 'pause_all'};
}
export function changedTrajectoryActions(previous, next) {
  const known=new Map(previous.map(action=>[action.id,action]));
  return next.filter(action=>{const old=known.get(action.id);return !old||old.outcome!==action.outcome||old.event_ids?.length!==action.event_ids?.length;}).map(action=>action.id);
}
// Snapshot and patch delivery share the same identity merge. Evidence subsets
// cannot resurrect a completed call; replay seeks deliberately replace the model.
export function mergeTrajectory(run, patch) {
  const changed = [];
  for (const key of ['agents', 'actions', 'episodes', 'edges', 'intervals', 'signals']) {
    const items = run[key] || (run[key] = []);
    let lookup = indexes.get(items);
    if (!lookup) {lookup = new Map(items.map(item => [item.id, item])); indexes.set(items, lookup);}
    for (const update of patch[key] || []) {
      const previous = lookup.get(update.id);
      if(previous&&key==='actions') {
        const oldIds=previous.event_ids||[],newIds=update.event_ids||[];
        if(Array.isArray(update.event_ids)&&oldIds.some(id=>!newIds.includes(id))&&newIds.every(id=>oldIds.includes(id)))continue;
        if(['success','failed','cancelled'].includes(previous.outcome)&&update.outcome==='running')continue;
      }
      if (key === 'actions' && (!previous || previous.outcome !== update.outcome || previous.event_ids?.join('|') !== update.event_ids?.join('|'))) changed.push(update.id);
      if (previous) Object.assign(previous, update);
      else {items.push({...update}); lookup.set(update.id, items.at(-1));}
    }
  }
  for (const [name, samples] of Object.entries(patch.counter_samples || patch.counters || {})) {
    const series = (run.counters ||= {})[name] ||= [];
    const seen=new Set(series.map(p=>JSON.stringify(p)));
    for(const p of samples){const key=JSON.stringify(p);if(!seen.has(key)){series.push(p);seen.add(key);}}
    series.sort((a,b)=>a[0]-b[0]);
  }
  Object.assign(run, patch.meta || {});
  return changed;
}
export function mergeTrajectorySnapshot(run, snapshot) {
  const {actions,agents,episodes,edges,intervals,signals,counters,...meta}=snapshot;
  return mergeTrajectory(run,{actions,agents,episodes,edges,intervals,signals,counters,meta});
}
// Bounds apply only to the browser's live projection. Durable source records,
// saved review and explicit historical pages retain their original evidence.
export function boundTrajectory(run, pins=[]) {
  const keep=new Set(pins.slice(0,2000));
  // High-volume syscalls may fill spare profiler capacity but cannot evict
  // actual model/tool calls from the live trajectory's retention window.
  const sorted=[...(run.actions||[])].sort((a,b)=>Number(a.kind==='syscall')-Number(b.kind==='syscall')||b.start-a.start||(a.id>b.id?1:a.id<b.id?-1:0));
  let completed=0,active=0;
  for(const a of sorted)if(a.outcome==='running'?active++<200:completed++<2000)keep.add(a.id);
  const trim=(key,predicate)=>{const items=run[key];if(!items)return;const retained=items.filter(predicate);if(retained.length!==items.length){items.splice(0,items.length,...retained);indexes.delete(items);}};
  const removed=(run.actions||[]).filter(a=>!keep.has(a.id)).length;
  trim('actions',a=>keep.has(a.id));
  const episodes=new Set(run.actions.map(a=>a.episode));
  trim('episodes',e=>episodes.has(e.id));for(const e of run.episodes||[])e.actions=e.actions.filter(id=>keep.has(id));trim('intervals',i=>keep.has(i.action)||keep.has(i.id));
  for(const key of ['signals','edges','messages','obligations'])if(run[key]?.length>1000){run[key].splice(0,run[key].length-1000);indexes.delete(run[key]);}
  let clipped=false;
  for(const points of Object.values(run.counters||{}))if(points.length>2000){points.splice(0,points.length-2000);clipped=true;}
  run.coverage ||= {};
  if(sorted.filter(a=>a.kind!=='syscall'&&a.outcome==='running').length>200)inflightWindowedRuns.add(run);
  if(inflightWindowedRuns.has(run))run.coverage.client_inflight_windowed=true;
  if(removed||clipped||run.coverage.client_window)windowedRuns.add(run);
  if(windowedRuns.has(run))run.coverage.client_window='Live view retains up to 2,000 recent completed calls, 200 in-flight calls, bounded inspection/reading pins and 2,000 samples per series. Use column paging for older calls; complete evidence remains in recorded review/export.';
}

export class TrajectoryStream {
  // `clock`/`play` open a replay at a chosen instant, already playing or held.
  constructor({runId, replay, onMessage, onTransport, clock, play=false, speed}) {
    Object.assign(this, {runId, replay, onMessage, onTransport, replayClock:clock, replayPlaying:play, replaySpeed:speed});
    this.cursor = 0;
    this.epoch = '';
    this.closed = false;
    this.attempt = 0;
    this.connect();
  }
  connect() {
    if (this.closed) return;
    this.onTransport(this.attempt ? 'reconnecting' : 'connecting');
    const query = new URLSearchParams({cursor: this.cursor, epoch: this.epoch});
    if (this.replay) {query.set('mode', 'replay'); query.set('run', this.runId);if(Number.isFinite(this.replayClock))query.set('clock',this.replayClock);if(this.replayPlaying){query.set('play','1');if(Number.isFinite(this.replaySpeed))query.set('speed',this.replaySpeed);}}
    if(this.replaySeekId!=null)query.set('seek_id',this.replaySeekId);
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/trajectory?${query}`);
    this.socket = ws;
    ws.onopen = () => {if (!this.closed) {this.attempt = 0; this.onTransport('connected');}};
    ws.onmessage = event => {
      if (this.closed || ws !== this.socket) return;
      const message = JSON.parse(event.data);
      if (message.type === 'execution_command_result' || message.type === 'measurement_command_result') {this.onMessage(message); return;}
      if(this.replaySeekId!=null&&message.replay?.seek_id!==this.replaySeekId)return;
      this.replaySeekId=null;
      // Ignore duplicate or stale revisions, including delayed frames after reconnect.
      if (!this.replay && this.received && message.epoch === this.epoch && (message.type==='updates'?message.cursor<=this.cursor:message.cursor<this.cursor)) return;
      this.received=true;
      this.cursor = message.cursor;
      if(message.replay){this.replayClock=message.replay.clock;this.replayPlaying=message.replay.playing;this.replaySpeed=message.replay.speed;}
      this.epoch = message.epoch || '';
      this.onMessage(message);
    };
    ws.onclose = () => {
      if (this.closed || ws !== this.socket) return;
      this.onTransport('disconnected');
      this.timer = setTimeout(() => {this.attempt++; this.connect();}, Math.min(10000, 500 * 2 ** this.attempt));
    };
    ws.onerror = () => ws.close();
  }
  send(type, extra = {}) {
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    try {this.socket.send(JSON.stringify({type, ...extra})); return true;}
    catch {return false;}
  }
  sendExecution(command, extra = {}) {
    return !this.replay && this.runId === 'live' && ['pause','resume','pause_all','resume_all','stop','kill','stop_all','kill_all','queue_message','redirect','run'].includes(command)
      && this.send('execution_command', {...extra, command});
  }
  // Wall-clock boundary edits are not execution commands: they never reach the agent relay.
  sendMeasurement(command, extra = {}) {
    return !this.replay && this.runId === 'live' && ['end','hold','reset'].includes(command)
      && this.send('measurement_command', {...extra, command});
  }
  sendReplay(type, extra = {}) {
    const sent=this.replay && ['play','pause','step','seek'].includes(type) && this.send(type, extra);
    if(sent&&type==='seek'){this.replayClock=extra.clock;this.replaySeekId=extra.seek_id??null;}
    return sent;
  }
  reconnect() {
    clearTimeout(this.timer);
    const previous = this.socket;
    this.socket = null;
    previous?.close();
    this.attempt++;
    this.connect();
  }
  close() {
    this.closed = true;
    clearTimeout(this.timer);
    this.socket?.close();
  }
}
