// A shared normalized model for snapshot, incremental live updates and replay.
const indexes = new WeakMap();
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
export function mergeTrajectory(run, patch) {
  const changed = [];
  for (const key of ['agents', 'actions', 'episodes', 'edges', 'intervals', 'signals']) {
    const items = run[key] || (run[key] = []);
    let lookup = indexes.get(items);
    if (!lookup) {lookup = new Map(items.map(item => [item.id, item])); indexes.set(items, lookup);}
    for (const update of patch[key] || []) {
      const previous = lookup.get(update.id);
      if (key === 'actions' && (!previous || previous.outcome !== update.outcome || previous.event_ids?.length !== update.event_ids?.length)) changed.push(update.id);
      if (previous) Object.assign(previous, update);
      else {items.push(update); lookup.set(update.id, update);}
    }
  }
  for (const [name, samples] of Object.entries(patch.counter_samples || {})) {
    const series = (run.counters ||= {})[name] ||= [];
    series.push(...samples);
    series.sort((a,b)=>a[0]-b[0]);
  }
  Object.assign(run, patch.meta || {});
  return changed;
}

export class TrajectoryStream {
  constructor({runId, replay, onMessage, onTransport}) {
    Object.assign(this, {runId, replay, onMessage, onTransport});
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
    if (this.replay) {query.set('mode', 'replay'); query.set('run', this.runId);if(Number.isFinite(this.replayClock))query.set('clock',this.replayClock);}
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/trajectory?${query}`);
    this.socket = ws;
    ws.onopen = () => {if (!this.closed) {this.attempt = 0; this.onTransport('connected');}};
    ws.onmessage = event => {
      if (this.closed || ws !== this.socket) return;
      const message = JSON.parse(event.data);
      if (message.type === 'execution_command_result') {this.onMessage(message); return;}
      // Ignore duplicate or stale revisions, including delayed frames after reconnect.
      if (!this.replay && message.type === 'updates' && message.epoch === this.epoch && message.cursor <= this.cursor) return;
      this.cursor = message.cursor;
      if(message.replay)this.replayClock=message.replay.clock;
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
    return !this.replay && this.runId === 'live' && ['pause','resume','pause_all','resume_all'].includes(command)
      && this.send('execution_command', {...extra, command});
  }
  sendReplay(type, extra = {}) {
    return this.replay && ['play','pause','step','seek'].includes(type) && this.send(type, extra);
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
