// A shared normalized model for snapshot, incremental live updates and replay.
const indexes = new WeakMap();
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
    if (this.replay) {query.set('mode', 'replay'); query.set('run', this.runId);}
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/trajectory?${query}`);
    this.socket = ws;
    ws.onopen = () => {if (!this.closed) {this.attempt = 0; this.onTransport('connected');}};
    ws.onmessage = event => {
      if (this.closed || ws !== this.socket) return;
      const message = JSON.parse(event.data);
      // Ignore duplicate or stale revisions, including delayed frames after reconnect.
      if (!this.replay && message.type === 'updates' && message.epoch === this.epoch && message.cursor <= this.cursor) return;
      this.cursor = message.cursor;
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
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({type, ...extra}));
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
