// app.js — Agency Web UI client
// Connects to the WebSocket, processes events, and renders the dashboard.

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  agents:      new Map(),  // agname -> { color, state, skill, tool }
  teams:       new Map(),  // team_name -> Set<agname>
  histories:   new Map(),  // agname -> msg[]
  // agname -> {ts, html}[] -- tool/llm/syscall event lines, interleaved into
  // the transcript at render time by real timestamp (see buildHistoryHtml()).
  systemLogs:  new Map(),
  fullLogsEnabled: false,
  tokenUsage:  new Map(),  // agname -> { inp, out, history: [{ts,inp,out}] }
  resources: { gpus_acquired: 0, gpus_total: 0, cpus_acquired: 0, cpus_total: 0, memory_acquired_mb: 0, memory_total_mb: 0 },
  agentOrder: [],          // [agname] ordered for display / Tab cycling
  pinnedAgents: [],        // [agname] dropped onto the interaction pane as side panels
  panelWidths: new Map(),  // agname -> side panel width in px (see .panel-resizer)
  focusedIdx: 0,
  autoSelected: false,     // true once the first agent has been auto-selected
  activeTab:  'all',       // 'all' | 'live' | 'idle' | 'finished'
  timeline: {
    liveMode:  true,
    indexLen:  0,
    firstTs:   null,
    lastTs:    null,
    samples:   [],         // [[index_pos, ts], ...] downsampled
  },
};

// ---------------------------------------------------------------------------
// ANSI → HTML
// ---------------------------------------------------------------------------

const ANSI16 = [
  '#000','#a00','#0a0','#880','#00a','#a0a','#0aa','#aaa',
  '#555','#f55','#5f5','#ff5','#55f','#f5f','#5ff','#fff',
];

function xterm256(n) {
  if (n < 16) return ANSI16[n];
  if (n < 232) {
    const i = n - 16;
    const c = l => l ? (55 + 40 * l).toString(16).padStart(2, '0') : '00';
    return '#' + c(~~(i / 36)) + c(~~(i / 6) % 6) + c(i % 6);
  }
  const v = (8 + (n - 232) * 10).toString(16).padStart(2, '0');
  return `#${v}${v}${v}`;
}

function ansiToHtml(text) {
  const parts = text.split(/(\x1b\[[0-9;]*m)/);
  let color = '', bold = false, dim = false;
  const out = [];
  for (const p of parts) {
    if (p.startsWith('\x1b[')) {
      const code = p.slice(2, -1);
      if (!code || code === '0') { color = ''; bold = false; dim = false; }
      else if (code === '1') bold = true;
      else if (code === '2') dim = true;
      else if (code.startsWith('38;5;')) color = xterm256(+code.slice(5));
      else {
        const n = +code;
        if (n >= 30 && n <= 37) color = ANSI16[n - 30];
        else if (n >= 90 && n <= 97) color = ANSI16[n - 82];
      }
    } else if (p) {
      const esc = p.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const css = [
        color && `color:${color}`,
        bold  && 'font-weight:bold',
        dim   && 'opacity:0.5',
      ].filter(Boolean).join(';');
      out.push(css ? `<span style="${css}">${esc}</span>` : esc);
    }
  }
  return out.join('');
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

const $tlSlider  = document.getElementById('timeline-slider');
const $tlFrom    = document.getElementById('timeline-from');
const $tlTo      = document.getElementById('timeline-to');
const $tlLiveBtn = document.getElementById('timeline-live-btn');

function fmtTs(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

// Find the timestamp for a given index_pos using the samples array.
function tsAtPos(pos) {
  const s = state.timeline.samples;
  if (!s.length) return null;
  for (let i = s.length - 1; i >= 0; i--) {
    if (s[i][0] <= pos) {
      if (i === s.length - 1) return s[i][1];
      // Linear interpolate between s[i] and s[i+1]
      const t0 = s[i][1], t1 = s[i + 1][1];
      const p0 = s[i][0], p1 = s[i + 1][0];
      return t0 + (t1 - t0) * (pos - p0) / (p1 - p0);
    }
  }
  return s[0][1];
}

function updateTimelineBar() {
  const tl = state.timeline;
  $tlSlider.max   = Math.max(0, tl.indexLen - 1);
  $tlSlider.disabled = tl.indexLen === 0;

  if (tl.liveMode) {
    $tlSlider.value = $tlSlider.max;
    $tlFrom.textContent = tl.firstTs ? fmtTs(tl.firstTs) : '';
    $tlTo.textContent   = '';
    $tlLiveBtn.classList.add('active');
  } else {
    const pos = parseInt($tlSlider.value, 10);
    const ts  = tsAtPos(pos);
    $tlFrom.textContent = ts ? fmtTs(ts) : '';
    $tlTo.textContent   = tl.lastTs ? fmtTs(tl.lastTs) : '';
    $tlLiveBtn.classList.remove('active');
  }
}

function applyTimelineSync(ev) {
  const tl = state.timeline;
  tl.indexLen = ev.index_len || 0;
  tl.firstTs  = ev.first_ts  || null;
  tl.lastTs   = ev.last_ts   || null;
  tl.liveMode = true;
  updateTimelineBar();
  // Fetch samples for hover timestamps.
  fetch('/api/timeline').then(r => r.json()).then(j => {
    tl.samples = j.samples || [];
  }).catch(() => {});
}

// Poll /api/timeline every 10s in live mode to keep slider max current.
setInterval(async () => {
  if (!state.timeline.liveMode) return;
  try {
    const r  = await fetch('/api/timeline');
    const j  = await r.json();
    const tl = state.timeline;
    tl.indexLen = j.index_len || 0;
    tl.firstTs  = j.first_ts  || tl.firstTs;
    tl.lastTs   = j.last_ts   || tl.lastTs;
    tl.samples  = j.samples   || tl.samples;
    updateTimelineBar();
  } catch {}
}, 10_000);

// ---------------------------------------------------------------------------
// Profiler artifacts
// ---------------------------------------------------------------------------

function fmtBytes(n) {
  if (n >= 1_048_576) return (n / 1_048_576).toFixed(1) + 'MB';
  if (n >= 1024)      return (n / 1024).toFixed(1) + 'KB';
  return n + 'B';
}

// agprof writes its output files (trace/summary/sqlite3) at its own pace,
// under names/timing this client has no reason to know or guess -- rather
// than tracking which specific files mean "done" (fragile: a rename or a
// newly added output file would silently stop showing up), just render
// whatever /api/profiler/files currently reports and keep refreshing it.
// One persistent line is updated in place (not re-appended) each poll, so
// a file list that's still growing never produces duplicate log lines.
let _profilerLogLineEl = null;
let _profilerPollTimer = null;

function _renderProfilerFiles(files) {
  if (!files.length) return;
  const links = files
    .map(f => `<a href="/api/profiler/download/${encodeURIComponent(f.name)}" download>${esc(f.name)}</a> (${fmtBytes(f.size)})`)
    .join('  ');
  const html = `<span style="color:var(--yellow)">Profiler output:</span>  ${links}`;
  if (_profilerLogLineEl) {
    _profilerLogLineEl.innerHTML = html;
  } else {
    _profilerLogLineEl = _appendLogLine(html);
  }
}

async function checkProfilerFiles() {
  if (wsClosed) {
    clearInterval(_profilerPollTimer);
    return;
  }
  try {
    const r = await fetch('/api/profiler/files');
    const j = await r.json();
    _renderProfilerFiles(j.files || []);
  } catch {}
}

// Armed once, the first time the run is known to be over -- from the
// "done" event, whether pushed live or replayed from this agent db's tail
// on a client that only connects after the run already finished (see
// server.py's _fetch_tail_events(): "done" is always that db's very last
// event, so it's always within the replayed tail window). Nothing polls
// before that: the profiler's own files (profile_data.sqlite3 especially)
// exist from early in the run, long before it's actually finished, so
// checking earlier would only mean showing/updating this line too soon.
function armProfilerPolling() {
  if (_profilerPollTimer !== null) return;
  _profilerPollTimer = setInterval(checkProfilerFiles, 3_000);
  checkProfilerFiles();
}

// Reset all agent/log state before replaying a historical window.
function clearAgentState() {
  state.agents.clear();
  state.teams.clear();
  state.histories.clear();
  state.systemLogs.clear();
  state.agentOrder = [];
  state.pinnedAgents = [];
  state.focusedIdx = 0;
  state.autoSelected = false;
  $sharedLog.innerHTML = '';
  $sidePanels.innerHTML = '';
  renderAgentList();
  renderHistory();
}

async function enterHistoricalMode(indexPos) {
  state.timeline.liveMode = false;
  updateTimelineBar();
  clearAgentState();
  appendLog('\x1b[33m[timeline] loading historical events…\x1b[0m');
  try {
    const endTs   = tsAtPos(indexPos) || state.timeline.lastTs || 0;
    const startTs = state.timeline.firstTs || 0;
    const r  = await fetch(`/api/events?start_ts=${startTs}&end_ts=${endTs}`);
    const j  = await r.json();
    clearAgentState();
    for (const line of (j.events || [])) {
      try { handleEvent(JSON.parse(line)); } catch {}
    }
    if (j.from_ts) {
      $tlFrom.textContent = fmtTs(j.from_ts);
      $tlTo.textContent   = j.to_ts ? fmtTs(j.to_ts) : '';
    }
  } catch (e) {
    appendLog(`\x1b[31m[timeline] fetch failed: ${e}\x1b[0m`);
  }
}

$tlSlider.addEventListener('input', () => {
  const pos = parseInt($tlSlider.value, 10);
  const max = parseInt($tlSlider.max,   10);
  if (pos >= max) {
    // Snap back to live — reload for clean state.
    window.location.reload();
  } else {
    enterHistoricalMode(pos);
  }
});

$tlLiveBtn.addEventListener('click', () => {
  if (!state.timeline.liveMode) window.location.reload();
});

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const $sharedLog        = document.getElementById('shared-log');
const $interaction      = document.getElementById('interaction');
const $agentHistory     = document.getElementById('agent-history');
const $sidePanels       = document.getElementById('side-panels');
const $fullLogsCheckbox = document.getElementById('full-logs-checkbox');
const $agentList        = document.getElementById('agent-list');
const $interactionTitle = document.getElementById('interaction-title');
const $navLabel         = document.getElementById('nav-label');
const $countAll         = document.getElementById('count-all');
const $countLive        = document.getElementById('count-live');
const $countIdle        = document.getElementById('count-idle');
const $countFinished    = document.getElementById('count-finished');
const $agentSearch      = document.getElementById('agent-search');
const $resourceStats    = document.getElementById('resource-stats');
const $btnPauseToggle   = document.getElementById('btn-pause-toggle');
const $btnPauseAll      = document.getElementById('btn-pause-all');
const $btnResumeAll     = document.getElementById('btn-resume-all');
const $btnUpdateConfig  = document.getElementById('btn-update-config');
const $configOverlay    = document.getElementById('config-modal-overlay');
const $configTitle      = document.getElementById('config-modal-title');
const $configBody       = document.getElementById('config-modal-body');
const $configCancel     = document.getElementById('config-cancel');
const $configUpdate     = document.getElementById('config-update');
const $configUpdateAll  = document.getElementById('config-update-all');

function updateResourceBadge() {
  const r = state.resources;
  const parts = [];
  if (r.gpus_total > 0) {
    parts.push(`GPU ${r.gpus_acquired}/${r.gpus_total}`);
  }
  if (r.cpus_acquired > 0) {
    parts.push(`CPU ${r.cpus_acquired}/${r.cpus_total}`);
  }
  if (r.memory_acquired_mb > 0) {
    const acqG = (r.memory_acquired_mb / 1024).toFixed(0);
    const totG = (r.memory_total_mb / 1024).toFixed(0);
    parts.push(`MEM ${acqG}/${totG}G`);
  }
  $resourceStats.textContent = parts.length ? parts.join('  ') + '  (Used/Total)' : '';
}


function fmtTokens(inp, out, history) {
  function compact(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'k';
    return String(n);
  }
  let s = `↑${compact(inp)} ↓${compact(out)}`;
  if (history && history.length >= 2) {
    const oldest  = history[0];
    const newest  = history[history.length - 1];
    const elapsed = (Date.now() / 1000) - oldest.ts;
    if (elapsed > 0) {
      const inpS = compact(Math.round((newest.inp - oldest.inp) / elapsed));
      const outS = compact(Math.round((newest.out - oldest.out) / elapsed));
      s += `  (↑${inpS}/s ↓${outS}/s)`;
    }
  }
  return s;
}

function updateInteractionTitle() {
  const agname  = currentAgent();
  const visible = visibleOrder();
  const n   = visible.length;
  const idx = n ? state.focusedIdx % n : 0;

  if (!agname) {
    $interactionTitle.innerHTML = 'No agents';
    $navLabel.textContent = '';
    return;
  }

  const usage   = state.tokenUsage.get(agname);
  const tokHtml = usage
    ? `<span class="token-badge">${fmtTokens(usage.inp, usage.out, usage.history)}</span>`
    : '';
  $interactionTitle.innerHTML =
    `${esc(agname)}  [${idx + 1}/${n}]  ← →${tokHtml}`;
  $navLabel.textContent = `${idx + 1} / ${n}`;
}

// ---------------------------------------------------------------------------
// Shared log
// ---------------------------------------------------------------------------

let logAutoScroll = true;

$sharedLog.addEventListener('scroll', () => {
  logAutoScroll = $sharedLog.scrollHeight - $sharedLog.scrollTop - $sharedLog.clientHeight < 40;
});

function _appendLogLine(html) {
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = html;
  $sharedLog.appendChild(div);
  if (logAutoScroll) $sharedLog.scrollTop = $sharedLog.scrollHeight;
  // Cap at 5000 lines to prevent unbounded growth
  while ($sharedLog.children.length > 5000) $sharedLog.removeChild($sharedLog.firstChild);
  return div;
}

function appendLog(line) {
  _appendLogLine(ansiToHtml(line));
}

// agterm-style colorization: term_message lines are plain text of the form
// "[agname] rest of the message" -- color just the agent tag, matching the
// old terminal-based agui's per-agent-colored `[agname]` prefix, and leave
// the rest to ansiToHtml (term_message never contains ANSI codes, but this
// keeps escaping consistent with every other log line). Every event carries
// its own `ts` (see server.py's _build_envelope), so a short HH:MM:SS
// prefix -- dim, like the old terminal-based agui's -- rides along whenever
// one's available.
function colorizeTermMessage(termMessage, color, ts) {
  const tsPrefix = ts ? `<span class="log-ts">${fmtTs(ts)}</span> ` : '';
  const m = /^(\[[^\]]+\])(.*)$/s.exec(termMessage);
  if (!m || !color) {
    return tsPrefix + ansiToHtml(termMessage);
  }
  const tag = `<span style="color:${color};font-weight:bold">${esc(m[1])}</span>`;
  return tsPrefix + tag + ansiToHtml(m[2]);
}

function appendAgentLog(termMessage, color, ts) {
  _appendLogLine(colorizeTermMessage(termMessage, color, ts));
}

// ---------------------------------------------------------------------------
// Per-agent system log (tool/LLM/syscall events) -- kept out of the shared
// log; interleaved into that agent's own transcript instead (buildHistoryHtml()
// below), gated by the "System Logs" checkbox. Lines are always buffered
// per-agent regardless of the checkbox, so toggling it on replays everything
// seen so far for the currently selected agent.
// ---------------------------------------------------------------------------

// 'tool_result'/'running_tool' are deliberately excluded here -- every
// *allowed* tool call already renders from the canonical transcript
// (llm_block's tool_use/tool_result, in buildHistoryHtml() below), ungated by
// the Full Logs checkbox, styled the same "TOOL name ▶/✓" way. Including
// them here too just double-rendered the same call. 'tool_denied' stays,
// since a denied call never runs and so has no transcript tool_result to
// show it any other way.
const SYSTEM_LOG_TYPES = new Set([
  'syscall_result', 'llm_stream_error', 'llm_stream_cancelled',
]);
const SYSTEM_LOG_AGENT_STATES = new Set([
  'tool_denied', 'running_syscall', 'syscall_denied',
]);
const SYSTEM_LOG_CAP = 2000;

function isSystemLogEvent(ev) {
  if (SYSTEM_LOG_TYPES.has(ev.type)) return true;
  return ev.type === 'agent_state' && SYSTEM_LOG_AGENT_STATES.has(ev.state);
}

// term_message text for these events is always "[agname] LABEL[pad] MARK  rest"
// (see host_interaction_server.py's _record_admission/_record_completion and
// llm_handler_server.py's _finalize_error/_finalize_cancelled) -- parsed back
// into a name/body split so these lines can render in the exact same
// header-row-then-newline-then-body shape as an assistant's own tool_use
// block (.msg-tool-call), instead of as one flat line.
const SYSLOG_ICON = { TOOL: '⚙', SYSCALL: '⌘', LLM: '✦' };
// Body brightness tier per label -- TOOL/SYSCALL get the same tiers their
// canonical transcript/system counterparts use; LLM (stream error/cancelled)
// is itself a system-level notice, not agent-authored content.
const SYSLOG_BODY_CLASS = { TOOL: 'body-tool', SYSCALL: 'body-syscall', LLM: 'body-system' };
const SYSLOG_LINE_RE = /^\[[^\]]+\]\s+(TOOL|SYSCALL|LLM)\s*([▶✓✗?])\s+([\s\S]*)$/;

function formatSyslogLine(termMessage, ts) {
  const tsHtml = ts ? `<span class="log-ts">${fmtTs(ts)}</span>` : '';
  const m = SYSLOG_LINE_RE.exec(termMessage);
  if (!m) {
    // Defensive fallback -- every event routed here is expected to match.
    return `${tsHtml} <span class="role-tool-call">${esc(termMessage)}</span>`;
  }
  const [, label, mark, rest] = m;
  const icon = SYSLOG_ICON[label] || '⚙';
  let name = '';
  let body = rest;
  if (label !== 'LLM') {
    // Tool/syscall names never contain spaces -- the first run of 2+ spaces
    // is always the field separator the backend used when building the line.
    const sepIdx = rest.search(/ {2,}/);
    if (sepIdx === -1) {
      name = rest;
      body = '';
    } else {
      name = rest.slice(0, sepIdx);
      body = rest.slice(sepIdx).trim();
    }
  }
  const nameHtml = name ? `  ${esc(name)}` : '';
  const header =
    `${tsHtml} <span class="role-tool-call">${icon} ${esc(label)}${nameHtml} ${esc(mark)}</span>`;
  const bodyClass = SYSLOG_BODY_CLASS[label] || 'body-system';
  return body ? `${header}\n<span class="${bodyClass}">${esc(body)}</span>` : header;
}

function recordSystemLog(agname, termMessage, ts) {
  const html = formatSyslogLine(termMessage, ts);
  if (!state.systemLogs.has(agname)) state.systemLogs.set(agname, []);
  const lines = state.systemLogs.get(agname);
  lines.push({ ts: ts || 0, html });
  if (lines.length > SYSTEM_LOG_CAP) lines.shift();
  refreshAgentView(agname);
}

try {
  const stored = localStorage.getItem('agency_full_logs_enabled');
  state.fullLogsEnabled = stored === null ? false : stored === '1';
} catch {}
$fullLogsCheckbox.checked = state.fullLogsEnabled;

$fullLogsCheckbox.addEventListener('change', () => {
  state.fullLogsEnabled = $fullLogsCheckbox.checked;
  try { localStorage.setItem('agency_full_logs_enabled', state.fullLogsEnabled ? '1' : '0'); } catch {}
  renderHistory();
});

// ---------------------------------------------------------------------------
// Agent list (right panel)
// ---------------------------------------------------------------------------

// The backend's own idle state (agent.py's record_state("agent_idle"),
// set at construction and again by orchestrator._update_agent_display_locked
// whenever an agent has no ready/blocked work left) is a different string
// than 'inactive' -- the client's own synthetic "nothing fetched yet"
// placeholder (agent_registered's initial state, and every empty-state
// fallback below). Both mean the same thing to a viewer, so normalize the
// backend's string to the client's at the two points backend state enters
// `state.agents` (loadAgentDetail's poll, and the 'agent_state' WS push) --
// every other check in this file only ever needs to know about 'inactive'.
function normalizeAgentState(st) {
  return st === 'agent_idle' ? 'inactive' : st;
}

function isLive(st)      { return st !== 'inactive' && st !== 'finished' && st !== 'skill' && st !== 'paused'; }
function isIdle(st)      { return st === 'inactive' || st === 'paused'; }
function isFinished(st)  { return st === 'finished'; }

function tabVisible(st) {
  if (state.activeTab === 'all')      return true;
  if (state.activeTab === 'live')     return isLive(st);
  if (state.activeTab === 'idle')     return isIdle(st);
  if (state.activeTab === 'finished') return isFinished(st);
  return true;
}

function updateCounts() {
  let live = 0, idle = 0, finished = 0;
  for (const ag of state.agents.values()) {
    if (isLive(ag.state))     live++;
    else if (isFinished(ag.state)) finished++;
    else                      idle++;
  }
  $countAll.textContent      = state.agents.size;
  $countLive.textContent     = live;
  $countIdle.textContent     = idle;
  $countFinished.textContent = finished;
}

function currentAgent() {
  const visible = visibleOrder();
  if (!visible.length) return null;
  return visible[state.focusedIdx % visible.length];
}

function visibleOrder() {
  const query = ($agentSearch ? $agentSearch.value : '').trim().toLowerCase();
  return state.agentOrder.filter(agname => {
    const ag = state.agents.get(agname);
    if (!(ag ? tabVisible(ag.state) : state.activeTab === 'all')) return false;
    return !query || agname.toLowerCase().includes(query);
  });
}

function renderAgentList() {
  updateCounts();

  // Build agname → team_name reverse map
  const agentTeam = new Map();
  for (const [tname, agents] of state.teams) {
    for (const ag of agents) agentTeam.set(ag, tname);
  }

  const visible = visibleOrder();
  const frags = [];
  const emittedTeams = new Set();
  const focused = currentAgent();

  for (const agname of visible) {
    const tname = agentTeam.get(agname);

    if (tname && !emittedTeams.has(tname)) {
      emittedTeams.add(tname);
      frags.push(`<div class="team-header">${esc(tname)}</div>`);
    }

    const indent = tname ? 12 : 0;
    const ag = state.agents.get(agname) || { color: '#d4d4d4', state: 'inactive', skill: null, tool: null };
    const isFocused = agname === focused;
    frags.push(renderAgentEntry(agname, ag, indent, isFocused));
  }

  $agentList.innerHTML = frags.join('');
  updateAgentActionsBar();
}

function updateAgentActionsBar() {
  const agname = currentAgent();
  if (!agname) {
    $btnPauseToggle.disabled = true;
    $btnPauseToggle.textContent = 'Pause';
    $btnPauseToggle.classList.remove('active');
    $btnUpdateConfig.disabled = true;
    return;
  }
  const ag = state.agents.get(agname);
  const paused = ag ? ag.state === 'paused' : false;
  $btnPauseToggle.disabled = false;
  $btnPauseToggle.textContent = paused ? 'Resume' : 'Pause';
  $btnPauseToggle.classList.toggle('active', paused);
  $btnUpdateConfig.disabled = false;
}

function renderAgentEntry(agname, ag, indent, isFocused) {
  const color = ag.color || '#d4d4d4';
  const focusedClass = isFocused ? ' focused' : '';
  const { state: st, skill, tool } = ag;
  let dot, statusHtml;

  if (st === 'finished') {
    dot = `<span class="dot-finished">✓</span>`;
    statusHtml = `<span class="status-finished">finished</span>`;
  } else if (st === 'paused') {
    dot = `<span class="dot-paused">⏸</span>`;
    statusHtml = `<span class="dim">${esc(skill || '')}</span>: <span class="status-paused">paused</span>`;
  } else if (st === 'inactive') {
    dot = `<span class="dot-inactive">○</span>`;
    statusHtml = `<span class="dim">idle</span>`;
  } else if (st === 'llm') {
    dot = `<span style="color:${color}">●</span>`;
    statusHtml = `<span class="dim">${esc(skill || '')}</span>: <span class="status-llm">LLM Wait</span>`;
  } else if (st === 'tool') {
    dot = `<span style="color:${color}">●</span>`;
    statusHtml = `<span class="dim">${esc(skill || '')}</span>: <span class="status-tool">${esc(tool || 'tool')}</span>`;
  } else if (st === 'proc_wait') {
    dot = `<span style="color:${color}">●</span>`;
    statusHtml = `<span class="dim">${esc(skill || '')}</span>: <span class="dim">Shell Wait</span>`;
  } else if (st === 'human') {
    dot = `<span style="color:${color}">●</span> <span class="status-human">?</span>`;
    statusHtml = `<span class="dim">${esc(skill || '')}</span>: <span class="status-human">Input Pending</span>`;
  } else {
    dot = `<span style="color:${color}">●</span>`;
    statusHtml = `<span class="dim">${esc(skill || '')} - running</span>`;
  }

  return `<div class="agent-entry${focusedClass}" data-agname="${esc(agname)}" draggable="true" title="Drag onto the log panel to compare side-by-side" style="padding-left:${indent}px">
    <div class="agent-name">${dot} <span style="color:${color}">${esc(agname)}</span></div>
    <div class="agent-status">${statusHtml}</div>
  </div>`;
}

// ---------------------------------------------------------------------------
// History pane (left bottom)
// ---------------------------------------------------------------------------

let histAutoScroll = true;

$agentHistory.addEventListener('scroll', () => {
  histAutoScroll = $agentHistory.scrollHeight - $agentHistory.scrollTop - $agentHistory.clientHeight < 40;
});

// Builds one agent's transcript HTML -- shared by the main (nav-driven)
// panel and every pinned side panel (see #side-panels below), so a pinned
// agent renders exactly the same way it would as the focused one.
function buildHistoryHtml(agname) {
  const msgs    = state.histories.get(agname) || [];
  const syslogs = state.fullLogsEnabled ? (state.systemLogs.get(agname) || []) : [];
  let syslogIdx = 0;
  const frags = [];

  // A tool_use call and its tool_result are two separate transcript
  // messages, rendered separately, each wherever it actually falls in
  // message order -- no attempt to reposition a result next to its call.
  // A tool_result block itself carries no `name` (only tool_call_id), so
  // this is just a label lookup for the "TOOL name ✓" row below, built
  // from every tool_use seen so far.
  const toolNamesByCallId = new Map();

  // Flush every buffered system-log line that happened at or before a given
  // real timestamp -- each reconstructed message and each syslog line now
  // carries its own accurate ts (see _compute_agent_messages), so a plain
  // timestamp merge places each syslog line right where it actually
  // happened relative to the surrounding transcript messages.
  function flushSyslogsUpToTs(ts) {
    while (syslogIdx < syslogs.length && syslogs[syslogIdx].ts <= ts) {
      frags.push(`<div class="msg-syslog">${syslogs[syslogIdx].html}</div>`);
      syslogIdx++;
    }
  }

  for (let msgIdx = 0; msgIdx < msgs.length; msgIdx++) {
    const msg    = msgs[msgIdx];
    flushSyslogsUpToTs(msg.ts ?? Infinity);
    const role   = msg.role   || '';
    const blocks = msg.blocks || [];
    // Every block carries an agency-native `type` (text/thinking/tool_use/
    // tool_result/metadata) regardless of role -- see
    // llm_handler_server.get_main_transcript() for the shape.
    const text = blocks.filter(b => b.type === 'text').map(b => b.text || '').join('');

    // One message can render as several divs (assistant thinking/tool_use/
    // text blocks each get their own) -- put the same timestamp on every
    // one of them rather than just the first, so it's never missing from
    // whichever div the reader actually looks at (e.g. the closing text
    // div after an earlier tool_use call in the same message).
    const tsHtml = msg.ts ? `<span class="log-ts">${fmtTs(msg.ts)}</span> ` : '';

    if (role === 'system') {
      if (text && state.fullLogsEnabled) {
        frags.push(
          `<div class="msg-system">${tsHtml}<span class="role-tool-call">⚑ system</span>\n` +
          `<span class="body-system">${esc(text)}</span></div>`
        );
      }

    } else if (role === 'user') {
      if (text) {
        frags.push(
          `<div class="msg-user">${tsHtml}<span class="role-user">▶ user</span>  ${esc(text)}</div>`
        );
      }

    } else if (role === 'assistant') {
      // Render each block at its own position instead of aggregating text
      // into one trailing div -- a turn can emit text *before* its tool
      // calls (e.g. "Verified: ..." then two submit_output calls in the
      // same response), and pushing that text last regardless of its real
      // position misrepresents what the model actually said relative to
      // the tool calls it made.
      for (const b of blocks) {
        if (b.type === 'thinking' && b.text) {
          frags.push(`<div class="msg-thinking">${tsHtml}💭 thinking\n${esc(b.text)}</div>`);
        } else if (b.type === 'tool_use') {
          if (b.id) toolNamesByCallId.set(b.id, b.name || '?');
          let argsText = '';
          try {
            const raw = JSON.parse(b.arguments || '{}');
            argsText = Object.entries(raw)
              .map(([k, v]) => `  ${esc(k)}: ${esc(String(v))}`)
              .join('\n');
          } catch {
            argsText = esc(b.arguments || '');
          }
          frags.push(
            `<div class="msg-tool-call">${tsHtml}<span class="role-tool-call">⚙ TOOL ${esc(b.name || '?')} ▶</span>\n` +
            `<span class="body-tool">${argsText}</span></div>`
          );
        } else if (b.type === 'metadata' && state.fullLogsEnabled) {
          frags.push(
            `<div class="msg-metadata">${tsHtml}<span class="role-tool-call">◇ metadata</span>\n` +
            `<span class="dim">${esc(JSON.stringify(b))}</span></div>`
          );
        } else if (b.type === 'text' && b.text) {
          frags.push(
            `<div class="msg-assistant">${tsHtml}<span class="role-assistant">◆ assistant</span>\n${esc(b.text)}</div>`
          );
        }
      }

    } else if (role === 'tool') {
      // Rendered as its own separate row, in whatever message position it
      // actually falls in -- not attached to/repositioned under its call.
      for (const b of blocks) {
        if (b.type === 'tool_result') {
          const name = toolNamesByCallId.get(b.tool_call_id) || '?';
          frags.push(
            `<div class="msg-tool-result">${tsHtml}<span class="role-tool-call">⚙ TOOL ${esc(name)} ✓</span>\n` +
            `<span class="body-tool">${esc(b.text || '')}</span></div>`
          );
        }
      }
    }
  }
  // Anything buffered past the last known message (e.g. arrived in the gap
  // between this render and the next transcript poll) still belongs here.
  flushSyslogsUpToTs(Infinity);

  // Running indicator
  const ag = state.agents.get(agname);
  if (ag && ag.state !== 'inactive' && ag.state !== 'finished') {
    const st = ag.state;
    let label;
    if      (st === 'llm')       label = 'LLM Thinking…';
    else if (st === 'tool')      label = `Tool Running: ${ag.tool || ''}…`;
    else if (st === 'proc_wait') label = 'Waiting for processes…';
    else if (st === 'human')     label = 'Input Pending…';
    else if (st === 'paused')    label = 'Paused';
    else                         label = 'Running…';
    frags.push(`<div class="msg-running${st === 'paused' ? ' msg-paused' : ''}">▶ ${esc(label)}</div>`);
  }

  return frags.join('');
}

function renderHistory() {
  const agname = currentAgent();
  updateInteractionTitle();

  if (!agname) {
    $agentHistory.innerHTML = '';
  } else {
    $agentHistory.innerHTML = buildHistoryHtml(agname);
    if (histAutoScroll) $agentHistory.scrollTop = $agentHistory.scrollHeight;
  }
  renderSidePanels();
}

// ---------------------------------------------------------------------------
// Side panels -- agents dragged from the list on the right and dropped onto
// the interaction pane, so their logs render next to the focused agent's
// for direct comparison. Each panel keeps its own scroll-lock (mirroring
// histAutoScroll above) and is refreshed whenever that agent's data changes,
// independent of whichever agent is currently focused.
// ---------------------------------------------------------------------------

const panelAutoScroll = new Map(); // agname -> bool, true = pinned to bottom
const panelElements   = new Map(); // agname -> { resizer, panel, historyEl }
const DEFAULT_PANEL_WIDTH = 300;
const MIN_PANEL_WIDTH = 160;

function pinAgent(agname) {
  if (!agname || !state.agents.has(agname)) return;
  if (state.pinnedAgents.includes(agname)) return;
  state.pinnedAgents.push(agname);
  renderSidePanels();
  // Don't wait for the next 200ms poll tick -- fetch this agent's messages
  // immediately so the new panel isn't blank for a beat.
  loadAgentDetail(agname);
}

function unpinAgent(agname) {
  state.pinnedAgents = state.pinnedAgents.filter(a => a !== agname);
  renderSidePanels();
}

// Drag .panel-resizer (immediately to *agname*'s panel's left) to resize
// just that panel -- #panel-main is flex:1 and every other panel keeps its
// own width, so only the dragged panel's width ever needs to change.
function attachPanelResizer(resizer, agname) {
  resizer.addEventListener('mousedown', e => {
    e.preventDefault();
    const els = panelElements.get(agname);
    if (!els) return;
    const startX = e.clientX;
    const startWidth = els.panel.getBoundingClientRect().width;
    resizer.classList.add('resizing');
    document.body.classList.add('resizing-panels');

    function setWidth(px) {
      const width = Math.max(MIN_PANEL_WIDTH, Math.round(px));
      state.panelWidths.set(agname, width);
      els.panel.style.flex = `0 0 ${width}px`;
      els.panel.style.width = `${width}px`;
    }
    // The resizer sits at this panel's LEFT edge -- dragging it right
    // narrows the panel (its left edge moves toward its fixed right edge),
    // dragging left widens it, hence the subtraction.
    function onMove(ev) { setWidth(startWidth - (ev.clientX - startX)); }
    function onUp() {
      resizer.classList.remove('resizing');
      document.body.classList.remove('resizing-panels');
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });
}

function renderSidePanels() {
  // Drop panels (and their resizer) for agents no longer pinned.
  for (const [agname, els] of [...panelElements]) {
    if (!state.pinnedAgents.includes(agname)) {
      els.resizer.remove();
      els.panel.remove();
      panelElements.delete(agname);
      panelAutoScroll.delete(agname);
      state.panelWidths.delete(agname);
    }
  }

  for (const agname of state.pinnedAgents) {
    let els = panelElements.get(agname);
    if (!els) {
      const ag = state.agents.get(agname) || { color: '#d4d4d4' };

      const resizer = document.createElement('div');
      resizer.className = 'panel-resizer';
      attachPanelResizer(resizer, agname);

      const panel = document.createElement('div');
      panel.className = 'side-panel';
      panel.dataset.agname = agname;
      panel.innerHTML =
        `<div class="panel-title side-panel-title">` +
          `<span style="color:${ag.color || '#d4d4d4'}">${esc(agname)}</span>` +
          `<button class="side-panel-close" title="Unpin">&times;</button>` +
        `</div>` +
        `<div class="side-panel-history"></div>`;
      panel.querySelector('.side-panel-close').addEventListener('click', () => unpinAgent(agname));
      const historyEl = panel.querySelector('.side-panel-history');
      panelAutoScroll.set(agname, true);
      historyEl.addEventListener('scroll', () => {
        panelAutoScroll.set(agname, historyEl.scrollHeight - historyEl.scrollTop - historyEl.clientHeight < 40);
      });

      els = { resizer, panel, historyEl };
      panelElements.set(agname, els);
    }
    // Re-appending an already-attached node moves it -- this keeps DOM
    // order in sync with state.pinnedAgents with no separate reorder step.
    $sidePanels.appendChild(els.resizer);
    $sidePanels.appendChild(els.panel);

    const width = state.panelWidths.get(agname) || DEFAULT_PANEL_WIDTH;
    els.panel.style.flex = `0 0 ${width}px`;
    els.panel.style.width = `${width}px`;

    els.historyEl.innerHTML = buildHistoryHtml(agname);
    if (panelAutoScroll.get(agname) !== false) els.historyEl.scrollTop = els.historyEl.scrollHeight;
  }
}

// An agent's data changed -- refresh wherever it's currently shown (the
// main panel if focused, a side panel if pinned, both, or neither).
function refreshAgentView(agname) {
  if (agname === currentAgent()) {
    renderHistory(); // also refreshes side panels
  } else if (state.pinnedAgents.includes(agname)) {
    renderSidePanels();
  }
}

// A custom MIME type (not plain 'text/plain') so the overlay only ever
// reacts to one of our own agent entries being dragged -- not some
// unrelated drag (a link, an image, a browser tab) passing over the pane.
const AGENT_DRAG_TYPE = 'application/x-agency-agname';

function isAgentDrag(e) {
  return e.dataTransfer.types.includes(AGENT_DRAG_TYPE);
}

$interaction.addEventListener('dragover', e => {
  if (!isAgentDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});
let _dragDepth = 0;
$interaction.addEventListener('dragenter', e => {
  if (!isAgentDrag(e)) return;
  e.preventDefault();
  _dragDepth++;
  $interaction.classList.add('drag-over');
});
$interaction.addEventListener('dragleave', e => {
  if (!isAgentDrag(e)) return;
  _dragDepth = Math.max(0, _dragDepth - 1);
  if (_dragDepth === 0) $interaction.classList.remove('drag-over');
});
$interaction.addEventListener('drop', e => {
  if (!isAgentDrag(e)) return;
  e.preventDefault();
  _dragDepth = 0;
  $interaction.classList.remove('drag-over');
  pinAgent(e.dataTransfer.getData(AGENT_DRAG_TYPE));
});

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

function handleEvent(ev) {
  switch (ev.type) {

    case 'timeline_sync':
      applyTimelineSync(ev);
      break;

    case 'log':
      appendLog(ev.line);
      break;

    case 'agent_registered':
      if (!state.agents.has(ev.agname)) {
        state.agents.set(ev.agname, {
          color: ev.color || '#d4d4d4',
          state: 'inactive', skill: null, tool: null, config: {},
        });
        state.agentOrder.push(ev.agname);
      }
      if (ev.team) {
        if (!state.teams.has(ev.team)) state.teams.set(ev.team, new Set());
        state.teams.get(ev.team).add(ev.agname);
        reorderAgents();
      }
      renderAgentList();
      // Auto-select the first agent this browser session ever sees --
      // otherwise the interaction panel stays blank until the user clicks
      // or tabs to an agent themselves.
      if (!state.autoSelected && currentAgent()) {
        state.autoSelected = true;
        renderHistory();
        loadAgentDetail(currentAgent());
      }
      break;

    case 'agent_config': {
      const existing = state.agents.get(ev.agname) || { color: '#d4d4d4', state: 'inactive', skill: null, tool: null };
      state.agents.set(ev.agname, { ...existing, config: ev.config || {} });
      if (!state.agentOrder.includes(ev.agname)) state.agentOrder.push(ev.agname);
      break;
    }

    case 'request_submitted':
    case 'request_blocked':
    case 'request_ready':
    case 'request_started':
    case 'request_completed':
    case 'request_failed':
    case 'request_cancelled':
    case 'request_destroyed': {
      const existing = state.agents.get(ev.agname) || { color: '#d4d4d4' };
      const stateMap = {
        request_submitted: 'queued',
        request_blocked: 'blocked_on_dependency',
        request_ready: 'queued',
        request_started: 'skill',
        request_completed: 'finished',
        request_failed: 'error',
        request_cancelled: 'cancelled',
        request_destroyed: 'destroyed',
      };
      state.agents.set(ev.agname, {
        ...existing,
        state: stateMap[ev.type],
        skill: ev.skill || existing.skill || null,
        tool: null,
      });
      if (!state.agentOrder.includes(ev.agname)) state.agentOrder.push(ev.agname);
      renderAgentList();
      break;
    }

    case 'agent_state': {
      const existing = state.agents.get(ev.agname) || { color: '#d4d4d4' };
      state.agents.set(ev.agname, {
        ...existing,
        color: ev.color || existing.color || '#d4d4d4',
        state: normalizeAgentState(ev.state),
        skill: ev.skill,
        tool:  ev.tool,
      });
      if (!state.agentOrder.includes(ev.agname)) {
        state.agentOrder.push(ev.agname);
      }
      if (ev.team) {
        if (!state.teams.has(ev.team)) state.teams.set(ev.team, new Set());
        state.teams.get(ev.team).add(ev.agname);
        reorderAgents();
      }
      renderAgentList();
      refreshAgentView(ev.agname);
      break;
    }

    case 'team_registered': {
      state.teams.set(ev.team_name, new Set(ev.agents || []));
      reorderAgents();
      renderAgentList();
      break;
    }

    case 'messages_snapshot':
      state.histories.set(ev.agname, ev.messages || []);
      refreshAgentView(ev.agname);
      break;

    case 'resource_update':
      state.resources = {
        gpus_acquired:      ev.gpus_acquired      || 0,
        gpus_total:         ev.gpus_total         || 0,
        cpus_acquired:      ev.cpus_acquired      || 0,
        cpus_total:         ev.cpus_total         || 0,
        memory_acquired_mb: ev.memory_acquired_mb || 0,
        memory_total_mb:    ev.memory_total_mb    || 0,
      };
      updateResourceBadge();
      break;

    case 'done':
      appendLog('\x1b[1;32m✓ All done\x1b[0m  —  press Ctrl+C in the terminal to exit');
      // The run's profiling session (if any) has already stopped by the
      // time this event fires -- see armProfilerPolling()'s docstring --
      // so start polling right now instead of never checking at all.
      armProfilerPolling();
      break;
  }

  // Any event can carry the same human-readable line agDataLogger prints to
  // stderr (e.g. "[agent] SKILL OK ..."). Nothing emits a dedicated `log`
  // event to the global stream anymore, so this is how the shared log panel
  // sees anything beyond the few cases above that append explicitly --
  // except per-agent tool/LLM/syscall events, which go to that agent's own
  // system log instead (see isSystemLogEvent()).
  if (ev.term_message) {
    if (ev.agname && isSystemLogEvent(ev)) {
      recordSystemLog(ev.agname, ev.term_message, ev.ts);
    } else {
      appendAgentLog(ev.term_message, ev.color, ev.ts);
    }
  }
}

// Keep team agents contiguous and before standalone agents
function reorderAgents() {
  const prevAgent = currentAgent();
  const inTeam = new Set();
  const teamFirst = [];
  for (const [, agents] of state.teams) {
    for (const ag of agents) {
      if (!inTeam.has(ag)) { inTeam.add(ag); teamFirst.push(ag); }
    }
  }
  const standalone = state.agentOrder.filter(a => !inTeam.has(a));
  state.agentOrder = [...teamFirst, ...standalone];
  // Restore focus to the same agent by name so adding/reordering agents
  // doesn't silently replace the user's selection.
  if (prevAgent) {
    const newIdx = visibleOrder().indexOf(prevAgent);
    if (newIdx >= 0) state.focusedIdx = newIdx;
  }
}

// agname -> generation counter, guarding against that agent's own responses
// landing out of order (see the overlap comment below) -- scoped per agent
// since the focused agent and every pinned side panel now poll
// concurrently and must not invalidate each other.
const agentDetailGenerations = new Map();
// agname -> signature of the last content actually rendered. Polled every
// 200ms, but the underlying data usually only changes once per LLM/tool
// exchange -- rebuilding a panel's entire innerHTML unconditionally on
// every tick was pure churn most of the time, and repeatedly replacing
// that subtree risked jittering scrollTop enough to flip its autoscroll
// flag to false, silently freezing the visible view while content kept
// accumulating underneath.
const agentDetailSignatures = new Map();

async function loadAgentDetail(agname) {
  if (!agname) return;
  const generation = (agentDetailGenerations.get(agname) || 0) + 1;
  agentDetailGenerations.set(agname, generation);
  try {
    const response = await fetch('/api/agents/' + encodeURIComponent(agname));
    const detail = await response.json();
    if (agentDetailGenerations.get(agname) !== generation) return;
    if (detail.error) {
      appendLog('[agent detail] ' + detail.error);
      return;
    }
    const newMessages = detail.messages || [];
    const agentState = detail.state || {};
    const configPayload = detail.config || {};
    // Cheap but correct: messages can update in place (streaming text
    // growing on the last one), not just grow in count, so compare full
    // content, not just length.
    const signature = JSON.stringify(newMessages) + ' ' + (agentState.state || '') +
      ' ' + (agentState.tool || '');
    const changed = agentDetailSignatures.get(agname) !== signature;
    agentDetailSignatures.set(agname, signature);

    state.histories.set(agname, newMessages);
    const existing = state.agents.get(agname) || { color: '#d4d4d4' };
    state.agents.set(agname, {
      ...existing,
      state: normalizeAgentState(agentState.state) || existing.state || 'inactive',
      skill: agentState.skill ?? existing.skill ?? null,
      tool: agentState.tool ?? existing.tool ?? null,
      config: configPayload.config || configPayload,
    });
    const tokenPayload = detail.tokens || {};
    if (Object.keys(tokenPayload).length) {
      state.tokenUsage.set(agname, {
        inp: tokenPayload.agent_input || tokenPayload.input || 0,
        out: tokenPayload.agent_output || tokenPayload.output || 0,
        history: [],
      });
    }
    renderAgentList();
    if (changed) refreshAgentView(agname);
    if (agname === currentAgent()) updateInteractionTitle();
  } catch (error) {
    if (agentDetailGenerations.get(agname) === generation) {
      appendLog('[agent detail] fetch failed: ' + error);
    }
  }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

document.getElementById('nav-prev').addEventListener('click', cyclePrev);
document.getElementById('nav-next').addEventListener('click', cycleNext);

document.getElementById('agent-tabs').addEventListener('click', e => {
  const btn = e.target.closest('.tab-btn');
  if (!btn) return;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  state.activeTab  = btn.dataset.tab;
  state.focusedIdx = 0;
  renderAgentList();
  renderHistory();
  loadAgentDetail(currentAgent());
});

if ($agentSearch) {
  $agentSearch.addEventListener('input', () => {
    state.focusedIdx = 0;
    renderAgentList();
  });
}

document.addEventListener('keydown', e => {
  if (e.key === 'Tab' && !e.shiftKey) { e.preventDefault(); cycleNext(); }
  if (e.key === 'Tab' &&  e.shiftKey) { e.preventDefault(); cyclePrev(); }
  if (e.key === '[') cyclePrev();
  if (e.key === ']') cycleNext();
});

function cycleNext() {
  const n = visibleOrder().length;
  if (!n) return;
  state.focusedIdx = (state.focusedIdx + 1) % n;
  renderAgentList();
  renderHistory();
  loadAgentDetail(currentAgent());
}

function cyclePrev() {
  const n = visibleOrder().length;
  if (!n) return;
  state.focusedIdx = (state.focusedIdx - 1 + n) % n;
  renderAgentList();
  renderHistory();
  loadAgentDetail(currentAgent());
}

// Click an agent in the right panel to focus it
$agentList.addEventListener('click', e => {
  const entry = e.target.closest('.agent-entry');
  if (!entry) return;
  const agname = entry.dataset.agname;
  const idx = visibleOrder().indexOf(agname);
  if (idx >= 0) {
    state.focusedIdx = idx;
    renderAgentList();
    renderHistory();
    loadAgentDetail(agname);
  }
});

// Drag an agent entry onto the interaction pane (see #interaction's drop
// handler above) to pin it there as a side panel.
$agentList.addEventListener('dragstart', e => {
  const entry = e.target.closest('.agent-entry');
  if (!entry) return;
  e.dataTransfer.effectAllowed = 'copy';
  e.dataTransfer.setData(AGENT_DRAG_TYPE, entry.dataset.agname);
});

// ---------------------------------------------------------------------------
// Pause / resume actions
// ---------------------------------------------------------------------------

$btnPauseToggle.addEventListener('click', () => {
  const agname = currentAgent();
  if (!agname) return;
  const ag = state.agents.get(agname);
  const type = (ag && ag.state === 'paused') ? 'resume' : 'pause';
  ws.send(JSON.stringify({ type, agname }));
});

$btnPauseAll.addEventListener('click', () => {
  ws.send(JSON.stringify({ type: 'pause_all' }));
});

$btnResumeAll.addEventListener('click', () => {
  ws.send(JSON.stringify({ type: 'resume_all' }));
});

// ---------------------------------------------------------------------------
// Config editor modal
// ---------------------------------------------------------------------------

function renderConfigField(owner, field, value) {
  const inputId = `cfgfield__${owner}__${field}`;
  const attrs = `id="${inputId}" data-owner="${esc(owner)}" data-field="${esc(field)}"`;
  let inputHtml;
  if (typeof value === 'boolean') {
    inputHtml = `<input type="checkbox" ${attrs} data-kind="bool" ${value ? 'checked' : ''}>`;
  } else if (typeof value === 'number') {
    inputHtml = `<input type="number" step="any" ${attrs} data-kind="number" value="${esc(String(value))}">`;
  } else if (value === null || typeof value === 'string') {
    inputHtml = `<input type="text" ${attrs} data-kind="nullable_string" value="${esc(value ?? '')}">`;
  } else {
    inputHtml = `<textarea ${attrs} data-kind="json" rows="2">${esc(JSON.stringify(value))}</textarea>`;
  }
  return `<div class="config-field">
    <label for="${inputId}">${esc(field)}</label>
    ${inputHtml}
  </div>`;
}

function openConfigModal(agname) {
  const ag = state.agents.get(agname);
  const config = (ag && ag.config) || {};
  $configTitle.textContent = `Config — ${agname}`;

  const owners = Object.keys(config).sort();
  const frags = [];
  for (const owner of owners) {
    frags.push(`<div class="config-owner">${esc(owner)}</div>`);
    const fields = config[owner];
    for (const field of Object.keys(fields).sort()) {
      frags.push(renderConfigField(owner, field, fields[field]));
    }
  }
  $configBody.innerHTML = frags.join('') || '<div class="dim">No editable config fields.</div>';
  $configOverlay.dataset.agname = agname;
  $configOverlay.classList.remove('hidden');
}

function closeConfigModal() {
  $configOverlay.classList.add('hidden');
  delete $configOverlay.dataset.agname;
}

function collectConfigEdits() {
  const result = {};
  $configBody.querySelectorAll('[data-owner]').forEach(el => {
    const { owner, field, kind } = el.dataset;
    let value;
    if      (kind === 'bool')            value = el.checked;
    else if (kind === 'number')          value = Number(el.value);
    else if (kind === 'nullable_string') value = el.value === '' ? null : el.value;
    else /* json */ {
      try { value = JSON.parse(el.value); }
      catch (e) { throw new Error(`Invalid JSON for ${owner}.${field}: ${e.message}`); }
    }
    (result[owner] = result[owner] || {})[field] = value;
  });
  return result;
}

$btnUpdateConfig.addEventListener('click', () => {
  const agname = currentAgent();
  if (agname) openConfigModal(agname);
});

$configCancel.addEventListener('click', closeConfigModal);

$configOverlay.addEventListener('click', e => {
  if (e.target === $configOverlay) closeConfigModal();
});

$configUpdate.addEventListener('click', () => {
  const agname = $configOverlay.dataset.agname;
  if (!agname) return;
  let config;
  try { config = collectConfigEdits(); } catch (e) { alert(e.message); return; }
  ws.send(JSON.stringify({ type: 'update_config', agname, config }));
  closeConfigModal();
});

$configUpdateAll.addEventListener('click', () => {
  let config;
  try { config = collectConfigEdits(); } catch (e) { alert(e.message); return; }
  ws.send(JSON.stringify({ type: 'update_config_all', config }));
  closeConfigModal();
});

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

const ws = new WebSocket(`ws://${location.host}/ws`);
let wsClosed = false;

ws.onmessage = e => {
  try {
    const ev = JSON.parse(e.data);
    // In historical mode, only accept timeline_sync and ignore live events.
    if (!state.timeline.liveMode && ev.type !== 'timeline_sync') return;
    handleEvent(ev);
  } catch {}
};

ws.onclose = () => {
  wsClosed = true;
  appendLog('\x1b[31m[web ui] connection closed — reload to reconnect\x1b[0m');
};

ws.onerror = () => {
  appendLog('\x1b[31m[web ui] connection error\x1b[0m');
};

// The interaction panel(s) (messages/state/config/tokens) only ever refresh
// on an explicit selection or pin -- /api/agents/{agname} is a pull, not
// something the live event stream pushes updates for. Poll the focused
// agent AND every pinned side panel while connected, so all of them keep
// advancing without the user re-clicking/re-dragging anything.
// loadAgentDetail() already guards a response landing after that agent's
// own generation moved on, so this is safe to fire even mid-fetch.
//
// Self-rescheduling setTimeout, NOT setInterval: setInterval fires on a
// fixed wall-clock cadence regardless of whether the previous call has
// resolved yet. If a round trip ever takes longer than the interval, calls
// start overlapping -- and since each overlapping call bumps that agent's
// own generation before the earlier one's fetch resolves, every single
// response for it arrives already stale and gets silently discarded.
// Confirmed this was happening for real: every response stale, latency
// ~600ms against a 200ms interval -- permanent starvation once overlap
// begins (each overlapping request competes for the browser's per-host
// connection limit, which slows every request further, which causes more
// overlap), not just an occasional race. Only scheduling the next poll
// after the current one finishes makes that starvation structurally
// impossible.
async function _pollAgentDetailLoop() {
  if (wsClosed) return;
  if (state.timeline.liveMode) {
    const agname = currentAgent();
    const toPoll = new Set(state.pinnedAgents);
    if (agname) toPoll.add(agname);
    await Promise.all([...toPoll].map(loadAgentDetail));
  }
  setTimeout(_pollAgentDetailLoop, 200);
}
_pollAgentDetailLoop();

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function esc(s) {
  return String(s)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;');
}
