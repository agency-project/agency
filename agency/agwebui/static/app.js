// app.js — Agency Web UI client
// Connects to the WebSocket, processes events, and renders the dashboard.

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  agents:      new Map(),  // agname -> { color, state, skill, tool }
  teams:       new Map(),  // team_name -> Set<agname>
  histories:   new Map(),  // agname -> msg[]
  tokenUsage:  new Map(),  // agname -> { inp, out, history: [{ts,inp,out}] }
  globalTokens: { inp: 0, out: 0, history: [] },
  resources: { gpus_acquired: 0, gpus_total: 0, cpus_acquired: 0, cpus_total: 0, memory_acquired_mb: 0, memory_total_mb: 0 },
  agentOrder: [],          // [agname] ordered for display / Tab cycling
  focusedIdx: 0,
  pendingAsk: null,        // { agname, ask_id, question } | null
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

// Reset all agent/log state before replaying a historical window.
function clearAgentState() {
  state.agents.clear();
  state.teams.clear();
  state.histories.clear();
  state.agentOrder = [];
  state.focusedIdx = 0;
  state.pendingAsk = null;
  $sharedLog.innerHTML = '';
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
const $agentHistory     = document.getElementById('agent-history');
const $agentList        = document.getElementById('agent-list');
const $interactionTitle = document.getElementById('interaction-title');
const $agentInput       = document.getElementById('agent-input');
const $navLabel         = document.getElementById('nav-label');
const $countAll         = document.getElementById('count-all');
const $countLive        = document.getElementById('count-live');
const $countIdle        = document.getElementById('count-idle');
const $countFinished    = document.getElementById('count-finished');
const $agentSearch      = document.getElementById('agent-search');
const $globalTokens     = document.getElementById('global-tokens');
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

const RATE_WINDOW_S = 60;

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

function updateGlobalTokenBadge() {
  const { inp, out, history } = state.globalTokens;
  $globalTokens.textContent = (inp || out) ? fmtTokens(inp, out, history) : '';
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

  const waiting = (state.pendingAsk?.agname === agname) ? '  ?' : '';
  const usage   = state.tokenUsage.get(agname);
  const tokHtml = usage
    ? `<span class="token-badge">${fmtTokens(usage.inp, usage.out, usage.history)}</span>`
    : '';
  $interactionTitle.innerHTML =
    `${esc(agname)}  [${idx + 1}/${n}]  ← →${esc(waiting)}${tokHtml}`;
  $navLabel.textContent = `${idx + 1} / ${n}`;
}

// ---------------------------------------------------------------------------
// Shared log
// ---------------------------------------------------------------------------

let logAutoScroll = true;

$sharedLog.addEventListener('scroll', () => {
  logAutoScroll = $sharedLog.scrollHeight - $sharedLog.scrollTop - $sharedLog.clientHeight < 40;
});

function appendLog(line) {
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = ansiToHtml(line);
  $sharedLog.appendChild(div);
  if (logAutoScroll) $sharedLog.scrollTop = $sharedLog.scrollHeight;
  // Cap at 5000 lines to prevent unbounded growth
  while ($sharedLog.children.length > 5000) $sharedLog.removeChild($sharedLog.firstChild);
}

// ---------------------------------------------------------------------------
// Agent list (right panel)
// ---------------------------------------------------------------------------

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

  return `<div class="agent-entry${focusedClass}" data-agname="${esc(agname)}" style="padding-left:${indent}px">
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

function renderHistory() {
  const agname = currentAgent();
  const visible = visibleOrder();
  const n   = visible.length;
  const idx = n ? state.focusedIdx % n : 0;

  if (!agname) {
    updateInteractionTitle();
    $agentHistory.innerHTML = '';
    return;
  }

  updateInteractionTitle();

  const msgs  = state.histories.get(agname) || [];
  const frags = [];

  for (const msg of msgs) {
    const role    = msg.role    || '';
    const content = msg.content || '';

    if (role === 'system') {
      frags.push(`<div class="msg-system">─── sys: ${esc(content)}</div>`);

    } else if (role === 'user') {
      frags.push(`<div class="msg-user"><span class="role-user">▶ user</span>  ${esc(content)}</div>`);

    } else if (role === 'assistant') {
      const thinking   = msg._thinking || '';
      const toolCalls  = msg.tool_calls || [];

      if (thinking) {
        frags.push(`<div class="msg-thinking">💭 thinking\n${esc(thinking)}</div>`);
      }
      for (const tc of toolCalls) {
        const fn    = tc.function || {};
        const fname = fn.name || '?';
        let argsText = '';
        try {
          const raw = JSON.parse(fn.arguments || '{}');
          argsText = Object.entries(raw)
            .map(([k, v]) => `  ${esc(k)}: ${esc(String(v))}`)
            .join('\n');
        } catch {
          argsText = esc(fn.arguments || '');
        }
        frags.push(
          `<div class="msg-tool-call"><span class="role-tool-call">⚙ ${esc(fname)}</span>\n` +
          `<span class="dim">${argsText}</span></div>`
        );
      }
      if (content) {
        frags.push(`<div class="msg-assistant"><span class="role-assistant">◆ asst</span>\n${esc(content)}</div>`);
      }

    } else if (role === 'tool') {
      frags.push(`<div class="msg-tool-result"><span class="dim">← ${esc(content)}</span></div>`);

    } else if (msg.type === 'skill_start') {
      frags.push(`<div class="msg-event">── skill: ${esc(msg.skill || '')} ──</div>`);

    } else if (msg.type === 'skill_error') {
      frags.push(
        `<div class="msg-error"><span class="role-error">✗ skill error</span>` +
        ` [${esc(msg.skill || '')}]\n${esc(msg.error || '')}</div>`
      );

    } else if (msg.type === 'llm_retry') {
      frags.push(
        `<div class="msg-warning"><span class="role-warning">⟳ LLM retry</span>` +
        ` attempt ${esc(String(msg.attempt || ''))}: ${esc(msg.error || '')}</div>`
      );

    } else if (msg.type === 'llm_error') {
      frags.push(
        `<div class="msg-error"><span class="role-error">✗ LLM error</span>` +
        `\n${esc(msg.error || '')}</div>`
      );
    }
  }

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

  // Pending ask_human question
  if (state.pendingAsk?.agname === agname) {
    frags.push(
      `<div class="msg-ask"><span class="role-ask">? ${esc(state.pendingAsk.question)}</span></div>`
    );
  }

  $agentHistory.innerHTML = frags.join('');
  if (histAutoScroll) $agentHistory.scrollTop = $agentHistory.scrollHeight;
}

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
      break;

    case 'agent_config': {
      const existing = state.agents.get(ev.agname) || { color: '#d4d4d4', state: 'inactive', skill: null, tool: null };
      state.agents.set(ev.agname, { ...existing, config: ev.config || {} });
      if (!state.agentOrder.includes(ev.agname)) state.agentOrder.push(ev.agname);
      break;
    }

    case 'agent_state': {
      const existing = state.agents.get(ev.agname) || { color: '#d4d4d4' };
      state.agents.set(ev.agname, {
        ...existing,
        color: ev.color || existing.color || '#d4d4d4',
        state: ev.state,
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
      if (ev.agname === currentAgent()) renderHistory();
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
      if (ev.agname === currentAgent()) renderHistory();
      break;

    case 'ask_human': {
      state.pendingAsk = { agname: ev.agname, ask_id: ev.ask_id, question: ev.question };
      const idx = state.agentOrder.indexOf(ev.agname);
      if (idx >= 0) state.focusedIdx = idx;
      renderAgentList();
      renderHistory();
      $agentInput.focus();
      break;
    }

    case 'human_reply':
      if (state.pendingAsk?.ask_id === ev.ask_id) state.pendingAsk = null;
      renderHistory();
      break;

    case 'token_update': {
      const prev    = state.tokenUsage.get(ev.agname) || { inp: 0, out: 0, history: [] };
      const history = prev.history;
      history.push({ ts: ev.ts, inp: ev.agent_input, out: ev.agent_output });
      const cutoff  = ev.ts - RATE_WINDOW_S;
      while (history.length > 1 && history[0].ts < cutoff) history.shift();
      state.tokenUsage.set(ev.agname, { inp: ev.agent_input, out: ev.agent_output, history });

      // Global token values from concurrent agents race: each agent reports
      // global_base + its_own_live_progress, so values can arrive out of order.
      // Track the running maximum as the displayed total and only append to
      // history when the value is a new high-water mark, ensuring the rate
      // window is always monotonically increasing (no negative rates).
      const gInp = Math.max(state.globalTokens.inp, ev.global_input);
      const gOut = Math.max(state.globalTokens.out, ev.global_output);
      const gHistory = state.globalTokens.history;
      if (ev.global_input >= state.globalTokens.inp) {
        gHistory.push({ ts: ev.ts, inp: ev.global_input, out: ev.global_output });
        while (gHistory.length > 1 && gHistory[0].ts < cutoff) gHistory.shift();
      }
      state.globalTokens = { inp: gInp, out: gOut, history: gHistory };
      updateGlobalTokenBadge();
      if (ev.agname === currentAgent()) updateInteractionTitle();
      break;
    }

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
      break;
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
});

if ($agentSearch) {
  $agentSearch.addEventListener('input', () => {
    state.focusedIdx = 0;
    renderAgentList();
  });
}

document.addEventListener('keydown', e => {
  if (document.activeElement === $agentInput) return;
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
}

function cyclePrev() {
  const n = visibleOrder().length;
  if (!n) return;
  state.focusedIdx = (state.focusedIdx - 1 + n) % n;
  renderAgentList();
  renderHistory();
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
  }
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
// Input / ask_human reply
// ---------------------------------------------------------------------------

$agentInput.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const text = $agentInput.value.trim();
  $agentInput.value = '';
  if (!text) return;

  const agname = currentAgent();
  if (!agname) return;

  if (state.pendingAsk?.agname === agname) {
    const { ask_id } = state.pendingAsk;
    ws.send(JSON.stringify({ type: 'human_reply', ask_id, text }));
    // Optimistically clear so the UI doesn't show stale ? state
    state.pendingAsk = null;
    renderHistory();
  }
  // Unsolicited messages are not yet forwarded to agents.
});

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = e => {
  try {
    const ev = JSON.parse(e.data);
    // In historical mode, only accept timeline_sync and ignore live events.
    if (!state.timeline.liveMode && ev.type !== 'timeline_sync') return;
    handleEvent(ev);
  } catch {}
};

ws.onclose = () => {
  appendLog('\x1b[31m[web ui] connection closed — reload to reconnect\x1b[0m');
};

ws.onerror = () => {
  appendLog('\x1b[31m[web ui] connection error\x1b[0m');
};

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
