// app.js — Agency Web UI client
// Connects to the WebSocket, processes events, and renders the dashboard.

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  agents:     new Map(),   // agname -> { color, state, skill, tool }
  teams:      new Map(),   // team_name -> Set<agname>
  histories:  new Map(),   // agname -> msg[]
  agentOrder: [],          // [agname] ordered for display / Tab cycling
  focusedIdx: 0,
  pendingAsk: null,        // { agname, ask_id, question } | null
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
// DOM references
// ---------------------------------------------------------------------------

const $sharedLog        = document.getElementById('shared-log');
const $agentHistory     = document.getElementById('agent-history');
const $agentList        = document.getElementById('agent-list');
const $interactionTitle = document.getElementById('interaction-title');
const $agentInput       = document.getElementById('agent-input');
const $navLabel         = document.getElementById('nav-label');

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

function currentAgent() {
  if (!state.agentOrder.length) return null;
  return state.agentOrder[state.focusedIdx % state.agentOrder.length];
}

function renderAgentList() {
  // Build agname → team_name reverse map
  const agentTeam = new Map();
  for (const [tname, agents] of state.teams) {
    for (const ag of agents) agentTeam.set(ag, tname);
  }

  const frags = [];
  const emittedTeams = new Set();
  const focused = currentAgent();

  for (const agname of state.agentOrder) {
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
}

function renderAgentEntry(agname, ag, indent, isFocused) {
  const color = ag.color || '#d4d4d4';
  const focusedClass = isFocused ? ' focused' : '';
  const { state: st, skill, tool } = ag;
  let dot, statusHtml;

  if (st === 'inactive') {
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
  const n   = state.agentOrder.length;
  const idx = n ? state.focusedIdx % n : 0;

  if (!agname) {
    $interactionTitle.textContent = 'No agents';
    $navLabel.textContent = '';
    $agentHistory.innerHTML = '';
    return;
  }

  const waiting = (state.pendingAsk?.agname === agname) ? '  ?' : '';
  $interactionTitle.textContent = `${agname}  [${idx + 1}/${n}]  ← →${waiting}`;
  $navLabel.textContent = `${idx + 1} / ${n}`;

  const msgs  = state.histories.get(agname) || [];
  const frags = [];

  for (const msg of msgs) {
    const role    = msg.role    || '';
    const content = msg.content || '';

    if (role === 'system') {
      const first = content.split('\n')[0].slice(0, 120);
      frags.push(`<div class="msg-system">─── sys: ${esc(first)}</div>`);

    } else if (role === 'user') {
      const preview = content.slice(0, 300).replace(/\n/g, ' ');
      frags.push(`<div class="msg-user"><span class="role-user">▶ user</span>  ${esc(preview)}</div>`);

    } else if (role === 'assistant') {
      const thinking   = msg._thinking || '';
      const toolCalls  = msg.tool_calls || [];

      if (thinking) {
        frags.push(`<div class="msg-thinking">💭 thinking\n${esc(thinking.slice(0, 800))}</div>`);
      }
      for (const tc of toolCalls) {
        const fn    = tc.function || {};
        const fname = fn.name || '?';
        let argsText = '';
        try {
          const raw = JSON.parse(fn.arguments || '{}');
          argsText = Object.entries(raw)
            .map(([k, v]) => `  ${esc(k)}: ${esc(String(v).slice(0, 120))}`)
            .join('\n');
        } catch {
          argsText = esc((fn.arguments || '').slice(0, 200));
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
      frags.push(`<div class="msg-tool-result"><span class="dim">← ${esc(content.slice(0, 600))}</span></div>`);
    }
  }

  // Running indicator
  const ag = state.agents.get(agname);
  if (ag && ag.state !== 'inactive') {
    const st = ag.state;
    let label;
    if      (st === 'llm')       label = 'LLM Thinking…';
    else if (st === 'tool')      label = `Tool Running: ${ag.tool || ''}…`;
    else if (st === 'proc_wait') label = 'Waiting for processes…';
    else if (st === 'human')     label = 'Input Pending…';
    else                         label = 'Running…';
    frags.push(`<div class="msg-running">▶ ${esc(label)}</div>`);
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

    case 'log':
      appendLog(ev.line);
      break;

    case 'agent_registered':
      if (!state.agents.has(ev.agname)) {
        state.agents.set(ev.agname, {
          color: ev.color || '#d4d4d4',
          state: 'inactive', skill: null, tool: null,
        });
        state.agentOrder.push(ev.agname);
      }
      renderAgentList();
      break;

    case 'agent_state': {
      const existing = state.agents.get(ev.agname) || { color: '#d4d4d4' };
      state.agents.set(ev.agname, {
        ...existing,
        state: ev.state,
        skill: ev.skill,
        tool:  ev.tool,
      });
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

    case 'done':
      appendLog('\x1b[1;32m✓ All done\x1b[0m  —  press Ctrl+C in the terminal to exit');
      break;
  }
}

// Keep team agents contiguous and before standalone agents
function reorderAgents() {
  const inTeam = new Set();
  const teamFirst = [];
  for (const [, agents] of state.teams) {
    for (const ag of agents) {
      if (!inTeam.has(ag)) { inTeam.add(ag); teamFirst.push(ag); }
    }
  }
  const standalone = state.agentOrder.filter(a => !inTeam.has(a));
  state.agentOrder = [...teamFirst, ...standalone];
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

document.getElementById('nav-prev').addEventListener('click', cyclePrev);
document.getElementById('nav-next').addEventListener('click', cycleNext);

document.addEventListener('keydown', e => {
  if (document.activeElement === $agentInput) return;
  if (e.key === 'Tab' && !e.shiftKey) { e.preventDefault(); cycleNext(); }
  if (e.key === 'Tab' &&  e.shiftKey) { e.preventDefault(); cyclePrev(); }
  if (e.key === '[') cyclePrev();
  if (e.key === ']') cycleNext();
});

function cycleNext() {
  if (!state.agentOrder.length) return;
  state.focusedIdx = (state.focusedIdx + 1) % state.agentOrder.length;
  renderAgentList();
  renderHistory();
}

function cyclePrev() {
  if (!state.agentOrder.length) return;
  state.focusedIdx = (state.focusedIdx - 1 + state.agentOrder.length) % state.agentOrder.length;
  renderAgentList();
  renderHistory();
}

// Click an agent in the right panel to focus it
$agentList.addEventListener('click', e => {
  const entry = e.target.closest('.agent-entry');
  if (!entry) return;
  const agname = entry.dataset.agname;
  const idx = state.agentOrder.indexOf(agname);
  if (idx >= 0) {
    state.focusedIdx = idx;
    renderAgentList();
    renderHistory();
  }
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
  try { handleEvent(JSON.parse(e.data)); } catch {}
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
