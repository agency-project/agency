// Browser side of an interactive CLI session (agency/sessions/session.py).
// Binary frames carry raw terminal bytes in both directions; text frames are
// JSON control messages. Every (re)connection starts with a full screen
// redraw from the server, so a reconnect needs no client-side state.

const TOKEN_KEY = 'agency-terminal-token';
const stateLabel = document.getElementById('state');
const titleLabel = document.getElementById('cli-label');
const terminalElement = document.getElementById('terminal');
const profilerPane = document.getElementById('profiler-pane');
const profilerToggle = document.getElementById('toggle-profiler');

function readToken() {
  const fragment = new URLSearchParams(location.hash.slice(1)).get('token');
  if (fragment) {
    // Keep the token out of the address bar (and screenshots of it) while
    // still letting a reload or another tab on this origin reconnect.
    try { localStorage.setItem(TOKEN_KEY, fragment); } catch {}
    history.replaceState(null, '', location.pathname + location.search);
    return fragment;
  }
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}

function showState(text, tone = '') {
  stateLabel.textContent = text;
  stateLabel.dataset.tone = tone;
}

function showNotice(html) {
  document.getElementById('terminal-pane').innerHTML = `<div id="notice">${html}</div>`;
}

const token = readToken();
const term = new Terminal({
  cursorBlink: true,
  fontFamily: "'Cascadia Code', 'Fira Code', 'JetBrains Mono', Menlo, monospace",
  fontSize: 13,
  scrollback: 5000,
  theme: { background: '#000000' },
});
const fit = new FitAddon.FitAddon();
term.loadAddon(fit);

const encoder = new TextEncoder();
let socket = null;
let exited = false;
let retryDelay = 500;

function send(data) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(data);
}

function connect() {
  showState('Connecting…');
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${protocol}://${location.host}/ws/terminal`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    socket.send(JSON.stringify({ token, cols: term.cols, rows: term.rows }));
  };
  socket.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      term.write(new Uint8Array(event.data));
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === 'attached') {
      retryDelay = 500;
      showState(message.paused ? 'Paused' : 'Connected', message.paused ? 'warn' : 'ok');
    } else if (message.type === 'exited') {
      exited = true;
      showState(`${titleLabel.textContent} exited (${message.returncode})`, 'error');
    } else if (message.type === 'input_dropped') {
      showState(message.paused ? 'Paused · input not delivered' : 'Input not delivered', 'warn');
    }
  };
  socket.onclose = (event) => {
    if (exited) return;
    if (event.code === 4401 || event.code === 4403) {
      showState('Not authorized', 'error');
      showNotice('This page needs the access link printed by <code>agency claude</code> or <code>agency codex</code> ' +
        '(it ends in <code>#token=…</code>). Open that link in this browser.');
      return;
    }
    showState('Disconnected · reconnecting', 'warn');
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 8000);
  };
}

term.onData((data) => send(encoder.encode(data)));
// Bytes xterm.js produces outside UTF-8 text (e.g. legacy mouse reports).
term.onBinary((data) => send(Uint8Array.from(data, (c) => c.charCodeAt(0))));
term.onResize(({ cols, rows }) => send(JSON.stringify({ type: 'resize', cols, rows })));
// xterm.js sends Shift+Enter as a plain carriage return, which submits the
// prompt. Claude Code and Codex insert a newline for the kitty keyboard
// protocol's Shift+Enter (CSI 13;2u), as they do in terminals that speak that
// protocol; ESC CR (Meta+Enter) was verified to submit in Claude Code 2.1.299.
term.attachCustomKeyEventHandler((event) => {
  if (event.key !== 'Enter' || !event.shiftKey) return true;
  if (event.type === 'keydown') send(encoder.encode('\x1b[13;2u'));
  return false;
});

profilerToggle.addEventListener('click', () => {
  const visible = profilerPane.hidden;
  profilerPane.hidden = !visible;
  profilerToggle.setAttribute('aria-pressed', String(visible));
  requestAnimationFrame(() => fit.fit());
});

// Opening a new access link in a tab already showing this page only changes
// the fragment, which does not reload it; the new token needs a fresh start.
window.addEventListener('hashchange', () => location.reload());

async function start() {
  if (!token) {
    showState('Not authorized', 'error');
    showNotice('Open the access link printed by <code>agency claude</code> or <code>agency codex</code> ' +
      '(it ends in <code>#token=…</code>).');
    return;
  }
  const status = await fetch('/api/terminal').then((r) => r.json()).catch(() => ({}));
  if (status.label) {
    titleLabel.textContent = status.label;
    document.title = `Agency · ${status.label}`;
  }
  if (!status.available) {
    showState('No interactive session', 'error');
    showNotice('No interactive session is running in this run. ' +
      'Start one with <code>agency claude</code> or <code>agency codex</code>.');
    return;
  }
  term.open(terminalElement);
  fit.fit();
  new ResizeObserver(() => fit.fit()).observe(terminalElement);
  term.focus();
  connect();
}

start();
