// Perfetto is served by this same web UI; no external viewer or server.
(() => {
  const byId = id => document.getElementById(id);
  const frame = byId('profiler-frame');
  const status = byId('profiler-status');
  let initialized = false;
  let loading = false;
  let generation = 0;

  function setBusy(value) {
    loading = value;
    byId('profiler-reload').disabled = value;
    byId('profiler-open').disabled = value;
  }

  async function openTrace(buffer, title) {
    frame.hidden = false;
    // A fresh document restores Perfetto's one-shot trace message handler.
    const ready = new Promise((resolve, reject) => {
      const cleanup = () => {
        clearInterval(ping);
        clearTimeout(timeout);
        window.removeEventListener('message', receive);
      };
      const receive = event => {
        if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data !== 'PONG') return;
        cleanup();
        resolve();
      };
      window.addEventListener('message', receive);
      const ping = setInterval(() => frame.contentWindow?.postMessage('PING', location.origin), 250);
      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error('Perfetto did not load. Try reloading the trace.'));
      }, 60000);
    });
    frame.src = `/perfetto/?load=${++generation}#!/?mode=embedded`;
    await ready;
    frame.contentWindow.postMessage({perfetto: {buffer, title}}, location.origin, [buffer]);
    status.textContent = `${title} · Start-to-start connections`;
  }

  async function load(file) {
    if (loading) return;
    setBusy(true);
    status.textContent = 'Loading profiler…';
    try {
      const response = await fetch('/api/profiler', {cache: 'no-store'});
      if (!response.ok) throw new Error('Could not check profiler availability.');
      const info = await response.json();
      if (!info.viewer_available) {
        status.textContent = 'Perfetto assets are missing. Restart the web UI to rebuild them automatically.';
        return;
      }
      if (file) {
        await openTrace(await file.arrayBuffer(), file.name);
      } else if (info.trace_available) {
        const trace = await fetch('/api/profiler/trace', {cache: 'no-store'});
        if (!trace.ok) throw new Error('Could not load this run’s trace.');
        await openTrace(await trace.arrayBuffer(), 'Current run');
      } else {
        status.textContent = 'No completed trace yet. Reload after profiling finishes, or open a trace file.';
      }
      initialized = true;
    } catch (error) {
      status.textContent = error.message;
    } finally {
      setBusy(false);
    }
  }

  function select(profiler) {
    byId('app').hidden = profiler;
    byId('profiler-view').hidden = !profiler;
    for (const [id, active] of [['view-run', !profiler], ['view-profiler', profiler]]) {
      byId(id).classList.toggle('active', active);
      byId(id).setAttribute('aria-pressed', String(active));
    }
    if (profiler && !initialized) load();
  }
  byId('view-run').onclick = () => select(false);
  byId('view-profiler').onclick = () => select(true);
  byId('profiler-reload').onclick = () => load();
  byId('profiler-open').onclick = () => byId('profiler-file').click();
  byId('profiler-file').onchange = event => {
    const file = event.target.files[0];
    if (file) load(file);
    event.target.value = '';
  };
})();
