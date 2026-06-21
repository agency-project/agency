"""Standalone web server for agwebui.

No agency imports — this process is completely isolated from the execution
process.  It tails ui_events.jsonl and pushes events to browsers over
WebSocket.  Run via:

    python -m agency.agwebui.server --run-dir <path> --port 7860
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time as _time
from contextlib import asynccontextmanager
from pathlib import Path

# Seconds east of UTC for the server's local timezone (accounts for DST).
_TZ_OFFSET: int = -(_time.altzone if _time.daylight and _time.localtime().tm_isdst else _time.timezone)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_STATIC = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TAIL_BYTES     = 2 * 1024 * 1024   # bytes replayed to new clients on connect
INDEX_INTERVAL = 1_000        # build one index entry per N events

# ---------------------------------------------------------------------------
# Mutable globals — set in __main__ before uvicorn starts
# ---------------------------------------------------------------------------

_run_dir:   Path = Path(".")
_reply_dir: Path = Path(".")

# Event index: list of (server_timestamp, byte_offset_of_batch_start)
# Gives O(1) seek to any position in the log without storing events in RAM.
_sparse_index:       list[tuple[float, int]] = []
_events_since_index: int   = 0
_file_offset:        int   = 0   # current tail read cursor (bytes)
_file_size:          int   = 0   # last known file size (bytes)
_first_ts:           float | None = None
_last_ts:            float | None = None

_clients: set[WebSocket] = set()
_lock:    asyncio.Lock | None = None   # created at startup

# In-memory registry rebuilt from the event stream as new events arrive.
# Used to inject a "state preamble" for clients that connect mid-run,
# so they always see the full agent/team roster even if those events have
# scrolled past the TAIL_BYTES window.
_agent_registry: dict[str, str] = {}   # agname -> raw JSON line
_team_registry:  dict[str, str] = {}   # team_name -> raw JSON line


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _lock
    _lock = asyncio.Lock()
    task = asyncio.create_task(_tail_events())
    yield
    task.cancel()


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Timeline API
# ---------------------------------------------------------------------------

@app.get("/api/timeline")
async def api_timeline():
    """Return sparse index metadata for the timeline scrubber."""
    assert _lock is not None
    async with _lock:
        total = len(_sparse_index)
        if total == 0:
            return JSONResponse({"index_len": 0, "first_ts": None, "last_ts": None,
                                 "file_size": _file_size, "samples": []})
        # Downsample to ≤500 entries for the client
        step    = max(1, total // 500)
        samples = [[i, _sparse_index[i][0]] for i in range(0, total, step)]
        if samples[-1][0] != total - 1:
            samples.append([total - 1, _sparse_index[-1][0]])
        return JSONResponse({
            "index_len": total,
            "first_ts":  _sparse_index[0][0],
            "last_ts":   _sparse_index[-1][0],
            "file_size": _file_size,
            "samples":   samples,   # [[index_pos, timestamp], ...]
        })


@app.get("/api/events")
async def api_events(index_pos: int = 0, window: int = 10):
    """Return up to window*INDEX_INTERVAL events starting from index_pos."""
    assert _lock is not None
    async with _lock:
        if not _sparse_index:
            return JSONResponse({"events": [], "from_ts": None, "to_ts": None})
        pos     = max(0, min(index_pos, len(_sparse_index) - 1))
        end_pos = min(pos + window, len(_sparse_index))
        start_off = _sparse_index[pos][1]
        from_ts   = _sparse_index[pos][0]
        end_off   = _sparse_index[end_pos][1] if end_pos < len(_sparse_index) else _file_size
        to_ts     = _sparse_index[end_pos - 1][0] if end_pos > 0 else from_ts
        event_file = _run_dir / "ui_events.jsonl"
        cur_size   = _file_size

    events: list[str] = []
    if event_file.exists() and cur_size > 0:
        max_read = max(0, min(end_off - start_off, 4 * 1024 * 1024))
        if max_read:
            with open(event_file, "rb") as f:
                f.seek(start_off)
                raw = f.read(max_read)
            for bline in raw.split(b"\n"):
                s = bline.strip()
                if s:
                    events.append(s.decode("utf-8", errors="replace"))

    return JSONResponse({"events": events, "from_ts": from_ts, "to_ts": to_ts})


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    assert _lock is not None

    # Snapshot index state, then read tail from file (outside lock).
    async with _lock:
        event_file  = _run_dir / "ui_events.jsonl"
        cur_size    = _file_size
        index_len   = len(_sparse_index)
        first_ts    = _first_ts
        last_ts     = _last_ts
        tail_offset = max(0, cur_size - TAIL_BYTES)

    tail_lines: list[bytes] = []
    if event_file.exists() and cur_size > 0:
        with open(event_file, "rb") as f:
            f.seek(tail_offset)
            raw = f.read(cur_size - tail_offset)
        tail_lines = [l for l in raw.split(b"\n") if l.strip()]

    # Atomically send timeline_sync + registry preamble + tail, then register for live updates.
    async with _lock:
        sync = json.dumps({
            "type":       "timeline_sync",
            "index_len":  index_len,
            "first_ts":   first_ts,
            "last_ts":    last_ts,
            "file_size":  cur_size,
            "tail_offset": tail_offset,
            "tz_offset":  _TZ_OFFSET,
        })
        preamble = list(_agent_registry.values()) + list(_team_registry.values())
        try:
            await ws.send_text(sync)
            for bline in tail_lines:
                await ws.send_text(bline.decode("utf-8", errors="replace"))
            # Send registration state after the tail so late-joining clients
            # always learn about every agent/team even if the original events
            # have scrolled past the TAIL_BYTES window.  Duplicates are
            # harmless — the frontend updates by agname/team_name key.
            for line in preamble:
                await ws.send_text(line)
        except Exception:
            return
        _clients.add(ws)

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "human_reply":
                    ask_id = str(msg.get("ask_id", ""))
                    text   = str(msg.get("text", ""))
                    if ask_id:
                        (_reply_dir / f"{ask_id}.txt").write_text(text, encoding="utf-8")
            except Exception:
                pass
    except WebSocketDisconnect:
        async with _lock:
            _clients.discard(ws)


# ---------------------------------------------------------------------------
# Tail loop
# ---------------------------------------------------------------------------

def _scan_registries_from_file(event_file: Path) -> None:
    """Sequential full-file scan to seed _agent_registry and _team_registry."""
    global _agent_registry, _team_registry
    with open(event_file, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                text = raw.decode("utf-8", errors="replace")
                ev = json.loads(text)
                t  = ev.get("type")
                if t == "agent_registered":
                    agn = ev.get("agname")
                    if agn:
                        _agent_registry[agn] = text
                elif t == "team_registered":
                    tn = ev.get("team_name")
                    if tn:
                        _team_registry[tn] = text
            except Exception:
                pass


def _build_index_from_file(event_file: Path, n_points: int = 1000) -> None:
    """Scan the existing log file and populate _sparse_index at startup."""
    global _file_offset, _file_size, _first_ts, _last_ts, _sparse_index
    size = event_file.stat().st_size
    if size == 0:
        return
    _file_size   = size
    _file_offset = size   # tail from end

    step = max(1, size // n_points)
    with open(event_file, "rb") as f:
        for i in range(n_points + 1):
            seek_to = min(i * step, size - 1)
            f.seek(seek_to)
            # Skip to the next complete line boundary.
            if seek_to > 0:
                f.readline()   # discard partial line
            line_offset = f.tell()
            if line_offset >= size:
                break
            raw = f.readline()
            if not raw:
                break
            try:
                ev = json.loads(raw.decode("utf-8", errors="replace"))
                ts = ev.get("ts")
                if ts:
                    _sparse_index.append((float(ts), line_offset))
                    if _first_ts is None or ts < _first_ts:
                        _first_ts = float(ts)
                    if _last_ts is None or ts > _last_ts:
                        _last_ts = float(ts)
            except Exception:
                pass

    # Deduplicate and sort by offset (multiple seeks may land on same line).
    seen_offsets: set[int] = set()
    deduped = []
    for ts, off in sorted(_sparse_index, key=lambda x: x[1]):
        if off not in seen_offsets:
            seen_offsets.add(off)
            deduped.append((ts, off))
    _sparse_index = deduped


async def _tail_events() -> None:
    global _file_offset, _file_size, _first_ts, _last_ts, _events_since_index
    event_file = _run_dir / "ui_events.jsonl"

    # Build index and seed registries from existing file before tailing new events.
    if event_file.exists():
        await asyncio.to_thread(_scan_registries_from_file, event_file)
        await asyncio.to_thread(_build_index_from_file, event_file)

    while True:
        if event_file.exists():
            # Read new bytes outside the lock.
            batch_start = _file_offset
            with open(event_file, "rb") as f:
                f.seek(_file_offset)
                raw = f.read(16 * 1024 * 1024)   # up to 16 MB per tick

            if raw:
                # Find the last complete line — handles events larger than buffer.
                last_nl = raw.rfind(b"\n")
                if last_nl < 0:
                    complete_lines = []
                    consumed       = 0
                else:
                    complete_lines = [l for l in raw[:last_nl].split(b"\n") if l.strip()]
                    consumed       = last_nl + 1

                if complete_lines:
                    now = _time.time()
                    if _first_ts is None:
                        _first_ts = now
                    _last_ts       = now
                    _file_offset  += consumed
                    _file_size     = _file_offset

                    assert _lock is not None
                    async with _lock:
                        _events_since_index += len(complete_lines)
                        if not _sparse_index or _events_since_index >= INDEX_INTERVAL:
                            _sparse_index.append((now, batch_start))
                            _events_since_index = 0

                        # Update in-memory registry for mid-run client connects.
                        for bline in complete_lines:
                            try:
                                ev = json.loads(bline.decode("utf-8", errors="replace"))
                                t  = ev.get("type")
                                if t == "agent_registered":
                                    agn = ev.get("agname")
                                    if agn:
                                        _agent_registry[agn] = bline.decode("utf-8", errors="replace")
                                elif t == "team_registered":
                                    tn = ev.get("team_name")
                                    if tn:
                                        _team_registry[tn] = bline.decode("utf-8", errors="replace")
                            except Exception:
                                pass

                        dead: set[WebSocket] = set()
                        for bline in complete_lines:
                            line = bline.strip()
                            if not line:
                                continue
                            text = line.decode("utf-8", errors="replace")
                            for ws in list(_clients):
                                try:
                                    await ws.send_text(text)
                                except Exception:
                                    dead.add(ws)
                        _clients.difference_update(dead)

        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="agwebui standalone server")
    parser.add_argument("--run-dir", required=True, help="Directory containing ui_events.jsonl")
    parser.add_argument("--port",    type=int, default=7860)
    parsed = parser.parse_args()

    _run_dir   = Path(parsed.run_dir)
    _reply_dir = _run_dir / "ui_replies"
    _reply_dir.mkdir(parents=True, exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=parsed.port, log_level="error")
