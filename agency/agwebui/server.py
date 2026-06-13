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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_STATIC = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Mutable globals — set in __main__ before uvicorn starts
# ---------------------------------------------------------------------------

_run_dir:   Path = Path(".")
_reply_dir: Path = Path(".")

_all_events: list[str]    = []       # every event line seen so far (for replay)
_clients:    set[WebSocket] = set()
_lock:       asyncio.Lock | None = None   # created at startup


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    assert _lock is not None
    # Replay all historical events, then register for live updates.
    # Hold the lock so the tail task can't interleave new events between
    # the replay and the client being added to _clients.
    async with _lock:
        for line in _all_events:
            try:
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
                        (_reply_dir / f"{ask_id}.txt").write_text(
                            text, encoding="utf-8"
                        )
            except Exception:
                pass
    except WebSocketDisconnect:
        async with _lock:
            _clients.discard(ws)


# ---------------------------------------------------------------------------
# Tail loop
# ---------------------------------------------------------------------------

async def _tail_events() -> None:
    event_file = _run_dir / "ui_events.jsonl"
    position   = 0
    while True:
        if event_file.exists():
            with open(event_file, "r", encoding="utf-8") as f:
                f.seek(position)
                lines    = f.readlines()
                position = f.tell()
            if lines:
                assert _lock is not None
                async with _lock:
                    for raw in lines:
                        line = raw.strip()
                        if not line:
                            continue
                        _all_events.append(line)
                        dead: set[WebSocket] = set()
                        for ws in list(_clients):
                            try:
                                await ws.send_text(line)
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
