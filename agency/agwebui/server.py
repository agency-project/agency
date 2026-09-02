"""Standalone web server for agwebui.

No agency imports — this process is completely isolated from the execution
process.  It polls the global agency.sqlite3 event stream and reads selected agent databases on demand.
Run via:

    python -m agency.agwebui.server --run-dir <path> --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time as _time
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Seconds east of UTC for the server's local timezone (accounts for DST).
_TZ_OFFSET: int = -(
    _time.altzone if _time.daylight and _time.localtime().tm_isdst else _time.timezone
)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_STATIC = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Events replayed to new clients on connect. Matches
# agwebui_emitter._PRUNE_KEEP_RAW + _PRUNE_EVERY (500 + 500) -- the maximum
# possible size of the always-fully-raw tail that mechanism guarantees (see
# _PRUNE_KEEP_RAW's docstring), so a connecting client's replay window is
# never partially bucket-compacted mid-window.
TAIL_EVENTS = 1000
INDEX_INTERVAL = 1_000  # events between sample points in the timeline index

# ---------------------------------------------------------------------------
# Mutable globals — set in __main__ before uvicorn starts
# ---------------------------------------------------------------------------

_run_dir: Path = Path(".")
_reply_dir: Path = Path(".")
_command_dir: Path = Path(".")

# Highest event id seen so far; 0 means nothing read yet.
_last_event_id: int = 0
_event_count: int = 0
_first_ts: float | None = None
_last_ts: float | None = None

_clients: set[WebSocket] = set()
_lock: asyncio.Lock | None = None  # created at startup


# ---------------------------------------------------------------------------
# SQLite helpers (synchronous — called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically.

    Path.write_text() opens in truncate mode and then writes -- there is a
    real window between the truncate and the write completing where a
    concurrent reader (the execution process's _poll_commands() poll loop,
    or a test polling the same directory) can glob the file and read back
    an empty or partial string, raising JSONDecodeError. Writing to a
    sibling temp file and then os.replace()-ing it into place means readers
    only ever see the file fully absent or fully written, never in between
    -- os.replace() is atomic on both POSIX and Windows.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _db_path() -> Path:
    pointer = _run_dir / "global_data_path.txt"
    if pointer.exists():
        try:
            return Path(pointer.read_text(encoding="utf-8").strip())
        except Exception:  # noqa: S110 - fall back to the run-local database
            pass
    return _run_dir / "agency.sqlite3"


def _open_db(path: Path):
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _seed_from_db(path: Path) -> tuple[int, int, float | None, float | None]:
    """Read initial event-count/timestamp bookkeeping from an existing
    database. Global registration/resource recovery is handled by
    _fetch_state_preamble() on every connection, so no second in-memory
    registry must be rebuilt at startup.

    Returns (last_event_id, event_count, first_ts, last_ts).
    """
    if not path.exists():
        return 0, 0, None, None
    try:
        con = _open_db(path)
        row = con.execute("SELECT MAX(id), COUNT(*), MIN(ts), MAX(ts) FROM events").fetchone()
        con.close()
        if row and row[0] is not None:
            return row[0], row[1], row[2], row[3]
    except Exception as _e:
        print(f"[agwebui] WARNING: failed to read event summary from {path}: {_e}")
    return 0, 0, None, None


# Lightweight global projections read on every client connection. Detailed
# state, config, tokens, and history are fetched from the selected agent's
# database through /api/agents/{agname}.
_STATE_TABLES = ("agent_registry", "team_registry")


def _fetch_state_preamble(path: Path) -> list[str]:
    """Return global agent/team registration and resource state."""
    if not path.exists():
        return []
    rows: list[str] = []
    try:
        con = _open_db(path)
        for table in _STATE_TABLES:
            for (data,) in con.execute(f"SELECT data FROM {table}"):
                rows.append(data)
        row = con.execute("SELECT data FROM resource_state WHERE id=1").fetchone()
        if row:
            rows.append(row[0])
        con.close()
    except Exception as _e:
        print(f"[agwebui] WARNING: failed to read state preamble from {path}: {_e}")
    return rows


def _fetch_new_events(path: Path, after_id: int) -> list[tuple[int, str]]:
    """Return all (id, data) rows with id > after_id, ordered by id."""
    if not path.exists():
        return []
    try:
        con = _open_db(path)
        rows = con.execute(
            "SELECT id, data FROM events WHERE id > ? ORDER BY id", (after_id,)
        ).fetchall()
        con.close()
        return rows
    except Exception:
        return []


def _fetch_tail_events(path: Path, n: int = TAIL_EVENTS) -> list[str]:
    """Return the last n events (in chronological order) for new-client replay."""
    if not path.exists():
        return []
    try:
        con = _open_db(path)
        rows = con.execute(
            "SELECT data FROM (SELECT id, data FROM events ORDER BY id DESC LIMIT ?) ORDER BY id",
            (n,),
        ).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _fetch_timeline(path: Path) -> dict:
    """Build timeline metadata and sample points from the database."""
    if not path.exists():
        return {"index_len": 0, "first_ts": None, "last_ts": None, "samples": []}
    try:
        con = _open_db(path)
        row = con.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM events").fetchone()
        first_ts, last_ts, count = row if row else (None, None, 0)
        if not count:
            con.close()
            return {"index_len": 0, "first_ts": None, "last_ts": None, "samples": []}
        # Sample up to 500 evenly spaced points across the event stream.
        step = max(1, count // 500)
        raw_samples = con.execute(
            "SELECT ts FROM events WHERE (id % ?) = 1 ORDER BY id", (step,)
        ).fetchall()
        con.close()
        samples = [[i, r[0]] for i, r in enumerate(raw_samples)]
        return {
            "index_len": len(samples),
            "first_ts": first_ts,
            "last_ts": last_ts,
            "samples": samples,
        }
    except Exception:
        return {"index_len": 0, "first_ts": None, "last_ts": None, "samples": []}


def _fetch_events_range(path: Path, start_ts: float, end_ts: float) -> list[str]:
    """Return all event JSON strings with ts BETWEEN start_ts AND end_ts."""
    if not path.exists():
        return []
    try:
        con = _open_db(path)
        rows = con.execute(
            "SELECT data FROM events WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (start_ts, end_ts),
        ).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _fetch_agent_detail(global_path: Path, agname: str) -> dict:
    """Read detailed data only from the explicitly selected agent database."""
    if not global_path.exists():
        return {"error": "global database is not available", "agname": agname}
    global_con: "sqlite3.Connection | None" = None
    try:
        global_con = _open_readonly(global_path)
        row = global_con.execute(
            "SELECT db_path FROM agent_registry WHERE agname=?", (agname,)
        ).fetchone()
    except Exception as exc:
        return {"error": f"agent catalog lookup failed: {exc}", "agname": agname}
    finally:
        if global_con is not None:
            global_con.close()
    if row is None:
        return {"error": "unknown agent", "agname": agname}

    agent_path = Path(row[0])
    if not agent_path.exists():
        return {"error": "agent database is not available", "agname": agname}
    con: "sqlite3.Connection | None" = None
    try:
        con = _open_readonly(agent_path)
        latest = {
            event_type: _json_object(payload)
            for event_type, payload in con.execute(
                "SELECT type,payload FROM latest_values "
                "WHERE type IN ('live_messages','agent_state','agent_config','token_update')"
            )
        }
        if "live_messages" not in latest:
            row = con.execute(
                "SELECT payload FROM events WHERE type='skill_call' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                skill_call = _json_object(row[0])
                latest["live_messages"] = {
                    "messages": skill_call.get("history_delta", [])
                }
        event_rows = con.execute(
            "SELECT type,timestamp,call_label,payload FROM events "
            "ORDER BY id DESC LIMIT 500"
        ).fetchall()
        span_rows = con.execute(
            "SELECT name,start_ts,end_ts,attributes FROM spans "
            "ORDER BY id DESC LIMIT 500"
        ).fetchall()
    except Exception as exc:
        return {"error": f"agent database read failed: {exc}", "agname": agname}
    finally:
        if con is not None:
            con.close()

    return {
        "agname": agname,
        "messages": latest.get("live_messages", {}).get("messages", []),
        "state": latest.get("agent_state", {}),
        "config": latest.get("agent_config", {}),
        "tokens": latest.get("token_update", {}),
        "events": [
            {
                "type": event_type,
                "timestamp": timestamp,
                "call_label": call_label,
                "payload": _json_object(payload),
            }
            for event_type, timestamp, call_label, payload in reversed(event_rows)
        ],
        "spans": [
            {
                "name": name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "attributes": _json_object(attributes),
            }
            for name, start_ts, end_ts, attributes in reversed(span_rows)
        ],
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _lock, _last_event_id, _event_count, _first_ts, _last_ts
    _lock = asyncio.Lock()
    # Seed event-count/timestamp bookkeeping from an existing database (e.g.
    # server restart mid-run). Registration/state recovery no longer needs
    # seeding here -- _fetch_state_preamble() reads the durable state tables
    # fresh on every connect regardless of server restarts.
    seed = await asyncio.to_thread(_seed_from_db, _db_path())
    _last_event_id, _event_count, _first_ts, _last_ts = seed
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
    """Return sample points and metadata for the timeline scrubber."""
    tl = await asyncio.to_thread(_fetch_timeline, _db_path())
    return JSONResponse(tl)


@app.get("/api/events")
async def api_events(start_ts: float = 0.0, end_ts: float = 0.0):
    """Return all events with ts BETWEEN start_ts AND end_ts."""
    if end_ts <= start_ts:
        return JSONResponse({"events": [], "from_ts": start_ts, "to_ts": end_ts})
    events = await asyncio.to_thread(_fetch_events_range, _db_path(), start_ts, end_ts)
    return JSONResponse({"events": events, "from_ts": start_ts, "to_ts": end_ts})


@app.get("/api/agents/{agname}")
async def api_agent_detail(agname: str):
    detail = await asyncio.to_thread(_fetch_agent_detail, _db_path(), agname)
    status = 404 if detail.get("error") == "unknown agent" else 200
    return JSONResponse(detail, status_code=status)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    assert _lock is not None

    # Fetch tail and state outside lock — read-only DB queries.
    tail_lines = await asyncio.to_thread(_fetch_tail_events, _db_path())
    state_preamble = await asyncio.to_thread(_fetch_state_preamble, _db_path())

    async with _lock:
        sync = json.dumps(
            {
                "type": "timeline_sync",
                "index_len": max(0, _event_count // INDEX_INTERVAL),
                "first_ts": _first_ts,
                "last_ts": _last_ts,
                "tz_offset": _TZ_OFFSET,
            }
        )
        try:
            await ws.send_text(sync)
            for line in tail_lines:
                await ws.send_text(line)
            # Global registration/resource state follows the tail. Detailed
            # agent data is loaded only when the client selects that agent.
            for line in state_preamble:
                await ws.send_text(line)
        except Exception:
            return
        _clients.add(ws)

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                mtype = msg.get("type")
                if mtype == "human_reply":
                    ask_id = str(msg.get("ask_id", ""))
                    text = str(msg.get("text", ""))
                    if ask_id:
                        _atomic_write_text(_reply_dir / f"{ask_id}.txt", text)
                elif mtype in ("pause", "resume", "pause_all", "resume_all"):
                    cmd = {"type": mtype, "agname": msg.get("agname")}
                    cmd_file = _command_dir / f"{_uuid.uuid4().hex}.json"
                    _atomic_write_text(cmd_file, json.dumps(cmd))
                elif mtype in ("update_config", "update_config_all"):
                    cmd = {
                        "type": mtype,
                        "agname": msg.get("agname"),
                        "config": msg.get("config") or {},
                    }
                    cmd_file = _command_dir / f"{_uuid.uuid4().hex}.json"
                    _atomic_write_text(cmd_file, json.dumps(cmd))
            except Exception as _e:
                # Reference the raw `data`, not `msg` -- json.loads(data) itself
                # may be what raised, in which case `msg` was never assigned.
                print(f"[agwebui] WARNING: failed to handle client message {data!r}: {_e}")
    except WebSocketDisconnect:
        async with _lock:
            _clients.discard(ws)


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------


async def _tail_events() -> None:
    global _last_event_id, _event_count, _first_ts, _last_ts

    while True:
        path = _db_path()
        if path.exists():
            rows = await asyncio.to_thread(_fetch_new_events, path, _last_event_id)
            if rows:
                now = _time.time()
                if _first_ts is None:
                    _first_ts = now
                _last_ts = now

                assert _lock is not None
                async with _lock:
                    for event_id, _ in rows:
                        _last_event_id = event_id
                        _event_count += 1

                    dead: set[WebSocket] = set()
                    for _, data in rows:
                        for ws in list(_clients):
                            try:
                                await ws.send_text(data)
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
    parser.add_argument("--run-dir", required=True, help="Directory containing agency.sqlite3")
    parser.add_argument("--port", type=int, default=7860)
    parsed = parser.parse_args()

    _run_dir = Path(parsed.run_dir)
    _reply_dir = _run_dir / "ui_replies"
    _command_dir = _run_dir / "ui_commands"
    _reply_dir.mkdir(parents=True, exist_ok=True)
    _command_dir.mkdir(parents=True, exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=parsed.port, log_level="error")
