"""Standalone web server for agwebui.

No agency imports — this process is completely isolated from the execution
process. It polls the orchestrator's global_data.sqlite3 event stream
(agDataLogger's schema, written by the live execution process) directly and
reads selected agents' own databases on demand. Run via:

    python -m agency.observability.agwebui.server --run-dir <path> --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import threading
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

# Events replayed to new clients on connect.
TAIL_EVENTS = 1000
INDEX_INTERVAL = 1_000  # events between sample points in the timeline index

# ---------------------------------------------------------------------------
# Mutable globals — set in __main__ before uvicorn starts
# ---------------------------------------------------------------------------

_run_dir: Path = Path(".")
_command_dir: Path = Path(".")

# Highest event id seen so far -- ids are zero-padded-sequence + uuid4 hex
# strings (agDataLogger._next_id_locked), sortable as TEXT; "" sorts before
# every real id.
_last_event_id: str = ""
_event_count: int = 0
_first_ts: float | None = None
_last_ts: float | None = None

_clients: set[WebSocket] = set()
_lock: asyncio.Lock | None = None  # created at startup

# Per-agent tail cursors for the shared-log relay (see _tail_and_broadcast) --
# agname -> last seen event id in that agent's own db.
_agent_log_cursors: "dict[str, str]" = {}
# Single cadence for both the global db and every known agent's own db --
# each tick's rows from all sources are merged and sorted by timestamp
# before broadcasting, so a scheduler event and an agent event that
# happened close together in real time arrive at the client in that same
# order (rather than each source polling and broadcasting independently,
# which could reorder them at the client).
TAIL_POLL_INTERVAL = 0.3
AGENT_LOG_BACKLOG_PER_AGENT = 200


# ---------------------------------------------------------------------------
# SQLite helpers (synchronous — called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically: write to a sibling temp file,
    then `os.replace()` it into place, so a concurrent reader never sees
    a truncated or partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _db_path() -> Path:
    """The orchestrator's global agDataLogger's db path, duplicated inline
    (not imported) since this process stays free of agency package imports."""
    return _run_dir / "global_data.sqlite3"


# agprof.trace.json/summary.json/summary.md/profile_data.sqlite3 land in a
# `profiler/` directory that is a *sibling* of `_run_dir` (both are
# `agency_runs/<run_id>/{logs,profiler}` -- see agprof._env_out_dir() and
# agutil._DEFAULT_LOG_DIR, which share the same `agency_run_dir_name()`).
# Only exists once the profiling session has stopped (session()/workload()
# write output at exit, not continuously), so a run still in progress has
# nothing here yet -- that's normal, not an error.
_PROFILER_FILENAMES = ("summary.md", "summary.json", "agprof.trace.json", "profile_data.sqlite3")


def _profiler_dir() -> Path:
    override = os.environ.get("AGENCY_PROFILE_DIR")
    return Path(override) if override else _run_dir.parent / "profiler"


def _list_profiler_files() -> dict:
    directory = _profiler_dir()
    files = []
    for name in _PROFILER_FILENAMES:
        path = directory / name
        if path.is_file():
            files.append({"name": name, "size": path.stat().st_size})
    return {"files": files}


def _xterm256_hex(n: int) -> str:
    """xterm-256 color index -> hex."""
    if n < 16:
        _ansi16 = [
            "#000000", "#aa0000", "#00aa00", "#aa8800",
            "#0000aa", "#aa00aa", "#00aaaa", "#aaaaaa",
            "#555555", "#ff5555", "#55ff55", "#ffff55",
            "#5555ff", "#ff55ff", "#55ffff", "#ffffff",
        ]  # fmt: skip
        return _ansi16[n]
    if n < 232:
        idx = n - 16

        def _c(lvl: int) -> int:
            return 0 if lvl == 0 else 55 + 40 * lvl

        return f"#{_c(idx // 36):02x}{_c((idx // 6) % 6):02x}{_c(idx % 6):02x}"
    v = 8 + (n - 232) * 10
    return f"#{v:02x}{v:02x}{v:02x}"


def _make_color_palette() -> "list[str]":
    """Sample the 6x6x6 xterm-256 cube at component levels {0,2,4,5}
    (values 0/135/215/255), drop the cube-greys (r==g==b) and the
    near-black corner (brightest component <= 135) -- 54 clearly visible,
    well-spread colors -- then shuffle once per process, so colors stay
    varied and stable for the life of one webui server run."""
    levels = (0, 2, 4, 5)
    indices: "list[int]" = []
    for r in levels:
        for g in levels:
            for b in levels:
                if r == g == b:
                    continue
                if max(r, g, b) <= 2:
                    continue
                indices.append(16 + 36 * r + 6 * g + b)
    colors = [_xterm256_hex(idx) for idx in indices]
    random.shuffle(colors)
    return colors


_AGENT_COLOR_PALETTE: "list[str]" = _make_color_palette()
_agent_color_lock = threading.Lock()
_agent_color_assignments: "dict[str, str]" = {}
_agent_color_next = 0


def _agent_color(agname: str) -> str:
    """Assign each agent the next unused palette color the first time its
    name is seen, then remember it -- round-robin in order of first
    appearance, via one counter incremented per agent, indexing into the
    shuffled palette."""
    global _agent_color_next
    with _agent_color_lock:
        color = _agent_color_assignments.get(agname)
        if color is None:
            color = _AGENT_COLOR_PALETTE[_agent_color_next % len(_AGENT_COLOR_PALETTE)]
            _agent_color_next += 1
            _agent_color_assignments[agname] = color
        return color


# The orchestrator's own events (global db) carry no term_message -- unlike
# agent-side events, nothing ever printed them to a terminal, so there was
# never a human-readable line to persist. Synthesized here, purely from
# already-persisted payload fields, so "global orchestrator actions, the
# scheduling events" show up on the shared log the same way agent-side
# activity does, with zero changes to orchestrator.py/agteam.py itself.
# scheduler_state deliberately excluded: it's a continuous snapshot (fires
# after nearly every transition), not a discrete action -- logging it would
# flood the shared log with near-duplicate resource-count dumps.
#
# All scheduling actions are tagged "[scheduler]" (the source of the
# action), not the agent name -- request_* events are about an agent but
# come from the orchestrator, so the agent name rides in the message text
# instead, next to the request_id-scoped skill.
_SCHEDULER_EVENT_TYPES = frozenset(
    {
        "scheduler_started",
        "scheduler_stopped",
        "request_submitted",
        "request_blocked",
        "request_ready",
        "request_started",
        "request_completed",
        "request_failed",
        "request_cancelled",
        "request_destroyed",
    }
)
# Fixed, not palette-assigned: "scheduler" isn't an agent identity, and
# colors are reserved for real agent tags (see agterm.py's _EVENT_STYLES,
# which kept event-tag styling separate from per-agent colors the same way).
_SCHEDULER_TAG_COLOR = "#ffffff"


def _synthesize_term_message(event_type: str, name: "str | None", payload: dict) -> "str | None":
    skill = payload.get("skill")
    if event_type == "scheduler_started":
        return (
            f"[scheduler] STARTED  max_concurrent_engines={payload.get('max_concurrent_engines')}"
        )
    if event_type == "scheduler_stopped":
        return "[scheduler] STOPPED"
    if event_type == "request_submitted":
        return f"[scheduler] {name}  REQUEST ▶  submitted   {skill}"
    if event_type == "request_blocked":
        return f"[scheduler] {name}  REQUEST ⏸  blocked     {skill}"
    if event_type == "request_ready":
        return f"[scheduler] {name}  REQUEST ▶  ready       {skill}"
    if event_type == "request_started":
        return f"[scheduler] {name}  REQUEST ▶  started     {skill}"
    if event_type == "request_completed":
        return f"[scheduler] {name}  REQUEST ✓  completed   {skill}"
    if event_type == "request_failed":
        return f"[scheduler] {name}  REQUEST ✗  failed      {skill}"
    if event_type == "request_cancelled":
        return f"[scheduler] {name}  REQUEST ⊘  cancelled   {skill}"
    if event_type == "request_destroyed":
        return f"[scheduler] {name}  REQUEST ⊘  destroyed   {skill}"
    if event_type == "team_registered":
        return f"[{name}] TEAM REGISTERED  agents={payload.get('agents')}"
    return None


_TERM_MESSAGE_DISPLAY_MAX_CHARS = 300


def _truncate_for_display(term_message: str) -> str:
    """Trim a term_message for the browser only -- the terminal print and
    the term_message column stored in the agent's own db both keep the full
    text; only what ships to the shared webui log gets shortened here."""
    if len(term_message) <= _TERM_MESSAGE_DISPLAY_MAX_CHARS:
        return term_message
    return f"{term_message[:_TERM_MESSAGE_DISPLAY_MAX_CHARS]}…"


def _build_envelope(
    event_type: str,
    timestamp: float,
    name: "str | None",
    payload_json: str,
    term_message: "str | None" = None,
) -> str:
    """Reconstruct the flat JSON envelope the client expects (`{"type":...,
    "ts":..., "agname":..., ...payload fields}`) from one `events`/
    `latest_values` row. Payload fields are spread first so the canonical
    type/ts/agname always win over any (currently nonexistent) same-named
    payload field. `term_message` (the same human-readable line agDataLogger
    also prints to stderr, e.g. "[agent] SKILL OK ...") rides along when the
    row has one, so the client's shared log can show it without a dedicated
    `type: "log"` event -- nothing emits those anymore; falls back to
    _synthesize_term_message() for global/orchestrator event types that
    never had one to begin with. `color` rides along whenever the row is
    identified by name (agent_registered, or any term_message-bearing line)
    so the client can colorize that tag consistently in both the roster and
    the shared log, agterm-style -- except scheduling actions, which are
    tagged "[scheduler]" (a fixed color, not a per-agent one)."""
    payload = _json_object(payload_json)
    envelope = {**payload, "type": event_type, "ts": timestamp, "agname": name}
    if term_message is None:
        term_message = _synthesize_term_message(event_type, name, payload)
    if event_type in _SCHEDULER_EVENT_TYPES:
        envelope["color"] = _SCHEDULER_TAG_COLOR
    elif name and (event_type == "agent_registered" or term_message):
        envelope["color"] = _agent_color(name)
    if term_message:
        envelope["term_message"] = _truncate_for_display(term_message)
    return json.dumps(envelope)


def _open_db(path: Path):
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _seed_from_db(path: Path) -> tuple[str, int, float | None, float | None]:
    """Read initial event-count/timestamp bookkeeping from an existing
    database. Global registration/resource recovery is handled by
    _fetch_state_preamble() on every connection, so no second in-memory
    registry must be rebuilt at startup.

    Returns (last_event_id, event_count, first_ts, last_ts).
    """
    if not path.exists():
        return "", 0, None, None
    try:
        con = _open_db(path)
        row = con.execute(
            "SELECT MAX(id), COUNT(*), MIN(timestamp), MAX(timestamp) FROM events"
        ).fetchone()
        con.close()
        if row and row[0] is not None:
            return row[0], row[1], row[2], row[3]
    except Exception as _e:
        print(f"[agwebui] WARNING: failed to read event summary from {path}: {_e}")
    return "", 0, None, None


# Lightweight global projections read on every client connection. Detailed
# state, config, tokens, and history are fetched from the selected agent's
# database through /api/agents/{agname}.
_PREAMBLE_TYPES = ("agent_registered", "team_registered", "resource_update")


def _fetch_state_preamble(path: Path) -> list[str]:
    """Return global agent/team registration and resource state -- the
    latest row of each, one per (type, name) key in `latest_values`."""
    if not path.exists():
        return []
    rows: list[str] = []
    try:
        con = _open_db(path)
        for event_type, timestamp, name, payload, term_message in con.execute(
            "SELECT type, timestamp, name, payload, term_message FROM latest_values "
            "WHERE type IN (?,?,?)",
            _PREAMBLE_TYPES,
        ):
            rows.append(_build_envelope(event_type, timestamp, name, payload, term_message))
        con.close()
    except Exception as _e:
        print(f"[agwebui] WARNING: failed to read state preamble from {path}: {_e}")
    return rows


def _known_agents(global_path: Path) -> "dict[str, str]":
    """agname -> db_path for every currently-registered agent."""
    if not global_path.exists():
        return {}
    try:
        con = _open_db(global_path)
        rows = con.execute(
            "SELECT name, payload FROM latest_values WHERE type='agent_registered'"
        ).fetchall()
        con.close()
    except Exception:
        return {}
    agents: "dict[str, str]" = {}
    for name, payload in rows:
        db_path = _json_object(payload).get("db_path")
        if name and db_path:
            agents[name] = db_path
    return agents


def _fetch_new_term_messages(path: Path, after_id: str) -> list[tuple[str, str]]:
    """Return (id, envelope_json) for rows with id > after_id that carry a
    term_message -- the compact human-readable status lines
    (agent_created/skill_start/skill_success/...), not the far more
    numerous llm_block/event rows the shared log has no use for.

    `agent_state` rows are relayed too, term_message or not: they're what
    corrects a client's displayed agent state (queued/running_skill/
    running_harness/agent_idle) after the coarser scheduler-level
    request_completed ("finished") fires. request_completed reaches the
    client via the global db's unfiltered _fetch_new_events, but the true
    agent_idle that immediately follows it (orchestrator._update_agent_
    display_locked) lives in this per-agent db -- without relaying it too,
    a client never learns the agent went idle (or started a new request)
    and the display sticks on "finished" forever. record_state() never
    sets a term_message (every state transition would otherwise flood the
    shared log), so these rows would otherwise never pass the term_message
    filter below; _build_envelope's synthesis has no case for agent_state
    either, so they still carry no term_message and never touch the log."""
    if not path.exists():
        return []
    try:
        con = _open_db(path)
        rows = con.execute(
            "SELECT id, type, timestamp, name, payload, term_message FROM events "
            "WHERE id > ? AND (term_message IS NOT NULL OR type = 'agent_state') ORDER BY id",
            (after_id,),
        ).fetchall()
        con.close()
        return [
            (event_id, _build_envelope(event_type, timestamp, name, payload, term_message))
            for event_id, event_type, timestamp, name, payload, term_message in rows
        ]
    except Exception:
        return []


def _fetch_agent_log_backlog(
    global_path: Path, limit_per_agent: int = AGENT_LOG_BACKLOG_PER_AGENT
) -> list[str]:
    """Replay each currently-registered agent's own term_message history so
    a newly-connecting client's shared log isn't empty until new activity
    happens. Bounded per agent -- this is short status-line text, not
    message content, but the cap keeps a very long-lived agent's replay
    from growing unbounded. Not globally time-sorted across agents (each
    agent's own lines stay chronological); a best-effort shared log, not a
    strictly merged one."""
    lines: list[str] = []
    for agname, db_path in _known_agents(global_path).items():
        path = Path(db_path)
        if not path.exists():
            continue
        try:
            con = _open_db(path)
            rows = con.execute(
                "SELECT type, timestamp, name, payload, term_message FROM events "
                "WHERE term_message IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit_per_agent,),
            ).fetchall()
            con.close()
        except Exception as _e:
            print(f"[agwebui] WARNING: failed to read log backlog for {agname}: {_e}")
            continue
        lines.extend(
            _build_envelope(event_type, timestamp, name, payload, term_message)
            for event_type, timestamp, name, payload, term_message in reversed(rows)
        )
    return lines


def _fetch_new_events(path: Path, after_id: str) -> list[tuple[str, str]]:
    """Return all (id, envelope_json) rows with id > after_id, ordered by id."""
    if not path.exists():
        return []
    try:
        con = _open_db(path)
        rows = con.execute(
            "SELECT id, type, timestamp, name, payload, term_message FROM events "
            "WHERE id > ? ORDER BY id",
            (after_id,),
        ).fetchall()
        con.close()
        return [
            (event_id, _build_envelope(event_type, timestamp, name, payload, term_message))
            for event_id, event_type, timestamp, name, payload, term_message in rows
        ]
    except Exception:
        return []


def _fetch_tail_events(path: Path, n: int = TAIL_EVENTS) -> list[str]:
    """Return the last n events (in chronological order) for new-client replay."""
    if not path.exists():
        return []
    try:
        con = _open_db(path)
        rows = con.execute(
            "SELECT type, timestamp, name, payload, term_message FROM "
            "(SELECT id, type, timestamp, name, payload, term_message FROM events "
            "ORDER BY id DESC LIMIT ?) ORDER BY id",
            (n,),
        ).fetchall()
        con.close()
        return [
            _build_envelope(event_type, timestamp, name, payload, term_message)
            for event_type, timestamp, name, payload, term_message in rows
        ]
    except Exception:
        return []


def _fetch_timeline(path: Path) -> dict:
    """Build timeline metadata and sample points from the database."""
    if not path.exists():
        return {"index_len": 0, "first_ts": None, "last_ts": None, "samples": []}
    try:
        con = _open_db(path)
        row = con.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM events").fetchone()
        first_ts, last_ts, count = row if row else (None, None, 0)
        if not count:
            con.close()
            return {"index_len": 0, "first_ts": None, "last_ts": None, "samples": []}
        # Sample up to 500 evenly spaced points across the event stream. `id`
        # is a TEXT sequence+uuid string, not usable with `%`, but the table's
        # implicit rowid still increases in insertion (== id) order.
        step = max(1, count // 500)
        raw_samples = con.execute(
            "SELECT timestamp FROM events WHERE (rowid % ?) = 1 ORDER BY rowid", (step,)
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
    """Return all event envelopes with timestamp BETWEEN start_ts AND end_ts."""
    if not path.exists():
        return []
    try:
        con = _open_db(path)
        rows = con.execute(
            "SELECT type, timestamp, name, payload, term_message FROM events "
            "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (start_ts, end_ts),
        ).fetchall()
        con.close()
        return [
            _build_envelope(event_type, timestamp, name, payload, term_message)
            for event_type, timestamp, name, payload, term_message in rows
        ]
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


# record_final_transcript() writes one `events` row per block (not one row per
# exchange holding a blocks array) -- so the metadata block for an exchange
# is its own row, type='llm_block', with its own $.type=='metadata' inside
# payload; new_prompt_tokens is top-level (llm_handler_server._tag_metadata_
# block) but completion_tokens stays nested under $.usage (anthropic.py/
# openai.py's usage_dict). Only those two numbers are ever pulled out here;
# the far more numerous text/tool_use/tool_result rows are filtered out by
# SQLite itself and never reach Python, since nothing in the frontend
# renders raw events anyway (only messages/state/config/tokens are).
_TOKEN_TOTALS_SQL = (
    "SELECT SUM(json_extract(payload, '$.new_prompt_tokens')), "
    "SUM(json_extract(payload, '$.usage.completion_tokens')), "
    "SUM(json_extract(payload, '$.usage.cache_read_tokens')), "
    "SUM(json_extract(payload, '$.usage.cache_write_tokens')) "
    "FROM events WHERE type = 'llm_block' AND json_extract(payload, '$.type') = 'metadata'"
)


def _compute_agent_messages(con: sqlite3.Connection) -> "list[dict]":
    """The full current transcript for one agent's db: llm_block events
    plus whatever exchange is still streaming, shared by both HTTP pull
    and push paths. Ignores orchestrator.py's live_messages snapshots (no
    per-message clock) and host_interaction_server's tool_result events
    (a duplicate record of what llm_block already covers)."""
    finalized = con.execute(
        "SELECT type, call_label, payload, timestamp FROM events "
        "WHERE type = 'llm_block' ORDER BY id"
    ).fetchall()
    streaming_rows = con.execute(
        "SELECT call_label, payload, timestamp FROM stream_deltas "
        "WHERE type='llm_stream_delta' ORDER BY id"
    ).fetchall()
    return _reconstruct_in_progress_messages(finalized) + _reconstruct_streaming_messages(
        streaming_rows
    )


def _fetch_agent_detail(global_path: Path, agname: str) -> dict:
    """Read detailed data only from the explicitly selected agent database."""
    if not global_path.exists():
        return {"error": "global database is not available", "agname": agname}
    global_con: "sqlite3.Connection | None" = None
    try:
        global_con = _open_readonly(global_path)
        row = global_con.execute(
            "SELECT payload FROM latest_values WHERE type='agent_registered' AND name=?",
            (agname,),
        ).fetchone()
    except Exception as exc:
        return {"error": f"agent catalog lookup failed: {exc}", "agname": agname}
    finally:
        if global_con is not None:
            global_con.close()
    db_path = _json_object(row[0]).get("db_path") if row is not None else None
    if not db_path:
        return {"error": "unknown agent", "agname": agname}

    agent_path = Path(db_path)
    if not agent_path.exists():
        return {"error": "agent database is not available", "agname": agname}
    con: "sqlite3.Connection | None" = None
    try:
        con = _open_readonly(agent_path)
        latest = {
            event_type: _json_object(payload)
            for event_type, payload in con.execute(
                "SELECT type,payload FROM latest_values WHERE type IN ('agent_state','agent_config')"
            )
        }
        messages = _compute_agent_messages(con)
        token_row = con.execute(_TOKEN_TOTALS_SQL).fetchone()
    except Exception as exc:
        return {"error": f"agent database read failed: {exc}", "agname": agname}
    finally:
        if con is not None:
            con.close()

    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens = (
        token_row if token_row is not None else (None, None, None, None)
    )
    return {
        "agname": agname,
        "messages": messages,
        "state": latest.get("agent_state", {}),
        "config": latest.get("agent_config", {}),
        "tokens": {
            "input": input_tokens or 0,
            "output": output_tokens or 0,
            "cache_read": cache_read_tokens or 0,
            "cache_write": cache_write_tokens or 0,
        },
    }


def _reconstruct_in_progress_messages(rows: "list[tuple[str, str, str, float]]") -> "list[dict]":
    """Rebuild one agent's transcript from every llm_block row on record
    (see _compute_agent_messages), each keeping its own real timestamp so
    the panel advances live instead of only at completion. Groups
    consecutive rows sharing one (call_label, role) into a single
    message, same as a real transcript alternating turns."""
    messages: "list[dict]" = []
    current_key: "object" = object()  # sentinel, never equals a real (call_label, role)
    current_blocks: "list[dict] | None" = None
    for _event_type, call_label, payload_json, ts in rows:
        payload = _json_object(payload_json)
        role = payload.pop("role", "assistant")
        key = (call_label, role)
        if key != current_key or current_blocks is None:
            current_blocks = []
            messages.append({"role": role, "blocks": current_blocks, "ts": ts})
            current_key = key
        current_blocks.append(payload)
    return messages


# Field-by-field accumulation rules for one streamed block, ported from
# llm_handler_server._run_stream_producer's own in-memory merge loop (the
# same block_delta stream_items, just read back from stream_deltas instead
# of consumed live) -- kept in exact lockstep with that loop; if it changes
# how a field accumulates, mirror the change here too.
def _merge_stream_item(block: dict, stream_item: dict) -> None:
    text_piece = stream_item.get("text") or ""
    if text_piece:
        block["text"] += text_piece
    citations_piece = stream_item.get("citations")
    if citations_piece:
        if block.get("citations") is None:
            block["citations"] = []
        block["citations"].extend(citations_piece)
    sig_piece = stream_item.get("signature") or ""
    if sig_piece:
        block["signature"] = block.get("signature", "") + sig_piece
    if stream_item.get("id"):
        block["id"] = stream_item["id"]
    name_piece = stream_item.get("name") or ""
    if name_piece:
        block["name"] = block.get("name", "") + name_piece
    args_piece = stream_item.get("arguments") or ""
    if args_piece:
        block["arguments"] = block.get("arguments", "") + args_piece


def _reconstruct_streaming_messages(rows: "list[tuple[str, str, float]]") -> "list[dict]":
    """The exchange (if any) that's still streaming right now -- not yet
    finalized into a permanent events row (record_final_transcript() only runs once
    the whole exchange completes), so without this the panel would freeze
    for however long that one exchange takes (can be several real seconds)
    even though the raw deltas are already landing on disk continuously.
    Replicates llm_handler_server's own text/tool_use/thinking block merge
    (see _merge_stream_item) purely on read; metadata blocks and bare
    "usage" stream_items are dropped, same as the finalized-event path."""
    by_call_label: "dict[object, dict[int, dict]]" = {}
    order: "list[object]" = []
    first_ts: "dict[object, float]" = {}
    for call_label, payload_json, ts in rows:
        stream_item = _json_object(payload_json)
        if stream_item.get("type") != "block_delta":
            continue
        block_type = stream_item.get("block_type")
        if block_type == "metadata":
            continue
        blocks = by_call_label.setdefault(call_label, {})
        if call_label not in order:
            order.append(call_label)
            first_ts[call_label] = ts
        idx = stream_item.get("index")
        block = blocks.setdefault(idx, {"type": block_type, "index": idx, "text": ""})
        _merge_stream_item(block, stream_item)
    return [
        {
            "role": "assistant",
            "blocks": [by_call_label[label][i] for i in sorted(by_call_label[label])],
            "ts": first_ts[label],
        }
        for label in order
        if by_call_label[label]
    ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _lock, _last_event_id, _event_count, _first_ts, _last_ts
    _lock = asyncio.Lock()
    # Seed event-count/timestamp bookkeeping from an existing database (e.g.
    # server restart mid-run). Registration/state recovery needs no seeding
    # here -- _fetch_state_preamble() reads the durable state tables fresh
    # on every connect regardless of server restarts.
    seed = await asyncio.to_thread(_seed_from_db, _db_path())
    _last_event_id, _event_count, _first_ts, _last_ts = seed
    task = asyncio.create_task(_tail_and_broadcast())
    yield
    task.cancel()


app = FastAPI(lifespan=_lifespan)
app.mount(
    "/perfetto",
    StaticFiles(directory=str(_STATIC / "perfetto"), html=True, check_dir=False),
    name="perfetto",
)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Browsers otherwise happily cache app.js/style.css/index.html across
    reloads with no revalidation -- painful during active iteration on this
    file, since a plain refresh can silently keep serving stale JS/CSS with
    no visible sign anything is wrong."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html")


def _profile_trace_path() -> Path:
    return _profiler_dir() / "agprof.trace.json"


@app.get("/api/profiler")
async def profiler_status():
    trace = _profile_trace_path()
    return {
        "viewer_available": (_STATIC / "perfetto" / "index.html").is_file(),
        "trace_available": trace.is_file(),
    }


@app.get("/api/profiler/trace")
async def profiler_trace():
    trace = _profile_trace_path()
    if not trace.is_file():
        return JSONResponse(
            {"error": "No completed profiler trace for this run yet."}, status_code=404
        )
    return FileResponse(trace, media_type="application/json", headers={"Cache-Control": "no-store"})


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
# Profiler artifacts
# ---------------------------------------------------------------------------


@app.get("/api/profiler/files")
async def api_profiler_files():
    """Which profiler output files exist yet, if any -- empty while the run
    is still in progress (see _profiler_dir()'s docstring)."""
    return JSONResponse(await asyncio.to_thread(_list_profiler_files))


@app.get("/api/profiler/download/{filename}")
async def api_profiler_download(filename: str):
    # Whitelist, not just a directory-scoped read: filename comes straight
    # from the URL path, so this must never resolve outside _profiler_dir()
    # (no "..", no absolute path, no symlink surprise) -- an exact match
    # against the fixed set of names agprof actually writes closes all of
    # that off at once, no path-sanitizing logic needed.
    if filename not in _PROFILER_FILENAMES:
        return JSONResponse({"error": "unknown profiler file"}, status_code=404)
    path = _profiler_dir() / filename
    if not path.is_file():
        return JSONResponse({"error": "not available yet"}, status_code=404)
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


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
    agent_log_backlog = await asyncio.to_thread(_fetch_agent_log_backlog, _db_path())

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
            # Per-agent shared-log backlog (see _fetch_agent_log_backlog).
            for line in agent_log_backlog:
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
                if mtype in ("pause", "resume", "pause_all", "resume_all"):
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


async def _tail_and_broadcast() -> None:
    """One merged poll loop for both the global db and every registered
    agent's own db (the latter relays term_message lines -- SKILL start/
    success/error, CREATED, ... -- into the same shared broadcast stream,
    since the global db by itself carries almost none of these).

    Each tick collects new rows from every source into one batch and sorts
    that batch by timestamp before broadcasting, so a scheduler event and
    an agent event that happened close together in real time always arrive
    at the client in that same order."""
    global _last_event_id, _event_count, _first_ts, _last_ts

    while True:
        batch: "list[tuple[float, str]]" = []  # (ts, envelope_json), unsorted

        global_path = _db_path()
        if global_path.exists():
            global_rows = await asyncio.to_thread(_fetch_new_events, global_path, _last_event_id)
            if global_rows:
                now = _time.time()
                if _first_ts is None:
                    _first_ts = now
                _last_ts = now
                for event_id, data in global_rows:
                    _last_event_id = event_id
                    _event_count += 1
                    batch.append((json.loads(data)["ts"], data))

        agents = await asyncio.to_thread(_known_agents, global_path)
        for agname, agent_db_path in agents.items():
            after_id = _agent_log_cursors.get(agname, "")
            rows = await asyncio.to_thread(_fetch_new_term_messages, Path(agent_db_path), after_id)
            for event_id, data in rows:
                _agent_log_cursors[agname] = event_id
                batch.append((json.loads(data)["ts"], data))

        if batch:
            batch.sort(key=lambda item: item[0])
            assert _lock is not None
            async with _lock:
                dead: set[WebSocket] = set()
                for _, data in batch:
                    for ws in list(_clients):
                        try:
                            await ws.send_text(data)
                        except Exception:
                            dead.add(ws)
                _clients.difference_update(dead)

        await asyncio.sleep(TAIL_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    if __package__:
        from .build_perfetto import ensure_viewer
    else:
        from build_perfetto import ensure_viewer

    parser = argparse.ArgumentParser(description="agwebui standalone server")
    parser.add_argument(
        "--run-dir", required=True, help="log_dir shared with the execution process's agents"
    )
    parser.add_argument("--port", type=int, default=7860)
    parsed = parser.parse_args()

    ensure_viewer()

    _run_dir = Path(parsed.run_dir)
    _command_dir = _run_dir / "ui_commands"
    _command_dir.mkdir(parents=True, exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=parsed.port, log_level="error")
