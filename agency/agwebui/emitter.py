"""agwebui_emitter — writes structured UI events to a SQLite database.

The execution process calls these methods; the standalone web server polls
the database and pushes events to connected browsers.  No agency imports here
so this module can be imported from both sides if needed.
"""

from __future__ import annotations

import json
import re as _re
import sqlite3
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI → hex colour conversion (mirrors agterm's palette)
# ---------------------------------------------------------------------------


def _xterm256_hex(n: int) -> str:
    if n < 16:
        _ANSI16 = [
            "#000000",
            "#aa0000",
            "#00aa00",
            "#aa8800",
            "#0000aa",
            "#aa00aa",
            "#00aaaa",
            "#aaaaaa",
            "#555555",
            "#ff5555",
            "#55ff55",
            "#ffff55",
            "#5555ff",
            "#ff55ff",
            "#55ffff",
            "#ffffff",
        ]
        return _ANSI16[n]
    if n < 232:
        idx = n - 16

        def _c(lvl: int) -> int:
            return 0 if lvl == 0 else 55 + 40 * lvl

        return f"#{_c(idx // 36):02x}{_c((idx // 6) % 6):02x}{_c(idx % 6):02x}"
    v = 8 + (n - 232) * 10
    return f"#{v:02x}{v:02x}{v:02x}"


def ansi_to_hex(ansi: str) -> str:
    """Convert an agterm ANSI escape code to a CSS hex colour string."""
    m = _re.match(r"\033\[38;5;(\d+)m", ansi)
    if m:
        return _xterm256_hex(int(m.group(1)))
    return "#d4d4d4"


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class agwebui_emitter:
    """Thread-safe SQLite event writer for the web UI."""

    def __init__(self, run_dir: Path) -> None:
        self._db_path = run_dir / "ui_events.db"
        self._reply_dir = run_dir / "ui_replies"
        self._reply_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Prevents concurrent prune threads from piling up.
        self._prune_lock = threading.Lock()
        # Latest cumulative token counts per agent; flushed again on done().
        self._token_state: dict[str, tuple[int, int, int, int]] = {}
        # Registration state re-emitted in done() for late-joining clients.
        self._agent_registry: dict[str, dict] = {}  # agname -> event dict
        self._team_registry: dict[str, dict] = {}  # team_name -> event dict
        self._init_db()

    # High-frequency event types that are upserted into state tables AND
    # pruned from the append log to keep the database small.
    _STATE_TYPES = frozenset({"token_update", "messages_snapshot", "resource_update"})
    _PRUNE_EVERY = 500  # prune after this many inserts into events
    # Time-bucket size for downsampling: keep the last event per
    # (type, agname, floor(ts / bucket)) so scrubbing always finds a sample
    # within one bucket of any position.
    _PRUNE_BUCKET_S: float = 60.0  # seconds

    def _init_db(self) -> None:
        con = sqlite3.connect(str(self._db_path), timeout=30)
        con.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS events (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                type   TEXT    NOT NULL,
                agname TEXT,
                ts     REAL    NOT NULL,
                data   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts);
            CREATE INDEX IF NOT EXISTS idx_events_type   ON events(type);
            CREATE INDEX IF NOT EXISTS idx_events_agname ON events(agname);
            CREATE TABLE IF NOT EXISTS agent_state (
                agname   TEXT PRIMARY KEY,
                tokens   TEXT,
                messages TEXT
            );
            CREATE TABLE IF NOT EXISTS resource_state (
                id   INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            );
        """)
        con.commit()
        con.close()
        self._insert_count = 0

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------

    def emit(self, event: dict) -> None:
        data = json.dumps(event, ensure_ascii=False, default=str)
        ts = float(event.get("ts") or time.time())
        etype = event.get("type", "")
        agname = event.get("agname")
        with self._lock:
            # sqlite3.connect()'s default busy_timeout is only 5s. Under heavy
            # load, _run_prune()'s DELETE (which deliberately runs outside
            # self._lock so it never blocks emit() callers) can hold the
            # write lock longer than that over a large events table, making
            # this connect()/execute() raise "database is locked" instead of
            # waiting it out -- silently dropping the event (caught and only
            # logged as a warning by callers like agent._push_live_messages).
            # Match _run_prune()'s own generous timeout below.
            con = sqlite3.connect(str(self._db_path), timeout=30)
            con.execute(
                "INSERT INTO events(type, agname, ts, data) VALUES(?,?,?,?)",
                (etype, agname, ts, data),
            )
            # Upsert into state tables for cold-start preamble on reconnect.
            if etype == "token_update" and agname:
                con.execute(
                    "INSERT INTO agent_state(agname, tokens) VALUES(?,?)"
                    " ON CONFLICT(agname) DO UPDATE SET tokens=excluded.tokens",
                    (agname, data),
                )
            elif etype == "messages_snapshot" and agname:
                con.execute(
                    "INSERT INTO agent_state(agname, messages) VALUES(?,?)"
                    " ON CONFLICT(agname) DO UPDATE SET messages=excluded.messages",
                    (agname, data),
                )
            elif etype == "resource_update":
                con.execute(
                    "INSERT INTO resource_state(id, data) VALUES(1,?)"
                    " ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                    (data,),
                )
            self._insert_count += 1
            should_prune = self._insert_count % self._PRUNE_EVERY == 0
            con.commit()
            con.close()
        # Dispatched on a background thread so emit() itself doesn't block
        # for the whole prune -- but _run_prune() below still serializes its
        # actual DELETE through self._lock, the same lock emit() uses (see
        # its docstring for why).
        if should_prune:
            threading.Thread(target=self._run_prune, daemon=True, name="emitter-prune").start()

    def _flush_prune(self) -> None:
        """Block until any in-flight background prune has completed. For tests only."""
        with self._prune_lock:
            pass

    def _run_prune(self) -> None:
        """Background worker: delete old high-frequency events.

        Uses a try-lock so at most one prune runs at a time; excess triggers are
        dropped rather than queued, which is fine because the next scheduled prune
        will clean up any remaining rows.

        The actual DELETE holds self._lock -- the same lock emit() holds for its
        own connection -- so this connection and emit()'s are never open and
        writing to the db file at the same time. Two separate sqlite3
        connections both able to write concurrently (this used to run on its
        own connection outside self._lock, specifically so it wouldn't block
        emit() callers) both touch page 1 (the file header -- schema cookie,
        page count, freelist pointers) on nearly every write; letting that
        happen from two connections at once, coordinated only by SQLite's own
        cross-connection locking, is exactly the kind of window a real
        corruption of that page (confirmed live: header bytes replaced by
        garbage, but every other page -- and all real event data -- still
        intact and recoverable) would come from. Fully serializing every
        writer in this process removes that risk; the cost is emit() callers
        occasionally waiting for a prune's DELETE to finish, which is
        infrequent (every _PRUNE_EVERY inserts) and fast.
        """
        if not self._prune_lock.acquire(blocking=False):
            return
        try:
            with self._lock:
                # Keep the last event per (type, agname, time-bucket).
                # This guarantees at most one sample per bucket per agent,
                # so timeline scrubbing always finds a sample within
                # _PRUNE_BUCKET_S seconds of any scrub position.
                con = sqlite3.connect(str(self._db_path), timeout=120)
                con.execute(
                    """
                    DELETE FROM events
                    WHERE type IN ('token_update','messages_snapshot','resource_update')
                      AND id NOT IN (
                        SELECT MAX(id) FROM events
                        WHERE type IN ('token_update','messages_snapshot','resource_update')
                        GROUP BY type, agname, CAST(ts / ? AS INTEGER)
                      )
                    """,
                    (self._PRUNE_BUCKET_S,),
                )
                con.commit()
                con.close()
        except Exception:
            pass
        finally:
            self._prune_lock.release()

    # ------------------------------------------------------------------
    # Typed emitters
    # ------------------------------------------------------------------

    def log(self, line: str) -> None:
        self.emit({"type": "log", "line": line, "ts": time.time()})

    def agent_registered(self, agname: str, hex_color: str, team: str | None = None) -> None:
        ev = {
            "type": "agent_registered",
            "agname": agname,
            "color": hex_color,
            "team": team,
            "ts": time.time(),
        }
        with self._lock:
            self._agent_registry[agname] = ev
        self.emit(ev)

    def agent_state(
        self,
        agname: str,
        state: str,
        skill: str | None,
        tool: str | None,
        color: str | None = None,
        team: str | None = None,
    ) -> None:
        self.emit(
            {
                "type": "agent_state",
                "agname": agname,
                "state": state,
                "skill": skill,
                "tool": tool,
                "color": color,
                "team": team,
                "ts": time.time(),
            }
        )

    def agent_config(self, agname: str, config: dict) -> None:
        """Push the agent's current dynamic-config snapshot (see
        agConfig.dynamic_snapshot()) so the webui's config editor can show
        it without a round trip into the (isolated) execution process."""
        self.emit(
            {
                "type": "agent_config",
                "agname": agname,
                "config": config,
                "ts": time.time(),
            }
        )

    def team_registered(self, team_name: str, agent_names: list[str]) -> None:
        ev = {
            "type": "team_registered",
            "team_name": team_name,
            "agents": agent_names,
            "ts": time.time(),
        }
        with self._lock:
            self._team_registry[team_name] = ev
        self.emit(ev)

    def push_messages(self, agname: str, messages: list[dict]) -> None:
        try:
            json.dumps(messages)
        except Exception:
            return
        self.emit(
            {
                "type": "messages_snapshot",
                "agname": agname,
                "messages": messages,
                "ts": time.time(),
            }
        )

    _ASK_TIMEOUT_REPLY = "[no human available — timed out]"

    def ask_human(
        self, agname: str, ask_id: str, question: str, timeout_s: float | None = 300
    ) -> str:
        """Emit ask event then block-poll until the web UI delivers a reply or timeout.

        Pass ``timeout_s=None`` to wait indefinitely (for interactive use cases).
        """
        self.emit(
            {
                "type": "ask_human",
                "agname": agname,
                "ask_id": ask_id,
                "question": question,
                "ts": time.time(),
            }
        )
        reply_file = self._reply_dir / f"{ask_id}.txt"
        deadline = (time.time() + timeout_s) if timeout_s is not None else None
        while not reply_file.exists():
            if deadline is not None and time.time() >= deadline:
                self.emit(
                    {
                        "type": "human_reply",
                        "ask_id": ask_id,
                        "agname": agname,
                        "reply": self._ASK_TIMEOUT_REPLY,
                    }
                )
                return self._ASK_TIMEOUT_REPLY
            time.sleep(0.2)
        text = reply_file.read_text(encoding="utf-8").strip()
        try:
            reply_file.unlink()
        except Exception:
            pass
        return text

    def token_update(
        self,
        agname: str,
        agent_input: int,
        agent_output: int,
        global_input: int,
        global_output: int,
    ) -> None:
        """Emit cumulative token counts for one agent and the framework total."""
        with self._lock:
            self._token_state[agname] = (agent_input, agent_output, global_input, global_output)
        self.emit(
            {
                "type": "token_update",
                "agname": agname,
                "agent_input": agent_input,
                "agent_output": agent_output,
                "global_input": global_input,
                "global_output": global_output,
                "ts": time.time(),
            }
        )

    def resource_update(
        self,
        gpus_acquired: int,
        gpus_total: int,
        cpus_acquired: float,
        cpus_total: int,
        memory_acquired_mb: int,
        memory_total_mb: int,
    ) -> None:
        """Emit current resource acquisition counts for the dashboard badge."""
        self.emit(
            {
                "type": "resource_update",
                "gpus_acquired": gpus_acquired,
                "gpus_total": gpus_total,
                "cpus_acquired": round(cpus_acquired, 1),
                "cpus_total": cpus_total,
                "memory_acquired_mb": memory_acquired_mb,
                "memory_total_mb": memory_total_mb,
                "ts": time.time(),
            }
        )

    def done(self) -> None:
        # Re-emit registration and token state so that late-joining clients
        # always receive a complete roster and current token counts.
        with self._lock:
            agents = list(self._agent_registry.values())
            teams = list(self._team_registry.values())
            tokens = dict(self._token_state)
        now = time.time()
        for ev in agents:
            self.emit({**ev, "ts": now})
        for ev in teams:
            self.emit({**ev, "ts": now})
        for agname, (ai, ao, gi, go) in tokens.items():
            self.emit(
                {
                    "type": "token_update",
                    "agname": agname,
                    "agent_input": ai,
                    "agent_output": ao,
                    "global_input": gi,
                    "global_output": go,
                    "ts": now,
                }
            )
        self.emit({"type": "done", "ts": now})
