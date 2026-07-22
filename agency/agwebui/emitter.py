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
        self._init_db()

    # Event types with a meaningful "current value" -- each gets its own
    # latest-value table (one row per key, upserted in place) IN ADDITION to
    # its normal append-only insert into `events` above. The append-only
    # insert is what a client already connected when the event fires sees
    # live; the state-table upsert is what a client connecting or
    # reconnecting LATER recovers, regardless of how long the run has been
    # going -- immune to both TAIL_EVENTS' fixed-size replay window and
    # _PRUNE_TYPES' downsampling below, neither of which these tables are
    # ever subject to (they hold one row per key, so there's nothing to
    # prune or age out). See _fetch_state_preamble() in server.py, which
    # reads all of them in full on every client connect.
    #
    # agent_registered/team_registered fire once per agent/team and never
    # again -- this replaces what used to be a bespoke in-memory dict on
    # both this class and server.py (_agent_registry/_team_registry),
    # rebuilt by re-scanning the full, never-pruned `events` log at server
    # startup. That was only ever correct because registration events are
    # never pruned -- it happened to work, not because it needed a
    # different mechanism. Folding them into the same upsert-table pattern
    # as token_update/agent_state removes that asymmetry: one mechanism for
    # every "current value" type, backed by SQLite instead of a second,
    # server-process-local cache that has to be kept in sync by hand.
    _UPSERT_TABLES = {
        "token_update": "agent_tokens",
        "messages_snapshot": "agent_messages",
        "agent_state": "agent_state",
        "agent_config": "agent_config_state",
        "agent_registered": "agent_registry",
        "team_registered": "team_registry",
    }
    # Genuinely high-frequency types (many events/sec/agent possible) that
    # ALSO get downsampled out of the append-only `events` log -- the
    # append-only copy exists only for live delivery, so once a newer row
    # for the same (type, agname, time bucket) exists, older ones in that
    # bucket are pure waste. agent_registered/team_registered/agent_config
    # fire at most a handful of times per agent for the whole run, same
    # order of magnitude as `log` -- not worth pruning, same as `log` isn't.
    _PRUNE_TYPES = frozenset(
        {"token_update", "messages_snapshot", "resource_update", "agent_state"}
    )
    _PRUNE_EVERY = 500  # prune after this many inserts into events
    # The most recent _PRUNE_KEEP_RAW rows (by id, across ALL types) are
    # never touched by a prune sweep, regardless of type or time bucket --
    # every sweep only evaluates rows older than (current max id -
    # _PRUNE_KEEP_RAW). This is what lets server.py's TAIL_EVENTS replay
    # assume the tail it requests is always fully raw, never partially
    # compacted mid-window: since a sweep fires every _PRUNE_EVERY inserts
    # and never prunes the newest _PRUNE_KEEP_RAW rows, the raw
    # (unpruned-by-this-mechanism) tail at any moment is at least
    # _PRUNE_KEEP_RAW rows (right after a sweep) and at most
    # _PRUNE_KEEP_RAW + _PRUNE_EVERY rows (right before the next one, when
    # a full _PRUNE_EVERY inserts' worth has accumulated past the previous
    # sweep's protected window without yet being swept themselves). With
    # both set to 500, that's a 500-1000 row raw tail -- see TAIL_EVENTS.
    _PRUNE_KEEP_RAW = 500
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

            -- Latest-value tables -- see _UPSERT_TABLES' docstring above.
            CREATE TABLE IF NOT EXISTS agent_tokens (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_messages (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_state (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_config_state (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_registry (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS team_registry (
                team_name TEXT PRIMARY KEY,
                data      TEXT NOT NULL
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
            try:
                con.execute(
                    "INSERT INTO events(type, agname, ts, data) VALUES(?,?,?,?)",
                    (etype, agname, ts, data),
                )
                # Upsert into this type's latest-value table (if it has one)
                # for cold-start preamble on reconnect -- see _UPSERT_TABLES'
                # docstring above. team_registered keys on team_name, not
                # agname (there is no per-agent column here); every other
                # entry in _UPSERT_TABLES keys on agname.
                table = self._UPSERT_TABLES.get(etype)
                if table == "team_registry":
                    key = event.get("team_name")
                    if key:
                        con.execute(
                            f"INSERT INTO {table}(team_name, data) VALUES(?,?)"
                            " ON CONFLICT(team_name) DO UPDATE SET data=excluded.data",
                            (key, data),
                        )
                elif table and agname:
                    con.execute(
                        f"INSERT INTO {table}(agname, data) VALUES(?,?)"
                        " ON CONFLICT(agname) DO UPDATE SET data=excluded.data",
                        (agname, data),
                    )
                if etype == "resource_update":
                    con.execute(
                        "INSERT INTO resource_state(id, data) VALUES(1,?)"
                        " ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                        (data,),
                    )
                self._insert_count += 1
                should_prune = self._insert_count % self._PRUNE_EVERY == 0
                con.commit()
            except sqlite3.DatabaseError:
                # Corrupt-file-class errors (e.g. "unsupported file format",
                # "database disk image is malformed", "file is not a
                # database") mean page 1 itself is unreadable -- retrying the
                # same file only produces the same error forever, silently
                # killing the live dashboard for the rest of the run. Leaving
                # a leaked, unclosed, possibly-uncommitted connection behind
                # on this path was itself a way to *cause* that corruption
                # under a long, high-throughput run (many never-closed
                # handles all still able to touch the file), so close() in
                # `finally` below always runs, then swap in a fresh db so
                # subsequent emit() calls succeed again. Historical rows are
                # unrecoverable once page 1 is gone -- that's an acceptable
                # loss for what is only a live-dashboard event log, not the
                # agent's own state.
                self._reinit_after_corruption()
                return
            finally:
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
                # Keep the last event per (type, agname, time-bucket) --
                # ONLY among rows older than the most recent _PRUNE_KEEP_RAW
                # (by id). The `id <= (MAX(id) - _PRUNE_KEEP_RAW)` guard
                # excludes the newest _PRUNE_KEEP_RAW rows from the DELETE
                # entirely, regardless of type or bucket -- so
                # server.py's TAIL_EVENTS replay can always assume its
                # requested tail is fully raw, never a mix of some rows
                # already bucket-compacted and others not, which a plain
                # id-agnostic sweep firing mid-tail could otherwise produce.
                con = sqlite3.connect(str(self._db_path), timeout=120)
                try:
                    max_id_row = con.execute("SELECT MAX(id) FROM events").fetchone()
                    max_id = max_id_row[0] if max_id_row else None
                    if max_id is not None:
                        cutoff = max_id - self._PRUNE_KEEP_RAW
                        placeholders = ",".join("?" for _ in self._PRUNE_TYPES)
                        prune_types = tuple(self._PRUNE_TYPES)
                        con.execute(
                            f"""
                            DELETE FROM events
                            WHERE type IN ({placeholders})
                              AND id <= ?
                              AND id NOT IN (
                                SELECT MAX(id) FROM events
                                WHERE type IN ({placeholders})
                                  AND id <= ?
                                GROUP BY type, agname, CAST(ts / ? AS INTEGER)
                              )
                            """,
                            prune_types + (cutoff,) + prune_types + (cutoff, self._PRUNE_BUCKET_S),
                        )
                    con.commit()
                except sqlite3.DatabaseError:
                    # See emit()'s matching handler -- page 1 is unreadable,
                    # so hand off to the same reinit rather than leaking this
                    # connection on every future prune too.
                    self._reinit_after_corruption()
                finally:
                    con.close()
        except Exception as _e:
            print(
                f"[agwebui] WARNING: event-log prune failed (best-effort, will retry next cycle): {_e}"
            )
        finally:
            self._prune_lock.release()

    def _reinit_after_corruption(self) -> None:
        """Recover from a corrupt ui_events.db by starting a fresh one.

        Called with self._lock already held (from emit() or _run_prune()),
        so this must not try to reacquire it. Once SQLite reports page 1 as
        unreadable ("unsupported file format" / "database disk image is
        malformed" / "file is not a database"), no query -- not even
        PRAGMA/.recover-style raw page access -- succeeds against that file
        again; the only way forward is a new file. The corrupt file is kept
        alongside (renamed, not deleted) in case a human wants to try
        forensic recovery (e.g. scraping embedded JSON text with `strings`)
        later. Historical dashboard events for this run are lost; the
        agent's own in-memory state is untouched since these pushes are
        best-effort.
        """
        try:
            corrupt_path = None
            if self._db_path.exists():
                corrupt_path = self._db_path.with_name(
                    self._db_path.name + f".corrupt-{int(time.time())}"
                )
                self._db_path.rename(corrupt_path)
            for suffix in ("-wal", "-shm"):
                stale = self._db_path.with_name(self._db_path.name + suffix)
                if stale.exists():
                    stale.unlink()
            self._init_db()
            print(
                f"[agwebui] WARNING: {self._db_path} was corrupted and has been "
                f"reinitialized; historical dashboard events for this run were lost "
                f"(corrupt file kept at {corrupt_path})."
            )
        except Exception as _e:
            print(f"[agwebui] WARNING: failed to recover corrupted {self._db_path}: {_e}")

    # ------------------------------------------------------------------
    # Typed emitters
    # ------------------------------------------------------------------

    def log(self, line: str) -> None:
        self.emit({"type": "log", "line": line, "ts": time.time()})

    def agent_registered(self, agname: str, hex_color: str, team: str | None = None) -> None:
        self.emit(
            {
                "type": "agent_registered",
                "agname": agname,
                "color": hex_color,
                "team": team,
                "ts": time.time(),
            }
        )

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
        self.emit(
            {
                "type": "team_registered",
                "team_name": team_name,
                "agents": agent_names,
                "ts": time.time(),
            }
        )

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
        except Exception as _e:
            print(f"[agwebui] WARNING: failed to clean up reply file {reply_file}: {_e}")
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
        """Emit the terminal marker.

        Used to re-emit registration/token state by hand here so a
        late-joining client would see a complete roster -- no longer
        necessary now that agent_registered/team_registered/token_update
        (and every other _UPSERT_TABLES entry) are durably upserted into
        their own latest-value table on every emit(), read in full by
        _fetch_state_preamble() on every connect regardless of when it
        happens. See _UPSERT_TABLES' docstring.
        """
        self.emit({"type": "done", "ts": time.time()})
