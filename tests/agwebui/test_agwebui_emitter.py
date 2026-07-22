"""Tests for agwebui_emitter — the execution-side event writer."""

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

from agency.agwebui.emitter import agwebui_emitter, ansi_to_hex, _xterm256_hex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_events(run_dir: Path) -> list[dict]:
    db = run_dir / "ui_events.db"
    if not db.exists():
        return []
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT data FROM events ORDER BY id").fetchall()
    con.close()
    return [json.loads(r[0]) for r in rows]


def read_upsert_table(run_dir: Path, table: str, key_col: str = "agname") -> dict:
    """Return {key: parsed_json_data} for any latest-value table (agent_tokens,
    agent_messages, agent_state, agent_config_state, agent_registry,
    team_registry, ...) -- each shares the same (key, data) shape."""
    db = run_dir / "ui_events.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(str(db))
    rows = con.execute(f"SELECT {key_col}, data FROM {table}").fetchall()
    con.close()
    return {key: json.loads(data) for key, data in rows}


def read_agent_state(run_dir: Path) -> dict:
    """Return {agname: {tokens: dict|None, messages: dict|None}}, mirroring
    the old combined-column shape callers below expect."""
    tokens = read_upsert_table(run_dir, "agent_tokens")
    messages = read_upsert_table(run_dir, "agent_messages")
    result = {}
    for agname in set(tokens) | set(messages):
        result[agname] = {
            "tokens": tokens.get(agname),
            "messages": messages.get(agname),
        }
    return result


def read_resource_state(run_dir: Path) -> dict | None:
    db = run_dir / "ui_events.db"
    if not db.exists():
        return None
    con = sqlite3.connect(str(db))
    row = con.execute("SELECT data FROM resource_state WHERE id=1").fetchone()
    con.close()
    return json.loads(row[0]) if row else None


# ---------------------------------------------------------------------------
# ansi_to_hex / _xterm256_hex
# ---------------------------------------------------------------------------


def test_xterm256_system_colors():
    assert _xterm256_hex(0) == "#000000"
    assert _xterm256_hex(15) == "#ffffff"


def test_xterm256_cube():
    # index 16 = black cube corner
    assert _xterm256_hex(16) == "#000000"
    # index 231 = white cube corner
    assert _xterm256_hex(231) == "#ffffff"


def test_xterm256_greyscale():
    grey = _xterm256_hex(232)
    assert grey == "#080808"
    assert _xterm256_hex(255) == "#eeeeee"


def test_ansi_to_hex_38_5():
    # \033[38;5;214m → xterm-256 index 214
    result = ansi_to_hex("\033[38;5;214m")
    assert result == _xterm256_hex(214)


def test_ansi_to_hex_fallback():
    assert ansi_to_hex("") == "#d4d4d4"
    assert ansi_to_hex("\033[1m") == "#d4d4d4"


# ---------------------------------------------------------------------------
# emit — core writer
# ---------------------------------------------------------------------------


def test_emit_creates_file(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.emit({"type": "test", "val": 42})
    events = read_events(tmp_path)
    assert len(events) == 1
    assert events[0] == {"type": "test", "val": 42}


def test_emit_appends_lines(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.emit({"type": "a"})
    em.emit({"type": "b"})
    em.emit({"type": "c"})
    events = read_events(tmp_path)
    assert [e["type"] for e in events] == ["a", "b", "c"]


def test_emit_thread_safe(tmp_path):
    em = agwebui_emitter(tmp_path)
    N = 100

    def _write(i):
        em.emit({"type": "t", "i": i})

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = read_events(tmp_path)
    assert len(events) == N
    assert {e["i"] for e in events} == set(range(N))


def test_emit_survives_external_write_lock_longer_than_default_timeout(tmp_path):
    """Regression test: sqlite3.connect()'s default busy_timeout is only 5s.
    Under heavy load, _run_prune()'s DELETE (which deliberately runs outside
    self._lock so it never blocks emit() callers) can hold the write lock
    longer than that over a large events table. emit()'s connection must use
    a longer timeout (matching _run_prune()'s own) so it waits out a lock
    held for, say, 7 seconds instead of failing with "database is locked"
    and silently dropping the event."""
    em = agwebui_emitter(tmp_path)

    released_at = [None]
    lock_taken = threading.Event()
    HOLD_S = 7  # longer than sqlite3.connect()'s default 5s timeout

    def _hold_lock_then_release():
        # The blocker connection must be created and used entirely within
        # this thread -- sqlite3 forbids using a connection from a different
        # thread than the one that created it.
        blocker = sqlite3.connect(str(tmp_path / "ui_events.db"))
        blocker.execute("BEGIN IMMEDIATE")  # takes the write lock without committing
        lock_taken.set()
        time.sleep(HOLD_S)
        released_at[0] = time.time()
        blocker.commit()
        blocker.close()

    releaser = threading.Thread(target=_hold_lock_then_release)
    releaser.start()
    lock_taken.wait(timeout=5)
    try:
        em.emit({"type": "t", "i": 1})  # must not raise, must wait out the lock
        emitted_at = time.time()
    finally:
        releaser.join()

    assert released_at[0] is not None
    assert emitted_at >= released_at[0] - 0.05, (
        "emit() returned before the competing lock was released -- "
        "it should have waited, not failed fast"
    )
    events = read_events(tmp_path)
    assert len(events) == 1


def test_emit_non_serialisable_uses_str(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.emit({"type": "x", "val": object()})  # default=str handles it
    events = read_events(tmp_path)
    assert events[0]["type"] == "x"


# ---------------------------------------------------------------------------
# Typed emitters
# ---------------------------------------------------------------------------


def test_log(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.log("hello world")
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "log"
    assert ev["line"] == "hello world"
    assert "ts" in ev


def test_agent_registered(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.agent_registered("MyAgent", "#ff8800")
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "agent_registered"
    assert ev["agname"] == "MyAgent"
    assert ev["color"] == "#ff8800"


def test_agent_state(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.agent_state("A", "llm", "design", None)
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "agent_state"
    assert ev["state"] == "llm"
    assert ev["skill"] == "design"
    assert ev["tool"] is None


def test_team_registered(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.team_registered("MyTeam", ["AgA", "AgB"])
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "team_registered"
    assert ev["team_name"] == "MyTeam"
    assert ev["agents"] == ["AgA", "AgB"]


def test_push_messages(tmp_path):
    em = agwebui_emitter(tmp_path)
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    em.push_messages("AgX", msgs)
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "messages_snapshot"
    assert ev["agname"] == "AgX"
    assert ev["messages"] == msgs


def test_push_messages_skips_non_serialisable(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.push_messages("AgX", [{"role": "user", "content": object()}])
    # Should write nothing (silently skipped)
    assert read_events(tmp_path) == []


def test_done(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.done()
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "done"


# ---------------------------------------------------------------------------
# ask_human — file-based request/reply
# ---------------------------------------------------------------------------


def test_ask_human_returns_reply(tmp_path):
    em = agwebui_emitter(tmp_path)
    reply_text = "proceed with option A"

    def _write_reply():
        time.sleep(0.1)
        (tmp_path / "ui_replies" / "abc123.txt").write_text(reply_text)

    threading.Thread(target=_write_reply, daemon=True).start()
    result = em.ask_human("Bot", "abc123", "Which option?")
    assert result == reply_text


def test_ask_human_emits_event(tmp_path):
    em = agwebui_emitter(tmp_path)

    def _write_reply():
        time.sleep(0.05)
        (tmp_path / "ui_replies" / "id99.txt").write_text("yes")

    threading.Thread(target=_write_reply, daemon=True).start()
    em.ask_human("Bot", "id99", "Continue?")

    ev = read_events(tmp_path)[0]
    assert ev["type"] == "ask_human"
    assert ev["agname"] == "Bot"
    assert ev["ask_id"] == "id99"
    assert ev["question"] == "Continue?"


def test_ask_human_removes_reply_file(tmp_path):
    em = agwebui_emitter(tmp_path)
    reply_file = tmp_path / "ui_replies" / "del42.txt"

    def _write_reply():
        time.sleep(0.05)
        reply_file.write_text("done")

    threading.Thread(target=_write_reply, daemon=True).start()
    em.ask_human("Bot", "del42", "Delete test?")
    assert not reply_file.exists()


# ---------------------------------------------------------------------------
# State tables — upsert and pruning
# ---------------------------------------------------------------------------


def test_token_update_upserts_agent_state(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.token_update("AgX", 10, 5, 100, 50)
    em.token_update("AgX", 20, 8, 200, 80)  # should overwrite
    state = read_agent_state(tmp_path)
    assert "AgX" in state
    assert state["AgX"]["tokens"]["agent_input"] == 20


def test_messages_snapshot_upserts_agent_state(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.push_messages("AgY", [{"role": "user", "content": "v1"}])
    em.push_messages("AgY", [{"role": "user", "content": "v2"}])
    state = read_agent_state(tmp_path)
    assert state["AgY"]["messages"]["messages"][0]["content"] == "v2"


def test_agent_config_upserts_agent_config_state(tmp_path):
    """Regression test: agent_config used to only ever land in the
    append-only events table, so its value silently vanished for a client
    connecting after it aged out of TAIL_EVENTS -- see agent_config_state
    in server.py's _STATE_TABLES."""
    em = agwebui_emitter(tmp_path)
    em.agent_config("AgC", {"agskill": {"react_max_steps": 3}})
    em.agent_config("AgC", {"agskill": {"react_max_steps": 42}})  # should overwrite
    state = read_upsert_table(tmp_path, "agent_config_state")
    assert state["AgC"]["config"]["agskill"]["react_max_steps"] == 42


def test_agent_state_upserts_agent_state_table(tmp_path):
    """agent_state (the real status/skill/tool, distinct from the
    old agent_state *column pair* that only held tokens/messages) now gets
    the same latest-value treatment."""
    em = agwebui_emitter(tmp_path)
    em.agent_state("AgS", "llm", "design", None)
    em.agent_state("AgS", "tool", "design", "bash")  # should overwrite
    state = read_upsert_table(tmp_path, "agent_state")
    assert state["AgS"]["state"] == "tool"
    assert state["AgS"]["tool"] == "bash"


def test_agent_registered_upserts_agent_registry(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.agent_registered("AgR", "#ff8800")
    state = read_upsert_table(tmp_path, "agent_registry")
    assert state["AgR"]["color"] == "#ff8800"


def test_team_registered_upserts_team_registry(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.team_registered("MyTeam", ["AgA", "AgB"])
    state = read_upsert_table(tmp_path, "team_registry", key_col="team_name")
    assert state["MyTeam"]["agents"] == ["AgA", "AgB"]


def test_resource_update_upserts_resource_state(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.resource_update(1, 4, 2.0, 8, 1024, 16384)
    em.resource_update(2, 4, 3.0, 8, 2048, 16384)
    rs = read_resource_state(tmp_path)
    assert rs is not None
    assert rs["gpus_acquired"] == 2
    # Only one row in resource_state
    db = tmp_path / "ui_events.db"
    import sqlite3 as _sq

    con = _sq.connect(str(db))
    count = con.execute("SELECT COUNT(*) FROM resource_state").fetchone()[0]
    con.close()
    assert count == 1


def test_pruning_removes_old_high_freq_events(tmp_path):
    em = agwebui_emitter(tmp_path)
    orig_every = agwebui_emitter._PRUNE_EVERY
    orig_bucket = agwebui_emitter._PRUNE_BUCKET_S
    orig_keep_raw = agwebui_emitter._PRUNE_KEEP_RAW
    agwebui_emitter._PRUNE_EVERY = 10
    agwebui_emitter._PRUNE_BUCKET_S = 1.0  # 1-second buckets
    # No raw-tail protection here -- this test is isolating bucket
    # compaction itself; test_prune_never_touches_most_recent_raw_rows
    # below covers _PRUNE_KEEP_RAW's own guarantee.
    agwebui_emitter._PRUNE_KEEP_RAW = 0
    try:
        # All 10 token_updates have the same ts (within the same second bucket)
        # → prune collapses them to 1 row (the last one).
        for i in range(10):
            em.token_update("AgZ", i, 0, i, 0)
        em._flush_prune()
        events = read_events(tmp_path)
        token_rows = [e for e in events if e["type"] == "token_update"]
        assert len(token_rows) == 1, f"expected 1 after prune, got {len(token_rows)}"
        assert token_rows[0]["agent_input"] == 9  # last value kept
    finally:
        agwebui_emitter._PRUNE_EVERY = orig_every
        agwebui_emitter._PRUNE_BUCKET_S = orig_bucket
        agwebui_emitter._PRUNE_KEEP_RAW = orig_keep_raw


def test_pruning_preserves_separate_time_buckets(tmp_path):
    import sqlite3 as _sq

    em = agwebui_emitter(tmp_path)
    orig_every = agwebui_emitter._PRUNE_EVERY
    orig_bucket = agwebui_emitter._PRUNE_BUCKET_S
    orig_keep_raw = agwebui_emitter._PRUNE_KEEP_RAW
    agwebui_emitter._PRUNE_EVERY = 10
    agwebui_emitter._PRUNE_BUCKET_S = 60.0
    agwebui_emitter._PRUNE_KEEP_RAW = 0
    try:
        # Insert 5 events in bucket 0 (ts 0-59) and 5 in bucket 1 (ts 60-119).
        # Prune should keep 1 per bucket = 2 rows total.
        db = tmp_path / "ui_events.db"
        for i in range(5):
            con = _sq.connect(str(db))
            con.execute(
                "INSERT INTO events(type,agname,ts,data) VALUES(?,?,?,?)",
                (
                    "token_update",
                    "AgZ",
                    float(i),
                    f'{{"type":"token_update","agname":"AgZ","agent_input":{i}}}',
                ),
            )
            con.commit()
            con.close()
        for i in range(5):
            con = _sq.connect(str(db))
            con.execute(
                "INSERT INTO events(type,agname,ts,data) VALUES(?,?,?,?)",
                (
                    "token_update",
                    "AgZ",
                    float(60 + i),
                    f'{{"type":"token_update","agname":"AgZ","agent_input":{10 + i}}}',
                ),
            )
            con.commit()
            con.close()
        # Trigger prune by emitting via emitter (its insert_count wraps at _PRUNE_EVERY)
        em._insert_count = em._PRUNE_EVERY - 1
        em.log("trigger")  # this is the Nth insert → prune fires
        em._flush_prune()
        events = read_events(tmp_path)
        token_rows = [e for e in events if e["type"] == "token_update"]
        # One per 60s bucket: bucket 0 keeps agent_input=4, bucket 1 keeps agent_input=14
        assert len(token_rows) == 2, f"expected 2 (one per bucket), got {len(token_rows)}"
        inputs = sorted(e["agent_input"] for e in token_rows)
        assert inputs == [4, 14]
    finally:
        agwebui_emitter._PRUNE_EVERY = orig_every
        agwebui_emitter._PRUNE_BUCKET_S = orig_bucket
        agwebui_emitter._PRUNE_KEEP_RAW = orig_keep_raw


def test_pruning_preserves_log_events(tmp_path):
    em = agwebui_emitter(tmp_path)
    orig_every = agwebui_emitter._PRUNE_EVERY
    orig_bucket = agwebui_emitter._PRUNE_BUCKET_S
    orig_keep_raw = agwebui_emitter._PRUNE_KEEP_RAW
    agwebui_emitter._PRUNE_EVERY = 10
    agwebui_emitter._PRUNE_BUCKET_S = 1.0
    agwebui_emitter._PRUNE_KEEP_RAW = 0
    try:
        # 5 logs + 5 token_updates = 10 inserts → prune fires; logs must survive.
        # Freeze time.time() so all 5 token_updates land in the same 1-second
        # bucket deterministically -- without this, real wall-clock emission
        # (5 SQLite inserts under _lock) can cross the bucket boundary under
        # load, letting more than one row survive and making this test flaky.
        with patch("time.time", return_value=1_000_000.0):
            for i in range(5):
                em.log(f"line{i}")
            for i in range(5):
                em.token_update("AgZ", i, 0, i, 0)
        em._flush_prune()
        events = read_events(tmp_path)
        log_rows = [e for e in events if e["type"] == "log"]
        token_rows = [e for e in events if e["type"] == "token_update"]
        assert len(log_rows) == 5
        assert len(token_rows) == 1
    finally:
        agwebui_emitter._PRUNE_EVERY = orig_every
        agwebui_emitter._PRUNE_BUCKET_S = orig_bucket
        agwebui_emitter._PRUNE_KEEP_RAW = orig_keep_raw


def test_prune_never_touches_most_recent_raw_rows(tmp_path):
    """_PRUNE_KEEP_RAW rows are protected from a sweep regardless of type or
    time bucket -- this is what lets TAIL_EVENTS assume its replay window is
    always fully raw (see _PRUNE_KEEP_RAW's docstring in emitter.py)."""
    em = agwebui_emitter(tmp_path)
    orig_every = agwebui_emitter._PRUNE_EVERY
    orig_bucket = agwebui_emitter._PRUNE_BUCKET_S
    orig_keep_raw = agwebui_emitter._PRUNE_KEEP_RAW
    agwebui_emitter._PRUNE_EVERY = 5
    agwebui_emitter._PRUNE_BUCKET_S = 1.0
    agwebui_emitter._PRUNE_KEEP_RAW = 5
    try:
        # All 10 land in the same 1-second bucket. Without protection, a
        # sweep would collapse all 10 down to 1. With the most recent 5
        # protected, only rows 0-4 (the older half) are eligible, and among
        # those 5 (all one bucket) bucket-compaction still collapses them
        # to 1 -- so 1 (compacted old) + 5 (protected raw) = 6 survive.
        with patch("time.time", return_value=1_000_000.0):
            for i in range(10):
                em.token_update("AgZ", i, 0, i, 0)
        em._flush_prune()
        events = read_events(tmp_path)
        token_rows = [e for e in events if e["type"] == "token_update"]
        assert len(token_rows) == 6, f"expected 6 (1 compacted + 5 raw), got {len(token_rows)}"
        # The 5 most recent (agent_input 5-9) must all still be present, raw.
        raw_inputs = sorted(e["agent_input"] for e in token_rows if e["agent_input"] >= 5)
        assert raw_inputs == [5, 6, 7, 8, 9]
    finally:
        agwebui_emitter._PRUNE_EVERY = orig_every
        agwebui_emitter._PRUNE_BUCKET_S = orig_bucket
        agwebui_emitter._PRUNE_KEEP_RAW = orig_keep_raw
