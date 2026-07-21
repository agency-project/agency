"""Tests for agterm — color assignment, log formatting, thread safety, and
sandbox container naming / run isolation."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import uuid
from unittest.mock import MagicMock


def _worker_get_run_id():
    """Top-level so ProcessPoolExecutor can pickle it."""
    from agency.agsandbox import _RUN_ID

    return _RUN_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_agterm(agname: str | None = None):
    """Return an agterm instance with a guaranteed-unique agname."""
    from agency.agterm import agterm

    return agterm(agname or f"test-{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------


class TestColorPalette:
    def test_palette_has_at_least_54_colors(self):
        from agency.agterm import _AGENT_COLORS

        assert len(_AGENT_COLORS) >= 54

    def test_palette_entries_are_ansi_escape_sequences(self):
        from agency.agterm import _AGENT_COLORS

        for color in _AGENT_COLORS:
            assert color.startswith("\033["), f"not an ANSI escape: {color!r}"

    def test_palette_entries_are_unique(self):
        from agency.agterm import _AGENT_COLORS

        assert len(set(_AGENT_COLORS)) == len(_AGENT_COLORS)

    def test_make_color_palette_excludes_greys(self):
        from agency.agterm import _make_color_palette

        palette = _make_color_palette()
        # xterm-256 grey diagonal: index = 16 + 36r + 6r + r = 16 + 43r
        grey_indices = {16 + 43 * r for r in range(6)}
        codes = {int(re.search(r"\d+", c).group()) for c in palette}
        assert not codes & grey_indices

    def test_make_color_palette_excludes_near_black(self):
        from agency.agterm import _make_color_palette

        palette = _make_color_palette()
        for color in palette:
            idx = int(re.search(r"\d+", color).group())
            r = (idx - 16) // 36
            g = ((idx - 16) % 36) // 6
            b = (idx - 16) % 6
            assert max(r, g, b) > 2, f"near-black color included: index {idx}"


# ---------------------------------------------------------------------------
# Color assignment
# ---------------------------------------------------------------------------


class TestColorAssignment:
    def test_each_agent_gets_a_color(self):
        term = _fresh_agterm()
        assert term._color is not None
        assert "\033[" in term._color

    def test_agname_registered_in_class_dict(self):
        from agency.agterm import agterm

        name = f"reg-{uuid.uuid4().hex[:8]}"
        _fresh_agterm(name)
        assert name in agterm._agname_colors

    def test_color_counter_increments(self):
        from agency.agterm import agterm

        before = agterm._color_counter
        _fresh_agterm()
        assert agterm._color_counter == before + 1

    def test_different_agents_get_different_colors_across_palette(self):
        # Create enough agents to cycle through a few palette slots and confirm
        # consecutive agents are assigned different indices.
        colors = [_fresh_agterm()._color for _ in range(5)]
        # At minimum, consecutive colors should differ (palette is shuffled but
        # has 54+ entries so 5 consecutive picks are all distinct).
        assert len(set(colors)) == len(colors)


# ---------------------------------------------------------------------------
# _colorize_agnames
# ---------------------------------------------------------------------------


class TestColorizeAgnames:
    def test_registered_agname_in_message_is_wrapped(self):
        from agency.agterm import agterm

        name = f"colorize-{uuid.uuid4().hex[:8]}"
        _fresh_agterm(name)
        result = agterm._colorize_agnames(f"agent {name} finished")
        assert name in result
        assert "\033[" in result  # some ANSI code was inserted

    def test_unknown_agname_not_modified(self):
        from agency.agterm import agterm

        msg = "no-agent-here-xyz"
        assert agterm._colorize_agnames(msg) == msg

    def test_colorize_does_not_mutate_during_concurrent_registration(self):
        """Snapshot via list() prevents RuntimeError on dict resize."""
        from agency.agterm import agterm

        errors: list[Exception] = []

        def _register():
            for _ in range(50):
                _fresh_agterm()

        def _colorize():
            for _ in range(200):
                try:
                    agterm._colorize_agnames("some message with text")
                except RuntimeError as e:
                    errors.append(e)

        t1 = threading.Thread(target=_register)
        t2 = threading.Thread(target=_colorize)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"RuntimeError during concurrent iteration: {errors[0]}"


# ---------------------------------------------------------------------------
# log() — output and format
# ---------------------------------------------------------------------------


class TestLog:
    def test_log_writes_to_stderr_when_no_webui(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            term = _fresh_agterm()
            term.log("CREATED  ", "hello-stderr")
            captured = capsys.readouterr()
            assert "hello-stderr" in captured.err
        finally:
            agwebui_mod._active = old

    def test_log_contains_agname(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            name = f"logname-{uuid.uuid4().hex[:8]}"
            term = _fresh_agterm(name)
            term.log("TOOL     ", "msg")
            captured = capsys.readouterr()
            assert name in captured.err
        finally:
            agwebui_mod._active = old

    def test_log_contains_event_tag(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            term = _fresh_agterm()
            term.log("SKILL ▶  ", "event-check")
            captured = capsys.readouterr()
            assert "SKILL ▶" in captured.err
        finally:
            agwebui_mod._active = old

    def test_log_contains_source_location(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            term = _fresh_agterm()
            term.log("TOOL     ", "loc-check")
            captured = capsys.readouterr()
            # source location format: (filename.py:lineno)
            assert re.search(r"\(\w+\.py:\d+\)", captured.err)
        finally:
            agwebui_mod._active = old

    def test_log_silent_when_disabled(self, capsys):
        import agency.agwebui as agwebui_mod
        from agency.agterm import agterm

        old = agwebui_mod._active
        agwebui_mod._active = None
        old_enabled = agterm.enabled
        agterm.enabled = False
        try:
            term = _fresh_agterm()
            term.log("TOOL     ", "should-not-appear")
            captured = capsys.readouterr()
            assert "should-not-appear" not in captured.err
        finally:
            agterm.enabled = old_enabled
            agwebui_mod._active = old

    def test_log_token_tag_absent_when_none(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            term = _fresh_agterm()
            assert term._tokens is None
            term.log("TOOL     ", "check-token-absent")
            captured = capsys.readouterr()
            assert " toks)" not in captured.err
        finally:
            agwebui_mod._active = old

    def test_log_token_tag_present_when_set(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            term = _fresh_agterm()
            term._tokens = 42
            term.log("TOOL     ", "with-toks")
            captured = capsys.readouterr()
            assert "42 toks" in captured.err
        finally:
            agwebui_mod._active = old

    def test_log_depth_affects_source_file(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            term = _fresh_agterm()

            # depth=1 → this file; depth=2 → one frame up (also this file in a wrapper)
            def _wrapper():
                term.log("TOOL     ", "depth-test", depth=2)

            _wrapper()
            captured = capsys.readouterr()
            assert "test_agterm.py" in captured.err
        finally:
            agwebui_mod._active = old


# ---------------------------------------------------------------------------
# log() — webui active: errors to stderr, non-errors to webui only
# ---------------------------------------------------------------------------


class TestLogWebui:
    """When webui is active, only error events (✗) should reach stderr."""

    def _make_mock_webui(self):
        mock = MagicMock()
        mock.emitter = MagicMock()
        mock.emitter.log = MagicMock()
        return mock

    def test_non_error_event_goes_to_webui_only_not_stderr(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = self._make_mock_webui()
        try:
            term = _fresh_agterm()
            term.log("SKILL ✓  ", "normal-event")
            captured = capsys.readouterr()
            assert "normal-event" not in captured.err
            agwebui_mod._active.emitter.log.assert_called()
        finally:
            agwebui_mod._active = old

    def test_error_event_goes_to_both_webui_and_stderr(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = self._make_mock_webui()
        try:
            term = _fresh_agterm()
            term.log("SKILL ✗  ", "error-event")
            captured = capsys.readouterr()
            assert "error-event" in captured.err
            agwebui_mod._active.emitter.log.assert_called()
        finally:
            agwebui_mod._active = old

    def test_prune_error_event_also_reaches_stderr(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = self._make_mock_webui()
        try:
            term = _fresh_agterm()
            term.log("PRUNE ✗  ", "prune-error")
            captured = capsys.readouterr()
            assert "prune-error" in captured.err
        finally:
            agwebui_mod._active = old

    def test_non_error_events_with_webui_do_not_clutter_stderr(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = self._make_mock_webui()
        try:
            term = _fresh_agterm()
            for ev in ("CREATED  ", "SKILL ▶  ", "LLM ▶    ", "LLM ✓    ", "TOOL     "):
                term.log(ev, f"msg-{ev.strip()}")
            captured = capsys.readouterr()
            assert captured.err == ""
        finally:
            agwebui_mod._active = old


# ---------------------------------------------------------------------------
# Thread safety of log()
# ---------------------------------------------------------------------------


class TestLogThreadSafety:
    def test_concurrent_log_calls_do_not_interleave_lines(self, capsys):
        import agency.agwebui as agwebui_mod

        old = agwebui_mod._active
        agwebui_mod._active = None
        try:
            terms = [_fresh_agterm() for _ in range(8)]
            barrier = threading.Barrier(8)

            def _spam(term):
                barrier.wait()
                for i in range(20):
                    term.log("TOOL     ", f"msg-{i}")

            threads = [threading.Thread(target=_spam, args=(t,)) for t in terms]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            captured = capsys.readouterr()
            lines = [ln for ln in captured.err.splitlines() if ln.strip()]
            # Every line should be a complete log entry (contains a bracket-enclosed tag)
            for line in lines:
                assert "[" in line and "]" in line, f"Malformed line: {line!r}"
        finally:
            agwebui_mod._active = old


# ---------------------------------------------------------------------------
# Sandbox container naming and run isolation
# ---------------------------------------------------------------------------


class TestSandboxNaming:
    def test_container_name_includes_run_id(self):
        from agency.agsandbox import agSandbox, _RUN_ID

        sb = agSandbox("myagent")
        assert _RUN_ID in sb._name

    def test_container_name_includes_agname(self):
        from agency.agsandbox import agSandbox

        sb = agSandbox("myagent")
        assert "myagent" in sb._name

    def test_container_name_format(self):
        """The agname component is deduplicated (see agsandbox.py's
        __init__) -- "myagent" becomes "sandbox_myagent_XXXX" via the
        shared agname registry, so the exact suffix isn't predictable
        (it depends on how many times this base has already been claimed
        elsewhere in this same test process), only the overall shape is."""
        from agency.agsandbox import agSandbox, _RUN_ID

        sb = agSandbox("myagent")
        assert re.fullmatch(rf"sandbox-{_RUN_ID}-sandbox_myagent_[0-9a-z]{{4}}", sb._name), sb._name

    def test_two_sandboxes_same_agname_get_deduplicated_names(self):
        """Every agSandbox construction claims its own unique name from the
        shared agname registry (see agsandbox.py's __init__) -- passing the
        same literal agname twice must NOT collide into the same container
        identity. Anyone who genuinely needs to reference an existing
        sandbox must keep the object/backend itself around, not re-pass its
        name string (see test_agsandbox.py's
        test_ensure_started_reuses_running_container for the supported way
        to do that)."""
        from agency.agsandbox import agSandbox

        sb1 = agSandbox("shared-agent")
        sb2 = agSandbox("shared-agent")
        assert sb1._name != sb2._name

    def test_two_sandboxes_different_agnames_differ(self):
        from agency.agsandbox import agSandbox

        sb1 = agSandbox("agent-alpha")
        sb2 = agSandbox("agent-beta")
        assert sb1._name != sb2._name

    def test_run_id_is_run_scoped_not_pid(self):
        """_RUN_ID must not be the current PID (we switched to UUID)."""
        import os
        from agency.agsandbox import _RUN_ID

        assert str(os.getpid()) not in _RUN_ID

    def test_run_id_format(self):
        """_RUN_ID must be 'r' followed by 8 hex chars."""
        from agency.agsandbox import _RUN_ID

        assert re.fullmatch(r"r[0-9a-f]{8}", _RUN_ID), f"unexpected _RUN_ID: {_RUN_ID!r}"


class TestRunIsolation:
    def test_separate_imports_produce_different_run_ids(self):
        """Two separate process invocations must never share a _RUN_ID."""
        script = "from agency.agsandbox import _RUN_ID; print(_RUN_ID)"
        r1 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        r2 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        id1 = r1.stdout.strip()
        id2 = r2.stdout.strip()
        assert id1 and id2
        assert id1 != id2, (
            f"Two separate processes got the same _RUN_ID: {id1!r} — "
            "UUID generation is broken or _RUN_ID is PID-based"
        )

    def test_separate_imports_produce_different_container_names(self):
        """Container names from two separate runs must not collide."""
        script = (
            "from agency.agsandbox import agSandbox; "
            "sb = agSandbox('DataGen_0000'); print(sb._name)"
        )
        r1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        name1 = r1.stdout.strip()
        name2 = r2.stdout.strip()
        assert name1 and name2
        assert name1 != name2, (
            f"Same container name across two runs: {name1!r} — cross-run isolation is broken"
        )

    def test_checkpoint_image_tag_differs_across_runs(self):
        """Lifecycle image tags must be run-scoped to prevent cross-run clobber."""
        script = (
            "from agency.agsandbox import agSandbox; "
            "sb = agSandbox('DataGen_0000'); "
            "print(sb._backend._lifecycle_tag())"
        )
        r1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        tag1 = r1.stdout.strip()
        tag2 = r2.stdout.strip()
        assert tag1 and tag2
        assert tag1 != tag2, (
            f"Same lifecycle image tag across two runs: {tag1!r} — "
            "a new run would clobber the previous run's checkpoint"
        )

    def test_worker_process_inherits_run_id(self):
        """Worker processes (fork/spawn) must see the same _RUN_ID as the main process."""
        import concurrent.futures
        from agency.agsandbox import _RUN_ID

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            worker_id = pool.submit(_worker_get_run_id).result(timeout=30)

        assert worker_id == _RUN_ID, (
            f"Worker got _RUN_ID={worker_id!r}, main has {_RUN_ID!r} — "
            "workers must inherit the parent's run ID"
        )
