"""Tests for environment-controlled profiler lifecycle scopes."""

from contextlib import contextmanager

import pytest

from agency.profiler import agprof


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "workload"),
        ("", "workload"),
        ("workload", "workload"),
        ("WORKLOAD", "workload"),
        ("invalid", "workload"),
        ("process", "process"),
        (" PROCESS ", "process"),
    ],
)
def test_profile_scope(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    else:
        monkeypatch.setenv("AGENCY_PROFILE_SCOPE", value)

    assert agprof.profile_scope() == expected


def test_workload_scope_owns_session_exactly_around_workload(monkeypatch):
    events = []
    profiler = object()
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    monkeypatch.setenv("AGENCY_PROFILE_DIR", "custom-trace")
    monkeypatch.setattr(agprof, "enabled", lambda: False)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: events.append(("start", out_dir)) or profiler,
    )
    monkeypatch.setattr(agprof, "stop", lambda: events.append(("stop", None)))

    with agprof.workload() as active:
        events.append(("workload", None))
        assert active is profiler

    assert events == [
        ("start", "custom-trace"),
        ("workload", None),
        ("stop", None),
    ]


def test_workload_scope_stops_session_when_workload_raises(monkeypatch):
    events = []
    monkeypatch.setenv("AGENCY_PROFILE", "true")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "workload")
    monkeypatch.setattr(agprof, "enabled", lambda: False)
    monkeypatch.setattr(agprof, "start", lambda out_dir: events.append("start"))
    monkeypatch.setattr(agprof, "stop", lambda: events.append("stop"))

    with pytest.raises(RuntimeError, match="boom"):
        with agprof.workload():
            events.append("workload")
            raise RuntimeError("boom")

    assert events == ["start", "workload", "stop"]


@pytest.mark.parametrize(
    ("profile_value", "scope"),
    [
        ("", "workload"),
        ("0", "workload"),
        ("1", "process"),
    ],
)
def test_workload_context_does_not_own_other_lifecycles(monkeypatch, profile_value, scope):
    monkeypatch.setenv("AGENCY_PROFILE", profile_value)
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", scope)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("workload context must not start profiling"),
    )
    monkeypatch.setattr(
        agprof,
        "stop",
        lambda: pytest.fail("workload context must not stop profiling"),
    )

    with agprof.workload():
        pass


def test_workload_context_preserves_explicit_active_session(monkeypatch):
    existing = object()
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "workload")
    monkeypatch.setattr(agprof, "enabled", lambda: True)
    monkeypatch.setattr(agprof, "_profiler", existing)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("an explicit session is already active"),
    )
    monkeypatch.setattr(
        agprof,
        "stop",
        lambda: pytest.fail("must not stop an explicit session"),
    )

    with agprof.workload() as active:
        assert active is existing


def test_process_scope_is_the_only_environment_autostart(monkeypatch):
    events = []
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "process")
    monkeypatch.setenv("AGENCY_PROFILE_DIR", "process-trace")
    monkeypatch.setattr(agprof, "start", lambda out_dir: events.append(("start", out_dir)))
    monkeypatch.setattr(agprof.atexit, "register", lambda fn: events.append(("register", fn)))

    agprof._maybe_autostart()

    assert events == [("start", "process-trace"), ("register", agprof.stop)]


@pytest.mark.parametrize("scope", [None, "workload", "invalid"])
def test_default_and_invalid_scopes_do_not_autostart(monkeypatch, scope):
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    if scope is None:
        monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    else:
        monkeypatch.setenv("AGENCY_PROFILE_SCOPE", scope)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("non-process scope must not autostart"),
    )
    monkeypatch.setattr(
        agprof.atexit,
        "register",
        lambda fn: pytest.fail("non-process scope must not register process cleanup"),
    )

    agprof._maybe_autostart()


def test_webui_marks_only_supplied_function_as_workload(monkeypatch, tmp_path):
    import agency.agwebui as agwebui_module

    events = []

    @contextmanager
    def workload():
        events.append("profile-start")
        try:
            yield
        finally:
            events.append("profile-stop")

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            events.append("server-stop")

        def wait(self, timeout):
            return 0

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def setsockopt(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(agwebui_module.agprof, "workload", workload)
    monkeypatch.setattr(agwebui_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(agwebui_module.urllib.request, "urlopen", lambda *args, **kwargs: None)
    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSocket())
    monkeypatch.setattr(agwebui_module.atexit, "register", lambda fn: None)

    def fn():
        events.append("workload")

    agwebui_module.agwebui.run(fn, run_dir=tmp_path, port=17860, linger=False)

    assert events[:3] == ["profile-start", "workload", "profile-stop"]
