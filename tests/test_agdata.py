import io
import json
import pytest
from unittest.mock import MagicMock, patch

import agency.agwebui as _agwebui_mod
from agency.agdata import agdata, agerror


def test_init_and_dot_access():
    d = agdata(x=1, y="hello")
    assert d.x == 1
    assert d.y == "hello"


def test_to_dict():
    d = agdata(a=1, b=[1, 2])
    assert d.to_dict() == {"a": 1, "b": [1, 2]}


def test_to_json():
    d = agdata(a=1)
    parsed = json.loads(d.to_json())
    assert parsed == {"a": 1}


def test_from_dict():
    d = agdata.from_dict({"x": 10, "y": 20})
    assert d.x == 10
    assert d.y == 20


def test_from_json():
    d = agdata.from_json('{"name": "test", "value": 42}')
    assert d.name == "test"
    assert d.value == 42


def test_from_json_normalizes_camel_case_keys():
    """Some LLMs emit tool-call arguments in camelCase even when the tool
    schema declares snake_case params -- from_json() tolerates this."""
    d = agdata.from_json(
        '{"filePath": "/tmp/x.txt", "oldString": "a", "newString": "b", "replaceAll": true}'
    )
    assert d.file_path == "/tmp/x.txt"
    assert d.old_string == "a"
    assert d.new_string == "b"
    assert d.replace_all is True


def test_from_json_snake_case_keys_are_unaffected():
    d = agdata.from_json('{"file_path": "/tmp/y.txt", "command": "ls"}')
    assert d.file_path == "/tmp/y.txt"
    assert d.command == "ls"


def test_from_json_does_not_normalize_nested_keys():
    """Normalization is shallow (top-level only) -- nested dict/list values
    are argument *data*, not argument *names*, and must pass through as-is."""
    d = agdata.from_json('{"todos": [{"someKey": "value"}]}')
    assert d.todos == [{"someKey": "value"}]


def test_roundtrip_json():
    original = agdata(items=[1, 2, 3], nested={"a": "b"})
    restored = agdata.from_json(original.to_json())
    assert restored.to_dict() == original.to_dict()


def test_roundtrip_dict():
    original = agdata(flag=True, count=0)
    restored = agdata.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_setattr():
    d = agdata(x=1)
    d.x = 99
    assert d.x == 99


def test_missing_attr_raises():
    d = agdata(x=1)
    with pytest.raises(AttributeError):
        _ = d.nonexistent


def test_equality():
    assert agdata(a=1) == agdata(a=1)
    assert agdata(a=1) != agdata(a=2)


def test_empty():
    d = agdata()
    assert d.to_dict() == {}
    assert d.to_json() == "{}"
    restored = agdata.from_json("{}")
    assert restored.to_dict() == {}


def test_messages_pattern():
    """History pattern used by agent."""
    h = agdata(messages=[])
    h.messages.append({"role": "user", "content": "hi"})
    assert len(h.messages) == 1


# ---------------------------------------------------------------------------
# Pending state
# ---------------------------------------------------------------------------


def test_pending_agdata_resolves_on_field_access():
    from concurrent.futures import Future

    f: Future[agdata] = Future()
    pending = agdata(_future=f)
    assert pending.is_pending() is True

    f.set_result(agdata(answer=42))
    assert pending.answer == 42
    assert pending.is_pending() is False


def test_pending_agdata_resolves_on_to_dict():
    from concurrent.futures import Future

    f: Future[agdata] = Future()
    f.set_result(agdata(x=1, y=2))
    pending = agdata(_future=f)
    assert pending.to_dict() == {"x": 1, "y": 2}


def test_pending_agdata_resolves_on_to_json():
    from concurrent.futures import Future

    f: Future[agdata] = Future()
    f.set_result(agdata(val="hello"))
    pending = agdata(_future=f)
    import json as _json

    assert _json.loads(pending.to_json()) == {"val": "hello"}


# ---------------------------------------------------------------------------
# Error emission
# ---------------------------------------------------------------------------


class TestErrorEmission:
    def test_error_string_prints_to_stderr(self):
        """agerror('...') emits to stderr immediately at construction."""
        buf = io.StringIO()
        with patch("agency.agterm.sys.stderr", buf):
            agerror("something went wrong")
        assert "something went wrong" in buf.getvalue()

    def test_error_string_contains_timestamp(self):
        """Emitted line includes a HH:MM:SS timestamp."""
        import re

        buf = io.StringIO()
        with patch("agency.agterm.sys.stderr", buf):
            agerror("ts check")
        assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", buf.getvalue())

    def test_non_error_agdata_does_not_print(self):
        """Normal agdata with no error field emits nothing."""
        buf = io.StringIO()
        with patch("agency.agterm.sys.stderr", buf):
            agdata(x=1, y="hello")
        assert buf.getvalue() == ""

    def test_non_string_agerror_raises(self):
        """agerror only accepts str — passing a type or non-str raises TypeError."""
        with pytest.raises(TypeError):
            agerror(str)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            agerror(42)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            agerror(None)  # type: ignore[arg-type]

    def test_error_emitted_to_webui_when_active(self):
        """When webui is active, the error line is sent to emitter.log via agterm."""
        mock_webui = MagicMock()
        mock_webui.emitter.log = MagicMock()
        old = _agwebui_mod._active
        _agwebui_mod._active = mock_webui
        try:
            agerror("webui error")
        finally:
            _agwebui_mod._active = old
        calls = [str(c) for c in mock_webui.emitter.log.call_args_list]
        assert any("webui error" in c for c in calls)

    def test_error_always_goes_to_stderr_even_with_webui(self):
        """Even when webui is active, errors (✗ events) still appear on stderr."""
        mock_webui = MagicMock()
        mock_webui.emitter.log = MagicMock()
        old = _agwebui_mod._active
        _agwebui_mod._active = mock_webui
        try:
            buf = io.StringIO()
            with patch("agency.agterm.sys.stderr", buf):
                agerror("always stderr")
        finally:
            _agwebui_mod._active = old
        assert "always stderr" in buf.getvalue()

    def test_multiple_errors_each_emit_once(self):
        """Each separate agerror(...) emits exactly one line."""
        buf = io.StringIO()
        with patch("agency.agterm.sys.stderr", buf):
            agerror("err A")
            agerror("err B")
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert sum("err A" in l for l in lines) == 1
        assert sum("err B" in l for l in lines) == 1


def test_pending_repr_before_resolution():
    from concurrent.futures import Future

    f: Future[agdata] = Future()
    pending = agdata(_future=f)
    assert "pending" in repr(pending)


def test_pending_repr_after_resolution():
    from concurrent.futures import Future

    f: Future[agdata] = Future()
    f.set_result(agdata(x=99))
    pending = agdata(_future=f)
    _ = pending.x  # trigger resolution
    assert "pending" not in repr(pending)


def test_pending_equality_resolves_both():
    from concurrent.futures import Future

    f1: Future[agdata] = Future()
    f2: Future[agdata] = Future()
    f1.set_result(agdata(v=1))
    f2.set_result(agdata(v=1))
    assert agdata(_future=f1) == agdata(_future=f2)


def test_normal_agdata_pending_is_false():
    d = agdata(x=1)
    assert d.is_pending() is False
