import json
import pytest
from src.agdata import agdata


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

