import asyncio
import json
from dataclasses import dataclass
from concurrent.futures import Future
import pytest

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
# agerror construction
# ---------------------------------------------------------------------------


def test_non_string_agerror_raises():
    """agerror only accepts str — passing a type or non-str raises TypeError."""
    with pytest.raises(TypeError):
        agerror(str)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        agerror(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        agerror(None)  # type: ignore[arg-type]


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


def _pending(value):
    future = Future()
    future.set_result(value)
    return agdata(_future=future)


def test_wait_all_and_recursive_dependency_resolution_accept_pending_data():
    first = _pending(agdata(answer=1))
    second = _pending(agdata(answer=2))
    values = [first, second]
    assert agdata.wait_all(values) is values

    nested = agdata(items=[first, (second,), {"again": first}])
    nested.resolve_input_dependencies()
    assert nested.items[0].answer == 1
    assert nested.items[1][0].answer == 2
    assert nested.items[2]["again"].answer == 1


@dataclass
class _StructuredValue:
    result: object


class _ModelValue:
    def __init__(self, result):
        self.result = result

    def model_dump(self):
        return {"result": self.result}


def test_nested_serialization_materializes_pending_data_tuples_and_models():
    wrapped = _pending(agdata(result="literal"))
    value = agdata(
        tuple_value=(wrapped,),
        dataclass_value=_StructuredValue(wrapped),
        model_value=_ModelValue(wrapped),
    )
    assert value.to_dict() == {
        "tuple_value": [{"result": "literal"}],
        "dataclass_value": {"result": {"result": "literal"}},
        "model_value": {"result": {"result": "literal"}},
    }


def test_wait_all_rejects_non_waitable_values():
    with pytest.raises(TypeError, match="not waitable"):
        agdata.wait_all([object()])


def test_wait_with_timeout_raises_when_not_yet_resolved():
    future: "Future[agdata]" = Future()
    pending = agdata(_future=future)
    with pytest.raises(TimeoutError):
        pending.wait(timeout=0.05)
    assert pending.is_pending()


def test_wait_with_timeout_returns_self_once_resolved():
    future: "Future[agdata]" = Future()
    pending = agdata(_future=future)
    future.set_result(agdata(answer="done"))
    assert pending.wait(timeout=1) is pending
    assert pending.answer == "done"


async def _await(value):
    return await value


def test_bare_agdata_is_awaitable_and_field_proxies_after_resolution():
    future: "Future[agdata]" = Future()
    pending = agdata(_future=future)
    future.set_result(agdata(result="literal", other=42))

    resolved = asyncio.run(_await(pending))
    assert resolved is pending
    assert pending.other == 42
    assert pending.result == "literal"
    assert pending.to_dict() == {"result": "literal", "other": 42}


def test_cancelling_one_async_waiter_does_not_cancel_the_shared_future():
    future: "Future[agdata]" = Future()
    pending = agdata(_future=future)

    async def scenario():
        waiter = asyncio.ensure_future(pending)
        await asyncio.sleep(0)
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("expected the shielded waiter task to be cancelled")
        future.set_result(agdata(answer="still running"))
        return await pending

    assert asyncio.run(scenario()).answer == "still running"
    assert future.cancelled() is False
