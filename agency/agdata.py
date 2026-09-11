from __future__ import annotations
import asyncio
import json
from dataclasses import fields, is_dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future

from .observability.profiler import agprof
from .agtype import agtype
from .utils.agutil import _camel_to_snake


class AgError(RuntimeError):
    """Raised when accessing a non-error field on an agerror instance."""


class agdata:
    """Generic data container with dict serialization.

    An agdata may be in a *pending* state when created by agent.run() or
    agteam.run().  In that case it wraps a Future[agdata] internally.  Any
    field access or serialization call automatically blocks until the future
    resolves and populates _data.  Use agdata.is_pending() to check without
    blocking.
    """

    def __init__(self, _future: "Future[agdata] | None" = None, **data):
        object.__setattr__(self, "_future", _future)
        object.__setattr__(self, "_data", data)

    # ------------------------------------------------------------------
    # Pending-state resolution
    # ------------------------------------------------------------------

    def _resolve(self) -> None:
        """Block until the internal future resolves, then populate _data."""
        f = object.__getattribute__(self, "_future")
        if f is None:
            return
        if f.done():
            resolved = f.result()
        else:
            with agprof.span("sync:result_wait"):
                resolved = f.result()
        if not isinstance(resolved, agdata):
            as_pending = getattr(resolved, "_as_pending_agdata", None)
            if not callable(as_pending):
                raise TypeError(
                    "pending agdata future resolved to an incompatible value: "
                    f"{type(resolved).__name__}"
                )
            resolved = as_pending()
        resolved._resolve()  # chain: future may resolve to another pending agdata
        object.__setattr__(self, "_data", object.__getattribute__(resolved, "_data"))
        object.__setattr__(self, "_future", None)

    def _as_pending_agdata(self) -> "agdata":
        """Return the common pending-data representation used by the scheduler."""
        return self

    def is_pending(self) -> bool:
        """Return True if this agdata is still waiting for a future result."""
        f = object.__getattribute__(self, "_future")
        return f is not None and not f.done()

    def wait(self, timeout: "float | None" = None) -> "agdata":
        """Block until this agdata is resolved and return self. *timeout*,
        if given, raises ``TimeoutError`` rather than blocking forever."""
        if timeout is not None:
            f = object.__getattribute__(self, "_future")
            if f is not None:
                f.result(timeout=timeout)
        self._resolve()
        return self

    def __await__(self):
        return self._await_self().__await__()

    async def _await_self(self) -> "agdata":
        # Shielded so cancelling one waiter's task never cancels the shared
        # underlying future for any other concurrent awaiter.
        f = object.__getattribute__(self, "_future")
        if f is not None:
            await asyncio.shield(asyncio.wrap_future(f))
        self._resolve()
        return self

    @staticmethod
    def wait_all(pending: "list") -> "list":
        """Block until every agdata in *pending* is resolved.

        Use as a barrier over a fan-out::

            teams   = [MyTeam(topic=t) for t in topics]
            results = [t.run() for t in teams]
            # ... do other work ...
            agdata.wait_all(results)    # barrier — wait for all teams
            for r in results:
                print(r.report_path)   # all resolved, no further blocking
        """
        for p in pending:
            resolver = getattr(p, "_resolve", None)
            if callable(resolver):
                resolver()
                continue
            waiter = getattr(p, "wait", None)
            if not callable(waiter):
                raise TypeError(f"object is not waitable: {type(p).__name__}")
            waiter()
        return pending

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_serializable(obj):
        """Recursively convert agdata objects (including nested ones) to plain types."""
        if not isinstance(obj, agdata):
            as_pending = getattr(obj, "_as_pending_agdata", None)
            if callable(as_pending):
                obj = as_pending()
        if isinstance(obj, type) and issubclass(obj, agtype):
            return obj.schema_type()
        if isinstance(obj, type):
            return obj.__name__
        # Handle generic aliases like list[agimage] → "list[image]"
        import typing

        origin = typing.get_origin(obj)
        if origin is list:
            args = typing.get_args(obj)
            if args:
                inner = agdata._to_serializable(args[0])
                return f"list[{inner}]"
        if isinstance(obj, agdata):
            obj._resolve()
            return {k: agdata._to_serializable(v) for k, v in obj._data.items()}
        if isinstance(obj, dict):
            return {k: agdata._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [agdata._to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [agdata._to_serializable(v) for v in obj]
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                field.name: agdata._to_serializable(getattr(obj, field.name))
                for field in fields(obj)
            }
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            return agdata._to_serializable(model_dump())
        return obj

    def to_dict(self) -> dict:
        self._resolve()
        return {k: agdata._to_serializable(v) for k, v in self._data.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "agdata":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "agdata":
        """Parse a JSON object into an agdata, tolerating camelCase keys
        (some LLMs emit tool-call args in camelCase)."""
        return cls(**{_camel_to_snake(k): v for k, v in json.loads(s).items()})

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        self._resolve()
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        available = list(data.keys())
        raise AttributeError(f"agdata has no field {name!r}. Available fields: {available}")

    def __setattr__(self, name: str, value):
        object.__getattribute__(self, "_data")[name] = value

    def __repr__(self) -> str:
        f = object.__getattribute__(self, "_future")
        if f is not None and not f.done():
            return "agdata(<pending>)"
        self._resolve()
        return f"agdata({self._data!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, agdata):
            self._resolve()
            other._resolve()
            return self._data == other._data
        return NotImplemented

    @staticmethod
    def _resolve_dependency(value):
        if not isinstance(value, agdata):
            as_pending = getattr(value, "_as_pending_agdata", None)
            if callable(as_pending):
                value = as_pending()
        if isinstance(value, agdata):
            value._resolve()
            return value
        if isinstance(value, list):
            return [agdata._resolve_dependency(item) for item in value]
        if isinstance(value, tuple):
            return tuple(agdata._resolve_dependency(item) for item in value)
        if isinstance(value, dict):
            return {key: agdata._resolve_dependency(item) for key, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            return replace(
                value,
                **{
                    field.name: agdata._resolve_dependency(getattr(value, field.name))
                    for field in fields(value)
                },
            )
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if not isinstance(dumped, dict):
                raise TypeError(f"{type(value).__name__}.model_dump() must return a dictionary")
            updates = {key: agdata._resolve_dependency(item) for key, item in dumped.items()}
            model_copy = getattr(value, "model_copy", None)
            if callable(model_copy):
                return model_copy(update=updates)
            try:
                for key, item in updates.items():
                    setattr(value, key, item)
            except (AttributeError, TypeError):
                # Generic immutable model-like values have no standard copy
                # protocol.  Preserve their resolved data rather than leaving
                # hidden pending handles behind.
                return updates
            return value
        return value

    def resolve_input_dependencies(self) -> None:
        """Resolve pending data/result handles nested inside self, in-place."""
        self._resolve()
        for key, value in tuple(self._data.items()):
            self._data[key] = self._resolve_dependency(value)


class agerror(agdata):
    """Returned by skills and tools to signal failure.

    Usage::

        return agerror("context limit exceeded")

    Callers check with ``isinstance(result, agerror)``.
    Accessing any field other than ``.error`` raises AgError.
    """

    def __init__(self, message: str):
        if not isinstance(message, str):
            raise TypeError(f"agerror message must be a str, got {type(message).__name__}")
        object.__setattr__(self, "_future", None)
        object.__setattr__(self, "_data", {"error": message})

    def __getattr__(self, name: str):
        if name == "error":
            return object.__getattribute__(self, "_data")["error"]
        raise AgError(object.__getattribute__(self, "_data")["error"])

    def __repr__(self) -> str:
        return f"agerror({self._data.get('error')!r})"


class agcanceled(agerror):
    """Returned when an invocation was cancelled via ``agent.cancel(handle)``.

    A typed subclass of ``agerror`` so callers can ``isinstance(result, agcanceled)``
    instead of string-matching ``.error``.
    """

    def __init__(self, message: str = "agent invocation cancelled"):
        super().__init__(message)
