from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future

from . import agpause
from .profiler import agprof
from .agtype import agtype
from .agutil import _camel_to_snake


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
        with agpause.note_blocked_on(agpause.producer_of(f)):
            if f.done():
                resolved = f.result()
            else:
                with agprof.span("sync:result_wait"):
                    resolved = f.result()
        resolved._resolve()  # chain: future may resolve to another pending agdata
        object.__setattr__(self, "_data", object.__getattribute__(resolved, "_data"))
        object.__setattr__(self, "_future", None)

    def is_pending(self) -> bool:
        """Return True if this agdata is still waiting for a future result."""
        f = object.__getattribute__(self, "_future")
        return f is not None and not f.done()

    def wait(self) -> "agdata":
        """Block until this agdata is resolved and return self.

        Use as a barrier on a single result::

            result = team.run()
            # ... do other work ...
            result.wait()   # block here until the team finishes
            print(result.report_path)   # guaranteed resolved
        """
        self._resolve()
        return self

    @staticmethod
    def wait_all(pending: "list[agdata]") -> "list[agdata]":
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
            p._resolve()
        return pending

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_serializable(obj):
        """Recursively convert agdata objects (including nested ones) to plain types."""
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
        """Parse a JSON object into an agdata, tolerating camelCase keys --
        some LLMs emit tool-call arguments in camelCase even when a tool's
        schema declares snake_case parameter names. Normalizes only
        top-level keys; nested dict/list values are left untouched."""
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

    def resolve_input_dependencies(self) -> None:
        """Resolve any pending agdata values nested inside self, in-place."""
        self._resolve()
        for val in self._data.values():
            if isinstance(val, agdata):
                val._resolve()
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, agdata):
                        item._resolve()


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
        from .agterm import agterm as _agterm_cls

        if not hasattr(agerror, "_term"):
            agerror._term = _agterm_cls("agerror")
        agerror._term.log("ERROR ✗  ", message, depth=2)

    def __getattr__(self, name: str):
        if name == "error":
            return object.__getattribute__(self, "_data")["error"]
        raise AgError(object.__getattribute__(self, "_data")["error"])

    def __repr__(self) -> str:
        return f"agerror({self._data.get('error')!r})"
