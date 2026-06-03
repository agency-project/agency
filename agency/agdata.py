from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future


class AgError(RuntimeError):
    """Raised when accessing a result field on an agdata that holds a skill error.

    Use ``agdata.error`` (a property) to read the error message without raising,
    or ``agdata.is_error()`` for an explicit boolean check.
    """


class agdata:
    """Generic data container with JSON/dict serialization.

    An agdata may be in a *pending* state when created by agent.submit().
    In that case it wraps a Future[agdata] internally.  Any field access or
    serialization call automatically blocks until the future resolves and
    populates _data.  Use agdata.is_pending() to check without blocking.
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
        resolved = f.result()
        object.__setattr__(self, "_data", object.__getattribute__(resolved, "_data"))
        object.__setattr__(self, "_future", None)

    def is_pending(self) -> bool:
        """Return True if this agdata is still waiting for a future result."""
        f = object.__getattribute__(self, "_future")
        return f is not None and not f.done()

    def is_error(self) -> bool:
        """Return True if this agdata holds a skill error (without raising)."""
        self._resolve()
        return "error" in object.__getattribute__(self, "_data")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_serializable(obj):
        """Recursively convert agdata objects (including nested ones) to plain types."""
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
        return cls(**json.loads(s))

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    @property
    def error(self) -> "str | None":
        """Return the error message without raising, or None if healthy.

        This is the one safe accessor on a failed agdata — all other field
        accesses raise AgError.  Use ``is_error()`` for a boolean check.
        """
        self._resolve()
        return object.__getattribute__(self, "_data").get("error")

    def __getattr__(self, name: str):
        self._resolve()
        data = object.__getattribute__(self, "_data")
        if "error" in data:
            raise AgError(data["error"])
        if name in data:
            return data[name]
        raise AttributeError(name)

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
