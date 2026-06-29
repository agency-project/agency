from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future
    from .agsandbox import agSandbox

from .agtype import agtype, agfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum length of a string field that will be auto-offloaded to a sandbox file.
INPUT_OFFLOAD_CHARS: int = 40_000

class AgError(RuntimeError):
    """Raised when accessing a non-error field on an agerror instance."""


class agdata:
    """Generic data container with JSON/dict serialization.

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
        return cls(**json.loads(s))

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        self._resolve()
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        available = list(data.keys())
        raise AttributeError(
            f"agdata has no field {name!r}. Available fields: {available}"
        )

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

    def check_schema(self, schema: "agdata") -> list[str]:
        """Return a list of error strings; empty list means the data is valid."""
        errors: list[str] = []
        for key, hint in schema._data.items():
            if key not in self._data:
                errors.append(f"missing required field '{key}'")
                continue
            actual = self._data[key]
            if isinstance(hint, type) and issubclass(hint, agtype):
                if not isinstance(actual, str):
                    errors.append(f"field '{key}' ({hint.__name__}) must be a string")
                continue
            if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
                item_template = hint[0]
                if not isinstance(actual, list):
                    errors.append(f"field '{key}': expected list, got {type(actual).__name__}")
                    continue
                for i, item in enumerate(actual):
                    if not isinstance(item, dict):
                        errors.append(f"field '{key}[{i}]': expected dict, got {type(item).__name__}")
                        continue
                    for item_key, item_type in item_template.items():
                        if item_key not in item:
                            errors.append(f"field '{key}[{i}]': missing key '{item_key}'")
                        elif isinstance(item_type, type) and not isinstance(item[item_key], item_type):
                            errors.append(
                                f"field '{key}[{i}].{item_key}': expected {item_type.__name__}, "
                                f"got {type(item[item_key]).__name__}"
                            )
                continue
            if isinstance(hint, type):
                if not isinstance(actual, hint):
                    errors.append(
                        f"field '{key}': expected {hint.__name__}, got {type(actual).__name__}"
                    )
        return errors

    def validate_input(
        self,
        input_schema: "agdata | None",
        _is_continuation: bool,
    ) -> "str | None":
        """Return an error string if input fails schema validation, else None."""
        if input_schema is None or _is_continuation:
            return None
        errors = self.check_schema(input_schema)
        if errors:
            return f"input schema error: {errors}"
        return None

    def offload_large_fields(
        self,
        sandbox: "agSandbox",
        skill_name: str,
        schema: "agdata | None" = None,
        suffix: str = "",
        context_limit: "int | None" = None,
    ) -> "tuple[list[str], list[str]]":
        """Write oversized string fields to /workspace/inputs/ in the sandbox.

        Called after agtype fields have already been prepared (so agfile inputs are
        already short file paths).  Each remaining field whose string value still
        exceeds _offload_threshold(context_limit) is replaced in-place with a short
        reference.  Returns (paths_written, field_names) so the caller can delete
        files and build an auto-offload note for the system prompt.

        Fields already managed by an agtype subclass (e.g. agimage data URLs,
        agfile/agbinary paths) are skipped — their prepared values must not be
        replaced by sandbox file references.  agrawstring is the exception: its
        prepare() is a no-op, so a long value arrives here at full length and
        should be offloaded like any plain string.
        """
        agtype_keys: set[str] = set()
        if schema is not None:
            for key, hint in schema._data.items():
                if agtype.in_hint(hint):
                    agtype_keys.add(key)

        _threshold = min(INPUT_OFFLOAD_CHARS, int(context_limit * 0.1 * 4)) if context_limit else INPUT_OFFLOAD_CHARS
        paths: list[str] = []
        fields: list[str] = []
        for key, val in list(self._data.items()):
            if key in agtype_keys:
                continue
            if isinstance(val, str):
                if len(val) <= _threshold:
                    continue
                path = f"/workspace/inputs/{skill_name}_{key}{suffix}.txt"
                try:
                    sandbox.write_file(path, val)
                    self._data[key] = (
                        f"(content saved to {path} — use the read tool to access it)"
                    )
                    paths.append(path)
                    fields.append(key)
                except Exception as _e:
                    print(f"[agent] WARNING: failed to offload input field '{key}' to {path}: {_e}")
            elif isinstance(val, list):
                new_vals = list(val)
                offloaded_any = False
                for i, item in enumerate(val):
                    if not isinstance(item, str) or len(item) <= _threshold:
                        continue
                    path = f"/workspace/inputs/{skill_name}_{key}_{i}{suffix}.txt"
                    try:
                        sandbox.write_file(path, item)
                        new_vals[i] = path
                        paths.append(path)
                        offloaded_any = True
                    except Exception as _e:
                        print(f"[agent] WARNING: failed to offload input list field '{key}[{i}]' to {path}: {_e}")
                if offloaded_any:
                    self._data[key] = new_vals
                    fields.append(key)
        return paths, fields

    def prepare_agtype_inputs(
        self,
        schema: "agdata | None",
        sandbox: "agSandbox",
        skill_name: str,
        suffix: str = "",
    ) -> list[str]:
        """Prepare agtype input fields before the skill runs.

        Recursively handles agtype subclasses nested inside list, dict, and tuple
        containers at any depth.  Calls ``hint.prepare()`` at each agtype leaf.
        Returns all sandbox paths written for cleanup.
        """
        if schema is None:
            return []
        paths: list[str] = []
        for key, hint in schema._data.items():
            def on_leaf(h, v, _key=key):
                try:
                    return h.prepare(v, sandbox, skill_name, _key, suffix=suffix)
                except Exception as _e:
                    print(f"[agent] WARNING: {h.__name__}.prepare failed for field '{_key}': {_e}")
                    return v, []
            new_val, written = agtype.walk(hint, self._data.get(key), on_leaf)
            if written or new_val is not self._data.get(key):
                self._data[key] = new_val
            paths.extend(written)
        return paths

    def recover_agtype_outputs(
        self,
        schema: "agdata | None",
        sandbox: "agSandbox",
    ) -> list[str]:
        """Recover agtype output fields after the skill finishes.

        Recursively handles agtype subclasses nested inside list, dict, and tuple
        containers at any depth.  Calls ``hint.recover()`` at each agtype leaf.
        Returns all sandbox paths for cleanup.
        """
        if schema is None or isinstance(self, agerror):
            return []
        paths: list[str] = []
        for key, hint in schema._data.items():
            def on_leaf(h, v, _key=key):
                try:
                    return h.recover(v, sandbox)
                except Exception as _e:
                    print(f"[agent] WARNING: {h.__name__}.recover failed for field '{_key}': {_e}")
                    return v, []
            new_val, written = agtype.walk(hint, self._data.get(key), on_leaf)
            if written or new_val is not self._data.get(key):
                self._data[key] = new_val
            paths.extend(written)
        return paths

    def resolve_input(self) -> None:
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
            raise TypeError(
                f"agerror message must be a str, got {type(message).__name__}"
            )
        object.__setattr__(self, "_future", None)
        object.__setattr__(self, "_data", {"error": message})
        from .agterm import agterm as _agterm_cls
        if not hasattr(agerror, "_term"):
            agerror._term = _agterm_cls("agdata")
        agerror._term.log("ERROR ✗  ", message, depth=2)

    def __getattr__(self, name: str):
        if name == "error":
            return object.__getattribute__(self, "_data")["error"]
        raise AgError(object.__getattribute__(self, "_data")["error"])

    def __repr__(self) -> str:
        return f"agerror({self._data.get('error')!r})"
