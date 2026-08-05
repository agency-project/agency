"""Minimal, standalone-loadable stand-in for `agdata`/`agerror`, used only to
call a host-authored `add_tools`/`replace_tools` closure (`fn: Callable[[agdata],
agdata]`) after it's been shipped into the container via cloudpickle (see
`_native_in_container_entrypoint.py`'s `_make_custom_tool_handler`).

Same convention as `agtool_pure.py`/`agllm_pure.py`: zero imports beyond
stdlib (`json` only), zero relative imports, loaded by raw file path
(`importlib.util.spec_from_file_location`) so the entrypoint never needs to
`import agency` (which would pull in `agency/__init__.py`'s eager
`openai`/`anthropic`/`boto3` chain -- see that module's own docstring).

**Not a full drop-in for `agency.agdata.agdata`.** A tool call's argument is
always already-resolved JSON by the time it reaches this shim -- there is no
async/pending state to model inside this synchronous, in-container loop --
so this deliberately omits `agdata`'s `Future`-wrapping, `agpause`
integration, and `resolve_input_dependencies`/`wait`/`wait_all`. Only what a
typical tool `fn` body actually uses is here: construction from kwargs,
attribute get/set, dict/JSON (de)serialization, equality, and `agerror`'s
error-only-field contract.

Why this is enough even though the shipped closure was authored against the
*real* `agdata`/`agerror` classes: pickle resolves any class reference by
`(cls.__module__, cls.__qualname__)`, which for both real classes is always
`("agency.agdata", "agdata")` / `("agency.agdata", "agerror")` regardless of
how the user's code imported them. The entrypoint pre-registers a
`sys.modules["agency.agdata"]` stub exposing *this* module's classes under
those same names before unpickling, so every such reference resolves here
transparently, without touching the shipped closure's bytecode at all.
"""

from __future__ import annotations

import json


class AgError(RuntimeError):
    """Raised when accessing a non-error field on an agerror instance."""


class agdata:
    def __init__(self, **data):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        available = list(data.keys())
        raise AttributeError(f"agdata has no field {name!r}. Available fields: {available}")

    def __setattr__(self, name: str, value) -> None:
        object.__getattribute__(self, "_data")[name] = value

    def to_dict(self) -> dict:
        return dict(object.__getattribute__(self, "_data"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "agdata":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "agdata":
        return cls(**json.loads(s))

    def __eq__(self, other) -> bool:
        if isinstance(other, agdata):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def __repr__(self) -> str:
        return f"agdata({self.to_dict()!r})"


class agerror(agdata):
    def __init__(self, message: str):
        if not isinstance(message, str):
            raise TypeError(f"agerror message must be a str, got {type(message).__name__}")
        object.__setattr__(self, "_data", {"error": message})

    def __getattr__(self, name: str):
        if name == "error":
            return object.__getattribute__(self, "_data")["error"]
        raise AgError(object.__getattribute__(self, "_data")["error"])

    def __repr__(self) -> str:
        return f"agerror({self.to_dict().get('error')!r})"
