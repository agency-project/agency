from __future__ import annotations
import base64
import json
import mimetypes
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Callable, get_args, get_origin
from .agutil import _looks_like_path

if TYPE_CHECKING:
    from .agsandbox import agSandbox


class agtype:
    """Base class for typed agdata field values.

    Subclass to add custom serialization, sandbox I/O, system prompts, or
    cleanup logic for a field type.  The default implementations pass plain
    Python values through unchanged.

    Class-level use
    ---------------
    An ``agtype`` subclass is used as a *type marker* in agskill schemas::

        skill = agskill(
            name="write",
            system_prompt="...",
            input_schema=agdata(theme=str, background=agfile),
            output_schema=agdata(report=agfile),
        )

    The class object itself (not an instance) is stored in the schema agdata.
    All interface methods are therefore ``@classmethod`` so they can be called
    on the class without instantiation.

    Interface methods
    -----------------
    schema_type() -> str
        Human-readable type label shown in the JSON format hint in the system
        prompt.  Default: ``"str"``.

    needs_sandbox() -> bool
        True if this type requires sandbox filesystem access during
        prepare/recover.  Default: ``False``.

    prepare(value, sandbox, skill_name, field_name) -> tuple[str, list[str]]
        Called before the skill's ReAct loop.  ``value`` is the raw Python
        value from agdata.  Returns (transformed_value, paths_to_cleanup).
        ``transformed_value`` replaces the field in the JSON sent to the LLM.
        Default: returns (value, []).

    recover(value, sandbox) -> tuple[str, list[str]]
        Called after the skill's ReAct loop on output fields.  ``value`` is
        whatever the LLM returned for this field.  Returns
        (recovered_value, paths_to_cleanup).  Default: returns (value, []).

    extra_input_prompt(field_name) -> str
        Additional instruction line added to the system prompt for this input
        field.  Return an empty string to add nothing.  Default: ``""``.

    extra_output_prompt(field_name, skill_name) -> str
        Additional instruction line added to the system prompt for this output
        field.  Return an empty string to add nothing.  Default: ``""``.
    """

    @classmethod
    def schema_type(cls) -> str:
        return "str"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[object, list[str]]:
        return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[object, list[str]]:
        return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return ""

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return ""

    @classmethod
    def return_tool_description(cls, field_name: str) -> str:
        """Description for the return_<field> tool itself."""
        return f"Register the '{field_name}' output field."

    @classmethod
    def return_value_description(cls, field_name: str) -> str:
        """Description for the `value` parameter of return_<field>."""
        return f"Value for '{field_name}'"

    @staticmethod
    def walk(hint, value, on_leaf):
        """Recursively walk a type hint/value pair, calling on_leaf(hint, value)
        at every agtype leaf.  Returns (new_value, paths).

        Handles list, dict, and tuple containers at any nesting depth.
        on_leaf must handle its own exceptions and always return (value, []).
        """
        origin = get_origin(hint)
        args   = get_args(hint)

        if isinstance(hint, type) and issubclass(hint, agtype):
            return on_leaf(hint, value)

        if origin is list and args:
            if not isinstance(value, list):
                return value, []
            new_vals, paths = [], []
            for v in value:
                nv, written = agtype.walk(args[0], v, on_leaf)
                new_vals.append(nv)
                paths.extend(written)
            return new_vals, paths

        if origin is dict and len(args) == 2:
            if not isinstance(value, dict):
                return value, []
            new_vals, paths = {}, []
            for k, v in value.items():
                nv, written = agtype.walk(args[1], v, on_leaf)
                new_vals[k] = nv
                paths.extend(written)
            return new_vals, paths

        if origin is tuple and args:
            if not isinstance(value, (list, tuple)):
                return value, []
            new_vals, paths = list(value), []
            for i, (type_arg, v) in enumerate(zip(args, value)):
                nv, written = agtype.walk(type_arg, v, on_leaf)
                new_vals[i] = nv
                paths.extend(written)
            return new_vals, paths

        return value, []

    @staticmethod
    def in_hint(hint) -> bool:
        """Return True if hint contains any agtype subclass other than agrawstring,
        at any nesting depth."""
        if isinstance(hint, type) and issubclass(hint, agtype):
            return not issubclass(hint, agrawstring)
        origin = get_origin(hint)
        args   = get_args(hint)
        if origin is list and args:
            return agtype.in_hint(args[0])
        if origin is dict and len(args) == 2:
            return agtype.in_hint(args[1])
        if origin is tuple and args:
            return any(agtype.in_hint(a) for a in args)
        return False

    @staticmethod
    def from_hint(hint: object) -> "type[agtype] | None":
        """Return the agtype subclass for a hint, handling both T and list[T]."""
        if isinstance(hint, type) and issubclass(hint, agtype):
            return hint
        if get_origin(hint) is list:
            args = get_args(hint)
            if args and isinstance(args[0], type) and issubclass(args[0], agtype):
                return args[0]
        return None


class agfile(agtype):
    """File-backed agskill schema field.

    Input fields
    ------------
    The framework writes the Python string value to a temporary file inside
    the agent's sandbox before the skill runs.  The LLM receives the file
    path and uses the ``read`` tool to access the content.

    Output fields
    -------------
    The agent writes its output to a file and returns the path.  The
    framework reads the file content back and stores it as a plain string
    in the result agdata.  The agent-side file is deleted after the skill
    ends.

    Example::

        design_skill = agskill(
            name="design",
            system_prompt="Create a story design document.",
            input_schema=agdata(theme=str),
            output_schema=agdata(design_doc=agfile),
        )
    """

    @classmethod
    def schema_type(cls) -> str:
        return "file"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return True

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        path = f"/workspace/inputs/{field_name}{suffix}.txt"
        try:
            sandbox.write_file(path, value)
            return path, [path]
        except Exception as _e:
            print(f"[agtype] WARNING: agfile.prepare failed to write {path}: {_e}")
            return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        try:
            content = sandbox.read_file(value)
            return content, [value]
        except Exception as _e:
            print(f"[agtype] WARNING: agfile.recover failed to read {value}: {_e}")
            return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: the JSON value is a path to a temporary "
            f"file in your sandbox. Use the read tool to access the full content "
            f"before completing the task. This input file is temporary and will be cleaned up automatically "
            f"after the task completes."
        )

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return (
            f"  - Output `{field_name}`: write your output to a file (e.g. "
            f"/workspace/outputs/{field_name}.txt) and return only "
            f"the file path as the field value. This output file is temporary and will be cleaned up automatically "
            f"after the task completes. "
            f"If the output is long, write it in multiple smaller tool calls (e.g. write the first portion, "
            f"then append subsequent portions) rather than generating it all in one response — "
            f"this avoids hitting generation length limits."
        )

    @classmethod
    def return_tool_description(cls, field_name: str) -> str:
        return (
            f"Register the '{field_name}' output file. "
            f"Write your output to a sandbox file first (e.g. /workspace/outputs/{field_name}.txt), "
            f"then call this tool with that file path as the value."
        )

    @classmethod
    def return_value_description(cls, field_name: str) -> str:
        return (
            f"Absolute path to the file you wrote in the sandbox "
            f"(e.g. /workspace/outputs/{field_name}.txt). Do NOT pass the file content — pass the path."
        )


class agimage(agtype):
    """Image input field for agskill schemas.

    The caller supplies a local file path, an http/https URL, or an existing
    data URL.  Local files are base64-encoded by the framework before the skill
    runs.  The image is injected directly into the multimodal content array of
    the user message — the LLM sees it as a visual input, not as text.

    Single image::

        skill = agskill(
            name="describe",
            system_prompt="Describe the image.",
            input_schema=agdata(question=str, photo=agimage),
        )

    List of images::

        skill = agskill(
            name="compare",
            system_prompt="Compare the images.",
            input_schema=agdata(question=str, frames=list[agimage]),
        )
    """

    @classmethod
    def schema_type(cls) -> str:
        return "image"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        # Already a URL or data URL — pass through unchanged.
        if value.startswith(("http://", "https://", "data:")):
            return value, []
        # Local file path — read and base64-encode.
        path = Path(value)
        raw = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        b64 = base64.b64encode(raw).decode()
        return f"data:{mime};base64,{b64}", []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: an image attached directly to this message "
            f"as a visual input. The JSON field value is a placeholder — the actual "
            f"image is visible to you in the message content."
        )

    @classmethod
    def return_tool_description(cls, field_name: str) -> str:
        return f"Register the '{field_name}' image output as a file path or data URL."

    @classmethod
    def return_value_description(cls, field_name: str) -> str:
        return f"File path, http/https URL, or data: URL of the image for '{field_name}'."


class agbinary(agtype):
    """Binary file-backed agskill schema field.

    Carries raw bytes through the skill boundary via the sandbox filesystem.
    The LLM never sees the binary content — it only sees a file path and is
    told to use shell tools (``file``, ``xxd``, domain-specific CLIs) to
    inspect or produce binary output.

    Caller value types
    ------------------
    Input  (``prepare``): ``bytes``, a local host file path (``str``), or a
    base64 data URL (``str`` starting with ``data:``).

    Output (``recover``): always ``bytes``.

    Example::

        process_skill = agskill(
            name="process_audio",
            system_prompt="Trim the audio to the first 10 seconds using ffmpeg.",
            input_schema=agdata(audio=agbinary),
            output_schema=agdata(trimmed=agbinary),
        )

        result = ag.run(process_skill, agdata(audio=Path("clip.wav").read_bytes()))
        Path("trimmed.wav").write_bytes(result.trimmed)

    Note: the agent's sandbox image must include whatever CLI tools are needed
    to process the binary format (e.g. ``ffmpeg``, ``imagemagick``, ``sox``).
    """

    @classmethod
    def schema_type(cls) -> str:
        return "binary_file"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return True

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[str, list[str]]:
        path = f"/workspace/inputs/{field_name}{suffix}.bin"
        try:
            raw = cls._to_bytes(value)
        except (TypeError, ValueError):
            return value, []
        try:
            sandbox.write_file_bytes(path, raw)
            return path, [path]
        except Exception as _e:
            print(f"[agtype] WARNING: agbinary.prepare failed to write {path}: {_e}")
            return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[bytes, list[str]]:
        if not isinstance(value, str):
            return value, []
        try:
            raw = sandbox.read_file_bytes(value)
            return raw, [value]
        except Exception as _e:
            print(f"[agtype] WARNING: agbinary.recover failed to read {value}: {_e}")
            return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: a binary file at the path shown in the JSON. "
            f"Do NOT read it as text — use shell tools (e.g. `file`, `xxd`, or "
            f"domain-specific CLIs) to inspect or process it. "
            f"This file is temporary and will be deleted after the task completes."
        )

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return (
            f"  - Output `{field_name}`: write your binary output to a file "
            f"(e.g. /workspace/outputs/{field_name}.bin) using shell tools, "
            f"then return only the file path as the field value. "
            f"Do NOT encode the content as text or base64 — write the raw binary file."
        )

    @classmethod
    def return_tool_description(cls, field_name: str) -> str:
        return (
            f"Register the '{field_name}' binary output file. "
            f"Write your output as a binary file first "
            f"(e.g. /workspace/outputs/{field_name}.bin), "
            f"then call this tool with that file path as the value."
        )

    @classmethod
    def return_value_description(cls, field_name: str) -> str:
        return (
            f"Absolute path to the binary file you wrote in the sandbox "
            f"(e.g. /workspace/outputs/{field_name}.bin). "
            f"Pass the path — do NOT pass encoded content."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _to_bytes(cls, value: object) -> bytes:
        """Normalise caller input to raw bytes."""
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            if value.startswith("data:") and ";base64," in value:
                _, b64 = value.split(";base64,", 1)
                return base64.b64decode(b64)
            # Treat as a local host file path.
            return Path(value).read_bytes()
        raise TypeError(f"agbinary.prepare: unsupported value type {type(value).__name__!r}")


class agrawstring(agtype):
    """Raw string field — bypasses JSON input/output formatting entirely.

    Input
    -----
    When used as the sole input field, the string value is sent directly as
    the user message content.  No JSON wrapping, no schema hint.

    Output
    ------
    When used as the sole output field, the model's complete text response is
    captured as-is.  No JSON parsing, no retry loop.

    Constraint
    ----------
    Must be the only field in its input or output schema.

    Example::

        write_skill = agskill(
            name="write_chapter",
            system_prompt="You are a novelist. Write the chapter as requested.",
            input_schema=agdata(prompt=agrawstring),
            output_schema=agdata(chapter=agrawstring),
        )

        result = ag.run(write_skill, agdata(prompt="Write a dark opening scene."))
        print(result.chapter)   # the model's prose, unmodified
    """

    @classmethod
    def schema_type(cls) -> str:
        return "str"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False


# ---------------------------------------------------------------------------
# Type-hint inspection and schema helpers
# ---------------------------------------------------------------------------

def _hint_to_json_type(hint) -> str:
    """Map a schema hint to a JSON Schema type string for tool parameter specs."""
    if isinstance(hint, type):
        if issubclass(hint, bool):   return "boolean"   # bool before int (bool is subclass of int)
        if issubclass(hint, int):    return "integer"
        if issubclass(hint, float):  return "number"
        if issubclass(hint, agtype): return "string"
        if issubclass(hint, (list, tuple)): return "array"
        if issubclass(hint, dict):   return "object"
        return "string"
    origin = get_origin(hint)
    if origin is list or origin is tuple: return "array"
    if origin is dict:                    return "object"
    if isinstance(hint, list):            return "array"   # [{"key": type, ...}] literal
    return "string"


def _example_for_hint(hint) -> str:
    """Return a short valid-JSON example for a type hint (used in tool descriptions/errors).

    Every returned string is parseable by json.loads.
    """
    if hint is bool:  return "true"
    if hint is int:   return "42"
    if hint is float: return "3.14"
    if hint is str:   return '"text"'
    origin = get_origin(hint)
    args   = get_args(hint)
    if hint is list or origin is list:
        return f"[{_example_for_hint(args[0])}]" if args else "[]"
    if hint is dict or origin is dict:
        if args and len(args) == 2:
            return "{" + f"{_example_for_hint(args[0])}: {_example_for_hint(args[1])}" + "}"
        return "{}"
    if hint is tuple or origin is tuple:
        return "[" + ", ".join(_example_for_hint(t) for t in args) + "]" if args else "[]"
    if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
        obj = "{" + ", ".join(f'"{k}": {_example_for_hint(t)}' for k, t in hint[0].items()) + "}"
        return f"[{obj}]"
    if isinstance(hint, type) and issubclass(hint, agtype):
        return '"value"'
    return "null"


def _value_desc_for_hint(field: str, hint) -> str:
    """Return a value description with a concrete format example and a 'pass directly' note."""
    origin = get_origin(hint)
    args   = get_args(hint)
    ex     = _example_for_hint(hint)
    direct = "Pass directly — do not JSON-encode into a string."

    if hint is str:
        return (
            f"The complete string value for '{field}'. "
            "Pass the full content directly — not a file path."
        )
    if hint is bool:
        return f"Boolean for '{field}' (true or false)."
    if hint is int:
        return f"Integer for '{field}'. Example: {ex}."
    if hint is float:
        return f"Floating-point number for '{field}'. Example: {ex}."
    if hint is list or origin is list:
        if args:
            elem_type = args[0]
            elem_name = getattr(elem_type, "__name__", repr(elem_type))
            return (
                f"JSON array of {elem_name} values for '{field}'. "
                f"Example: {ex}. {direct}"
            )
        return f"JSON array for '{field}'. Example: {ex}. {direct}"
    if hint is dict or origin is dict:
        if args and len(args) == 2:
            kn = getattr(args[0], "__name__", repr(args[0]))
            vn = getattr(args[1], "__name__", repr(args[1]))
            return (
                f"JSON object with {kn} keys and {vn} values for '{field}'. "
                f"Example: {ex}. {direct}"
            )
        return f"JSON object for '{field}'. Example: {ex}. {direct}"
    if hint is tuple or origin is tuple:
        if args:
            types_str = ", ".join(getattr(t, "__name__", repr(t)) for t in args)
            return (
                f"JSON array of {len(args)} element(s) ({types_str}) for '{field}'. "
                f"Example: {ex}. {direct}"
            )
        return f"JSON array for '{field}'. Example: {ex}. {direct}"
    if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
        keys = ", ".join(f'"{k}"' for k in hint[0])
        obj_ex = "{" + ", ".join(f'"{k}": {_example_for_hint(t)}' for k, t in hint[0].items()) + "}"
        return (
            f"JSON array of objects for '{field}'. Each object must have keys: {keys}. "
            f"Example: [{obj_ex}]. {direct}"
        )
    return f"Value for '{field}'."


def _return_tool_descriptions(field: str, hint) -> "tuple[str, str]":
    """Return (tool_description, value_description) for a return_<field> tool."""
    tool_desc = (
        f"Return the final value for the '{field}' output field. "
        f"Pass the actual output content as the '{field}' argument — "
        f"do not call this tool with empty or placeholder arguments."
    )
    # agtype subclass — delegate to its classmethods
    if isinstance(hint, type) and issubclass(hint, agtype):
        return hint.return_tool_description(field), hint.return_value_description(field)
    # list[agtype] — delegate to the inner type, but include a JSON array example
    if get_origin(hint) is list:
        args = get_args(hint)
        if args and isinstance(args[0], type) and issubclass(args[0], agtype):
            inner = args[0]
            ex = _example_for_hint(hint)
            return (
                inner.return_tool_description(field) + " (as a JSON array)",
                inner.return_value_description(field) + f" Provide as a JSON array. Example: {ex}.",
            )
    return tool_desc, _value_desc_for_hint(field, hint)


def _validate_value(hint, value) -> "str | None":
    """Recursively validate value against hint. Returns an error string or None."""
    origin = get_origin(hint)
    args   = get_args(hint)

    if isinstance(hint, type):
        if issubclass(hint, agtype):
            return None if isinstance(value, str) else f"expected str, got {type(value).__name__}"
        if issubclass(hint, bool):
            return None if isinstance(value, bool) else f"expected bool, got {type(value).__name__}"
        if issubclass(hint, (int, float, str)):
            return None if isinstance(value, hint) else f"expected {hint.__name__}, got {type(value).__name__}"
        # bare list/tuple/dict without type args
        if issubclass(hint, (list, tuple)):
            return None if isinstance(value, (list, tuple)) else f"expected array, got {type(value).__name__}"
        if issubclass(hint, dict):
            return None if isinstance(value, dict) else f"expected dict, got {type(value).__name__}"
        return None if isinstance(value, hint) else f"expected {hint.__name__}, got {type(value).__name__}"

    if origin is list:
        if not isinstance(value, list):
            return f"expected list, got {type(value).__name__}"
        if args:
            for i, item in enumerate(value):
                err = _validate_value(args[0], item)
                if err:
                    return f"item {i}: {err}"
        return None

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            return f"expected array, got {type(value).__name__}"
        if args:
            for i, (type_arg, item) in enumerate(zip(args, value)):
                err = _validate_value(type_arg, item)
                if err:
                    return f"item {i}: {err}"
        return None

    if origin is dict:
        if not isinstance(value, dict):
            return f"expected dict, got {type(value).__name__}"
        if len(args) == 2:
            for k, v in value.items():
                err = _validate_value(args[1], v)
                if err:
                    return f"key {k!r}: {err}"
        return None

    # [{"key": type, ...}] literal list-of-dicts schema
    if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
        if not isinstance(value, list):
            return f"expected list, got {type(value).__name__}"
        template = hint[0]
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                return f"item {i}: expected dict, got {type(item).__name__}"
            for k, t in template.items():
                if k not in item:
                    return f"item {i}: missing key '{k}'"
                if isinstance(t, type) and not isinstance(item[k], t):
                    return f"item {i}.{k}: expected {t.__name__}, got {type(item[k]).__name__}"
        return None

    return None


def _validate_output_field(field: str, value, schema) -> "str | None":
    """Validate a single (field, value) pair against the output schema hint.

    Returns an error string, or None if valid.
    schema is an agdata instance; accessed via duck typing to avoid circular imports.
    """
    return _validate_value(schema._data[field], value)


# ---------------------------------------------------------------------------
# Type-hint traversal helpers  (now static methods on agtype)
# ---------------------------------------------------------------------------


def output_field_desc(hint: object) -> str:
    """Return a human-readable type description with usage guidance for an output field."""
    if isinstance(hint, type) and issubclass(hint, agtype):
        return hint.schema_type()
    if hint is str:
        return "string — pass the complete text content as the value (not a file path)"
    if hint is int:
        return "integer — pass the numeric value directly"
    if hint is float:
        return "float — pass the numeric value directly"
    if hint is bool:
        return "boolean — pass true or false"
    if hint is list or hint is dict:
        return hint.__name__
    if get_origin(hint) is list:
        args = get_args(hint)
        inner = output_field_desc(args[0]) if args else "any"
        return f"array of {inner}"
    return str(hint)


def raw_schema_key(schema: "object | None") -> "str | None":
    """Return the single field key if schema has exactly one agrawstring field, else None."""
    if schema is None:
        return None
    items = list(schema._data.items())
    if len(items) == 1:
        key, hint = items[0]
        if isinstance(hint, type) and issubclass(hint, agrawstring):
            return key


def make_field_handler(
    field: str,
    output_schema: object,
    sandbox: "agSandbox",
    collected_outputs: dict,
    required_fields: set,
    exec_timeout: int,
) -> "Callable[[dict], str]":
    """Build a handler for a single return_<field> intercept tool call.

    The returned callable validates the value, runs agfile/agbinary checks
    against the sandbox, and records the field in collected_outputs.
    """
    hint = output_schema._data[field]
    _is_agfile   = isinstance(hint, type) and issubclass(hint, agfile)
    _is_agbinary = isinstance(hint, type) and issubclass(hint, agbinary)
    _is_str      = hint is str

    def _handle(args: dict) -> str:
        value = next(iter(args.values()), None)
        err = _validate_output_field(field, value, output_schema)
        if err is not None:
            _hint    = output_schema._data[field]
            ex       = _example_for_hint(_hint)
            got_str  = isinstance(value, str)
            exp_arr  = _hint_to_json_type(_hint) == "array"
            exp_obj  = _hint_to_json_type(_hint) == "object"
            if value is None:
                fix = (
                    f"You called return_{field}() with no arguments. "
                    f"Pass the actual output as the '{field}' argument. "
                    f"Example: return_{field}({field}={ex})"
                )
            elif got_str and exp_arr:
                fix = (
                    f"You passed a JSON-encoded string; pass a JSON array directly. "
                    f"Example: {ex}"
                )
            elif got_str and exp_obj:
                fix = (
                    f"You passed a JSON-encoded string; pass a JSON object directly. "
                    f"Example: {ex}"
                )
            else:
                fix = f"Expected format: {ex}"
            return json.dumps({"error": f"field '{field}': {err}. {fix}"})

        if _is_agfile and isinstance(value, str):
            try:
                content = sandbox.read_file(value)
            except IsADirectoryError:
                return json.dumps({"error": (
                    f"field '{field}': '{value}' is a directory, not a file. "
                    f"Pass the path to a specific output file "
                    f"(e.g. {value}/{field}.txt)."
                )})
            except UnicodeDecodeError:
                return json.dumps({"error": (
                    f"field '{field}': file at '{value}' contains binary data "
                    f"and cannot be read as text. Write a UTF-8 text file instead."
                )})
            except Exception:
                return json.dumps({"error": (
                    f"field '{field}': no file found at path '{value}'. "
                    f"Write your output to a file first, then call this "
                    f"tool with that file's path."
                )})
            if not content or not content.strip():
                return json.dumps({"error": (
                    f"field '{field}': file at '{value}' is empty. "
                    f"Write the actual content to the file before "
                    f"registering the path."
                )})
            if _looks_like_path(content.strip()):
                return json.dumps({"error": (
                    f"field '{field}': file at '{value}' contains only a "
                    f"path reference ('{content.strip()}'), not real content. "
                    f"Write the actual content to a file and return that "
                    f"file's path."
                )})

        if _is_agbinary and isinstance(value, str):
            _, dir_rc = sandbox._container_exec(
                f"test -d {shlex.quote(value)}", timeout=exec_timeout, shell="sh"
            )
            if dir_rc == 0:
                return json.dumps({"error": (
                    f"field '{field}': '{value}' is a directory, not a file. "
                    f"Pass the path to a specific binary output file "
                    f"(e.g. {value}/{field}.bin)."
                )})
            _, exist_rc = sandbox._container_exec(
                f"test -s {shlex.quote(value)}", timeout=exec_timeout, shell="sh"
            )
            if exist_rc != 0:
                _, found_rc = sandbox._container_exec(
                    f"test -e {shlex.quote(value)}", timeout=exec_timeout, shell="sh"
                )
                if found_rc != 0:
                    return json.dumps({"error": (
                        f"field '{field}': no file found at path '{value}'. "
                        f"Write your binary output to a file first, then call "
                        f"this tool with that file's path."
                    )})
                return json.dumps({"error": (
                    f"field '{field}': file at '{value}' is empty. "
                    f"Write the actual binary content to the file before "
                    f"registering the path."
                )})

        if _is_str and isinstance(value, str) and _looks_like_path(value):
            try:
                resolved = sandbox.read_file(value)
                if resolved and resolved.strip() and not _looks_like_path(resolved.strip()):
                    value = resolved
            except Exception:
                pass

        collected_outputs[field] = value
        remaining = required_fields - set(collected_outputs)
        if remaining:
            _remaining_tools = ", ".join(f"return_{f}" for f in sorted(remaining))
            return json.dumps({"result": (
                f"[HARNESS SYSTEM] ✓ '{field}' registered. "
                f"Still needed: {sorted(remaining)}, call {_remaining_tools} tool(s)."
            )})
        return json.dumps({"result": (
            f"[HARNESS SYSTEM] ✓ '{field}' registered. "
            f"All required fields complete, please end your response now."
        )})

    return _handle