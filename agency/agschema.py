"""agschema — schema wrapper for agskill input/output schemas.

Users write ``agdata(task=str)`` at call sites (making it clear that skills
receive/return ``agdata``).  ``agskill.__init__`` converts these to ``agschema``
internally.  All internal schema operations use ``agschema``.
"""

from __future__ import annotations
import json
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .agsandbox import agSandbox
    from .agconfig import agConfig

from .agdata import agdata, agerror
from .agtype import (
    agtype,
    agrawstring,
    type_hint_to_string_type,
    get_return_tool_description_prompt,
    validate_value_against_type_hint,
    get_json_example_for_type_hint,
    output_field_desc,
)
from .agutil import _looks_like_path
from .agconfig import DynamicConfigParam, _AgConfigViewBase


# Exists only to register agschema's config fields (via __set_name__ at
# import time). Reads use a throwaway instance -- _AgSchemaFields(agconfig)
# -- since agschema instances don't hold their own agconfig, so there's no
# self to hang a descriptor on.
class _AgSchemaFields:
    # Maximum length of a string field that will be auto-offloaded to a
    # sandbox file. Other code needing this same value (e.g. tests) reads
    # the descriptor's frozen default directly: _AgSchemaFields.input_offload_chars.default
    input_offload_chars = DynamicConfigParam("agschema", default=40_000)
    # Cap the per-field offload threshold at a fraction of the context window
    # (converted from tokens to chars) so a single oversized field can't eat
    # the whole context on small-context models. Shared with agskill.py's
    # tool-output offload sizing -- this class is the source of truth for both.
    offload_context_fraction = DynamicConfigParam("agschema", default=0.1)
    chars_per_token = DynamicConfigParam("agschema", default=4)

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agSchemaConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agschema tunables in one call::

        cfg = agConfig(agSchemaConfig(input_offload_chars=2000))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agschema"


def _type_error_fix(field_name: str, type_hint, value) -> str:
    """Return a corrective hint string for a type-validation failure."""
    ex = get_json_example_for_type_hint(type_hint)
    got_str = isinstance(value, str)
    if value is None:
        return (
            f"You called return_{field_name} without providing the required argument. "
            f"You must pass your output as '{field_name}' keyed argument to the tool. Calling this tool without passing an argument will not work. "
            f'Format Example (JSON): {{"{field_name}": {ex}}}'
        )
    if got_str and type_hint_to_string_type(type_hint) == "array":
        return f"You passed a JSON-encoded string; pass a JSON array directly. Example: {ex}"
    if got_str and type_hint_to_string_type(type_hint) == "object":
        return f"You passed a JSON-encoded string; pass a JSON object directly. Example: {ex}"
    return f"Expected format: {ex}"


class agschema:
    """Schema wrapper for agskill input/output schemas.

    Wraps a ``_data: dict[str, type_hint]``.  Construction accepts an
    ``agdata`` (or another ``agschema``).
    """

    def __init__(self, source):
        if isinstance(source, agschema):
            self._data = source._data
        elif isinstance(source, agdata):
            self._data = dict(source._data)
        else:
            raise TypeError(f"agschema requires agdata or agschema, got {type(source).__name__}")

    # ------------------------------------------------------------------
    # Serialization helpers (used by agskill system prompt)
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the schema as if it were an agdata with the same keys."""
        return agdata(**self._data).to_json()

    # ------------------------------------------------------------------
    # Schema checking
    # ------------------------------------------------------------------

    def check(self, data: agdata) -> list[str]:
        """Return a list of error strings; empty list means the data is valid."""
        errors: list[str] = []
        for key, hint in self._data.items():
            if key not in data._data:
                errors.append(f"missing required field '{key}'")
                continue
            actual = data._data[key]
            if isinstance(hint, type) and issubclass(hint, agtype):
                err = hint.validate_input_value(actual)
                if err is not None:
                    errors.append(f"field '{key}' ({hint.__name__}): {err}")
                continue
            if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
                item_template = hint[0]
                if not isinstance(actual, list):
                    errors.append(f"field '{key}': expected list, got {type(actual).__name__}")
                    continue
                for i, item in enumerate(actual):
                    if not isinstance(item, dict):
                        errors.append(
                            f"field '{key}[{i}]': expected dict, got {type(item).__name__}"
                        )
                        continue
                    for item_key, item_type in item_template.items():
                        if item_key not in item:
                            errors.append(f"field '{key}[{i}]': missing key '{item_key}'")
                        elif isinstance(item_type, type) and not isinstance(
                            item[item_key], item_type
                        ):
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

    def check_field(self, field_name: str, value) -> "str | None":
        """Validate a single (field_name, value) pair against the schema type hint.

        Returns an error string, or None if valid.
        """
        return validate_value_against_type_hint(self._data[field_name], value)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(self, data: agdata) -> "str | None":
        """Return an error string if input fails schema validation, else None."""
        errors = self.check(data)
        if errors:
            return f"input schema error: {errors}"
        return None

    # ------------------------------------------------------------------
    # agtype field preparation and recovery
    # ------------------------------------------------------------------

    def prepare_inputs_in_sandbox(
        self,
        data: agdata,
        sandbox: "agSandbox",
        skill_name: str,
        suffix: str = "",
        context_limit: "int | None" = None,
        agconfig: "agConfig | None" = None,
    ) -> "tuple[list[str], list[str]]":
        """Prepare all input fields that require sandbox access, in one pass.

        For each schema field:
        - agtype fields (agfile, agbinary, agimage, …): call agtype.prepare() via
          walk, which transforms the value (e.g. writes file content to a sandbox
          path). agrawstring is excluded — its prepare() is a no-op and its value
          may still be size-offloaded below.
        - Plain string / agrawstring fields whose value exceeds the offload
          threshold: write content to a sandbox file and replace the value with a
          short path reference so the context window stays manageable.

        Returns (all_paths, auto_offloaded_fields) where auto_offloaded_fields are
        the names of fields that were size-offloaded (used for the system prompt
        warning telling the LLM to read those files).
        """
        _schema_fields = _AgSchemaFields(agconfig)
        _input_offload_chars = _schema_fields.input_offload_chars
        _threshold = (
            min(
                _input_offload_chars,
                int(
                    context_limit
                    * _schema_fields.offload_context_fraction
                    * _schema_fields.chars_per_token
                ),
            )
            if context_limit
            else _input_offload_chars
        )
        all_paths: list[str] = []
        auto_offloaded_fields: list[str] = []

        for key, hint in self._data.items():
            if agtype.in_hint(hint):
                # agtype field (not agrawstring) — transform via prepare().
                def on_leaf(h, v, _key=key):
                    try:
                        return h.prepare(v, sandbox, skill_name, _key, suffix=suffix)
                    except Exception as _e:
                        print(
                            f"[agent] WARNING: {h.__name__}.prepare failed for field '{_key}': {_e}"
                        )
                        return v, []

                new_val, written = agtype.walk(hint, data._data.get(key), on_leaf)
                if written or new_val is not data._data.get(key):
                    data._data[key] = new_val
                all_paths.extend(written)
            else:
                # Plain field or agrawstring — size-based offload.
                val = data._data.get(key)
                if isinstance(val, str):
                    if len(val) > _threshold:
                        path = f"/workspace/inputs/{skill_name}_{key}{suffix}.txt"
                        try:
                            sandbox.write_file(path, val)
                            data._data[key] = (
                                f"(content saved to {path} — use the read tool to access it)"
                            )
                            all_paths.append(path)
                            auto_offloaded_fields.append(key)
                        except Exception as _e:
                            print(
                                f"[agent] WARNING: failed to offload input field '{key}' to {path}: {_e}"
                            )
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
                            all_paths.append(path)
                            offloaded_any = True
                        except Exception as _e:
                            print(
                                f"[agent] WARNING: failed to offload input list field '{key}[{i}]' to {path}: {_e}"
                            )
                    if offloaded_any:
                        data._data[key] = new_vals
                        auto_offloaded_fields.append(key)

        return all_paths, auto_offloaded_fields

    def recover_outputs(
        self,
        data: agdata,
        sandbox: "agSandbox",
    ) -> list[str]:
        """Recover agtype output fields after the skill finishes.

        Recursively handles agtype subclasses nested inside list, dict, and tuple
        containers at any depth.  Calls ``hint.recover()`` at each agtype leaf.
        Returns all sandbox paths for cleanup.
        """
        if isinstance(data, agerror):
            return []
        paths: list[str] = []
        for key, hint in self._data.items():

            def on_leaf(h, v, _key=key):
                try:
                    return h.recover(v, sandbox)
                except Exception as _e:
                    print(f"[agent] WARNING: {h.__name__}.recover failed for field '{_key}': {_e}")
                    return v, []

            new_val, written = agtype.walk(hint, data._data.get(key), on_leaf)
            if written or new_val is not data._data.get(key):
                data._data[key] = new_val
            paths.extend(written)
        return paths

    # ------------------------------------------------------------------
    # raw_schema_key equivalent
    # ------------------------------------------------------------------

    def raw_key(self) -> "str | None":
        """Return the single field key if schema has exactly one agrawstring field, else None."""
        items = list(self._data.items())
        if len(items) == 1:
            key, type_hint = items[0]
            if isinstance(type_hint, type) and issubclass(type_hint, agrawstring):
                return key
        return None

    # ------------------------------------------------------------------
    # Field description helpers (for system prompt)
    # ------------------------------------------------------------------

    def field_desc(self, field_name: str) -> str:
        """Return a human-readable type description with usage guidance for an output field."""
        return output_field_desc(self._data[field_name])

    # ------------------------------------------------------------------
    # Return tool descriptions
    # ------------------------------------------------------------------

    def get_return_tool_descriptions(self, field_name: str) -> "tuple[str, str]":
        """Return (tool_description, value_description) for a return_<field_name> tool."""
        return get_return_tool_description_prompt(field_name, self._data[field_name])

    # ------------------------------------------------------------------
    # Return output tools
    # ------------------------------------------------------------------

    def make_return_output_agtool(
        self,
        sandbox: "agSandbox",
        collected_outputs: dict,
        required_fields: set,
        exec_timeout: int,
    ) -> list:
        """Build one agtool per output field, wired to collect into collected_outputs.

        Each tool runs in the calling thread (run_in_subprocess=False) so the handler
        closure can mutate collected_outputs and required_fields directly.
        """
        from .agtool import agtool as _agtool

        def _return_log_fn(tool, arg, result, _elapsed_ms):
            if tool._term is None:
                return
            arg_str = arg.to_json()
            if "error" in result._data:
                tool._term.log("TOOL ✗   ", f"{tool.name}({arg_str})  → {result._data['error']}")
            else:
                tool._term.log("TOOL ✓   ", f"{tool.name}({arg_str})")

        tools = []
        for field, hint in self._data.items():
            json_type = type_hint_to_string_type(hint)
            tool_desc, value_desc = get_return_tool_description_prompt(field, hint)
            value_schema: dict = {"type": json_type, "description": value_desc}
            handler = self.make_field_handler(
                field, sandbox, collected_outputs, required_fields, exec_timeout
            )

            def _fn(arg, _h=handler):
                return agdata.from_json(_h(arg._data))

            tools.append(
                _agtool(
                    name=f"return_{field}",
                    description=tool_desc,
                    fn=_fn,
                    params={
                        "type": "object",
                        "properties": {field: value_schema},
                        "required": [field],
                    },
                    log_fn=_return_log_fn,
                    run_in_subprocess=False,
                )
            )
        return tools

    def make_return_output_tools(self) -> list[dict]:
        """Return raw OpenAI-wire dicts for each output field. Used by tests."""
        tools = []
        for field, hint in self._data.items():
            json_type = type_hint_to_string_type(hint)
            tool_desc, value_desc = get_return_tool_description_prompt(field, hint)
            value_schema: dict = {"type": json_type, "description": value_desc}
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"return_{field}",
                        "description": tool_desc,
                        "parameters": {
                            "type": "object",
                            "properties": {field: value_schema},
                            "required": [field],
                        },
                    },
                }
            )
        return tools

    # ------------------------------------------------------------------
    # Field handler factory
    # ------------------------------------------------------------------

    def make_field_handler(
        self,
        field_name: str,
        sandbox: "agSandbox",
        collected_outputs: dict,
        required_fields: set,
        exec_timeout: int,
    ) -> "Callable[[dict], str]":
        """Build a handler for a single return_<field_name> intercept tool call."""
        type_hint = self._data[field_name]
        agtype_cls = agtype.from_hint(type_hint)

        def _handle(args: dict) -> str:
            value = next(iter(args.values()), None)

            err = self.check_field(field_name, value)
            if err is not None:
                return json.dumps(
                    {
                        "error": (
                            f"field_name '{field_name}': {err}. "
                            f"{_type_error_fix(field_name, type_hint, value)}"
                        )
                    }
                )

            if agtype_cls is not None:
                err = agtype_cls.validate_output(field_name, value, sandbox, exec_timeout)
                if err is not None:
                    return json.dumps({"error": err})

            if type_hint is str and isinstance(value, str) and _looks_like_path(value):
                try:
                    resolved = sandbox.read_file(value)
                    if resolved and resolved.strip() and not _looks_like_path(resolved.strip()):
                        print(
                            f"[agschema] WARNING: output field '{field_name}' looked like a "
                            f"path ('{value}') and was auto-resolved to that file's contents "
                            f"because its type hint is plain str. If '{field_name}' is meant "
                            f"to hold a path rather than content, declare it as agpath instead."
                        )
                        value = resolved
                except Exception as _e:
                    # Expected whenever the str value just isn't an actual
                    # readable path in the sandbox -- leave it as a plain
                    # string rather than auto-resolved content.
                    print(
                        f"[agschema] '{field_name}' looked like a path but could not be read: {_e}"
                    )

            if type_hint is float and isinstance(value, int):
                value = float(value)
            collected_outputs[field_name] = value
            remaining = required_fields - set(collected_outputs)
            if remaining:
                _remaining_tools = ", ".join(f"return_{f}" for f in sorted(remaining))
                return json.dumps(
                    {
                        "result": (
                            f"[HARNESS SYSTEM] ✓ '{field_name}' registered. "
                            f"Still needed: {sorted(remaining)}, call {_remaining_tools} tool(s)."
                        )
                    }
                )
            return json.dumps(
                {
                    "result": (
                        f"[HARNESS SYSTEM] ✓ '{field_name}' registered. "
                        f"All required fields complete, please end your response now."
                    )
                }
            )

        return _handle

    def __repr__(self) -> str:
        return f"agschema({self._data!r})"
