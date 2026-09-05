"""agschema — schema wrapper for agskill input/output schemas.

Users write ``agdata(task=str)`` at call sites (making it clear that skills
receive/return ``agdata``).  ``agskill.__init__`` converts these to ``agschema``
internally.  All internal schema operations use ``agschema``.
"""

from __future__ import annotations
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sandbox.agsandbox import agSandbox

from .agdata import agdata, agerror
from .agtype import (
    agtype,
    agrawstring,
    get_return_tool_description_prompt,
    validate_value_against_type_hint,
    output_field_desc,
)
from .configs.agconfig import agconfig as agconfig_cls


def _lenient_json_object(raw_text: str) -> dict:
    """Parse *raw_text* as a JSON object, tolerating a harness's model
    wrapping its final answer in prose and/or a markdown code fence
    despite being asked for raw JSON only
    (`agharness.build_output_format_instruction`'s instruction is not
    always followed strictly -- confirmed against a real response from a
    real Claude model: 'Perfect! All tasks have been completed
    successfully. Let me provide the final status:\\n\\n```json\\n{...}\\n```').

    Tries, in order: the raw text as-is; the contents of a ```...```
    fence if one is present; the substring from the first '{' to the
    last '}'. Raises the ORIGINAL `json.JSONDecodeError` (from the
    raw-text attempt) if every strategy fails, so a genuinely non-JSON
    response still reports its own real parse error instead of a
    fallback attempt's more confusing one."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # `except ... as name` is implicitly deleted once this block exits
        # (Python avoids a traceback reference cycle) -- keep it alive
        # under a different name so it's still raiseable at the bottom.
        original_exc = exc

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw_text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise original_exc


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
        agconfig: "agconfig_cls | None" = None,
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
        _cfg = agconfig if agconfig is not None else agconfig_cls()
        _input_offload_chars = _cfg.schema.input_offload_chars
        _threshold = (
            min(
                _input_offload_chars,
                int(
                    context_limit
                    * _cfg.schema.offload_context_fraction
                    * _cfg.schema.chars_per_token
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

    def validate_and_recover(
        self,
        raw_text: str,
        sandbox: "agSandbox",
    ) -> "tuple[agdata | agerror, list[str]]":
        """Validate and recover a harness's single raw final-answer text
        against this schema, in one call.

        Harness adapters return one raw text blob, so this provides the
        whole-schema validation and recovery step as a pure composition of
        `check()` (whole-schema field presence/type validation, already
        used for input validation despite the name) and `recover_outputs()`
        (per-agtype-field `.recover()`) -- no new validation logic.

        Returns `(data, paths)` on success (`paths` are the sandbox paths
        `recover_outputs()` produced, for parity with `execute_react()`'s
        own cleanup bookkeeping), or `(agerror(...), [])` if *raw_text*
        isn't valid JSON, isn't a JSON object, or fails schema validation.
        """
        try:
            parsed = _lenient_json_object(raw_text)
        except json.JSONDecodeError as exc:
            return agerror(f"could not parse harness output as JSON: {exc}"), []
        if not isinstance(parsed, dict):
            return (
                agerror(f"harness output must be a JSON object, got {type(parsed).__name__}"),
                [],
            )

        data = agdata(**parsed)
        errors = self.check(data)
        if errors:
            return agerror(f"output schema error: {errors}"), []

        paths = self.recover_outputs(data, sandbox)
        return data, paths

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

    def __repr__(self) -> str:
        return f"agschema({self._data!r})"
