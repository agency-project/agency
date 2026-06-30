# agschema

Validates and manages input/output type contracts for `agskill`. Users declare schemas as `agdata(field=type)` at call sites; `agskill.__init__` converts them to `agschema` internally. All runtime schema operations — validation, sandbox preparation, output collection — go through `agschema`.

## When to use / when not to use

Use `agschema` when writing framework-level code that needs to inspect, validate, or transform skill data. Do not construct `agschema` directly in application code; write `agdata(field=type)` and let `agskill` handle the conversion.

---

## Constructor

```python
agschema(source: agdata | agschema)
```

Wraps an `agdata` or copies another `agschema`. Raises `TypeError` for any other input type. The internal representation is a plain `dict[str, type_hint]` stored as `_data`.

```python
from agency import agdata, agschema

schema = agschema(agdata(query=str, max_results=int))
```

---

## Public methods

### `to_json() -> str`

Serializes the schema to a JSON string in the same format as `agdata.to_json()`. Used by `agskill` to inject schema information into system prompts.

```python
schema = agschema(agdata(result=str, count=int))
print(schema.to_json())
# '{"result": "str", "count": "int"}'
```

### `check(data: agdata) -> list[str]`

Validates `data` against the schema. Returns a list of error strings; an empty list means the data is valid.

Checks:
- All required fields are present.
- `agtype` subclass fields (e.g. `agfile`, `agimage`) must carry string values.
- Typed list-of-dict fields validate each item's keys and value types.
- Plain `type` fields use `isinstance` matching.

```python
errors = schema.check(agdata(result="hello", count=3))
# errors == []

errors = schema.check(agdata(result="hello"))
# errors == ["missing required field 'count'"]
```

### `check_field(field_name: str, value) -> str | None`

Validates a single field value against its type hint. Returns an error string on failure, `None` if valid. Used by `make_field_handler` when an agent calls a `return_<field>` tool.

```python
err = schema.check_field("count", "not_an_int")
# err == "expected int, got str"
```

### `validate_input(data: agdata) -> str | None`

Convenience wrapper around `check`. Returns a formatted error string if validation fails, `None` if valid. Called by `agskill` before running a skill.

```python
err = schema.validate_input(agdata(query="search term", max_results=5))
# err is None
```

### `raw_key() -> str | None`

Returns the field name if the schema has exactly one field typed as `agrawstring`, otherwise returns `None`. Used to detect raw-string output schemas where the agent response body is captured directly rather than through a structured tool call.

```python
raw_schema = agschema(agdata(output=agrawstring))
raw_schema.raw_key()   # "output"

multi_schema = agschema(agdata(a=str, b=int))
multi_schema.raw_key() # None
```

### `prepare_inputs_in_sandbox(data, sandbox, skill_name, suffix="", context_limit=None) -> tuple[list[str], list[str]]`

Transforms input fields before a skill runs in a sandbox. Two modes of transformation:

- **`agtype` fields** (e.g. `agfile`, `agimage`): calls `agtype.prepare()` on each leaf, which may write files into the sandbox and replace the value with a path reference.
- **Plain string / `agrawstring` fields**: if the string length exceeds the offload threshold (scaled down when `context_limit` is set), writes the content to `/workspace/inputs/<skill_name>_<field><suffix>.txt` and replaces the value with a short path hint. List fields with oversized string items are handled item-by-item.

Returns `(all_paths, auto_offloaded_fields)`. `auto_offloaded_fields` names the fields that were size-offloaded so the system prompt can instruct the agent to read those files.

The `context_limit` parameter (in tokens) reduces the offload threshold to 10% of the context window (in characters) when provided, preventing large inputs from consuming disproportionate context.

### `recover_outputs(data: agdata, sandbox: agSandbox) -> list[str]`

Recovers `agtype` output fields after a skill completes. Recursively walks any nested containers (list, dict, tuple) and calls `hint.recover()` on each `agtype` leaf. Returns all sandbox paths touched, which are used for cleanup.

No-ops if `data` is an `agerror`.

### `make_return_output_agtool(sandbox, collected_outputs, required_fields, exec_timeout) -> list`

Builds one `agtool` per output field. Each tool is named `return_<field_name>` and runs in the calling thread (not a subprocess) so the handler closure can write directly into `collected_outputs`. `required_fields` stays unchanged throughout and is used to check completeness via set subtraction (`required_fields - set(collected_outputs)`).

After each successful call the tool returns a harness message listing any remaining fields the agent still needs to call, or confirms completion when all required fields have been collected.

```python
collected = {}
required = {"result", "count"}
tools = output_schema.make_return_output_agtool(
    sandbox, collected, required, exec_timeout=60
)
# tools is a list of agtool objects, one per output field
```

### `field_desc(field_name: str) -> str`

Returns a human-readable type description with usage guidance for a field. Used when constructing system prompts.

---

## Internal / framework methods

The following methods are used by the framework internally. Application code should not need to call them directly.

| Method | Purpose |
|---|---|
| `get_return_tool_descriptions(field_name)` | Returns `(tool_description, value_description)` strings for a `return_<field>` tool. |
| `make_return_output_tools()` | Returns raw OpenAI-wire `dict` tool definitions for each output field. Used by tests. |
| `make_field_handler(field_name, sandbox, collected_outputs, required_fields, exec_timeout)` | Builds the handler closure for a single `return_<field>` tool call. Validates the value, resolves file paths for plain `str` fields, and writes into `collected_outputs`. |

---

## Common patterns

### Defining a skill with typed inputs and outputs

```python
from agency import agskill, agdata

skill = agskill(
    name="summarize",
    input_schema=agdata(text=str, max_words=int),
    output_schema=agdata(summary=str),
)
```

`agskill.__init__` wraps both `agdata` arguments in `agschema` automatically.

### Validating data before passing it to a skill

```python
schema = agschema(agdata(text=str, max_words=int))
errors = schema.check(agdata(text="hello world", max_words=50))
if errors:
    raise ValueError(f"Bad input: {errors}")
```

### Detecting a raw-string output schema

```python
if output_schema.raw_key() is not None:
    # Capture the agent's response body directly
    pass
else:
    # Collect structured fields via return_<field> tools
    pass
```

---

## Constraints and gotchas

- `agschema` copies the field dict from `agdata` at construction time. Later mutations to the original `agdata` do not affect the schema.
- `check` validates structure but does not call `agtype.validate_output`; that deeper check happens inside `make_field_handler` at tool-call time.
- `prepare_inputs_in_sandbox` mutates `data._data` in place. Do not reuse the same `agdata` instance across multiple sandbox invocations.
- The offload path pattern is `/workspace/inputs/<skill_name>_<field><suffix>.txt`. If two parallel skill invocations share the same sandbox, use distinct `suffix` values to avoid collisions.
- `recover_outputs` is a no-op on `agerror` instances; always check for errors before assuming output fields are populated.
- `raw_key()` returns `None` for any schema with more than one field, even if all fields are `agrawstring`. The single-field restriction is intentional.
