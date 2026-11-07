# Schemas and data types

```python
from agency import agschema, agdata, agtype, agfile, agpath, agbinary, agimage, agrawstring
```

## Schemas and validation

`agschema(source)` accepts an agdata template or another agschema (`TypeError` otherwise). Template fields are shallow-copied from agdata; copying from a schema shares its field mapping. Use field values as Python types or agtype class objects, not instances. Text descriptions require presence but do not enforce a Python type.

| Method | Purpose, return and important limits |
| --- | --- |
| `check(data)` | Returns a list of missing/type errors, empty means valid for implemented checks. Extra fields allowed. Checks Python types, agtype input validators and one-item list-of-dict templates. Does not generally enforce arbitrary recursive typing expressions. Reads the payload directly: resolve pending input first. |
| `check_field(field_name, value)` | Returns an error string or `None`, using field type validation. Unknown field raises `KeyError`. |
| `validate_input(data)` | Returns `None` or a single formatted input-schema error string. Explicit caller operation; not automatically invoked by skill execution. |
| `to_json()` | Returns the schema JSON/type descriptions for prompts. |
| `prepare_inputs_in_sandbox(data, sandbox, skill_name, suffix="", context_limit=None, agconfig=None)` | Blocking preparation; mutates fields, writes files/offloads long text; returns `(cleanup_paths, offloaded_fields)`. Default text threshold is 40,000 characters, reduced by context fraction × chars/token when a context limit is supplied. Failures are best-effort warnings and can retain original values. |
| `recover_outputs(data, sandbox)` | Blocking file recovery; mutates output, returns cleanup paths. Direct agerror is untouched. Failed recovery warns and can retain the model's original value. |
| `validate_outputs(data, sandbox, exec_timeout=5)` | Returns per-type sandbox validation errors. Performs file probes for file/binary markers; exec_timeout applies to binary shell probes; text-file reads use backend I/O limits. Neither bounds the whole invocation. |
| `validate_and_recover(raw_text, sandbox, exec_timeout=5)` | Accepts JSON object text, including fenced/embedded JSON. Returns `(agdata, paths)` or `(agerror, [])`; runs shape and output-file checks then recovery. Recovery itself is best-effort. |
| `raw_key()` | Returns the single agrawstring field name, or `None`. |
| `field_desc(field_name)` | Returns the prompt description; unknown field raises `KeyError`. |
| `get_return_tool_descriptions(field_name)` | Returns `(tool_description, value_description)` for field submission; unknown field raises `KeyError`. |

```python
from agency import agdata, agschema

schema = agschema(agdata(title=str, rows=[{"score": int}]))
assert schema.validate_input(agdata(title="Results", rows=[{"score": 3}])) is None
assert schema.check(agdata(title="Results"))  # missing rows
```

## File, path, binary, image and raw text meanings

These are schema markers. Input preparation/recovery may perform synchronous I/O; sandbox paths refer to that sandbox's filesystem, host paths to the caller's machine. Values are not universally portable across machines.

| Marker | Caller input | Model-facing value | Successful caller output |
| --- | --- | --- | --- |
| `agfile` | Text contents as str, **not a host filename** | Sandbox text file path under `/workspace/inputs/` | Reads a returned sandbox path into text contents; validates a nonempty text file. |
| `agpath` | Path-shaped str | The same path | The same path; checks shape only, no read or existence check. |
| `agbinary` | bytes, local host file path, or base64 data URL | Sandbox `.bin` path | Reads a returned nonempty sandbox file into bytes. Failed recovery can leave a path. |
| `agimage` | Local host image path, URL or data URL | Host-read image encoded as a multimodal image_url block, or supplied URL | No specialized output file recovery. Model/provider must support image input. |
| `agrawstring` | Text str | Raw text for a single-field schema (long text can still be offloaded) | Raw response text for a single-field output schema. |
| Plain `str` | Text | Text or offloaded sandbox path | Text remains text, including path-looking strings; agpath additionally enforces path shape. |

Preparation checks can fail for missing host files or malformed binary data. Schema preparation warns and may leave the original value; input validation must be explicit when an application requires a failure before execution. `agpath.validate_input_value` tests path shape, not filesystem availability.

```python
from agency import agdata, agskill, agfile, agbinary, agpath

convert = agskill("convert", "Read the text and write a binary artifact.",
                  input_schema=agdata(source=agfile),
                  output_schema=agdata(artifact=agbinary, location=agpath))
input_payload = agdata(source="This is file content, not /host/input.txt")
```

## Custom type protocol

Subclass `agtype` and put the class in the schema. `schema_type()` supplies the JSON prompt label (default `str`), `needs_sandbox()` defaults false. `prepare(value, sandbox, skill_name, field_name, suffix="")` and `recover(value, sandbox)` return `(transformed_value, cleanup_paths)`; default passes the value through. Callers of these direct hooks own the returned cleanup paths. Pipeline helpers perform best-effort cleanup.

`validate_input_value(value)` and `validate_output(field_name, value, sandbox, exec_timeout)` return `None` or an error string. `extra_input_prompt`, `extra_output_prompt` and return-tool description helpers return prompt strings. `build_content_prompt(key, value)` returns `(replacement_placeholder_or_None, content_blocks)`.

`walk(type_hint, value, on_leaf)` transforms supported nested list/dict/tuple shapes and combines cleanup paths; its callback must return `(value, paths)` and handle its own errors. `in_hint` detects special types except agrawstring; `from_hint` resolves a direct marker or `list[marker]`, else `None`. Concrete markers override this same protocol; their behavior is specified in the table rather than repeated for every override.

## Source signatures

[Source: agschema.py](../../agency/agschema.py)

::: agency.agschema.agschema
    options:
      members: ["__init__", "to_json", "check", "check_field", "validate_input", "prepare_inputs_in_sandbox", "recover_outputs", "validate_outputs", "validate_and_recover", "raw_key", "field_desc", "get_return_tool_descriptions"]

[Source: agtype.py](../../agency/agtype.py)

::: agency.agtype.agtype
    options:
      members: ["schema_type", "needs_sandbox", "prepare", "recover", "extra_input_prompt", "extra_output_prompt", "build_content_prompt", "get_return_tool_description", "get_return_tool_value_description", "validate_output", "validate_input_value", "walk", "in_hint", "from_hint"]

[Source: agtype.py](../../agency/agtype.py)

::: agency.agtype.agfile
    options:
      members: []

[Source: agtype.py](../../agency/agtype.py)

::: agency.agtype.agpath
    options:
      members: []

[Source: agtype.py](../../agency/agtype.py)

::: agency.agtype.agbinary
    options:
      members: []

[Source: agtype.py](../../agency/agtype.py)

::: agency.agtype.agimage
    options:
      members: []

[Source: agtype.py](../../agency/agtype.py)

::: agency.agtype.agrawstring
    options:
      members: []
