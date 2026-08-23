# Data Schema: I/O Between the LLM and the Framework

This document describes how Python-side data travels to and from the LLM: how
`agdata` is serialized into prompt text and LLM tool calls, how `agtype`
subclasses intercept that path to do file I/O, image injection, and other
transformations, and how long-form content is handled without flooding the
context window.

---

## `agdata` — the universal value container

`agdata` is a thin dict wrapper. Every value crossing a skill boundary (inputs
from the caller, outputs returned by the LLM) is an `agdata`.

```python
result = agdata(summary="...", score=42, tags=["ml", "nlp"])
result.summary   # "..."
result.score     # 42
```

**Schema-as-instance.** agskill schemas are also `agdata` objects — their
`_data` dict maps field names to *type hints* instead of values:

```python
input_schema  = agdata(theme=str, background=agfile)
output_schema = agdata(report=agfile, score=int)
```

This dual use (schema vs value) means the same serialization path handles both.

**Serialization.** `agdata.to_json()` calls `_to_serializable()` recursively:

| Python type | Serialized form |
|---|---|
| `str`, `int`, `float`, `bool` | JSON primitive |
| `agdata` | JSON object (nested) |
| `list` | JSON array |
| `dict` | JSON object |
| `agtype` subclass (as class object) | `cls.schema_type()` string, e.g. `"file"` |
| generic alias `list[agimage]` | `"list[image]"` |
| any other type object | `type.__name__` |

When serializing a *schema* `agdata` (field → type hint), each hint is
serialized to its human-readable label.  This is used in the system prompt:

```
Input JSON format:
{"theme": "str", "background": "file"}
```

---

## End-to-end I/O flow for a skill run

```
Caller side                 Framework                       LLM side
──────────                  ─────────                       ────────

agdata(theme="AI",          agschema.prepare_inputs_in_sandbox()  system prompt:
       background="<text>") ↳ agfile.prepare() writes             - Input JSON format
                               text → /workspace/inputs/    - Output format hints
                               background.txt               - File-backed field hints
                               replaces value with path
                                                            user message:
                            build_prompt_payload()          {"theme":"AI",
                            ↳ injects images (agimage)       "background":
                            ↳ sends JSON for text fields      "/workspace/inputs/background.txt"}

                                                            LLM runs ReAct loop,
                                                            calls return_<field> tools

                            _handle() per return_<field>
                            ↳ validates type/content
                            ↳ agfile: reads file back        return_report("/workspace/outputs/report.txt")
                            ↳ agbinary: reads bytes back

                            agschema.recover_outputs()
                            ↳ agfile.recover() reads file
                               content → plain string
                            ↳ agbinary.recover() reads bytes

                            cleanup: delete temp paths

agdata(report="<content>",  ←─────────────────────────────
       score=7)
```

---

## `agschema` — the schema layer

Users write `agdata(field=type)` at call sites. `agskill.__init__` converts these to `agschema` objects internally. All schema operations — validation, return-tool generation, input preparation, output recovery — go through `agschema`, not directly through `agskill` or `agtype`. Key methods: `prepare_inputs_in_sandbox()`, `recover_outputs()`, `make_return_output_agtool()`, `check()`, `raw_key()`.

---

## `agtype` — the serialization extension point

Every special field type is an `agtype` subclass.  The class object itself is
stored in the schema (never an instance).  All interface methods are
`@classmethod`.

```
agtype
  ├─ agfile        text I/O through sandbox filesystem
  ├─ agbinary      binary I/O through sandbox filesystem
  ├─ agpath        path-only string -- never reads/writes sandbox files
  ├─ agimage       multimodal image injection
  └─ agrawstring   bypass JSON wrapping entirely
```

### Interface

| Method | Called when | Purpose |
|---|---|---|
| `schema_type() → str` | Schema serialization | Human-readable label in the system prompt (e.g. `"file"`, `"image"`) |
| `needs_sandbox() → bool` | Before skill starts | Whether sandbox filesystem access is needed before `prepare()` |
| `prepare(value, sandbox, skill, field, suffix="") → (new_value, paths)` | Before ReAct loop | Transform Python value → LLM-visible value; write sandbox files |
| `recover(value, sandbox) → (new_value, paths)` | After ReAct loop | Transform LLM-returned string → Python value; read sandbox files |
| `extra_input_prompt(field) → str` | System prompt build | Extra instruction injected for this input field |
| `extra_output_prompt(field, skill) → str` | System prompt build | Extra instruction injected for this output field |
| `get_return_tool_description(field_name) → str` | Tool spec generation | Description of the `return_<field>` tool |
| `get_return_tool_value_description(field_name) → str` | Tool spec generation | Description of the `value` parameter |
| `build_content_prompt(key, value) → tuple[str|None, list[dict]]` | User-message construction | Returns optional JSON placeholder and extra multimodal content blocks (e.g. image_url entries for agimage) |
| `validate_output(field_name, value, sandbox, exec_timeout) → str|None` | Agent calls `return_<field>` | Sandbox-side checks (file exists, well-formed, ...) run while the agent is still alive to correct an error; default no-op |
| `validate_input_value(value) → str|None` | `agschema.check()`/`check_field()`, before the skill runs | Validates a caller-supplied input value; default requires `str`, `agpath` additionally requires it to look like a path |

`prepare()` and `recover()` each return `(new_value, cleanup_paths)`.  The
framework replaces the field value in `_data` with `new_value` and accumulates
all `cleanup_paths` for deletion after the skill ends.

`suffix` is a timestamp string appended to sandbox file paths so that repeated
runs on the same persistent agent always write to distinct paths (e.g.
`/workspace/inputs/field_1751234567890.txt`).  Subclasses that do not write
files (e.g. `agimage`, `agrawstring`) accept and ignore it.

---

## `agfile` — text through the sandbox filesystem

**Use case.** Long text inputs or outputs that would overflow the LLM context if
inlined in the JSON message.

**Input flow.**

```
Caller passes: agdata(doc=agfile, content="<10k tokens of text>")

agfile.prepare():
  1. write content → /workspace/inputs/content.txt
  2. replace field value with "/workspace/inputs/content.txt"
  3. return cleanup path

LLM receives: {"doc": "...", "content": "/workspace/inputs/content.txt"}

System prompt adds:
  "Input `content`: the JSON value is a path to a temporary file in your
   sandbox. Use the read tool to access the full content..."
```

**Output flow.**

```
LLM calls: return_content("/workspace/outputs/content.txt")

_handle() validates:
  - path is a file (not directory)
  - file exists and is non-empty
  - file is UTF-8 text (not binary)
  - file does not contain only another path (double-indirection guard)

agfile.recover():
  1. read file content from sandbox
  2. return plain string
  3. return cleanup path

Caller receives: agdata(content="<actual text>")
```

**str-field path shortcut.** When an output field is typed `str` but the agent
returns something that looks like a file path (matched by `_looks_like_path`),
the framework reads the file and returns the content, printing a
`[agschema] WARNING` to stderr every time this fires. This allows agents to
write large outputs to files even for plain `str` fields without an explicit
`agfile` type — but it also means a plain `str` field can never reliably hold
a path-shaped *value* (a field genuinely meant to return a path, not content).
Use `agpath` for those fields instead — see below.

---

## `agbinary` — binary through the sandbox filesystem

**Use case.** Raw bytes — audio, images as files, compiled artifacts, compressed
data — that the LLM should never see as text.

**Input flow.**

```
Caller passes: agdata(audio=agbinary, content=b"<raw wav bytes>")
               OR a local host path: "/data/clip.wav"
               OR a base64 data URL: "data:audio/wav;base64,..."

agbinary.prepare():
  1. normalise to bytes (_to_bytes: handles bytes | path | data URL)
  2. write bytes → /workspace/inputs/content.bin via write_file_bytes()
  3. replace field value with "/workspace/inputs/content.bin"

LLM receives: {"audio": "...", "content": "/workspace/inputs/content.bin"}

System prompt adds:
  "Input `content`: a binary file at the path shown in the JSON.
   Do NOT read it as text — use shell tools (file, xxd, ...) to inspect or process it."
```

**Output flow.**

```
LLM calls: return_content("/workspace/outputs/content.bin")

_handle() validates (via shell exec):
  - path is not a directory (test -d)
  - path exists and is non-empty (test -s)

agbinary.recover():
  1. read raw bytes from sandbox via read_file_bytes()
  2. return bytes object

Caller receives: agdata(content=b"<raw bytes>")
```

---

## `agpath` — path-only string, no sandbox I/O

**Use case.** A field whose value **is** a path — a destination, a location to
hand to another tool — as opposed to content that happens to be file-backed.
Unlike every other built-in `agtype`, `agpath` never reads or writes anything
in the sandbox; `prepare()`/`recover()` are the inherited no-op passthrough.
Only the *shape* of the value is checked, on both the input and output side.

**Why this type exists.** `agfile`/`agbinary` output fields, and the
`str`-field path shortcut above, are both built around the assumption that a
path-looking value is a *pointer to the real content*, which the framework
should resolve. That assumption breaks for a field whose value is meant to
stay a path — e.g. `output_schema=agdata(path=str, content=str)`, where the
agent correctly calls `return_path("/data/note.txt")` and the framework
"helpfully" replaces `"/data/note.txt"` with the note's own file content,
because that's indistinguishable from the `content` field's shortcut case.
`agpath` opts a field out of that resolution entirely.

**Input flow.**

```
Caller passes: agdata(dest=agpath, value="/data/out.txt")

agpath.prepare(): passthrough, unchanged (inherited default)

LLM receives: {"dest": "/data/out.txt"}

check()/validate_input_value() rejects the call before the skill runs if the
value doesn't look like a path:
  "field 'dest' (agpath): 'not a path at all' does not look like a path"
```

**Output flow.**

```
LLM calls: return_moved_to("/data/out.txt")

_handle() → agpath.validate_output():
  - value must look like a path (_looks_like_path) — no file is read
  - otherwise: {"error": "field_name 'moved_to': '<value>' does not look
    like a path. Pass the path string itself, not file content."}

agpath.recover(): passthrough, unchanged (inherited default)

Caller receives: agdata(moved_to="/data/out.txt")
```

```python
move_skill = agskill(
    name="move_file",
    system_prompt="Move the file to the given destination and confirm.",
    input_schema=agdata(src=agpath, dest=agpath),
    output_schema=agdata(moved_to=agpath),
)
```

---

## `agimage` — multimodal image injection

**Use case.** Visual inputs that the LLM should see directly, not read as text.

**Input flow.**

```
Caller passes: agdata(photo=agimage, value="/path/to/photo.jpg")
               OR "https://example.com/image.png"
               OR "data:image/jpeg;base64,..."

agimage.prepare():
  - http/https/data URL → pass through unchanged
  - local path → base64-encode → "data:image/jpeg;base64,..."

build_prompt_payload():
  - collects all agimage field values (prepared data URLs)
  - replaces image field in the text JSON with "[image attached]" placeholder
  - builds a multimodal content array:
    [
      {"type": "text",      "text": "New Skill Input:\n{...json with placeholder...}"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      ...
    ]
```

**list[agimage].** Multiple images are all extracted and appended as separate
`image_url` content parts.  The text placeholder becomes `"[N image(s) attached]"`.

**Container nesting.** `agtype` subclasses can appear inside `list`, `dict`,
and `tuple` containers at any nesting depth in both input and output schemas.
`agschema.prepare_inputs_in_sandbox()` and `agschema.recover_outputs()` recurse through the
container structure and call `prepare()`/`recover()` at every agtype leaf.
Plain Python values at non-agtype positions pass through unchanged.

```python
# All of these are valid schema hints:
input_schema=agdata(frames=list[agimage])          # list of images
input_schema=agdata(batches=list[list[agimage]])   # nested list of images
output_schema=agdata(reports=dict[str, agfile])    # dict of file outputs
output_schema=agdata(result=tuple[agfile, int])    # tuple with agtype position
```

**No recover step.** `agimage` has no output path — it is input-only. An agent
that generates images would use `agbinary` or `agfile` for the output.

---

## `agrawstring` — bypass JSON entirely

**Use case.** Free-form prose in, free-form prose out. No schema, no JSON
parsing, no retry loop.

**Constraint.** Must be the **only** field in its input or output schema.

**Input path** (`_raw_input_key()` detects this).

```
Caller passes: agdata(prompt="Write a haiku about autumn.")

build_prompt_payload() → returns the string directly, no JSON wrapper.

System prompt does NOT include the "Input JSON format:" section.
```

**Large input offloading.** Unlike other `agtype` subclasses, `agrawstring` is
**not** excluded from `_offload_large_fields`.  Because its `prepare()` is a
no-op, the raw string arrives at the offload step at full length.  If it exceeds
the offload threshold it is written to a sandbox file and replaced with a
reference, exactly like a plain `str` field.  Other agtype subclasses are
excluded from offloading because their `prepare()` has already transformed the
value into a short sandbox path or data URL.

The offload threshold is `max(40 000, context_limit × 0.1 × 4)` characters —
10 % of the model context window expressed in characters (4 chars/token), with a
40 000-character floor.  The same formula applies to both input field offloading
(`_offload_large_fields`) and tool-output offloading in `_dispatch_tools`.

**Output path** (`_raw_output_key()` detects this).

```
System prompt adds:
  "Respond with plain text only — no JSON wrapping, no markdown code fences."

The raw text content is returned directly without JSON parsing.
The return_<field> tool mechanism is bypassed entirely.

Caller receives: agdata(chapter="Crimson leaves descend / ...")
```

---

## System prompt construction

`_build_system_prompt()` assembles the full system prompt in this order:

```
1. skill.system_prompt                          (always)

2. File-backed fields block                     (if any agtype field has extra_*_prompt)
   "File-backed fields — WARNING: these files are temporary..."
   - Input `background`: the JSON value is a path...
   - Output `report`: write your output to a file...

3. extra (caller-injected string)               (if provided)

4. Input JSON format:                           (if input_schema exists and not agrawstring)
   {"theme": "str", "background": "file"}

5. Output instructions                          (if output_schema exists)
   - agrawstring: "Respond with plain text only..."
   - otherwise:   "To return your results, call return_<field> tool(s)..."
                  Required fields:
                    - report: file
                    - score: integer — pass the numeric value directly
```

---

## Return tool generation

For each output field, one `return_<field>` tool is generated with a `value`
parameter whose JSON Schema type is derived from the field's type hint:

| Python hint | JSON Schema type | Example in description |
|---|---|---|
| `str` | `"string"` | `"text"` |
| `int` | `"integer"` | `42` |
| `float` | `"number"` | `3.14` |
| `bool` | `"boolean"` | `true` |
| `list`, `list[T]` | `"array"` | `["text"]`, `[42]` |
| `tuple`, `tuple[T,...]` | `"array"` | `["text", 42]` |
| `dict`, `dict[K,V]` | `"object"` | `{}`, `{"text": 42}` |
| `agtype` subclass | `"string"` | type-specific via `return_value_description` |
| `list[agtype]` | `"array"` | `["value"]` |
| `dict[K, agtype]` | `"object"` | `{"text": "value"}` |
| `tuple[agtype, ...]` | `"array"` | `["value", ...]` |
| `[{"k": T, ...}]` | `"array"` | `[{"k": "text"}]` |

The description of the `value` parameter includes a concrete valid-JSON example
and an explicit "Pass directly — do not JSON-encode into a string" note. This
prevents models from double-encoding array/object values as strings.

---

## Validation and error feedback

After the LLM calls `return_<field>`, `_handle()` validates the value and
returns an immediate tool result:

```
SUCCESS: {"result": "✓ 'field' registered. Still needed: [...] / All complete."}
FAILURE: {"error": "field 'field': <error>. <fix hint with example>"}
```

The fix hint distinguishes the most common mistake:

- **Wrong container type** (e.g. string where array expected): `"You passed a JSON-encoded string; pass a JSON array directly. Example: [...]"`
- **Type mismatch**: `"Expected format: <example>"`
- **agfile not found / empty / binary / double-path**: specific diagnostic per case
- **agbinary not found / empty**: specific diagnostic per case
- **agpath not path-shaped**: `"'<value>' does not look like a path. Pass the path string itself, not file content."` — no sandbox file is read for this check

The LLM retries up to `max_output_schema_retries` times (default 10) before
the skill returns an error result.

---

## Custom `agtype` subclasses

To add a new field type, subclass `agtype` and override the methods you need:

```python
class agcsv(agtype):
    """CSV field: input is a list[dict], written to a .csv file in the sandbox."""

    @classmethod
    def schema_type(cls) -> str:
        return "csv_file"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return True

    @classmethod
    def prepare(cls, value, sandbox, skill_name, field_name, suffix=""):
        import csv, io
        if not isinstance(value, list):
            return value, []
        buf = io.StringIO()
        if value:
            w = csv.DictWriter(buf, fieldnames=value[0].keys())
            w.writeheader(); w.writerows(value)
        path = f"/workspace/inputs/{field_name}.csv"
        sandbox.write_file(path, buf.getvalue())
        return path, [path]

    @classmethod
    def recover(cls, value, sandbox):
        import csv, io
        if not isinstance(value, str):
            return value, []
        content = sandbox.read_file(value)
        rows = list(csv.DictReader(io.StringIO(content)))
        return rows, [value]

    @classmethod
    def extra_input_prompt(cls, field_name):
        return (
            f"  - Input `{field_name}`: a CSV file at the path shown. "
            f"Use the read tool or pandas to process it."
        )
```

The same `agtype` subclass works for `field=agcsv`, `field=list[agcsv]`,
`field=dict[str, agcsv]`, `field=tuple[agcsv, int]`, and any deeper nesting —
`agschema.prepare_inputs_in_sandbox()` and `agschema.recover_outputs()` recurse through
`list`, `dict`, and `tuple` containers at any depth and call `prepare()`/`recover()`
at every agtype leaf automatically.
