# agtype

`agtype` is the base class for typed agdata field values that need type-specific custom behaviour beyond plain Python values. Subclass it to control how a schema field is serialised to JSON, transferred to/from the agent's sandbox filesystem, represented in the system prompt, and cleaned up after the skill ends.

## Why it exists

Some skill fields are too large or too structured to inline in the LLM context window.  The `agfile` subclass is the built-in example: its content lives in a sandbox file, and the LLM receives only a path.  `agimage` is the multimodal example: the image is injected directly into the message content array so the LLM sees it visually.  `agrawstring` is the escape-hatch example: it bypasses JSON formatting entirely so the model receives and returns raw text.  `agtype` makes this pattern extensible — custom field types (encrypted blobs, binary data, remote-fetched content, etc.) follow the same interface without touching the core framework.

## Interface

All methods are `@classmethod` because `agtype` subclasses are used as *type markers* in agskill schemas — the class object itself is stored, not an instance:

```python
skill = agskill(
    name="write",
    system_prompt="...",
    input_schema=agdata(theme=str, background=agfile),
    output_schema=agdata(report=agfile),
)
```

### `schema_type() -> str`

Human-readable type label shown in the JSON format hint appended to the system prompt.

| Class | Returns |
|---|---|
| `agtype` (base) | `"str"` |
| `agfile` | `"file"` |
| `agimage` | `"image"` |
| `agrawstring` | `"str"` |

### `needs_sandbox() -> bool`

Return `True` if `prepare` or `recover` require sandbox filesystem access.  The framework uses this for documentation; `prepare`/`recover` always receive the sandbox object regardless.

### `prepare(value, sandbox, skill_name, field_name, suffix="") -> tuple[transformed_value, paths]`

Called **before** the skill's ReAct loop on input fields.  `value` is the raw Python value from the caller's agdata.  Returns a `(transformed_value, paths_to_cleanup)` tuple:

- `transformed_value` replaces the field in the JSON sent to the LLM.
- `paths_to_cleanup` is a list of sandbox file paths that will be deleted in the `finally` block after the skill ends.

`suffix` is a timestamp string appended to sandbox file paths so that repeated runs on the same persistent agent always write to distinct paths.  Subclasses that write sandbox files (e.g. `agfile`, `agbinary`) use it; subclasses that do not write files (e.g. `agimage`, `agrawstring`) accept and ignore it.

Default: `(value, [])` — pass through unchanged.

### `recover(value, sandbox) -> tuple[recovered_value, paths]`

Called **after** the skill's ReAct loop on output fields.  `value` is whatever the LLM returned for this field (typically a file path or other reference).  Returns `(recovered_value, paths_to_cleanup)`.

Default: `(value, [])` — pass through unchanged.

## Container nesting

`agtype` subclasses can be placed inside `list`, `dict`, and `tuple` container hints at any nesting depth.  The framework recursively walks the hint/value structure and calls `prepare` or `recover` at every agtype leaf.  Plain Python values at non-agtype positions pass through unchanged.

```python
# list of images — each element encoded before the skill runs
input_schema=agdata(frames=list[agimage])

# dict of file outputs — each value recovered after the skill ends
output_schema=agdata(reports=dict[str, agfile])

# nested list of images
input_schema=agdata(batches=list[list[agimage]])

# tuple with mixed types — only the agfile position is prepared/recovered
output_schema=agdata(result=tuple[agfile, int])
```

The same recursive walk applies to `_offload_large_fields`: a field whose hint contains any non-`agrawstring` agtype at any depth is excluded from large-string offloading, so prepared values (data URLs, sandbox paths) are never replaced by file references.

`agrawstring` is the only agtype subclass that is **not** excluded from offloading, because its `prepare` is a no-op — the raw string arrives at the offload step unchanged and is written to a sandbox file when it exceeds the offload threshold (`max(40 000, context_limit × 0.1 × 4)` characters).

Output schema validation (`validate_value_against_type_hint`) is also recursive: it descends into `list`, `dict`, and `tuple` containers and checks each element against the corresponding inner type, reporting the exact path (e.g. `item 2: key 'x': expected str, got int`) on mismatch.

### `extra_input_prompt(field_name) -> str`

One-line instruction added to the system prompt for this input field.  Return `""` to add nothing.  Collected into a "File-backed fields" block if any non-empty lines exist.

### `extra_output_prompt(field_name, skill_name) -> str`

One-line instruction added to the system prompt for this output field.

## Built-in subclass: `agfile`

`agfile` is the standard file-backed field type.

```python
from agency import agfile
```

| Method | Behaviour |
|---|---|
| `schema_type()` | `"file"` |
| `needs_sandbox()` | `True` |
| `prepare(value, ...)` | Writes `value` to `/workspace/inputs/<field><suffix>.txt`; returns the path and adds it to cleanup. |
| `recover(value, ...)` | Reads the file at `value` (the path returned by the LLM); returns the content and adds the path to cleanup. |
| `extra_input_prompt` | Tells the agent to use the `read` tool to access the file. |
| `extra_output_prompt` | Tells the agent to write output to a file and return the path. |
| `get_return_tool_description(field_name)` | Tool description for `return_<field>` — explicitly instructs the agent to write to a file first and then pass the path. |
| `get_return_tool_value_description(field_name)` | Parameter description for the `value` argument — asks for an absolute file path, not raw content. |

### Live validation during return

When the agent calls `return_<field>` for an `agfile` field, the framework reads the file from the sandbox **immediately** (while the agent is still alive) using `sandbox.read_file()`. This is a strict UTF-8 read via base64 round-trip — it detects the three failure modes that would otherwise produce silent garbage:

| Exception raised by `read_file` | Meaning | Error sent to agent |
|---|---|---|
| `IsADirectoryError` | Path exists but is a directory (`test -d` confirmed) | Points to a file, not a directory |
| `UnicodeDecodeError` | File exists but raw bytes are not valid UTF-8 | Cannot read binary as text |
| `FileNotFoundError` | Path does not exist in the container | Write the file first |

After the exception checks pass, two content checks run:

- **Empty file** → error; the agent must write actual content.
- **File contains only another path** → error; the agent is chaining paths instead of writing content.

On success the path is stored in the collected outputs. `recover()` re-reads the content after the skill completes, so the caller always receives text, not a path.

### `sandbox.read_file` implementation

`read_file` in `agSandbox` uses `base64` encoding to capture raw bytes from the container, then decodes with strict UTF-8. This means:

- A **directory** path raises `IsADirectoryError` (verified with `test -d`), not the generic "not found" error.
- A **binary file** raises `UnicodeDecodeError` rather than silently returning garbled text via `errors="replace"`.
- A **missing path** raises `FileNotFoundError` as before.

### Example

```python
from agency import agskill, agdata, agfile

design_skill = agskill(
    name="design",
    system_prompt="Create a story design document.",
    input_schema=agdata(theme=str, background=agfile),
    output_schema=agdata(design_doc=agfile),
)
```

The framework:

1. Writes `background` content to `/workspace/inputs/background.txt` in the sandbox.
2. Sends `{"theme": "...", "background": "/workspace/inputs/background.txt"}` to the LLM.
3. Appends instructions telling the agent to read that path and to write `design_doc` to a file.
4. When the agent calls `return_design_doc(value="/workspace/outputs/design_design_doc.txt")`, validates the file immediately — directory, binary, missing, empty, and path-in-file checks all run now, while the agent can still fix them.
5. After the loop, `recover()` reads the file content from the sandbox.
6. Deletes both sandbox files in `finally`.

The caller always passes and receives plain strings — file paths are an internal detail.

## Built-in subclass: `agbinary`

`agbinary` is the binary file-backed field type. Use it when a skill consumes or produces raw bytes — audio, images, compiled binaries, PDFs, etc. — that should never be decoded as text.

```python
from agency import agbinary
```

| Method | Behaviour |
|---|---|
| `schema_type()` | `"binary_file"` |
| `needs_sandbox()` | `True` |
| `prepare(value, ...)` | Normalises the caller value to `bytes` (accepts `bytes`, a local host path `str`, or a base64 data URL `str`), writes to `/workspace/inputs/<field>.bin`, returns the path. |
| `recover(value, ...)` | Reads raw bytes from the path via `sandbox.read_file_bytes()`; returns `bytes`. |
| `extra_input_prompt` | Tells the agent the field is a binary file and to use shell tools (`file`, `xxd`, domain-specific CLIs) rather than text tools. |
| `extra_output_prompt` | Tells the agent to write a binary output file and return the path. |
| `get_return_tool_description` | Instructs the agent to write a binary file first, then pass its path. |
| `get_return_tool_value_description` | Asks for an absolute file path — explicitly says not to encode content as text. |

### Caller value types

**Input**: the caller may pass any of:
- `bytes` — raw binary data
- `str` that is a local host file path — framework reads it as bytes
- `str` that is a base64 data URL (`data:<mime>;base64,...`) — framework decodes it

**Output**: `recover` always returns `bytes`. The caller is responsible for handling the binary result (e.g. writing it to disk or passing it to another tool).

### Live validation during return

When the agent calls `return_<field>`, the framework checks the path using lightweight shell tests rather than reading the file content (binary files can be large):

| Check | Shell command | Error sent to agent |
|---|---|---|
| Is a directory | `test -d <path>` | Points to a specific binary output file |
| Does not exist | `test -e <path>` fails | Write the file first |
| Exists but empty | `test -s <path>` fails, `test -e` passes | Write actual binary content |

On success the path is stored. `recover()` reads the raw bytes via `sandbox.read_file_bytes()` after the skill completes.

### Note on agent tooling

The agent's sandbox image must include whatever CLI tools are needed to process the binary format — for example `ffmpeg` for audio/video, `ImageMagick` for images, or `sox` for audio. The base image does not include domain-specific tools by default.

### Example

```python
from agency import agskill, agdata, agbinary
from pathlib import Path

trim_skill = agskill(
    name="trim_audio",
    system_prompt="Trim the audio clip to the first 10 seconds using ffmpeg.",
    input_schema=agdata(audio=agbinary),
    output_schema=agdata(trimmed=agbinary),
)

raw = Path("full_clip.wav").read_bytes()
result = ag.run(trim_skill, agdata(audio=raw))
Path("trimmed.wav").write_bytes(result.trimmed)
```

The framework:

1. Writes `audio` bytes to `/workspace/inputs/audio.bin` in the sandbox.
2. Sends `{"audio": "/workspace/inputs/audio.bin"}` to the LLM with a note to use shell tools.
3. When the agent calls `return_trimmed(value="/workspace/outputs/trimmed.bin")`, validates the path with `test -d` / `test -s` / `test -e` — no content read.
4. After the loop, `recover()` reads the raw bytes via `read_file_bytes()`.
5. Deletes both sandbox files in `finally`.

The caller receives `bytes`, not a file path.

## Built-in subclass: `agimage`

`agimage` is the multimodal image field type. The image is injected into the OpenAI-compatible message content array so the model receives it as a visual input rather than text.

```python
from agency import agimage
```

| Method | Behaviour |
|---|---|
| `schema_type()` | `"image"` |
| `needs_sandbox()` | `False` |
| `prepare(value, ...)` | Local file paths are base64-encoded into a data URL. HTTP/HTTPS URLs and existing data URLs pass through unchanged. |
| `recover(value, ...)` | No-op — images are input-only. |
| `extra_input_prompt` | Tells the agent the field is an image attached to the message. |

### Single image

```python
from agency import agskill, agdata, agimage

describe_skill = agskill(
    name="describe",
    system_prompt="Describe what you see in the image.",
    input_schema=agdata(question=str, photo=agimage),
)

# Local file, URL, or data URL all work
result = agent.run(describe_skill, agdata(
    question="What is in this diagram?",
    photo="/tmp/architecture.png",         # encoded to data URL automatically
))
```

### List of images

```python
compare_skill = agskill(
    name="compare",
    system_prompt="Compare these frames and describe what changed.",
    input_schema=agdata(question=str, frames=list[agimage]),
)

result = agent.run(compare_skill, agdata(
    question="What changed between frames?",
    frames=["https://example.com/before.png", "https://example.com/after.png"],
))
```

### What the LLM receives

The user message content becomes a multimodal array. Image field values are replaced with a short placeholder in the text portion so no raw base64 appears in the JSON:

```json
[
  {"type": "text",      "text": "{\"question\": \"What is this?\", \"photo\": \"[image attached]\"}"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]
```

For `list[agimage]` with N images, the placeholder reads `"[N image(s) attached]"` and N `image_url` entries follow in the array.

### Requirements

The model in `llm_config` must support multimodal input (vision). Passing `agimage` fields to a text-only model will produce a provider-side error or silently ignored images depending on the backend.

## Built-in subclass: `agrawstring`

`agrawstring` bypasses the JSON input/output contract entirely. The caller's string is sent as the raw user message; the model's full text response is captured as-is, with no JSON parsing or retry loop.

```python
from agency import agrawstring
```

| Method | Behaviour |
|---|---|
| `schema_type()` | `"str"` |
| `needs_sandbox()` | `False` |
| `prepare(value, ...)` | No-op — value passes through unchanged. |
| `recover(value, ...)` | No-op — output is captured at the skill level, not via recover. |
| `extra_input_prompt` | None — JSON format hint is suppressed entirely. |
| `extra_output_prompt` | None — replaced by "Respond with plain text only." |

### Large input offloading

Because `prepare` is a no-op, a long `agrawstring` value arrives at the offload step at full length. Unlike other `agtype` subclasses (whose `prepare` already transforms the value into a short path or data URL), `agrawstring` is **not** excluded from `_offload_large_fields`. If the value exceeds the offload threshold it is written to `/workspace/inputs/<skill>_<field>.txt` in the sandbox and replaced with a reference, exactly like a plain oversized `str` field.

The threshold is `max(40 000, context_limit × 0.1 × 4)` characters — 10 % of the model context window in characters (4 chars/token), with a 40 000-character floor.

### Constraint

`agrawstring` must be the **only** field in its input or output schema. If multiple fields are present, the framework falls back to normal JSON mode silently.

### Input

```python
from agency import agskill, agdata, agrawstring

write_skill = agskill(
    name="write_chapter",
    system_prompt="You are a novelist. Write the chapter as requested.",
    input_schema=agdata(prompt=agrawstring),
    output_schema=agdata(chapter=agrawstring),
)

result = ag.run(write_skill, agdata(prompt="Write a dark opening scene set on a space station."))
print(result.chapter)   # the model's prose, unmodified
```

The user message sent to the model is `"Write a dark opening scene set on a space station."` — no JSON wrapping, no schema hint.

### Output

When the output schema is a single `agrawstring` field, the model's complete text response (including newlines, quotes, and any markdown) is stored verbatim under that field. The JSON parsing step and retry loop are both skipped.

### Input-only or output-only

`agrawstring` can appear on either side independently:

```python
# Raw input, structured output
extract_skill = agskill(
    name="extract",
    system_prompt="Extract the key facts from the text.",
    input_schema=agdata(text=agrawstring),
    output_schema=agdata(facts=str, confidence=float),
)

# Structured input, raw output
prose_skill = agskill(
    name="prose",
    system_prompt="Write prose based on the outline.",
    input_schema=agdata(outline=str, tone=str),
    output_schema=agdata(prose=agrawstring),
)
```

### When to use

Use `agrawstring` when:

- The input is a long freeform prompt that should reach the model as-is (no JSON quoting artifacts).
- The output is long prose, code, or markdown that the model cannot reliably produce inside a JSON string (escaping issues, generation-length pressure).
- You want to avoid the JSON retry loop for generative tasks where any response is acceptable.

Avoid it when the output needs structured fields that downstream code will inspect — use `agfile` or plain schema fields instead.

## Module-level helpers

### `validate_output_field_against_schema(field_name, value, schema) -> str | None`

Validates a single output field value against its type hint declared in `schema._data`.  Returns `None` when the value is valid, or a human-readable error string when it is not.

```python
from agency.agtype import validate_output_field_against_schema

error = validate_output_field_against_schema("count", "oops", output_schema)
# error -> "field 'count': expected int, got str"
```

This is the public replacement for the former private `_validate_output_field_against_schema`.

> **Note**: `_type_error_fix()` and `make_field_handler()` were removed from `agtype.py` during the refactor and now live in `agschema.py`.

## Writing a custom agtype

```python
from agency.agtype import agtype

class agencrypted(agtype):
    """Field type that transparently encrypts/decrypts content."""

    @classmethod
    def schema_type(cls) -> str:
        return "encrypted_str"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False

    @classmethod
    def prepare(cls, value, sandbox, skill_name, field_name):
        # encrypt before sending to the LLM
        encrypted = _my_encrypt(value)
        return encrypted, []

    @classmethod
    def recover(cls, value, sandbox):
        # decrypt after the LLM returns it
        return _my_decrypt(value), []
```

Use it in a schema exactly like `agfile`:

```python
skill = agskill(
    name="secure_summary",
    system_prompt="Summarise the encrypted document.",
    input_schema=agdata(doc=agencrypted),
    output_schema=agdata(summary=agencrypted),
)
```
