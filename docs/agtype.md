# agtype

`agtype` is the base class for typed agdata field values that need type-specific custom behaviour beyond plain Python values. Subclass it to control how a schema field is serialised to JSON, transferred to/from the agent's sandbox filesystem, represented in the system prompt, and cleaned up after the skill ends.

## Why it exists

Some skill fields are too large or too structured to inline in the LLM context window.  The `agfile` subclass is the built-in example: its content lives in a sandbox file, and the LLM receives only a path.  `agimage` is the multimodal example: the image is injected directly into the message content array so the LLM sees it visually.  `agtype` makes this pattern extensible — future field types (encrypted blobs, binary data, remote-fetched content, etc.) follow the same interface without touching the core framework.

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

### `needs_sandbox() -> bool`

Return `True` if `prepare` or `recover` require sandbox filesystem access.  The framework uses this for documentation; `prepare`/`recover` always receive the sandbox object regardless.

### `prepare(value, sandbox, skill_name, field_name) -> tuple[transformed_value, paths]`

Called **before** the skill's ReAct loop on input fields.  `value` is the raw Python value from the caller's agdata.  Returns a `(transformed_value, paths_to_cleanup)` tuple:

- `transformed_value` replaces the field in the JSON sent to the LLM.
- `paths_to_cleanup` is a list of sandbox file paths that will be deleted in the `finally` block after the skill ends.

Default: `(value, [])` — pass through unchanged.

### `recover(value, sandbox) -> tuple[recovered_value, paths]`

Called **after** the skill's ReAct loop on output fields.  `value` is whatever the LLM returned for this field (typically a file path or other reference).  Returns `(recovered_value, paths_to_cleanup)`.

Default: `(value, [])` — pass through unchanged.

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
| `prepare(value, ...)` | Writes `value` to `/workspace/inputs/<skill>_<field>.txt`; returns the path and adds it to cleanup. |
| `recover(value, ...)` | Reads the file at `value` (the path returned by the LLM); returns the content and adds the path to cleanup. |
| `extra_input_prompt` | Tells the agent to use the `read` tool to access the file. |
| `extra_output_prompt` | Tells the agent to write output to a file and return the path. |

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

1. Writes `background` content to `/workspace/inputs/design_background.txt` in the sandbox.
2. Sends `{"theme": "...", "background": "/workspace/inputs/design_background.txt"}` to the LLM.
3. Appends instructions telling the agent to read that path and to write `design_doc` to a file.
4. After the loop, reads the file at the path the LLM returned for `design_doc`.
5. Deletes both sandbox files in `finally`.

The caller always passes and receives plain strings — file paths are an internal detail.

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
