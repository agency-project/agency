# agdata

`agdata` is the universal data container used for skill inputs, skill outputs, and schema definitions. It behaves like a plain `**kwargs` bag with JSON serialization, lazy resolution of async results, and first-class support for Python type objects as schema field markers.

## Construction

```python
from agency import agdata

# Plain data bag
d = agdata(text="hello", count=3)

# From dict or JSON
d = agdata.from_dict({"text": "hello", "count": 3})
d = agdata.from_json('{"text": "hello", "count": 3}')
```

## Field access

```python
d = agdata(summary="short summary", word_count=42)
print(d.summary)      # "short summary"
print(d.word_count)   # 42
d.new_field = "added" # set a field
```

Accessing a missing field raises `AttributeError`. Accessing any field on a failed result raises `AgError`. Use `agdata.error` (a property) to read the error message without raising, or `is_error()` for a boolean check.

## Pending state

`agent.run()` and `agteam.run()` return a **pending** `agdata` immediately without blocking. The result resolves lazily when a field is first accessed.

```python
result = ag.run(skill, agdata(text="..."))
# result is pending here — no blocking yet

print(result.summary)   # blocks here until the skill finishes
```

Use `is_pending()` to check without triggering resolution. Use `wait()` or `wait_all()` as explicit barriers:

```python
result.wait()                  # block until resolved, return self
agdata.wait_all([r1, r2, r3])  # block until all three resolve
```

## Serialization

```python
d.to_dict()   # → plain Python dict
d.to_json()   # → JSON string
```

Nested `agdata` objects, dicts, and lists are serialized recursively. Python type objects stored as field values (used in schema agdata) are converted to their string names or `schema_type()` labels.

## Schema agdata

When used as `input_schema` or `output_schema` in an `agskill`, field values are **Python type objects** rather than data:

```python
from agency import agdata, agfile

input_schema  = agdata(theme=str, word_count=int, enabled=bool)
output_schema = agdata(report=agfile, score=float)
```

Supported schema types:

| Type | JSON hint sent to LLM |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"float"` |
| `bool` | `"boolean"` |
| `list` | `"list"` |
| `dict` | `"object"` |
| `agfile` | `"file"` |
| any `agtype` subclass | `cls.schema_type()` |

See [agskill.md](agskill.md) for how schemas are used in validation and system prompt generation, and [agtype.md](agtype.md) for defining custom typed field values.

## Error handling

```python
result = ag.run(skill, agdata(text="..."))

if result.is_error():
    print(result.error)   # error message string, no raise
else:
    print(result.summary) # raises AgError if is_error() is True
```

`AgError` is a subclass of `RuntimeError`. It is raised when any field other than `.error` is accessed on a failed result.

## API summary

| Method / property | Description |
|---|---|
| `agdata(**kwargs)` | Construct with keyword fields |
| `agdata.from_dict(d)` | Construct from plain dict |
| `agdata.from_json(s)` | Construct from JSON string |
| `d.to_dict()` | Serialize to plain dict |
| `d.to_json()` | Serialize to JSON string |
| `d.<field>` | Read a field (blocks if pending; raises `AgError` on error) |
| `d.<field> = v` | Write a field |
| `d.error` | Read error message without raising (`None` if healthy) |
| `d.is_error()` | `True` if the result holds a skill error |
| `d.is_pending()` | `True` if the future has not yet resolved |
| `d.wait()` | Block until resolved; return `self` |
| `agdata.wait_all(list)` | Block until all items resolve; return the list |
