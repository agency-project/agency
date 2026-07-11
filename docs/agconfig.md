# agconfig

`agconfig.py` implements the framework's configuration system: `agConfig` (the storage object), three `ConfigParam` descriptor tiers (`GlobalConfigParam`/`StaticConfigParam`/`DynamicConfigParam`), `_OwnerView` (the `cfg.<owner>.<field>` nested-attribute syntax), and `_AgConfigViewBase` (the `agXXXConfig` classes — `agLLMBackendConfig`, `agAgentConfig`, `agSandboxConfig`, ...).

This doc covers the implementation. For the user-facing perspective — how to configure an agent, how to add your own custom tunables, what the three tiers mean when you're just calling the framework rather than extending it — see [`Design_configuration.md`](Design_configuration.md).

## Storage model

`agConfig` is a nested override store: `data[owner][name] = value`. `owner` is a short string identifying which class a field belongs to (`"agllm_backend"`, `"agSandbox"`, `"agtool"`, ...); `name` is the field name. There is no schema beyond what's registered in `FIELD_REGISTRY` (below) — `agConfig` itself doesn't know or care what any given owner/name pair means.

```python
cfg = agConfig()
cfg.data                      # {} initially
cfg.set("agllm_backend", "model", "claude-sonnet-5")
cfg.data                      # {"agllm_backend": {"model": "claude-sonnet-5"}}
```

Four primitive operations, all on `agConfig` itself:

| Method | Tier | Behavior |
|---|---|---|
| `get(owner, name, default)` | 3 (dynamic) | Live read. Never locks. |
| `get_static(owner, name, default)` | 2 (static) | Locks `(owner, name)` **on this instance** against further `set()`. |
| `set(owner, name, value)` | — | Writes the value. Raises `ValueError` if that key is locked on this instance. |
| `clone()` | — | `agConfig(self.data)` — same data, no lock history. |

Nothing above tier/descriptor machinery cares which tier a field is — `get_static`/`set`/lock-tracking are generic. The three `ConfigParam` classes just decide *which* of `get`/`get_static`/`set` to call and *which* `agConfig` instance to call it on.

## The three tiers

A consuming class declares its tunables as class attributes, using one of three descriptor types — real Python descriptors (`__get__`/`__set__`/`__set_name__`), not plain values:

```python
class _AgToolFields:
    pool_max_workers = GlobalConfigParam("agtool", default=256)
    timeout_s        = DynamicConfigParam("agtool", default=1800)
```

| Tier | Class | Scope | Locks on | Settable through the descriptor |
|---|---|---|---|---|
| 1 | `GlobalConfigParam` | Process-wide — every `agConfig` instance sees the same value | First **read**, process-wide | Yes, until first read |
| 2 | `StaticConfigParam` | Resolved once per **consuming instance**, then cached on that instance | First **read**, on the `agConfig` instance backing that consumer | No — `__set__` always raises `AttributeError` |
| 3 | `DynamicConfigParam` | Live — re-read from the `agConfig` on every access | Never | Yes, any time |

### `_ConfigParam` base

```python
class _ConfigParam:
    def __init__(self, owner: str, default: Any) -> None:
        self.owner = owner
        self.default = default          # frozen at class-body-execution time
        self.name: str | None = None

    def __set_name__(self, objtype, name) -> None:
        self.name = name                 # PEP 487 -- told our attribute name automatically
        agConfig.FIELD_REGISTRY[(self.owner, name)] = self   # (simplified; see below)
```

`default` is evaluated **once**, when the `ConfigParam(...)` call executes (i.e. at class-body/import time), and stored as a plain attribute. This is the single most important implementation detail in the whole module:

> **A `ConfigParam`'s `default=` never re-reads anything. If you write `default=SOME_CONSTANT`, `self.default` becomes a frozen copy of whatever `SOME_CONSTANT` held at import time — reassigning `SOME_CONSTANT` afterward has zero effect on the descriptor.**

This is why the framework never writes `default=SOME_MODULE_CONSTANT` for a value meant to be tunable — the literal is inlined directly into the `ConfigParam(...)` call (`default="agency-sandbox:latest"`, `default=1800`, ...). A real bug from exactly this pattern: `agSandbox.BASE_IMAGE = "my-image:latest"` used to be a class attribute that `base_image = StaticConfigParam("agSandbox", default=BASE_IMAGE)` read once, at import time. Code elsewhere that reassigned `agSandbox.BASE_IMAGE` later, expecting to change the sandbox's image, was a complete no-op — every sandbox kept using the frozen import-time default, since nothing ever reads `BASE_IMAGE` again after `__set_name__` runs. The fix was twofold: inline the literal into the descriptor's `default=`, and — for code that legitimately needs the same frozen value elsewhere (docs, tests, a function's own default argument) — read it off the descriptor itself: `_AgSandboxFields.base_image.default`, not a separate plain constant. Accessing a `ConfigParam` on the **class** (not an instance) returns the descriptor object unchanged (see `__get__` below), so `.default` is always available this way.

### `GlobalConfigParam` (tier 1)

```python
class GlobalConfigParam(_ConfigParam):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self                                        # class-level access
        return agConfig.GLOBAL.get_static(self.owner, self.name, self.default)

    def __set__(self, obj, value):
        agConfig.GLOBAL.set(self.owner, self.name, value)       # raises if already locked
```

Every `GlobalConfigParam` read/write goes to the single shared `agConfig.GLOBAL` instance, **ignoring** whichever `agConfig` the object holds. It's implemented as tier-2 machinery (`get_static`/`set`) permanently pointed at one instance — there's no separate tier-1 code path. The consequence: once *any* object *anywhere in the process* reads a given `(owner, name)` global field, it's locked for the rest of the process. Two unrelated objects, with two different (or no) `agConfig`s, both see and are both blocked from changing the same value — by design, since a tier-1 field represents something that's physically process-wide (a worker-pool size, a semaphore limit), not something that can differ per agent.

### `StaticConfigParam` (tier 2)

```python
class StaticConfigParam(_ConfigParam):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cache = obj.__dict__.setdefault("_static_cache", {})
        if self.name not in cache:
            agconfig = getattr(obj, "_agconfig", None)
            cache[self.name] = (
                agconfig.get_static(self.owner, self.name, self.default)
                if agconfig is not None else self.default
            )
        return cache[self.name]

    def __set__(self, obj, value):
        raise AttributeError(f"{self.owner}.{self.name} is a tier-2 field, fixed once at construction; "
                              f"construct a new instance with a different (or cloned) agconfig instead")
```

Resolved on **first access** per consuming instance, then cached in `obj.__dict__["_static_cache"]` — every later read on that instance returns the cached value without touching `agConfig` again. This also locks that key on the *backing* `agConfig` instance (via `get_static`), so a later `.set(...)` on that same `agConfig` raises. The intent: a tier-2 field represents a resource that's physically fixed once the consumer is constructed (a running container's image, its bind mounts) — trying to change it after the fact would silently do nothing to the actual resource, so the framework raises instead of pretending it worked. To get a different value, construct a new consumer against a different (or `.clone()`d) `agConfig` — see "Changing a tier-2 field" below.

### `DynamicConfigParam` (tier 3)

```python
class DynamicConfigParam(_ConfigParam):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        agconfig = getattr(obj, "_agconfig", None)
        return agconfig.get(self.owner, self.name, self.default) if agconfig is not None else self.default

    def __set__(self, obj, value):
        agconfig = getattr(obj, "_agconfig", None)
        if agconfig is None:
            raise AttributeError(f"{self.owner}.{self.name} can't be set -- this instance has no agconfig")
        agconfig.set(self.owner, self.name, value)
```

No caching, no locking. Every read hits `agConfig.get(...)` fresh; every write goes straight to `agConfig.set(...)`. Two instances sharing the same `agConfig` see each other's writes immediately; instances with different (or no) `agConfig`s never interfere. This is the tier almost every per-call tunable uses (LLM sampling params, timeouts, retry counts) — anything that's safe to change on a live object because nothing has "already used" the old value in a way that can't be revisited.

In practice, framework classes (`agent`, `agllm`, `agSandbox`, `aglog`, `agResourcePool`, `agteam`, ...) each `.clone()` whatever `agConfig` they're given at construction time rather than storing it as-is — so two framework objects never end up sharing the literal same `agConfig` instance just by being built from a common source, even though the "two instances sharing the same `agConfig`" behavior described above is real if you deliberately hand one `agConfig` object to two constructors that don't clone it (e.g. two test-only classes, or your own custom owner). See [Design_configuration.md](Design_configuration.md#changing-a-dynamic-field-live) for how to update a framework object's config live given this.

## `FIELD_REGISTRY` — registration without instantiation

```python
class agConfig:
    FIELD_REGISTRY: ClassVar[dict[tuple[str, str], "_ConfigParam"]] = {}
```

`_ConfigParam.__set_name__` is a Python data-model hook: it fires automatically for every class attribute assigned a descriptor, at class-**body**-execution time — i.e. the moment the module defining `_AgToolFields`/`agtool`/etc. is imported, before any instance of that class is ever constructed. This is how `agConfig` discovers every framework owner and field just by `import agency` having happened, with no explicit registration step and no need to construct a throwaway instance of every consuming class. `__set_name__` also raises `ValueError` immediately if `(owner, name)` is already registered — field names must be unique per owner, checked at import time, not silently overwritten.

```python
agConfig.FIELD_REGISTRY[("agtool", "timeout_s")]        # the DynamicConfigParam instance itself
agConfig.FIELD_REGISTRY[("agtool", "timeout_s")].default # 1800
```

## `_OwnerView` — `cfg.<owner>.<field>` nested syntax

```python
cfg = agConfig()
cfg.agllm_backend.model = "claude-sonnet-5"   # pre-configure before any agllm_backend exists
```

`agConfig.__getattr__` checks whether `name` is a known owner (any owner with at least one registered field) and, if so, returns an `_OwnerView(self, name)` — a thin proxy whose `__getattr__`/`__setattr__` dispatch to the right tier:

- **Read**: dispatches on the field's registered tier — `GlobalConfigParam` reads `agConfig.GLOBAL.get_static(...)`; `StaticConfigParam` reads `self._agconfig.get_static(...)`; `DynamicConfigParam` reads `self._agconfig.get(...)`. Reading a tier-1 or tier-2 field this way locks it, exactly as if a real consuming instance had read it through the descriptor — `cfg.agSandbox.base_image` and `_AgSandboxFields(agconfig=cfg).base_image` have the same locking effect. Only a tier-3 read is ever lock-free.
- **Write**: for a `GlobalConfigParam`, writes to `agConfig.GLOBAL` (never the wrapped instance — writing a global field onto a non-GLOBAL instance would create a dead override nothing ever reads). For anything else, writes straight to the underlying store via `self._agconfig.set(...)`, bypassing the descriptor's `__set__` entirely — this is deliberately how a tier-2 field can be *pre-configured* before any consumer exists (`cfg.agSandbox.base_image = "..."` works even though the descriptor's own `__set__` always raises `AttributeError`; the view talks to the store, not the descriptor). This only works *before* anything has locked that key — the usual `ValueError` applies once something has.
- Unknown owner or unknown field: `AttributeError`, naming what was missing.

## `_AgConfigViewBase` — the `agXXXConfig` classes

Every owner with tunable fields has a paired `agXXXConfig` class (`agLLMBackendConfig`, `agAgentConfig`, `agSandboxConfig`, ...) subclassing `_AgConfigViewBase`:

```python
class _AgConfigViewBase:
    _OWNER: ClassVar[str]                                   # set by each subclass
    _ALLOWED_FIELDS: "ClassVar[frozenset[str] | None]" = None  # None = every field under _OWNER

    def __init__(self, agconfig: "agConfig | None" = None, **fields: Any) -> None:
        self._agconfig = agconfig if agconfig is not None else agConfig()
        if fields:
            self.update(**fields)

    @property
    def agconfig(self) -> "agConfig":
        return self._agconfig

    def update(self, **fields: Any) -> "_AgConfigViewBase":
        known = {name: knob for (owner, name), knob in agConfig.FIELD_REGISTRY.items() if owner == self._OWNER}
        if self._ALLOWED_FIELDS is not None:
            known = {name: knob for name, knob in known.items() if name in self._ALLOWED_FIELDS}
        unknown = set(fields) - set(known)
        if unknown:
            raise TypeError(f"{type(self).__name__} has no field(s) {sorted(unknown)}")
        for name, value in fields.items():
            if isinstance(known[name], GlobalConfigParam):
                agConfig.GLOBAL.set(self._OWNER, name, value)
            else:
                self._agconfig.set(self._OWNER, name, value)
        return self
```

Defining one of these takes one line beyond the class statement itself:

```python
class agToolConfig(_AgConfigViewBase):
    _OWNER = "agtool"
```

Every subclass gets, for free:

- **Construction with a fresh `agConfig`**: `agToolConfig(timeout_s=60)` — same as `_AgConfigViewBase.__init__(agconfig=None, timeout_s=60)`, which creates a private `agConfig()` and calls `update(timeout_s=60)` on it.
- **Construction wrapping an existing `agConfig`**: `agToolConfig(cfg, timeout_s=60)` — sets the field(s) on `cfg` directly (still routing `GlobalConfigParam` fields to `agConfig.GLOBAL`, never onto `cfg` itself).
- **Field-name validation**: `update()` raises `TypeError` naming every unknown field at once — not a partial write, not a silent no-op. This is the main reason `agXXXConfig(...)` exists at all instead of several `cfg.owner.field = value` lines: a typo'd keyword raises immediately instead of the framework quietly reading a default forever.
- **Composability**: since `.agconfig` is just a property, and `agConfig(*sources)` (below) accepts anything with an `.agconfig` property, any `agXXXConfig(...)` instance can be passed directly into `agConfig(...)` to merge with others.

### `_ALLOWED_FIELDS` — restricting a view to a subset of its owner's fields

`_ALLOWED_FIELDS` exists so a subclass can expose only *some* of an owner's registered fields, instead of everything `FIELD_REGISTRY` has under that owner. The motivating case: `agllm_backend`'s single owner has ~30 registered fields (every parameter any backend might read — OpenAI-style generation params, vLLM sampling extensions, AWS credentials, ...), but any *one* concrete backend only reads a subset of them. `agllm_backend.py` defines four provider-specific subclasses:

```python
class _AgProviderBackendConfig(agLLMBackendConfig):
    _PROVIDER: "ClassVar[str]"

    def __init__(self, agconfig=None, **fields) -> None:
        super().__init__(agconfig)                              # no fields yet -- bypasses _ALLOWED_FIELDS
        self._agconfig.set(self._OWNER, "provider", self._PROVIDER)  # set directly, not through update()
        if fields:
            self.update(**fields)                                # NOW validated against _ALLOWED_FIELDS


class agAnthropicBackendConfig(_AgProviderBackendConfig):
    _PROVIDER = "anthropic"
    _ALLOWED_FIELDS = frozenset({
        "model", "api_key", "base_url", "context_limit", "workspace_id",
        "temperature", "top_p", "max_completion_tokens", "max_tokens", "extra_body",
    })
```

Two things worth noting in this pattern:

1. **`provider` is set directly via `self._agconfig.set(...)`, not through `self.update(provider=...)`.** `update()` enforces `_ALLOWED_FIELDS`, and `"provider"` is deliberately *not* in `agAnthropicBackendConfig._ALLOWED_FIELDS` — the whole point of picking this class is that it guarantees `agllm_backend.for_config()` routes to `_AnthropicBackend`, so the class fixes `provider` itself rather than accepting it as a settable field the caller could override to something inconsistent with the class name.
2. **Fields outside `_ALLOWED_FIELDS` raise `TypeError` even though they're registered under the same owner.** `agAnthropicBackendConfig(frequency_penalty=0.5)` raises, even though `("agllm_backend", "frequency_penalty")` is a perfectly valid `FIELD_REGISTRY` entry — because `_AnthropicBedrockCompletions.create()` (the adapter every Anthropic-family backend shares) silently drops that parameter if it's set. Restricting the *Config* class turns a value that would otherwise be silently ignored at the API-call layer into an immediate, clear construction-time error.

`_ALLOWED_FIELDS` is a generic hook on `_AgConfigViewBase`, not specific to LLM backends — any owner with fields that only make sense in certain combinations can use the same pattern.

## `agConfig(*sources)` — the variadic merge constructor

```python
def __init__(self, *sources: "agConfig | dict[str, dict[str, Any]] | _AgConfigViewBase") -> None:
    self.data: dict[str, dict[str, Any]] = {}
    self._locked_keys: set[tuple[str, str]] = set()
    self._lock = threading.Lock()
    for src in sources:
        if src is None:
            continue
        if isinstance(src, dict):
            src_data = src
        elif isinstance(src, agConfig):
            src_data = src.data
        elif hasattr(src, "agconfig"):
            src_data = src.agconfig.data
        else:
            raise TypeError(...)
        for owner, fields in src_data.items():
            self.data.setdefault(owner, {}).update(fields)
```

Each source contributes its data; a later source's field wins over an earlier one's on a conflicting `(owner, name)` — the same semantics as `{**a, **b}`, applied per-owner. A source can be:

- a plain nested dict — `agConfig({"agllm_backend": {"model": "..."}})` — the original, still-supported single-argument form;
- another `agConfig` — merges its `.data`;
- anything with an `.agconfig` property — every `agXXXConfig(...)` view qualifies automatically, with no special-casing needed beyond `hasattr(src, "agconfig")`;
- `None` — skipped, so optional views can be threaded through without an `if` at the call site.

This constructor is *always* how an `agConfig` should be built, whether from one view or several — see "The canonical form" in [`Design_configuration.md`](Design_configuration.md).

Merging copies each owner's field dict with `.update(...)` — a shallow copy per owner, not a deep copy of every value. Two `agConfig`s built by merging the same source therefore share any mutable field **value** (e.g. `agSandboxConfig`'s `mounts` dict) until one of them calls `.set(...)` on it, which replaces that owner's dict entry outright rather than mutating it in place — see `agSandboxConfig.add_mount()`'s `{**current, name: (...)}` pattern. In practice this means merging never aliases in a way that lets one merged `agConfig`'s later `.set()` calls leak into another's.

## Locking and `clone()`

Every `agConfig` instance tracks its own `_locked_keys: set[tuple[str, str]]`, populated by `get_static()` (called by tier-1 and tier-2 descriptor reads). `set()` raises `ValueError` naming the field if it's in that set — the message always says `"...clone() it first to change it for new objects"`, since that's the actual fix:

```python
cfg = agConfig(agSandboxConfig().add_mount("data", host_dir_1, "/data"))
ag = agent(agconfig=cfg)
ag.run(some_skill, some_input)          # first sandbox resolves + locks "mounts" on cfg

agSandboxConfig(cfg).add_mount("data", host_dir_2, "/data")   # raises ValueError

cfg2 = cfg.clone()                       # fresh agConfig, same data, no lock history
agSandboxConfig(cfg2).add_mount("data", host_dir_2, "/data")  # succeeds
```

`clone()` is `agConfig(self.data)` — it goes through the same variadic constructor, so it inherits the "shallow copy per owner" aliasing behavior described above: `cfg2` starts with the same field values as `cfg`, but a `.set()` on either one only ever replaces that instance's own copy of the owner dict, never the other's.

Note that `GlobalConfigParam` locks are **not** per-instance — they live on `agConfig.GLOBAL`, a single process-wide singleton, so `clone()` cannot undo a tier-1 lock. There is no "fresh GLOBAL" to get back to within one process; a tier-1 field really is fixed for the rest of the process once anything has read it.

Every consuming class that holds an `agconfig` (`agent`, `agteam`, `agllm`, `agllm_backend`, `aglog`, `agSandbox`, `agResourcePool`) builds this same `clone()` call into a symmetric pair of public methods: `change_config(new_cfg)` replaces the object's agconfig with a clone of `new_cfg` (propagating to sub-objects where relevant), and `get_config_copy()` returns a clone of the object's current agconfig. See [`Design_configuration.md`](Design_configuration.md#changing-a-dynamic-field-live) for usage.

## How to add a new owner's config fields

This is the recipe every existing owner (`agtool`, `agllm`, `agsandbox`, ...) follows, and the one to copy for a new one:

1. Define a `_AgXXXFields` class (or reuse the consuming class directly, if it doesn't need a separate registration-only stand-in) with one `GlobalConfigParam`/`StaticConfigParam`/`DynamicConfigParam` per tunable, all sharing one owner string:
   ```python
   class _AgFooFields:
       retry_count = DynamicConfigParam("agfoo", default=3)
   ```
   Inline the literal directly into `default=` — never `default=SOME_CONSTANT` (see "The three tiers" above for why).
2. Have the real consuming class inherit `_AgXXXFields` (or, if it can't hold its own `_agconfig` per-instance for some structural reason, use a throwaway instance — `_AgFooFields(agconfig)` — purely to read through the descriptors; see `agtool.py`/`agschema.py` for this pattern).
3. Add a one-line `agXXXConfig` view:
   ```python
   class agFooConfig(_AgConfigViewBase):
       _OWNER = "agfoo"
   ```
   Add `_ALLOWED_FIELDS` only if some fields don't make sense together (see the provider-specific `agllm_backend` classes above) — omit it entirely otherwise, which is the common case.

Nothing else needs registering anywhere — `__set_name__` populates `FIELD_REGISTRY` the moment the module is imported, and `agConfig(agFooConfig(...))` composes with every other owner's view immediately.

## Registered owners (reference)

| Owner | View class(es) | Tiers used |
|---|---|---|
| `agllm_backend` | `agLLMBackendConfig` (generic), `agVLLMBackendConfig`, `agOpenAIBackendConfig`, `agAnthropicBackendConfig`, `agBedrockBackendConfig` (each `_ALLOWED_FIELDS`-restricted, `provider` fixed) | Dynamic (per-call params), Global (`model_listing_timeout_seconds`, `default_max_tokens`) |
| `agllm` | `agLLMConfig` | Dynamic, Global (`call_max_concurrency`) |
| `agent` | `agAgentConfig` | Dynamic |
| `agskill` | `agSkillConfig` | Dynamic |
| `agschema` | `agSchemaConfig` | Dynamic |
| `agtool` | `agToolConfig` | Dynamic, Global (`pool_max_workers`) |
| `aglog` | `agLogConfig` | Dynamic |
| `agutil` | `agUtilConfig` | Global |
| `agResourcePool` | `agResourcePoolConfig` | Global (detection timeouts/fallbacks), Dynamic (`idle_cpus`, `idle_memory` — also the sandbox's starting CPU/memory limits) |
| `agSandbox` | `agSandboxConfig` (bespoke — adds `add_mount`/`remove_mount`/`mounts`, an unregistered structural field layered on top of `update()`) | Static (`base_image`), Global (everything else) |

See each owning module (`agllm_backend.py`, `agsandbox.py`, ...) for the exact field list — this table is about which mechanics apply, not a field reference.
