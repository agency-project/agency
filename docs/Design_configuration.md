# Design: Configuration

How to configure an agent, team, or any other framework object — and how to add your own tunable when the framework's aren't enough — from a user's perspective. For how any of this is implemented (descriptor mechanics, locking, `FIELD_REGISTRY`), see [`agconfig.md`](agconfig.md).

## The mental model

One `agConfig` object carries every tunable for everything it's passed to. You build it once, hand it to `agent(agconfig=cfg)` (or an `agteam` subclass's `agconfig` class attribute), and it flows to that agent's LLM backend, sandbox, and every skill/tool it runs — no separate config object per subsystem, no threading dozens of keyword arguments through constructors.

```python
from agency import agent
from agency.agconfig import agConfig
from agency.agllm_backend import agVLLMBackendConfig
from agency.agsandbox import agSandboxConfig

cfg = agConfig(
    agVLLMBackendConfig(base_url="http://localhost:8000/v1", model="...", api_key="EMPTY"),
    agSandboxConfig().add_mount("data", "/host/path", "/data"),
)
ag = agent(agconfig=cfg)
```

Every object that takes `agconfig=` clones it at construction time, so `ag.agconfig`, `ag.llm._agconfig`, and `ag.sandbox._agconfig` are independent copies of `cfg`, not the same object — this is what lets two agents built from one shared `cfg` diverge safely later. See "Changing a Dynamic field live" below for how to update config after construction now that mutating `cfg` itself no longer reaches anything already built from it.

## The canonical form: `agConfig(agXXXConfig(...), ...)`

Every framework owner (the LLM backend, the agent itself, the sandbox, tools, skills, ...) has a matching `agXXXConfig` class — `agLLMBackendConfig`, `agAgentConfig`, `agSandboxConfig`, `agToolConfig`, and so on. Always build your `agConfig` by passing one or more of these into `agConfig(...)`:

```python
cfg = agConfig(agVLLMBackendConfig(model="...", api_key="..."))              # one owner
cfg = agConfig(agVLLMBackendConfig(model="..."), agAgentConfig(react_max_steps=20))  # several
```

**Never** write `agVLLMBackendConfig(model="...").agconfig` — chaining `.agconfig` directly off a view. Both forms produce a working `agConfig`, but only the first scales cleanly from one owner to several without changing shape: adding a second owner to an existing call is a one-line diff (`agConfig(view_a, view_b)`), whereas the `.agconfig` form has no natural way to combine two views at all short of switching to the `agConfig(...)` form anyway. Use one convention everywhere and this question never comes up again.

Each `agXXXConfig(...)` call validates its keyword arguments against that owner's real fields — a typo'd field name raises `TypeError` immediately, naming the bad field, rather than the framework silently reading a default forever:

```python
agVLLMBackendConfig(mdel="claude-...")   # TypeError: agVLLMBackendConfig has no field(s) ['mdel']
```

## The tiered parameter system, from where you're standing

Every field belongs to one of three tiers. The tier isn't something you choose when *using* a field — it's fixed by whoever defined it — but it determines what you're allowed to do with it and when:

| Tier | What it means for you | Can I change it after first use? |
|---|---|---|
| **Dynamic** (most fields — LLM sampling params, timeouts, retry counts, `react_max_steps`, ...) | Read fresh every time it's used. Change it on a live `agConfig` and the very next thing that reads it sees the new value. | Yes, any time, on the same `agConfig`. |
| **Static** (a handful of fields tied to a physical resource — the sandbox's Docker image, its bind mounts) | Resolved once, the first time something that uses it is created (e.g. the first sandbox container), then fixed for that thing's whole lifetime. | Not on the same `agConfig` once something has read it — see below. |
| **Global** (process-wide tunables — worker-pool sizes, semaphore limits) | Shared by every agent/team in the process, not scoped to your `agConfig` at all. | Not once anything, anywhere in the process, has read it. |

You can always tell which tier a field is from its behavior, but you don't need to look it up ahead of time — just try the field you want to change:

- **It's a normal field on a running agent** (`cfg.agllm_backend.temperature = 0.9`, `cfg.agent.react_max_steps = 50`) → almost always Dynamic. It just works, immediately, on the next call.
- **It's a sandbox image/mount, and you've already run something on this agent** → Static. Setting it raises `ValueError` with a message telling you to `clone()` first (see below) — that's not a bug, it's the framework telling you the old value already went into a running container that can't retroactively change.
- **It's a global worker-pool/semaphore-style setting, and *anything* in the process has already used it** → Global. Same `ValueError`-on-write-after-read behavior, but process-wide instead of per-`agConfig` — cloning your `agConfig` doesn't help, since a tier-1 field never lived on your `agConfig` in the first place. Set these, if you're going to, before constructing anything.

### Changing a Dynamic field live

Every framework object that takes `agconfig=` clones it at construction time — `ag.agconfig`, `ag.llm._agconfig`, `ag.sandbox._agconfig`, etc. are each independent copies of whatever you passed in, not the same object. This means mutating your original `cfg` (or even `ag.agconfig`) after construction does **not** reach `ag.llm` — each object only sees writes made through its *own* `agconfig`. This is deliberate: it's what stops two agents built from the same `cfg` from silently changing each other's behavior.

To change a Dynamic field live, reach into the specific object whose behavior you want to change:

```python
ag = agent(agconfig=cfg)
ag.run(skill, agdata(...))                    # call 1, temperature=0.7 (say)
ag.llm._agconfig.agllm_backend.temperature = 0.2
ag.run(skill, agdata(...))                    # call 2, sees temperature=0.2 immediately
```

No new agent, no sandbox teardown — the field is re-read from `ag.llm._agconfig` on every LLM call. Mutating `cfg` or `ag.agconfig` instead would have no effect on `ag.llm`, since `ag.llm._agconfig` is its own clone.

You can also replace an object's config wholesale instead of mutating it in place:

```python
ag.llm._agconfig = agConfig(agVLLMBackendConfig(model="...", temperature=0.2))
```

See `examples/dynamic_config_example.py` for this pattern end to end.

### Changing a Static field — the clone-and-recreate pattern

A sandbox's mounts (and its base image) are resolved once, when the sandbox container is first created, and then fixed for that container's whole lifetime — a running container's bind mounts genuinely can't change without recreating it, so the framework raises rather than silently ignoring your change:

```python
agSandboxConfig(cfg).add_mount("data", new_host_dir, "/data")
# ValueError: agSandbox.mounts was already read as static on this agConfig;
# clone() it first to change it for new objects
```

The fix is exactly what the message says — `clone()` for a fresh `agConfig` with no lock history, apply the change there, and make sure whatever creates the *next* sandbox uses the clone instead of the original. Since `ag.agconfig` is already an independent clone (see above), mutate it directly rather than the `cfg` you originally built:

```python
cfg2 = ag.agconfig.clone()
agSandboxConfig(cfg2).add_mount("data", new_host_dir, "/data")
ag.sandbox.destroy()      # tear down the old container -- it's still on the old mount
ag.sandbox  = None        # agskill only provisions a new sandbox when this is None
ag.agconfig = cfg2        # the next run() creates a fresh sandbox that resolves "data" from cfg2
```

See `examples/config_example.py` for this pattern end to end, contrasted directly against a Dynamic field update on the same agent.

### Global fields

Set these, if you need non-default values, before constructing anything that might read them — ideally right after building your `agConfig`, before the first `agent(...)`/`agteam(...)` call in your process:

```python
cfg = agConfig(agToolConfig(pool_max_workers=64))   # fine, if nothing has used agtool's pool yet
ag = agent(agconfig=cfg)
```

If you're not sure whether something in the framework has already read a given global field (another agent constructed earlier in the same process, a test that ran before yours, ...), the write will simply raise and tell you — there's no way to "guess wrong" silently.

## Passing config to an `agteam`

An `agteam` subclass takes `agconfig` as a class attribute (a default shared by every instance) or a constructor argument (a per-instance override):

```python
class MyTeam(agteam):
    agconfig = agConfig(agVLLMBackendConfig(model="...", api_key="..."))

    def setup(self) -> None:
        self.main_agent = agent()   # no agconfig= given -- inherits the team's agconfig outright
```

Any `agent(...)` created inside `setup()`/`run()` with no explicit `agconfig=` inherits the active team's `agconfig` wholesale — not just LLM fields, but sandbox mounts, log/output dirs, and anything else set on it, at the moment the agent is constructed. Like every other framework object, the agent clones the team's `agconfig` rather than sharing it, so a later change to `team.agconfig` (or to the original `cfg` the team was built from) does not retroactively affect agents already constructed — only agents created *after* the change pick it up. Pass `agconfig=` explicitly to an `agent(...)` call inside a team to give that one agent a different config (e.g. a `.clone()` with one field overridden) instead.

## Adding your own custom config param

The framework's fields cover the framework's own classes. If you're building something on top — a custom `agteam` subclass, a custom tool that wants its own tunable retry count, a custom skill wrapper — and you want that tunable to live on the same `agConfig` your agents already carry (so it composes with everything else, shows up in the same place, and gets the same typo-checking), define your own owner the same way the framework defines its own:

```python
from agency.agconfig import DynamicConfigParam, _AgConfigViewBase

class _MyToolFields:
    retry_count = DynamicConfigParam("my_tool", default=3)

class myToolConfig(_AgConfigViewBase):
    _OWNER = "my_tool"
```

Now `myToolConfig` composes with the framework's own views exactly like any other:

```python
cfg = agConfig(
    agVLLMBackendConfig(model="...", api_key="..."),
    myToolConfig(retry_count=5),
)
```

And your tool reads it the same way any framework class reads its own tunables — inherit `_MyToolFields` (so `self.retry_count` works as a plain attribute once `self._agconfig` is set), or use a throwaway instance (`_MyToolFields(agconfig).retry_count`) if your class doesn't hold a persistent `_agconfig` of its own. Pick `DynamicConfigParam` unless your tunable genuinely represents something physically fixed once created (→ `StaticConfigParam`) or something process-wide (→ `GlobalConfigParam`) — see [`agconfig.md`](agconfig.md) ("The three tiers") for the full tier semantics, and always inline the literal default (`default=3`), never a separate named constant — a `default=SOME_CONSTANT` is a trap the moment something later reassigns `SOME_CONSTANT` expecting it to matter.

Pick an owner string that won't collide with an existing one — `agConfig.FIELD_REGISTRY` raises `ValueError` at import time if two classes try to register the same `(owner, name)` pair, so a collision is a loud, immediate error, not a silent field takeover — but the string itself is otherwise just a namespace you choose.

## Restricting a view to a subset of fields

If your custom config view has fields that only make sense in certain combinations (the way `agAnthropicBackendConfig` only accepts the subset of `agllm_backend`'s fields the Anthropic backend actually reads), set `_ALLOWED_FIELDS` on your subclass:

```python
class myRestrictedToolConfig(myToolConfig):
    _ALLOWED_FIELDS = frozenset({"retry_count"})
```

Anything passed to `update()`/the constructor outside that set raises `TypeError`, even if it's a field registered under the same owner elsewhere. See [`agconfig.md`](agconfig.md) ("`_ALLOWED_FIELDS`") for the full mechanics, including how `provider` gets fixed non-overridably on the framework's own provider-specific backend classes using this same hook.

## Quick reference: what to do when

| I want to... | Do this |
|---|---|
| Set several fields on one owner in one call | `agConfig(agXXXConfig(field=value, ...))` |
| Set fields on several owners at once | `agConfig(agXXXConfig(...), agYYYConfig(...))` |
| Change an LLM param, timeout, or similar mid-run | `ag.llm._agconfig.owner.field = value` (target the specific object's own `agconfig`, not `cfg`/`ag.agconfig`) — works immediately if it's Dynamic (most fields are) |
| Change a sandbox mount/image after the first sandbox exists | `cfg.clone()`, apply the change to the clone, tear down and let the next `run()` recreate the sandbox from the clone |
| Set a process-wide tunable (a worker-pool size, ...) | Do it once, early, before constructing anything that might read it |
| Add a tunable for my own tool/team | Define a `_AgXXXFields` class + a matching `agXXXConfig(_AgConfigViewBase)` with a unique `_OWNER` string, same as any framework owner |
| Restrict which fields a config view accepts | Set `_ALLOWED_FIELDS` (a `frozenset` of field names) on the view subclass |
