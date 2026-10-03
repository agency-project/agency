# Agents and lifecycle

```python
from agency import Agent, agent, agdata, agskill
```

`Agent is agent`. Construct with `Agent(name=None, *, sandbox=None, agconfig=None, harness=None)`. A supplied base name is allocated a unique suffix in the live registry; omitted names are generated. Configuration must select an LLM via provider or model. Precedence is explicit config, `Agent.default_agconfig`, then an active team's configured LLM; missing configuration raises `TypeError`. The chosen config is cloned. `harness` overrides its agent namespace; the default is `native`. Annotation experiment modes require native (`ValueError` otherwise).

Construction registers the agent and opens its logger/process orchestrator. The sandbox and execution engine are created when a request is admitted. `sandbox` optionally supplies an existing sandbox; share it only with deliberate resource/lifecycle coordination. `agname`, `harness`, `agconfig`, `context`, `sandbox`, `engine` are observable instance state; use the config methods for updates, rather than mutating an in-flight engine.

## Submission and ordering

| API | Return, blocking and effects |
| --- | --- |
| `run(skill, skill_input, max_steps=None)` | Pending `agdata`; waits briefly for scheduler acknowledgement, then execution continues in the background. Requires an `agdata` input (`TypeError`). A supplied `max_steps` must be a positive integer excluding bool (`ValueError`); `None` uses `skill.react_max_steps` from config, default 4096 for native. External CLI adapters may not enforce this step limit. |
| `await asyncio_run(skill, skill_input, max_steps=None)` | Waits asynchronously and returns the resolved wrapper. Same submission contract as `run`. Cancelling the waiter leaves execution running. |
| `queue_message(message)` | Returns `None`. Enqueues a retained user message in the same context sequence as runs, without starting an engine/container. Non-string or empty strings raise `TypeError`/`ValueError`. Only submissions after this enqueue inherit it. |
| `history` | Getter blocks for preceding work and returns `agdata(messages=[...])`, a snapshot of the most recent reconstructed transcript, not a cumulative history of every invocation. Setter accepts `agdata(messages=...)`, resolves it and pending context, then replaces the transcript. |
| `record_state(state, skill=None, tool=None)` | Returns `None`; writes and flushes a state event. Logging/persistence errors can propagate. |

Same-agent submissions depend on their predecessor context: they execute in sequence. Different agents can run concurrently up to orchestrator capacity. Pending results nested in input structures become dependencies; blocked requests do not occupy execution workers or start containers. Dependencies are materialized before execution. Failure/cancellation of a dependency fails dependent work; dependency cycles are rejected with error payloads.

Successful execution invokes sandbox commit before publishing output/context; commit exceptions fail the request, but a false commit return is currently ignored. Failure or cancellation preserves predecessor context. Ordinary execution failure adds a rollback notice; controlled cancellation does not. Writes through host mounts, including the output directory, survive rollback. See [results](results.md) for why field access is not a reliable error discriminator.

```python
# worker is an Agent configured as in getting started.
from agency import agdata, agskill

summarize = agskill("summarize", "Summarize the text.",
                    input_schema=agdata(text=str), output_schema=agdata(summary=str))
result = worker.run(summarize, agdata(text="An application can compose agents."))
result.wait(timeout=60)  # timeout leaves execution alive
payload = result.to_dict()
if "error" in payload:
    raise RuntimeError(payload["error"])
print(payload["summary"])
```

## Controls and configuration

`pause()`, `resume()`, `cancel(handle)`, `redirect(handle, message)` return `None`; `is_paused()` returns bool. Pause is agent-wide, persistent across requests and best-effort at the harness boundary. Paused work can still be admitted and consume a slot. See [targeting and timing](results.md#targeting-and-timing) before relying on cancellation or redirect delivery.

`get_config_copy()` returns an independent deep clone. `change_config(config)` clones the replacement and cascades to this agent's logger, existing sandbox and engine. It does not reconfigure the process orchestrator or change the constructor-selected `harness` attribute. Apply updates between runs. Class defaults `log_dir`, `output_dir` and `default_agconfig` affect construction/path selection; prefer explicit config for reproducibility. [Configuration](configuration.md) explains scopes.

`output_path` is a host `Path` or `None`; `container_output_path` is `/agent_output` or `None`. Configured output is mounted automatically when a sandbox is created. Output-path properties read current config; changing output_dir after sandbox creation does not move its existing mount. Caller owns files there. There is no `Agent.close()` or agent context manager. Keep references to background work and wait before leaving your workload; explicitly `destroy()` directly owned sandboxes when finished. Process/agent cleanup also performs best-effort teardown.

## Stable save, load and fork

| API | Contract |
| --- | --- |
| `save(path=None)` | Blocks pending context and retries until context/sandbox form a consistent snapshot. Returns `None`; creates parent directories and writes a gzip tar checkpoint. Default: `agency_runs/saves/<agname>.ckpt`. Includes transcript, retained messages, harness sessions and, if present, exported sandbox image. Filesystem/archive/runtime errors propagate. |
| `Agent.load(path, agconfig=None)` | Returns a new live agent, restores saved name, harness/context and optional image. Can block image import. Duplicate saved live names raise ValueError. Supply fresh credentials/config. Saved LLM settings fill fields whose supplied values equal namespace defaults; an explicitly supplied default value cannot override saved non-default settings. Restores saved harness. |
| `Agent.fork(src, name=None)` | Blocks until a stable source snapshot, returns an independent agent with cloned config/context and a sandbox fork if one exists. Optional name otherwise generated. No live uncommitted sandbox changes beyond the captured checkpoint; caller owns new agent's lifecycle. |
| `Agent.all()` | Returns live agents from a weak registry. Does not wait. |
| `Agent.save_all(directory=None)` | Saves live agents sequentially; returns paths. Default `agency_runs/saves`. |
| `Agent.load_all(directory=None, agconfig=None)` | Loads sorted `*.ckpt`; returns agents. Reuses already-live agents of the saved name instead of replacing them. Supplied config is used for newly loaded agents. |

Saved LLM metadata omits API keys and AWS credentials; the whole configuration is not saved. Transcripts, session blobs and filesystem contents can still contain application data. Host mounts are external to the checkpoint. Image-based Docker/Podman saves require a compatible target runtime/image; chroot saves restore with chroot. No universal cross-host portability guarantee. `cow_zfs` does not support portable agent save/export/fork; see [checkpoint restrictions](sandboxes.md#checkpoint-backends).

## Source signatures

[Source: agent.py](../../agency/agent.py)

::: agency.agent.agent
    options:
      members: ["__init__", "run", "asyncio_run", "queue_message", "redirect", "cancel", "pause", "resume", "is_paused", "history", "output_path", "container_output_path", "change_config", "get_config_copy", "record_state", "fork", "all", "save", "load", "save_all", "load_all"]
