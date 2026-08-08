# Design: Profiler Integration with the Harness Abstraction

Status: proposal, nothing implemented.
Canonical profiler spec: branch `tony/profiler` @ `f46f00b`.
Target branch: `eric/harness-profiler-integration` (HEAD, `6c87e92`).

---

## 1. Repository architecture overview

Agency composes five swappable layers. The profiler has to sit across all of
them without knowing which implementation is plugged in.

| Layer | Entry point | Notes |
|---|---|---|
| Agent | `agency/agent.py` | owns `agname`, `agconfig`, `terminal`, `log`, `sandbox`, `engine` |
| Skill run | `agskill.run()` → `_task()`, [agskill.py:450](../agency/agskill.py:450) | one daemon thread per run; the outer lifecycle boundary |
| Engine (harness) | `agharness_backend.for_config()`, [base.py:112](../agency/agharness_internal/agharness_backends/base.py:112) | 5 backends: `native`, `claude_code`, `codex`, `opencode`, `grok` |
| LLM | `agLLMTerminus`, [agllm_terminus.py:196](../agency/agharness_internal/agllm_terminus.py:196) | the **only** place a credentialed provider client is constructed |
| Sandbox | `agSandbox` → `agsandbox_backends/{container,chroot}.py` | container lifecycle, cgroups, layer commits |

Cross-cutting services, all following the same shape (host-side FastAPI class +
own token→agent registry + TCP listener + UDS listener bind-mounted into the
container):

- `agLLMTerminus` — LLM dispatch, [agllm_terminus.py](../agency/agharness_internal/agllm_terminus.py)
- `agProxyLLM` — wire-format routing, runs **in-container**, [agproxy_llm.py](../agency/agharness_internal/agproxy_llm.py)
- `agMCPServer` — control-plane tools, [agmcp_server.py](../agency/agharness_internal/agmcp_server.py)
- `agHarnessMessenger` — inbox/pause bridge, [agharness_messenger.py](../agency/agharness_internal/agharness_messenger.py)
- `agProxyPtrace` — syscall mediation, [agproxy_ptrace.py](../agency/agharness_internal/agproxy_ptrace.py)

Logging/events are separate from profiling and stay that way: `aglog._record()`
/ `_tool_call()` / `_lifecycle()` ([aglog.py:89](../agency/aglog.py:89)) write
JSONL for the user; `agwebui/emitter.py` writes SQLite for the dashboard.
Neither carries timing suitable for a profile.

**Benchmarks are not on this branch.** `benchmarks/` exists only as
`__pycache__` here; the sources live on `eric/benchmark-tests` and
`sunga/benchmark-integration`, and neither references `agprof`. See §10.

---

## 2. Current profiler architecture (`tony/profiler`, canonical)

`agency/profiler/agprof.py`, 2101 lines. Three independent subsystems that get
conflated because they share one module:

### 2.1 Span recording — *not* torch

```python
# _TimedSpan.__enter__/__exit__, agprof.py:284-330
self._t0   = time.perf_counter_ns()
self._cpu0 = time.thread_time_ns()
self._rq0  = _read_schedstat()
self._rf.__enter__()                      # the only torch call
...
_records.append((tid, name, t0, wall, cpu, runq, metadata))
```

Every number in `summary.json` comes from `_records`. torch's
`record_function` contributes **one labeled box in the kineto trace** and
nothing else; `_inject_trace_args()` ([agprof.py:959](https://github.com/)) then
writes agprof's own numbers back into those boxes as `args`. torch is a
trace-file format and a viewer, not the measurement mechanism.

Parenthood is **thread-local nesting**: `_tls.span_stack`
([agprof.py:291](https://github.com/)). The README states this as rule 1:
"Nesting = parenthood… nobody declares hierarchy explicitly."

Open spans are tracked in `_open_spans` / `_open_spans_lock`, so a session
stopped mid-flight reports `incomplete_spans` with `outcome: "interrupted"`
rather than silently dropping them.

### 2.2 Resource sampling — pure `/proc` + cgroup v2 + NVML

`_Sampler` ([agprof.py:439-806](https://github.com/)), 367 lines, zero torch.
Per tick: recursive scan of the workload cgroup tree, per-PID
`/proc/<pid>/{stat,cmdline,io}`, per-registered-container cgroup files, NVML
device + per-process GPU sampling. Containers are registered from the host by
`container_started(label, cgroup_dir, daemon_dir, daemon_kind)`
([agprof.py:212](https://github.com/)), called from
`container.py`'s `_register_prof_container()`.

Container observation is **entirely host-side**. Nothing runs inside the
container. This matters for §5.

### 2.3 Summary construction

`_build_run_summary()` (377 lines) + `_render_summary_markdown()` (~310 lines)
+ `_resource_observations()` (~125) + `_percentile`/`_latency_stats` (~40).
All torch-free. Produces `summary.json` / `summary.md` with sections for runs,
spans, LLM, tools, workload, processes, sandboxes, GPUs, GPU leases, sampling
health, and `incomplete_spans`.

### 2.4 Instrumentation inventory (57 call sites)

`git grep -n "agprof\." tony/profiler -- '*.py'` gives the full list. Grouped:

- **Lane roots:** `agskill.py:501-503` (`run{N}:{skill}:{agname}`),
  `agmap.py:63-68` (`agmap:{fn}[{i}]`)
- **Skill phases:** `agskill.py:321,345,404,417,481,555` (`resolve`,
  `sandbox:provision`, `teardown:discard`, `teardown:commit`, `prune`,
  `input:prepare`)
- **Turn/LLM/tool:** `agskill.py:606,643,656,732` (`turn{i}`, `llm:{skill}`,
  `tool_dispatch:{skill}`), `agllm.py:164,315,350,539,870,883` (`llm:sync`,
  `llm:attempt[n]`, `llm:retry_backoff`, `llm:compact` + TTFT/token
  annotations), `agtool.py:353,374,377` (`tool:{name}`)
- **Sandbox:** `agsandbox.py:148,279-366` (11 spans),
  `container.py:103,131,882` (`sync:container`, `runtime:detect`,
  `sandbox:start`)
- **Resources:** `agresources.py:440,451,491` (`sync:gpu_wait`, GPU leases)
- **Sync:** `agsync.py:95`, `agdata.py:45`
- **Lifecycle:** `agent.py:235`, `container.py:990-1033,1703`,
  `agwebui/__init__.py:292`

---

## 3. Harness abstraction architecture

Three changes matter to the profiler.

### 3.1 One `execute()` seam, five engines

```python
# base.py:112
agharness_backend.for_config(engine, agconfig) -> _NativeBackend
                                               | _ClaudeCodeBackend
                                               | _CodexBackend
                                               | _OpencodeBackend
                                               | _GrokBackend
```

`agskill.execute_harness()` ([agskill.py:477](../agency/agskill.py:477)) calls
`backend.execute()` unconditionally for **every** engine.

### 3.2 The ReAct loop left the host process

[agskill.py:468-475](../agency/agskill.py:468) states it plainly:

> `execute_react()` (the old host-process ReAct loop — LLM calls direct from
> the host, tool dispatch via `agtool.py`'s `dispatch_tools()` …) was retired
> here. Every engine, native included, now runs through `execute_harness()`
> below — native's own loop lives in a persistent in-container process.

Native's loop is `_run_react_loop()`
([_native_in_container_entrypoint.py:1026](../agency/agharness_internal/agharness_backends/_native_in_container_entrypoint.py:1026)) —
`for _ in range(max_steps)` at line 1080 is the turn boundary,
`handler(fn_args)` at line 1115 is the tool boundary. The other four engines run
a third-party CLI whose loop is opaque.

Consequences visible in the tree:

- `agllm.py`: 968 → 341 lines. `agllm.call()`, its retry loop, streaming
  reassembly, `compact()`, and `_llm_call_semaphore` are **all gone**
  ([agllm.py:88-99](../agency/agllm.py:88)).
- `agtool.py`: `dispatch_tools()` gone; 176 lines remain.
- `agency/tools/`: `bash.py`, `edit.py`, `write.py`, `resource.py` deleted.
- `agency/agmap.py`: **deleted entirely** from HEAD.

### 3.3 One credentialed choke point, and a token that already identifies runs

Every engine's LLM traffic reaches `agLLMTerminus./internal/dispatch`
([agllm_terminus.py:289](../agency/agharness_internal/agllm_terminus.py:289)).
That is architecturally guaranteed — it is the whole reason the terminus
exists. Two properties the profiler can exploit:

1. `client.chat.completions.create(**kwargs)` at line 314, with
   `next(stream_iter)` forced before the HTTP response commits — an **exact
   TTFT measurement point** for all five engines.
2. `kwargs["messages"]` carries the *full* conversation each dispatch
   ([agllm_terminus.py:221-232](../agency/agharness_internal/agllm_terminus.py:221)),
   because every engine resends it whole.

The per-run bearer token is minted and registered at, e.g.,
[claude_code.py:184-185](../agency/agharness_internal/agharness_backends/claude_code.py:184):

```python
token = uuid.uuid4().hex
terminus.register(token, ag)
```

This is the natural correlation key: it already maps 1:1 to (agent, skill run).

---

## 4. Incompatibilities

`A` = architectural (the thing being measured moved or ceased to exist).
`I` = implementation mismatch (code deleted with the profiler; restore verbatim).

| # | Profiler feature | Canonical site | What changed | HEAD target | Kind |
|---|---|---|---|---|---|
| 1 | `run{N}` lane root, `resolve`, `sandbox:provision`, `teardown:*`, `prune` | `agskill.py:321-503` | nothing — `_task()` is structurally the same | [agskill.py:288-450](../agency/agskill.py:288) | I |
| 2 | 11 `sandbox:*` spans | `agsandbox.py:148,279-366` | nothing — facade is byte-identical minus the `with` blocks | [agsandbox.py:328-364](../agency/agsandbox.py:328) | I |
| 3 | `sync:container`, `runtime:detect`, `sandbox:start` | `container.py:103,131,882` | nothing | [container.py:85,107,809](../agency/agsandbox_backends/container.py:85) | I |
| 4 | GPU leases, `sync:gpu_wait` | `agresources.py:440-491` | nothing | [agresources.py:417,451](../agency/agresources.py:417) | I |
| 5 | `agsync:join`, `sync:result_wait`, `agent:create`, `agprof.workload()` | `agsync.py:95`, `agdata.py:45`, `agent.py:235`, `agwebui/__init__.py:292` | nothing | same files | I |
| 6 | **Container cgroup registration** | `container.py:990-1033`, `_register_prof_container()` | method deleted wholesale; no cgroup discovery survives on HEAD (`grep cgroup container.py` → 1 unrelated hit at line 1057) | must be re-added to [container.py](../agency/agsandbox_backends/container.py) | I (but blocks **all** container/process/GPU-attribution metrics) |
| 7 | `input:prepare` | `agskill.py:555` | operation hoisted from `execute_react` into `execute_harness` | [agskill.py:527](../agency/agskill.py:527) | I (relocate) |
| 8 | `proc_wait` | `agskill.py:821,858` | now native-only; the other 4 engines skip it because ptrace PID exit events never arrive | [agskill.py:582-592](../agency/agskill.py:582) | A (narrowed scope) |
| 9 | **`turn{i}`** | `agskill.py:606` | host loop deleted | native: [entrypoint.py:1080](../agency/agharness_internal/agharness_backends/_native_in_container_entrypoint.py:1080); other 4: inside a third-party CLI | **A** |
| 10 | **`tool:{name}`, `tool_dispatch:{skill}`** | `agtool.py:374`, `agskill.py:732` | `dispatch_tools()` deleted; host tool set reduced to glob/grep/read/todowrite/webfetch/human | native: [entrypoint.py:1110-1119](../agency/agharness_internal/agharness_backends/_native_in_container_entrypoint.py:1110); other 4: harness-internal | **A** |
| 11 | **`llm:attempt[n]` + TTFT/token annotations** | `agllm.py:315,350` | `agllm.call()` deleted | [agllm_terminus.py:314](../agency/agharness_internal/agllm_terminus.py:314) — *better placed than before* | **A** (favourable) |
| 12 | **`llm:retry_backoff`** | `agllm.py:539` | host retry loop deleted. Retries now live in 3 unrelated layers: `_dispatch_retry_backoff_s` ([entrypoint.py:791](../agency/agharness_internal/agharness_backends/_native_in_container_entrypoint.py:791), native only), each harness CLI's internal retry (invisible — arrives as a fresh dispatch), and nowhere else. The terminus deliberately does **one** attempt, no loop ([agllm_terminus.py:47-70](../agency/agharness_internal/agllm_terminus.py:47)) | no single owner | **A** |
| 13 | **`llm:sync`** (256-slot semaphore) | `agllm.py:164` | semaphore *deleted as a concept* ([agllm.py:88](../agency/agllm.py:88)) — there is no host LLM throttle to queue behind | none; metric is meaningless | **A** (retire) |
| 14 | **`llm:compact`** | `agllm.py:870` | moved in-container, native only ([entrypoint.py:958](../agency/agharness_internal/agharness_backends/_native_in_container_entrypoint.py:958)); external harnesses compact internally and invisibly | native only | **A** |
| 15 | **`agmap:{fn}[{i}]` lane root** | `agmap.py:63-68` | `agency/agmap.py` deleted from HEAD; lives on branch `tony/agmap` | none | **A** (see §10) |
| 16 | Per-span `cpu_ms` / `runqueue_ms` / `blocked_ms` | `_TimedSpan.__exit__` | `time.thread_time_ns()` and `/proc/self/schedstat` are **host-thread-local**. Any span originating in a container has no host thread | unrecoverable host-side | **A** |

**Summary:** 5 of 16 rows (≈60% of the 57 call sites) restore verbatim. Row 6 is
mechanical but gates the entire resource half. Rows 9–16 are architectural.

---

## 5. Proposed integration architecture

### 5.1 Principle: three observability tiers, declared per engine

The profiler cannot be uniformly complete across five engines, and pretending
otherwise produces silently wrong benchmark numbers. Make the tiers explicit.

**Tier 1 — engine-independent, host-observed.** Outer run, sandbox lifecycle,
cgroup/process/GPU sampling, GPU leases, and **every real provider call**. One
instrumentation site each; works for all five engines by construction.

**Tier 2 — engine-independent, syscall-observed.** Process spawn/exit and
`execve` argv via `agProxyPtrace`. Requires no harness cooperation.
`agProxyPtraceHandle.on_spawn()` / `.on_exit()`
([agproxy_ptrace.py:176-186](../agency/agharness_internal/agproxy_ptrace.py:176))
are **existing callbacks** — no new plumbing.

**Tier 3 — engine-specific, semantic.** Turn and tool boundaries. Native emits
them precisely. The other four require per-harness adapters.

`summary.json` gains a `coverage` block stating, per section and per engine,
one of `complete` / `derived` / `unavailable(reason)`. An absent section must
never be readable as zero.

### 5.2 The turn/tool problem is mostly solvable from Tier 1

This is the key insight and it should drive the first increment.

Every engine resends the whole conversation on every dispatch, and the terminus
already records it
([agllm_terminus.py:221-232, 380](../agency/agharness_internal/agllm_terminus.py:221)).
Therefore, host-side and with **zero** container code:

- Dispatch *n* for a token **is** turn *n*. Turn count, per-turn LLM latency,
  TTFT, and token usage are exact.
- `messages[len(prev_messages):]` between consecutive dispatches yields the
  assistant's `tool_calls` and the matching `role: "tool"` results — so **tool
  names, arguments, and outputs are recoverable for all five engines**.
- Tool *duration* is derived: `(dispatch_{n+1}.start − dispatch_n.end)`,
  attributed across that turn's tool calls. Exact when a turn has one tool call;
  an upper bound when it has several in parallel.

That is `derived` coverage — honest, engine-agnostic, and free. Precise
per-tool timing then becomes an *upgrade* per engine (Tier 3), not a
precondition for having any tool data at all.

### 5.3 Span model: OpenTelemetry spans, exact stats via a custom SpanProcessor

The one thing the harness abstraction genuinely broke is thread-local
parenthood (`_tls.span_stack`). OTel replaces it with `(trace_id, span_id,
parent_span_id)` — a **stored field**, not geometry inferred from containment.
A span created in a container declares its parent explicitly and lands in the
right place even when the clocks disagree. That is the whole reason to switch;
torch's kineto model cannot express it at all (§6, alternative B).

**The usual objection — that OTel costs statistical precision — is avoidable,
and the avoidance is specific: do not compute summaries from exported
metrics.** Register a custom `SpanProcessor` whose `on_end(span)` receives
exact start/end nanos, and append an agprof-shaped row:

```python
class _SummaryProcessor(SpanProcessor):
    def on_end(self, span):
        _records.append((
            span.attributes.get("agency.thread_id"),
            span.name,
            span.start_time,                          # ns, exact
            span.end_time - span.start_time,
            span.attributes.get("agency.cpu_ns"),     # None for remote spans
            span.attributes.get("agency.runq_ns"),
            dict(span.attributes),
            span.context.span_id,
            span.parent.span_id if span.parent else None,
        ))
```

`_build_run_summary()`, `_latency_stats()`, and `_percentile()` then run
**unchanged, on exact values** — no histogram buckets, no p99 approximation.
OTel supplies transport and correlation; the summary math stays agprof's.

Remote events (§5.4) become spans through the same path — the host calls
`tracer.start_span(..., context=parent_ctx)` with `agency.provenance =
"container_asserted"` and omits `agency.cpu_ns`/`runq_ns` (per row 16: `None`,
never `0`).

**Three OTel defaults must be overridden deliberately**, or the canonical spec
regresses:

1. **Unended spans are never exported.** A run that hangs or is OOM-killed
   would vanish, and `incomplete_spans` — a documented canonical feature —
   would silently become "the run was fast." Keep agprof's existing
   `_open_spans` / `_open_spans_lock` registry and force-end open spans with
   `outcome="interrupted"` at session stop.
2. **Sampling must be `AlwaysOn`, explicitly.** Resource metrics are always-on;
   if traces are sampled, the `llm` and `resources` sections of one
   `summary.json` describe different populations.
3. **`contextvars` do not cross `threading.Thread`.** A new thread starts with a
   fresh context, so the implicit parent is lost. Every skill run is spawned as
   a bare daemon thread — [agskill.py:450](../agency/agskill.py:450),
   [agteam.py:208](../agency/agteam.py:208),
   [agsandbox_backends/base.py:230](../agency/agsandbox_backends/base.py:230).
   Without a fix, every run is a disconnected root and the `parent_agent_id`
   hierarchy never forms. One helper, applied at all three sites:

   ```python
   def spawn_traced(fn, *a, **kw):
       ctx = otel_context.get_current()
       def run():
           token = otel_context.attach(ctx)
           try: fn(*a, **kw)
           finally: otel_context.detach(token)
       return threading.Thread(target=run, daemon=True)
   ```

### 5.3b Outputs without torch

Three artifacts from one span source:

| Artifact | Produced by | Reuse |
|---|---|---|
| `summary.json` / `summary.md` | `_build_run_summary()` + `_render_summary_markdown()` off `_records` | **verbatim** |
| Perfetto timeline (`*.trace.json`) | new `agprof_trace.py`: Chrome-trace writer from `_records` + `_samples` | `_inject_timelines()`'s counter-track logic ports nearly as-is |
| OTLP export | standard exporter → Jaeger / Grafana Tempo / Langfuse | new, ~0 lines |

`_inject_trace_args()` (~70 lines) is **deleted, not ported**. Its entire job
was writing agprof's own numbers back into kineto boxes as `args`, and its
`(tid, name, order)` matching key is exactly the thread-nesting assumption
that broke. Under OTel those numbers are span attributes at creation time.

### 5.4 Container emission: neutral JSON, host mints the span

`_native_in_container_entrypoint.py`'s docstring is explicit that it is
**stdlib + httpx only** — running it as `python3 <path>` specifically to avoid
importing `agency/__init__.py` and thus `openai`/`anthropic`/`boto3`. Any
in-container profiler dependency violates that constraint.

So the container writes facts, not spans:

```
{"ev":"turn_start","i":3,"ts":1770000123.456}
{"ev":"tool_start","id":"tc_7","name":"bash","ts":1770000123.501}
{"ev":"tool_end","id":"tc_7","ok":true,"ts":1770000125.902}
```

…to a UDS. The host turns each pair into a span via
`agprof.ingest_remote_span(event, parent_ctx, source)`, a thin wrapper over
`tracer.start_span(..., context=parent_ctx)` (§5.3). Benefits that fall out
structurally rather than by discipline:

- No dependency added to the sandbox image, and the entrypoint's stated
  constraint holds.
- **Clock skew is corrected once, at ingest.** Container clocks drift from the
  host under Docker Desktop; the host applies a measured offset when it stamps
  the span.
- **Provenance cannot be forged.** The host stamps `container_asserted` on
  everything arriving on that socket, because the host is the one calling
  `ingest_remote_span()`.
- Nothing is lost to an OOM kill — there is no in-container buffer to die with
  the container.

### 5.5 New host-side module: `agharness_internal/agprof_ingest.py`

Follows the established pattern exactly — see
[agharness_messenger.py:22-25](../agency/agharness_internal/agharness_messenger.py:22)
("each bridged service should own its own registry rather than reaching into a
sibling's in-process state"). Own class, own token→(agent, run span id)
registry, own UDS listener.

**A separate listener, not a route on the terminus.** Telemetry must never
share an event loop with `/internal/dispatch`; a span burst that stalls a
streaming LLM call would corrupt the TTFT number the profiler exists to measure.

### 5.6 Correlation

Extend the existing registration call, not a new mechanism:

```python
token = uuid.uuid4().hex
terminus.register(token, ag)
agprof_ingest.register(token, ag, run_span_id=agprof.current_span_id())
```

`run_span_id` is the `run{N}` lane root's id, captured on the host thread that
called `backend.execute()`. Everything ingested for that token parents to it.

The token is a **credential** and must never appear in a span attribute or in
`summary.json` — it is the ingest's auth header only, and the mapping to
`(agent, run)` is resolved host-side and discarded.

### 5.7 Where each engine's data comes from

| Signal | native | claude_code | codex | opencode | grok |
|---|---|---|---|---|---|
| run / sandbox / resources | T1 | T1 | T1 | T1 | T1 |
| LLM call, TTFT, tokens | T1 | T1 | T1 | T1 | T1 |
| turn boundaries | T3 exact | T1 derived → T3 via hooks | T1 derived | T1 derived | T1 derived |
| tool name/args/result | T3 exact | T1 derived → T3 via hooks | T1 derived | T1 derived | T1 derived |
| tool duration | T3 exact | T3 via hooks | T1 derived | T1 derived | T1 derived |
| process spawn/exit | T2 | T2 | T2 | T2 | T2 |
| compaction | T3 | unavailable | unavailable | unavailable | unavailable |
| retries | in-container | terminus attempt count only | " | " | " |

Claude Code is the first Tier-3 adapter because the wiring already exists:
`--settings` hook registration at
[claude_code.py:291-295](../agency/agharness_internal/agharness_backends/claude_code.py:291),
`_harness_permission_hook.py`, and
`_native_hooks.hook_payload_to_syscallevent()`
([_native_hooks.py:38](../agency/agharness_internal/agharness_backends/_native_hooks.py:38))
which already parses the `PreToolUse`/`PostToolUse` payload shape.

Two seams were reserved for this in advance and should be used rather than
duplicated: `agsyscallevent.tool_name` / `.tool_args`
([agproxy_ptrace.py:107-113](../agency/agharness_internal/agproxy_ptrace.py:107)),
and `_AgPtraceFields.profiler`
([agproxy_ptrace.py:135](../agency/agharness_internal/agproxy_ptrace.py:135),
"reserved for a future profiler hook; unused so far").

---

## 6. Alternative designs considered

**A. OpenTelemetry as transport and correlation, agprof as the summary math.**
Spans become OTel spans; a custom `SpanProcessor` feeds `_records` so
`_build_run_summary` is untouched (§5.3). Containers emit neutral JSON; the
host mints spans. torch is dropped entirely. *Pro:* `parent_span_id` is a
stored field, so cross-process parentage is native and survives clock skew;
`gen_ai.*` conventions make traces readable in Jaeger/Grafana/Langfuse with no
importer; removes a ~2–3 GB dependency whose only remaining job was a viewer
format. *Con:* three OTel defaults must be overridden deliberately (§5.3);
`_inject_trace_args` is discarded; the Perfetto timeline needs a small
hand-written Chrome-trace emitter (§5.3b). **Recommended.**

**B. Keep torch, synthesize a fake `tid` per container.** *Pro:* zero change to
the span model. *Con:* `_inject_trace_args` matches records to kineto events by
`(tid, name, order)` and its own docstring justifies this by "same-thread spans
can only nest." Remote spans nest with nothing on a host thread, and containment
is the only hierarchy torch has — so a clock-skewed child silently escapes its
parent with no field available to correct it. **Rejected: it is the one option
that cannot express the thing that broke.**

**C. Keep `_records` + torch, add an explicit `parent_id` field, ingest remote
events host-side.** *Pro:* smallest possible diff; preserves the TensorBoard
output path. *Con:* preserves it at the cost of keeping torch for nothing else
— torch never measured anything here (§2.1), and with TensorBoard confirmed
**not** a requirement, the dependency buys only `_inject_trace_args`, ~70 lines
that must be rewritten regardless. Also leaves the framework with a private,
hand-rolled trace schema and no interop. **Rejected once TensorBoard was
dropped as a requirement** (§10, Q7 — resolved).

**D. Host-only, no container emission at all.** Derive everything from terminus
transcripts (§5.2). *Pro:* zero container code, uniform across all five engines.
*Con:* tool durations are derived, parallel tool calls collapse, no compaction
visibility. **Adopted as a component of A, not as the whole answer** — it is
what makes non-native engines useful on day one.

---

## 7. Recommended solution

**A + D.** OTel for span identity, parentage, and export; agprof's `_Sampler`
and `_build_run_summary` retained wholesale for resource collection and
`summary.json`; turn/tool structure derived host-side from the terminus for all
five engines; precise in-container emission added only where it is cheap and
correct (native) or already wired (Claude Code hooks). torch is removed.

Rationale in one line: the harness abstraction destroyed the *host-thread
nesting* assumption, not the measurement code — so replace the parenthood
mechanism with one that has a real parent field, and leave the ~65% of
`agprof.py` that never depended on it alone.

What OTel is **not** being used for: inferring ReAct or tool structure (it
cannot — §5.2 and Tier 3 do that), and computing latency distributions (§5.3 —
those stay exact, off span timestamps).

Explicitly out of scope: reviving `llm:sync` (row 13 — the semaphore it
measured no longer exists).

---

## 8. Implementation roadmap

Difficulty: **S** ≈ hours, **M** ≈ 1–2 days, **L** ≈ 3–5 days.

### M0 — Restore the module, swap the span backend to OTel — *required*
- **Objective:** `AGENCY_PROFILE=1` produces a `summary.json` again, covering
  runs, sandbox ops, sync, GPU leases — with OTel spans underneath.
- **Files:**
  - restore `agency/profiler/{__init__,agprof}.py` and `tests/test_agprof.py`
    from `f46f00b`;
  - replace `_TorchSession` / `_TimedSpan._rf` with a `tracer.start_span()`
    wrapper; delete `_inject_trace_args()`; keep `_TimedSpan`'s
    `perf_counter_ns` / `thread_time_ns` / `_read_schedstat` capture and attach
    them as span attributes;
  - add `_SummaryProcessor` (§5.3) so `_records` still fills;
  - `pyproject.toml`: `profiler` extra becomes
    `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `nvidia-ml-py` —
    **no torch**;
  - re-apply spans at incompatibility rows 1–5 and 7 (`agskill.py`,
    `agsandbox.py`, `container.py`, `agresources.py`, `agsync.py`, `agdata.py`,
    `agent.py`, `agwebui/__init__.py`).
- **Difficulty:** M — the span-backend swap is contained to ~120 lines of
  `agprof.py`; the risk is in `agskill.py`'s restructured `_task()`, where
  `input:prepare` moves to `execute_harness` (row 7).
- **Dependencies:** none.
- **Outcome:** LLM/tool/turn sections present but empty. Expected; M3 fixes it.

### M0b — Perfetto trace emitter — *recommended*
- **Objective:** restore the timeline view without torch.
- **Files:** new `agency/profiler/agprof_trace.py` — Chrome-trace JSON
  (`ph:"X"` spans, `ph:"C"` counters) from `_records` + `_samples`. Port
  `_inject_timelines()`'s counter-track and GPU-lease-interval logic
  (`agprof.py:1169-1272`) nearly verbatim; it already hand-writes this format.
- **Difficulty:** S–M.
- **Dependencies:** M0.
- **Outcome:** drag-and-drop into ui.perfetto.dev, same as before.

### M1 — Restore container cgroup registration — *required*
- **Objective:** unblock every container, per-process, and GPU-attribution
  metric.
- **Files:** re-add `_register_prof_container()` and `_prof_container_label()`
  to `agsandbox_backends/container.py` (~50 lines from
  `tony/profiler:container.py:985-1033`); call from `_ensure_started()`
  ([container.py:809](../agency/agsandbox_backends/container.py:809)) and
  `container_stopped()` from `stop()`/`rm_container()`
  ([container.py:1546,1600](../agency/agsandbox_backends/container.py:1546)).
- **Difficulty:** S.
- **Dependencies:** M0.
- **Outcome:** `sandbox:*` resource tracks and per-process attribution return.

### M2 — Instrument the terminus — *required*
- **Objective:** exact LLM metrics for **all five engines** from one site.
- **Files:** [agllm_terminus.py:289-395](../agency/agharness_internal/agllm_terminus.py:289).
  Wrap `client.chat.completions.create()` in `llm:attempt[0]`; TTFT is the
  interval to `next(stream_iter)` at line 315 (already forced before the
  response commits); annotate `input_tokens`/`output_tokens` from
  `_serialize_usage`, plus `model`, `provider` (`type(ag.llm.backend).__name__`),
  and `outcome` from the existing 400/503 classification.
- **Difficulty:** S.
- **Dependencies:** M0. Needs M4 for correct parenting; land it unparented
  first and let M4 attach it.
- **Outcome:** the `llm` section of `summary.json` is complete and
  engine-independent.

### M3 — Derive turns and tools from terminus transcripts — *required*
- **Objective:** non-empty `turns` and `tools` sections for every engine.
- **Files:** new `agency/profiler/agprof_derive.py`; hook into
  `_record_transcript()`
  ([agllm_terminus.py:266](../agency/agharness_internal/agllm_terminus.py:266)).
  Diff consecutive `messages` arrays per token; emit `turn{i}` and
  `tool:{name}` records with `metadata["timing"] = "derived"`.
- **Difficulty:** M — the message-diff needs care around compaction (the array
  shrinks) and around `_record_transcript` being called on **every chunk**
  (line 380), not once per dispatch.
- **Dependencies:** M2.
- **Outcome:** all five engines report turn counts and tool names. This is the
  milestone that makes the profiler harness-agnostic.

### M4 — Context propagation and the correlation registry — *required*
- **Objective:** one connected trace per run instead of disjoint roots; remote
  and terminus spans parent to the right `run{N}`.
- **Files:**
  - `spawn_traced()` helper (§5.3, item 3) applied at
    [agskill.py:450](../agency/agskill.py:450),
    [agteam.py:208](../agency/agteam.py:208),
    [agsandbox_backends/base.py:230](../agency/agsandbox_backends/base.py:230)
    — **without this, every skill run is a separate root and nothing else in
    this milestone matters**;
  - new `agharness_internal/agprof_ingest.py` holding token →
    `(agent, run span context)`, patterned on `agharness_messenger.py`;
  - register alongside `terminus.register()` at
    [claude_code.py:184](../agency/agharness_internal/agharness_backends/claude_code.py:184)
    and the equivalent in `native.py`, `codex.py`, `opencode.py`, `grok.py`;
  - `agency.run_id` / `agent_id` / `parent_agent_id` as span attributes; the
    bearer token is the ingest's auth header **only** and must never reach a
    span attribute or `summary.json` (§5.6).
- **Difficulty:** M.
- **Dependencies:** M0.
- **Outcome:** `parent_agent_id` hierarchy forms; W3C `traceparent` can ride the
  existing UDS HTTP bridge for free.

### M5 — Native in-container emission — *recommended*
- **Objective:** exact turn/tool timing for `native`.
- **Files:** new `agency/profiler/agprof_emit.py` (**stdlib only** — see §5.4);
  bind-mount + emit at
  [entrypoint.py:1080](../agency/agharness_internal/agharness_backends/_native_in_container_entrypoint.py:1080)
  (turn), `:1110-1119` (tool), `:958` (compact), `:791` (retry backoff);
  ingest host-side in `agprof_ingest.py`.
- **Difficulty:** M.
- **Dependencies:** M4.
- **Outcome:** native reaches parity with the canonical spec, including
  `llm:compact` and `llm:retry_backoff`.

### M6 — ptrace process lifecycle — *recommended*
- **Objective:** child-process spans for the four external engines.
- **Files:** `agProxyPtraceHandle.on_spawn`/`on_exit`
  ([agproxy_ptrace.py:176-186](../agency/agharness_internal/agharness_backends/../agproxy_ptrace.py:176))
  → `agprof` records; resolve `execve` argv for span names.
- **Difficulty:** S — the callbacks already exist.
- **Dependencies:** M4.
- **Outcome:** "what did the harness actually run" becomes visible without
  harness cooperation.

### M7 — Claude Code semantic adapter — *recommended*
- **Objective:** exact tool timing for the highest-traffic external engine.
- **Files:** extend the hook script at
  [claude_code.py:291-295](../agency/agharness_internal/agharness_backends/claude_code.py:291)
  and `_harness_permission_hook.py` to also POST `PreToolUse`/`PostToolUse` to
  the ingest; reuse `_native_hooks.hook_payload_to_syscallevent()`.
- **Difficulty:** M.
- **Dependencies:** M4, M6.
- **Outcome:** `claude_code` upgrades from `derived` to `complete`.

### M8 — Coverage declaration + golden test — *required*
- **Objective:** make partial coverage legible and regressions detectable.
- **Files:** `coverage` block in `_build_run_summary()`; new
  `tests/test_agprof_harness.py` running each engine against the mock endpoint
  and diffing against a checked-in golden `summary.json`.
- **Difficulty:** M.
- **Dependencies:** M3.
- **Outcome:** a summary can no longer look complete while being empty.

### M9 — Optional cleanup
- Restore `agmap` instrumentation if `tony/agmap` merges (§10).
- Retire `llm:sync` from the span glossary and `summary.md` (row 13).
- Adopt `gen_ai.*` semantic-convention attribute names on LLM spans
  (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, …) so Langfuse/Braintrust
  read them without mapping. Deliberately deferred: it is a rename, and doing it
  before M8's golden `summary.json` exists means rewriting the fixture twice.
- Stand up a real OTLP collector target (Jaeger/Tempo) for interactive
  exploration. Not on the critical path — M0 writes traces to a local file
  exporter, which is enough for `summary.json` and Perfetto.

---

## 9. Risks and mitigations

### Architectural

| Risk | Mitigation |
|---|---|
| Tier-3 coverage never lands for codex/opencode/grok, leaving benchmark comparisons subtly unequal across engines | M3 makes `derived` coverage uniform first; M8's `coverage` block makes any residual asymmetry explicit in the artifact, so a comparison can be rejected rather than silently believed |
| Container-emitted spans are *claims*, not observations — a harness can under-report | Host stamps `provenance` at ingest (§5.4). Cross-check container-asserted turn counts against the terminus's own host-side `request_log` count; disagreement is a summary-level warning |
| Per-span `cpu_ms`/`runq_ms` unrecoverable for remote spans (row 16) | Emit `None`, never `0`. Optionally have the in-container emitter read its own `/proc/self/schedstat` and include the delta as a distinct field |

### Performance

| Risk | Mitigation |
|---|---|
| Telemetry stalls the LLM hot path and inflates the TTFT being measured | Separate UDS listener (§5.5); bounded queue that **drops** rather than blocks; record the drop count in `sampling health` |
| Unbounded ingest from a hostile or looping container | Per-token span-rate and attribute-size caps; log violations |
| `_Sampler` at 10 Hz recursively scanning cgroups under 32-agent fan-out | Already the canonical default; measure before changing. Report `effective` vs `configured` frequency, which `_build_run_summary` already does |

### Correctness

| Risk | Mitigation |
|---|---|
| Host↔container clock skew (Docker Desktop on macOS drifts seconds) reorders spans | Measure offset via a UDS round-trip at container start; apply once at ingest (§5.4). Assert no child starts before its parent in M8's golden test |
| `_record_transcript` fires on **every chunk** ([agllm_terminus.py:380](../agency/agharness_internal/agllm_terminus.py:380)); a naive M3 differ will emit a turn per chunk | Derive on dispatch completion, not on transcript write |
| Compaction shrinks the messages array, so M3's diff sees a spurious rewind | Detect a shrink and restart the diff baseline; native already signals compaction directly (M5) |
| Retries have no single owner (row 12): a harness CLI retry looks like a new turn | Report `llm.attempts_observed` (host truth, from the terminus) separately from `llm.retries_reported` (harness-asserted, usually absent). Benchmarks use the former |
| **OTel default: unended spans are never exported.** A hung or OOM-killed run would produce no spans and read as *fast*, silently losing canonical `incomplete_spans` | Keep agprof's existing `_open_spans` registry; force-end with `outcome="interrupted"` at session stop (§5.3). Covered by M8's interruption test |
| **OTel default: `contextvars` do not cross `threading.Thread`.** Every skill run spawns a bare daemon thread, so the implicit parent is lost and each run becomes a disjoint root | `spawn_traced()` at the three spawn sites, landed in M4. M8 asserts exactly one root span per `team.run()` under `agmap`-style fan-out |
| **OTel default: sampling.** If traces are sampled while resource metrics are always-on, one `summary.json`'s sections describe different populations | Pin `AlwaysOn` explicitly and record the sampler name in the `sampling health` section |
| Percentiles silently degrade if someone later routes latency through OTel histogram metrics | `_SummaryProcessor` reads exact `start_time`/`end_time` (§5.3). Add a test asserting p99 matches a hand-computed value on a fixed span set |

### Backwards compatibility

- `summary.json` keeps its existing keys; `coverage` and `provenance` are
  additive. `_unpack_record()` is the single decode point, so the 2 new tuple
  fields (`span_id`, `parent_id`) do not ripple.
- **`.pt.trace.json` / TensorBoard output is dropped** (confirmed not required,
  §10 Q7). M0b's Chrome-trace file replaces it for Perfetto;
  `torch-tb-profiler` no longer works. `agency/profiler/README.md`'s Usage and
  Installation sections need rewriting — they currently instruct
  `uv pip install -e ".[profiler]"` for torch and `tensorboard --logdir`.
- `llm:sync` disappears from output. Anything consuming it must be updated —
  the metric no longer has a referent.
- `agprof.session()`'s public API and `AGENCY_PROFILE` / `AGENCY_PROFILE_SCOPE`
  env handling are unchanged; `start()` stops returning a torch profiler object
  (it returned `prof` at `agprof.py:884`). Any caller using the return value
  breaks — `agwebui/__init__.py:292` uses `agprof.workload()` and does not.

### Testing

- `tests/test_agprof.py` restored from `f46f00b` as the unit baseline.
- New `tests/test_agprof_harness.py`: per engine, against the mock endpoint,
  assert (a) exactly one root span per run, (b) turn count matches the
  terminus's `request_log` length, (c) no child precedes its parent, (d) golden
  `summary.json` diff.
- Interruption test: kill a run mid-flight, assert `incomplete_spans` is
  non-empty and the run is **not** reported as fast-and-successful.

### Migration

- Linux-only remains ( `_require_linux()` at `agprof.py:104`), but the dev
  environment here is darwin. M0–M4 and M8 are testable on macOS *except* the
  cgroup/NVML paths; M1's tests need a Linux host or CI runner. Plan for that
  rather than discovering it at M1.
- `AGENCY_PROFILE_SCOPE=process` re-execs through `systemd-run` and needs
  non-interactive sudo — unchanged, but worth re-confirming it still composes
  with container-backed harness launches.

---

## 10. Open questions and assumptions

1. **`agmap` is deleted from HEAD.** It exists on `tony/agmap`. Is it returning?
   If yes, M0 should include its lane-root instrumentation; if no, the
   `agmap:{fn}[{i}]` glossary entry should be removed from the canonical spec.
   *Assumed: returning.*
2. **`benchmarks/` is not on this branch** — only `__pycache__`. Sources are on
   `eric/benchmark-tests` / `sunga/benchmark-integration`, and neither
   references `agprof`. Who owns wiring the profiler into the benchmark runner,
   and against which branch? *Assumed: out of scope here.*
3. **Is `llm:sync` genuinely dead?** The 256-slot semaphore is gone
   ([agllm.py:88](../agency/agllm.py:88)) and nothing replaced it. Confirming
   there is no host-side LLM throttle at all would let the span be retired
   cleanly rather than left as a permanently-zero row.
4. **Does the chroot backend need resource attribution?** `_register_prof_container`
   is container-only. A chroot-backed harness launch runs directly on the host,
   so its processes land in the workload cgroup and are covered by
   per-process tracks — but with no `sandbox:` aggregate. Acceptable?
5. **Row 8 / `proc_wait`:** `execute_harness` restricts `wait_for_processes()`
   to native because ptrace PID exit events never arrive for the other four
   engines ([agskill.py:566-572](../agency/agskill.py:566)). That is flagged in
   the code as a pre-existing gap. M6 touches the same callback path — worth
   confirming whether M6 fixes it as a side effect or must route around it.
6. **Assumed:** the per-run bearer token is 1:1 with a skill run for every
   backend. Verified for `claude_code.py`; the other four mint tokens in their
   own `execute()` and should be confirmed before M4 relies on it.
7. ~~**Assumed:** `tony/profiler`'s TensorBoard output is a hard requirement.~~
   **Resolved (user, 2026-08-07): TensorBoard is not a requirement.** This is
   what flips the recommendation from C to A — torch's only remaining job was
   the kineto trace and the `torch-tb-profiler` plugin, and Perfetto reads
   generic Chrome-trace JSON that M0b writes directly. torch is dropped.
8. **OTel SDK version pinning.** Only the host imports it (§5.4 — the container
   emits neutral JSON precisely so the sandbox image stays stdlib+httpx), so
   there is no host/container skew to manage. Worth confirming
   `opentelemetry-sdk` does not conflict with `fastapi`/`uvicorn` pins already
   in `pyproject.toml`.
9. **Do we want OTel auto-instrumentation for `httpx`?** It would trace the
   UDS hops between `agproxy_llm` → terminus for free. Probably yes, but it
   adds a span per internal RPC and will dominate span counts under fan-out —
   default it **off** and make it a config flag.
