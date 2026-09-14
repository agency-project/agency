# Automatic Python tracing

`agprof.session()` records two complementary timeline layers:

- Explicit `agprof.span()` intervals are semantic overlays chosen by Agency or
  application code. They remain the authoritative names for phases such as a
  run, LLM attempt, tool call, or sandbox operation.
- Automatic Python intervals come from Python 3.12's `sys.monitoring` API and
  use the `python.auto` Perfetto category. Functions do not need decorators or
  profiler calls to appear in this layer.

Automatic tracing is enabled by default. The defaults are:

| Option | Default | Meaning |
|---|---:|---|
| `auto_include` | current working directory and `agency` | Python source roots eligible for capture |
| `auto_include_dependencies` | `False` | Include installed dependencies and Python standard-library source |
| `auto_exclude` | empty | Additional path or module fragments to omit |
| `auto_min_duration_ms` | `1.0` | Drop shorter completed intervals |
| `auto_max_depth` | `32` | Maximum recorded call-stack depth per thread |
| `auto_max_events` | `250000` | Session-wide retained-event ceiling |

By default, virtual environments, site packages, and the standard library are
filtered out. Set `auto_include_dependencies=True` to include them; explicit
`auto_exclude` patterns still apply. Agprof itself remains excluded.
Generator and coroutine activity is recorded
as resume-to-yield/return segments. The summary reports captured and dropped
event counts. Set `auto_functions=False` to retain only explicit spans.

Agency's native harness also runs the same collector in its own Python
interpreter, including when launched inside a container. It loads the collector
without importing Agency's host package or OpenTelemetry. The host session's
limits and dependency option are forwarded through the authenticated attempt
bridge. Captured calls are sent in batches of at most 128 at native teardown;
the host enforces its session event limit too. Native records use distinct
process lanes to avoid collisions between container PID namespaces.

This does not inject Python into arbitrary subprocesses or instrument the
JavaScript/Rust internals of external harness CLIs. Native capture requires
Python 3.12+, just like host capture. The session reports collector failures,
rejected events, and transport drops; bounded capture is not lossless.

```python
from agency import agprof

with agprof.session("profile", auto_include_dependencies=True,
                    auto_min_duration_ms=0.1, auto_max_events=250_000):
    team.run()
```

Depth and duration filters deliberately omit calls. Host filtered counts and
retained-event overflow are reported separately. Capture across a generator's
yield is segmented; it does not claim that an async task ran while suspended.

## Timeline naming

Perfetto lane names retain their native TID while choosing the best available
human label in this order:

1. an explicit `agprof.thread_name()` label;
2. a semantic run or map span;
3. a role inferred from the automatic call tree;
4. Python's meaningful runtime thread name;
5. the generic `Python worker` fallback.

The Perfetto trace contains one process group for the Agency harness. Its
semantic spans, host Python calls, resource counters, and GPU leases appear
there. Helper-process groups, sampled process-lifetime bars, and automatic
Python calls from other processes are not exported. Semantic agent and tool
spans reported from containers remain on the harness timeline.

The resource sampler still collects per-process metrics for the JSON and
Markdown summaries; those metrics do not create extra Perfetto process groups.

Parent-child spans, including same-track tool and Python spans, are connected
by Perfetto flow arrows (`agprof.relationship`). Select a related span to inspect
its incoming/outgoing connections. Explicit semantic parent IDs take precedence;
automatic Python and legacy spans use their nearest enclosing slice on the same
track. Flow arguments distinguish `explicit_parent` from `same_track_nesting`.
Missing explicit parents and unrelated top-level spans are not linked. Nesting
does not imply causality across threads.

## LLM and tool details

Select an `llm:attempt[...]` span in Perfetto to inspect `llm.messages`,
`llm.response`, the model, stop reason, and reported input/output/total token
counts. Streamed responses collect usage from provider metadata trailers;
missing usage remains unavailable rather than being reported as zero.
The `call_label` links the attempt to its logged exchange.

Tool spans include `tool.arguments`, `tool.result`, and `tool.error`, alongside
their existing outcome and timing. Arguments are captured at admission so
interrupted tools still show their inputs. These transcript and tool fields
contain application content and are exported with the trace.

Each detail field is limited to 32,768 characters. The accompanying
`<field>_truncated` and `<field>_chars` attributes indicate whether the content
was cut and its original serialized length. Structured details are JSON text.
