# Automatic host-side Python tracing

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
| `auto_exclude` | empty | Additional path or module fragments to omit |
| `auto_min_duration_ms` | `1.0` | Drop shorter completed intervals |
| `auto_max_depth` | `32` | Maximum recorded call-stack depth per thread |
| `auto_max_events` | `250000` | Session-wide retained-event ceiling |

Virtual environments, site packages, the standard library, and agprof's own
implementation are filtered out. Generator and coroutine activity is recorded
as resume-to-yield/return segments. The summary reports captured and dropped
event counts. Set `auto_functions=False` to retain only explicit spans.

This capture runs only in the host Python interpreter. It does not inject code
into Docker containers. Container-side Python profiling remains a separate
extension so it can use a dedicated collector without multiplying host
profiler overhead across sandboxes.

## Timeline naming

Perfetto lane names retain their native TID while choosing the best available
human label in this order:

1. an explicit `agprof.thread_name()` label;
2. a semantic run or map span;
3. a role inferred from the automatic call tree;
4. Python's meaningful runtime thread name;
5. the generic `Python worker` fallback.

The resource sampler also assigns semantic process names from the command,
entrypoint, and sandbox registration. Every sampled PID receives an
`alive (sampled)` bar bounded by its first and last sampler observations. This
bar describes process lifetime; it does not replace function or semantic spans.
