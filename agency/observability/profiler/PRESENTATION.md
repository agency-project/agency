# Trace presentation schema, version 1

The embedded Perfetto viewer groups recorded work by conceptual owner. The
exporter retains ordinary Chrome slices, counters, details and flow events;
the viewer extension only arranges their tracks and adds activity summaries.

```text
Workflow / Main
  Main-thread work
  Host process counters
Global orchestrator
Agent <agent ID>
  Host
    Packed execution lanes
  Sandbox <sandbox ID>
    Processes
      <sampled process identity>
        Packed execution spans and Python calls
        Counters
      Harness attempt <reporter ID> (only when process identity is unavailable)
        Reported spans and Python calls
    Sandbox counters
  ↗ Shared sandbox (when this agent uses one)
Shared resources
  Shared or independently created sandboxes
  Workload / hardware counters
  GPUs (when sampled)
Unattributed
```

Only groups with recorded data appear. Groups start collapsed. Their summary
is the union of descendant slice intervals: nested work is counted once and
gaps remain visible. These intervals include waiting; they are not CPU
utilization, object lifetime or hardware bandwidth measurements. Counter-only
groups have no synthetic activity summary.

## Ownership and execution

| Field | Meaning |
| --- | --- |
| `agency.agent_id` | Unique agent object name within the run |
| `agency.sandbox_id` | Unique sandbox object name, including its deduplication suffix |
| `agency.component` | Explicit non-agent component, currently `orchestrator` |
| `agency.execution_side` | `host` or `sandbox` |
| `agency.reporter_id` | Attempt-scoped remote reporter identity |
| `agency.context_span_id` | Active semantic span captured when an automatic call starts |
| `agency.source_pid`, `agency.source_tid` | Original collector process/lane IDs before presentation routing |
| `agency.source_thread_name` | Original collector thread label |
| `agency.namespace_pid`, `agency.namespace_tid` | PID/TID reported inside the sandbox |
| `agency.pid_namespace` | Linux PID namespace identity, such as `pid:[4026533335]` |
| `agency.process_start_ticks` | Process start time from `/proc/<pid>/stat`; guards against PID reuse |
| `agency.host_pid`, `agency.process_identity` | Verified host PID and sampler identity after matching |

Local source IDs are OS IDs. Negative remote reporter IDs and negative
host-observed span lanes are synthetic, **not OS PIDs/TIDs**. Native harnesses
report their namespace identity, PID and process start ticks. The host sampler
reads the corresponding namespace information and `NSpid` from `/proc` for
processes in registered sandbox cgroups, including sibling systemd scopes.
An exact, unique match on sandbox, namespace, namespace PID and start ticks
puts reported spans and process counters under one process. Missing, ambiguous
or mismatched identities retain the attempt group; executable names are never
used to guess a match. Host-side child spans do not inherit sandbox process IDs.

Ownership follows explicit span parentage, with child annotations overriding
inherited values. Local timed spans explicitly select the host side, even when
their causal parent is a sandbox span. Automatic calls capture their semantic
context at entry, so reusing a worker does not reassign earlier work. Ownership
is also carried through traced thread creation, pool callbacks and scoped host
HTTP requests. These snapshots contain values, not mutable parent span handles,
and are reset when a request finishes. The main OS thread and its inherited
workflow tasks default to Workflow; other unknown work is Unattributed.

## Row packing

Within each ownership path, overlapping or nested intervals on one source lane
form an indivisible stack block. Blocks are assigned to the first available
display lane, reusing lanes only after their previous block ends. Idle gaps
between blocks can be filled by work from another source. Overlapping blocks
from different sources always occupy separate lanes. Source nesting and causal
edges are preserved; counters retain their original scope and are never packed.

Rows have logical labels such as `Host · Lane 1`; synthetic display TIDs identify
the packed rows, not OS threads. Each slice retains its source IDs and label.
Packing never crosses ownership paths, including process group boundaries.
The exporter plans packing from compact timing tuples, then streams the full
events without keeping a second array of transcripts or tool results.

A sandbox used by one agent appears under that agent. A sandbox used by multiple
agents has one canonical location under Shared resources. Each agent has an
`↗` shortcut with an Open button that expands and scrolls to that location.
Shortcuts carry no slices, counters or activity summaries. Spans retain their
agent IDs, so activity in that shared sandbox remains attributable. Sandbox
identity is also the sampler registry key: two sandboxes must never collide
because their names have a common prefix. Unrelated runtime helper processes
remain excluded; explicitly owned remote samples and sandbox process counters
are included.

## Transport and viewer

Each slice carries `args.agency_presentation`, a JSON string containing
`{"version":1,"path":[["stable-id","Display name"],...]}`. Every lane has one
path. Group IDs are scoped to their complete ancestor path; display labels
are never used as identity. Counters encode the same descriptor plus `label`
in their name after `agency-counter-v1:` because Chrome counter import does
not preserve arbitrary per-counter arguments. The viewer decodes this into
ordinary readable track labels. Original source IDs remain in slice details.
Shared sandbox descriptors also carry `shared_with`, a list of agent IDs used
to construct the navigation shortcuts without copying their data.

The embedded plugin reparents existing Perfetto tracks, retaining their IDs,
selection, search, details and causal arrows. Explicit parent edges connect
the starts of their slices across ownership groups. Inferred nesting edges
never cross ownership paths. Legacy traces with no ownership annotations use
the original layout. Standard Perfetto can still open exported JSON, but the
object hierarchy and decoded counter labels require the embedded extension.

The plugin source lives beside `build_perfetto.py` and participates in the
build fingerprint. Starting the web UI rebuilds stale generated assets.
