# Library API

Start with [getting started](../getting-started.md) for installation, provider credentials and a native first run. This reference describes the current checkout, including demonstrated limitations. Signatures below each group are extracted statically from source; behavioral explanations are maintained by hand.

| Task | Reference |
| --- | --- |
| Submit work, manage history, save or fork agents | [Agents and lifecycle](agents.md) |
| Wait, inspect failures, control requests | [Results and cancellation](results.md) |
| Define skills and executable tools | [Skills, tools and policies](skills.md) |
| Validate shapes and move files or images | [Schemas and types](types.md) |
| Compose parallel work and teams | [Workflows](workflows.md) |
| Set or update configuration | [Configuration](configuration.md) |
| Execute commands and own checkpoints | [Sandboxes and checkpoints](sandboxes.md) |
| Inspect scheduling, resources, logs and traces | [Advanced runtime and observability](runtime.md) |

Application entry points are `Agent`/`agent`, `agdata`, skills, tools, schemas, types, workflows and namespaced configuration. Direct sandbox use requires owning its cleanup. Scheduler, context internals and instrumentation APIs are advanced: avoid modifying them during submitted work. [Known contract issues](contracts.md) records differences between intended and demonstrated behavior.

## Root exports

The [inventory manifest](public-api.json) maps every current root export and additional supported import to a destination. The check derives `agency.__all__` from source each run, and checks explicit member selections and example imports. `Agent` is the exact alias of `agent`, not a subclass.

<!-- API_INVENTORY -->
