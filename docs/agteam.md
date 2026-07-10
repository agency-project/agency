# agteam

`agteam` is a base class for bundling tools, skills, agents, configuration, and execution logic into a single reusable object. Each subclass defines one cohesive unit of agentic work; multiple instances of the same class can run concurrently without sharing state.

## Motivation

Without `agteam`, a workflow is a loose collection of module-level objects:

```python
search_tool  = agtool(...)
find_skill   = agskill(..., tools=[search_tool])
main_agent   = agent(agconfig=cfg)
main_agent.run(find_skill, agdata(topic="KV cache"))
```

This works for a single run but scales poorly. To run the same workflow on multiple topics concurrently you have to manually manage separate tool/skill/agent instances for each. `agteam` encapsulates that structure so each instance is fully self-contained.

## Defining a team

Subclass `agteam` and override two methods:

| Method | Purpose |
|---|---|
| `setup()` | Create tools, skills, and agents. Called once at construction. |
| `run()` | Implement the workflow. Always non-blocking — returns a pending `agdata`. |

Configuration specific to the team (topic, file paths, limits, …) is passed as keyword arguments to `__init__` and becomes a plain instance attribute.

```python
from agency import agteam, agent, agskill, agdata, agsync
from agency.agtool import agtool
from agency.agconfig import agConfig
from agency.agllm_backend import agLLMBackendConfig

cfg = agConfig(agLLMBackendConfig(base_url="...", model="...", api_key="..."))

class PaperCrawlerTeam(agteam):
    agconfig = cfg                   # class-level default; overridable per instance

    def setup(self) -> None:
        self.search_arxiv = agtool(name="search_arxiv", ...)
        self.find_papers   = agskill(name="find_papers",   ..., tools=[self.search_arxiv])
        self.summarise     = agskill(name="summarise_paper", ..., tools=[])
        self.compile       = agskill(name="compile_report", ...)

        # agconfig injected automatically from the team
        self.main_agent = agent()

    def run(self) -> agdata:
        papers = self.main_agent.run(self.find_papers, agdata(topic=self.topic)).papers
        summaries = [
            agent.fork(self.main_agent).run(
                self.summarise,
                agdata(title=p["title"], url=p["url"], abstract=p["abstract"]),
            )
            for p in papers
        ]
        return self.main_agent.run(
            self.compile,
            agdata(topic=self.topic, summaries=summaries, output_path=self.output_path),
        )
```

## Construction

```python
team = PaperCrawlerTeam(topic="KV cache quantization")
```

Keyword arguments are set as instance attributes before `setup()` is called, so `self.topic` is available inside `setup()` and `run()`.

An explicit `agconfig` overrides the class-level default:

```python
team = PaperCrawlerTeam(
    topic="flash attention",
    agconfig=other_cfg,        # per-instance override
)
```

## run() — always non-blocking

`run()` starts the workflow in a background thread and returns a pending `agdata` immediately. Field access on the returned value blocks until the team finishes:

```python
result = PaperCrawlerTeam(topic="KV cache").run()  # returns immediately
print(result.report_path)                           # blocks here
```

If the workflow produces no meaningful return value (e.g. it only writes files), you can ignore the return value and use `agsync` as the only synchronisation point:

```python
team = BuildTeam(output_path="/workspace/out")
team.run()       # fire and forget
agsync(team)     # wait for completion
```

## Parallel fan-out

Since `run()` is non-blocking, parallel execution is just a list comprehension:

```python
from agency import agsync

topics  = ["KV cache", "flash attention", "speculative decoding"]
teams   = [PaperCrawlerTeam(topic=t) for t in topics]
pending = [t.run() for t in teams]   # all three start immediately

agsync(teams)                        # wait for all to finish
for topic, result in zip(topics, pending):
    print(f"{topic}: {result.report_path}")
```

## Error handling

If `run()` raises, the exception is captured and re-raised when any field on the pending `agdata` is first accessed, or when `agsync` is called on the team:

```python
try:
    agsync(team)
except Exception as e:
    print(f"team failed: {e}")
```

## Thread model

Each `run()` call executes in its own daemon thread. There is no shared pool to exhaust, so recursive team spawning — a team that creates and runs child teams inside its own `run()` — is safe by construction. Thread creation overhead (~100 µs) is negligible relative to any LLM call.

## Auto agent tracking

Any `agent(...)` call made inside `setup()` or `run()` is automatically registered with the team. The `agconfig` argument is optional — an agent created with no explicit `agconfig=` inherits the active team's `agconfig` at the moment it's constructed (not just LLM fields — log_dir/output_dir/sandbox settings set on it apply too):

```python
def setup(self) -> None:
    self.main_agent = agent()   # agconfig injected automatically
```

Like every other framework object, the agent clones `team.agconfig` rather than sharing it — so a later change to `team.agconfig` (or to the `cfg` the team itself was built from) does not retroactively affect agents already constructed, only ones created afterward.

`self.agents` returns a snapshot list of all agents currently registered with this team instance. Completed anonymous agents (fork agents with no other live reference) are GC'd automatically — only agents held via `self.*` or still in-flight are visible.

## agconfig class attribute

Declaring `agconfig` at the class level provides a default shared by all instances:

```python
from agency.agconfig import agConfig
from agency.agllm_backend import agLLMBackendConfig

_cfg = agConfig(agLLMBackendConfig(
    base_url="https://my-vllm/v1",
    api_key="...",
    model="my-model",
))

class MyTeam(agteam):
    agconfig = _cfg
```

Passing `agconfig=` at construction time creates an instance attribute that shadows the class default, leaving other instances unaffected.

## Log and output directories

`agteam` does not manage `agent.log_dir` or `agent.output_dir` itself — those are class variables on `agent` and apply to all agents created after they are set. Configure them before constructing your teams:

```python
agent.log_dir    = run_dir / "logs"
agent.output_dir = run_dir / "agent_output"

teams = [PaperCrawlerTeam(topic=t) for t in topics]
```

## API reference

### `agteam.__init__(agconfig=None, **config)`

Creates the team. Sets all `config` kwargs as instance attributes, then calls `setup()`.

### `agteam.setup()`

Override to define tools, skills, and agents. Default implementation does nothing.

### `agteam.run() → agdata`

Override with the team's workflow. Always non-blocking — runs in a background thread and returns a pending `agdata` immediately. Raises `NotImplementedError` if not overridden on the base class.

### `agteam.agents → list[agent]`

Snapshot of all agents currently registered with this team instance (auto-tracked from `setup()` and `run()`).
