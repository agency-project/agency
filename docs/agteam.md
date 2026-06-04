# agteam

`agteam` is a base class for bundling tools, skills, agents, configuration, and execution logic into a single reusable object. Each subclass defines one cohesive unit of agentic work; multiple instances of the same class can run concurrently without sharing state.

## Motivation

Without `agteam`, a workflow is a loose collection of module-level objects:

```python
search_tool  = agtool(...)
find_skill   = agskill(..., tools=[search_tool])
main_agent   = agent(llm_config=LLM_CONFIG, agskills=[find_skill])
main_agent.run("find_papers", agdata(topic="KV cache"))
```

This works for a single run but scales poorly. To run the same workflow on multiple topics concurrently you have to manually manage separate tool/skill/agent instances for each. `agteam` encapsulates that structure so each instance is fully self-contained.

## Defining a team

Subclass `agteam` and override two methods:

| Method | Purpose |
|---|---|
| `setup()` | Create tools, skills, and agents. Called once at construction. |
| `run()` | Implement the workflow. Called by the user. |

Configuration specific to the team (topic, file paths, limits, …) is passed as keyword arguments to `__init__` and becomes a plain instance attribute.

```python
from agency import agteam, agent, agskill, agdata
from agency.agtool import agtool

class PaperCrawlerTeam(agteam):
    llm_config = LLM_CONFIG          # class-level default; overridable per instance

    def setup(self) -> None:
        # Tools
        self.search_arxiv = agtool(name="search_arxiv", ...)

        # Skills
        self.find_papers   = agskill(name="find_papers",   ..., tools=[self.search_arxiv])
        self.summarise     = agskill(name="summarise_paper", ..., tools=[])
        self.compile       = agskill(name="compile_report", ...)

        # Agent — registered automatically via make_agent()
        self.main_agent = self.make_agent(
            [self.find_papers, self.summarise, self.compile]
        )

    def run(self) -> agdata:
        papers = self.main_agent.run("find_papers", agdata(topic=self.topic)).papers
        summaries = [
            agent(self.main_agent).run(
                "summarise_paper",
                agdata(title=p["title"], url=p["url"], abstract=p["abstract"]),
            )
            for p in papers
        ]
        return self.main_agent.run(
            "compile_report",
            agdata(topic=self.topic, summaries=summaries, output_path=self.output_path),
        )
```

## Construction

```python
team = PaperCrawlerTeam(topic="KV cache quantization")
```

Keyword arguments are set as instance attributes before `setup()` is called, so `self.topic` is available inside `setup()` and `run()`.

An explicit `llm_config` dict overrides the class-level default:

```python
team = PaperCrawlerTeam(
    topic="flash attention",
    llm_config={...},        # per-instance override
)
```

## make_agent()

```python
self.agent = self.make_agent(agskills, tools=None, **kwargs)
```

A thin wrapper around `agent(llm_config=self.llm_config, agskills=..., tools=..., **kwargs)` that also registers the new agent in `self.agents`. Keyword arguments are forwarded to `agent.__init__` unchanged.

`self.agents` is a list of all agents created by `make_agent()` for this team instance.

## Scaling

Because each team instance owns its own agents and sandbox containers, you can instantiate as many teams as you need and run them concurrently:

```python
topics = ["KV cache", "flash attention", "speculative decoding"]
teams  = [PaperCrawlerTeam(topic=t) for t in topics]

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor() as ex:
    results = list(ex.map(lambda t: t.run(), teams))
```

Each team gets its own agents (with distinct names), its own sandbox containers, and its own log files. There is no shared mutable state between instances.

## llm_config class attribute

Declaring `llm_config` at the class level provides a default shared by all instances:

```python
class MyTeam(agteam):
    llm_config = {
        "base_url": "https://my-vllm/v1",
        "api_key":  "...",
        "model":    "my-model",
    }
```

Passing `llm_config=` at construction time creates an instance attribute that shadows the class default, leaving other instances unaffected.

## Log and output directories

`agteam` does not manage `agent.log_dir` or `agent.output_dir` itself — those are class variables on `agent` and apply to all agents created after they are set. Configure them before constructing your teams:

```python
agent.log_dir    = run_dir / "logs"
agent.output_dir = run_dir / "agent_output"

teams = [PaperCrawlerTeam(topic=t) for t in topics]
```

## API reference

### `agteam.__init__(llm_config=None, **config)`

Creates the team. Sets all `config` kwargs as instance attributes, then calls `setup()`.

### `agteam.setup()`

Override to define tools, skills, and agents. Default implementation does nothing.

### `agteam.run()`

Override with the team's workflow. Raises `NotImplementedError` if not overridden.

### `agteam.make_agent(agskills, tools=None, **kwargs) → agent`

Create and register an agent. `llm_config` defaults to `self.llm_config`.

### `agteam.agents → list[agent]`

All agents created via `make_agent()` for this instance.
