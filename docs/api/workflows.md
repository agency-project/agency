# Workflows and composition

```python
from agency import agmap, agtask, agsync, agteam, agdata, sigterm_as_exit
```

## Parallel mapping

`agmap(fn, items, *, is_asynchronous=False)` runs Python callables in parallel traced daemon threads. **The default joins all tasks before returning**; it is concurrent but synchronous to its caller. `is_asynchronous=True` returns pending handles immediately. Neither mode launches an agent unless your callable does so.

| Shape | Return |
| --- | --- |
| One callable, one scalar item | One agtask |
| One callable, list/tuple of items | List of agtask |
| List/tuple of callables, one scalar | List of agtask |
| Both collections | Zipped pairs, equal lengths required (`ValueError` otherwise); list of agtask |

`agtask` is an agdata subclass and inherits its constructor, waiting, awaiting and conversion methods. Callable agdata results preserve payload fields. Other return values are wrapped under `.result`. Callable exceptions become error payloads; siblings continue. The pending wrapper retains agtask even on error, so inspect `to_dict()["error"]` rather than `isinstance(..., agerror)`. There is no Agent.cancel control for arbitrary agmap threads. Join before process/workload exit because daemon threads do not keep the process alive.

```python
from agency import agmap, agsync

tasks = agmap(lambda n: n * n, [2, 3], is_asynchronous=True)
agsync(tasks)
assert [task.result for task in tasks] == [4, 9]
```

## Synchronization

`agsync(*targets) -> None` blocks on agents, teams, agtasks or one-level lists containing those types. An empty list is harmless. It rejects ordinary agdata (including Agent.run results), tuples, nested lists and other values with `TypeError`; use `result.wait()` or `await result` for an agent output.

It joins active team-run futures before collecting team agents, then settles ordered agent contexts and tasks. Team exceptions are collected and raised after joining: one original exception, or an `ExceptionGroup` for multiple failures. Ordinary agent/agmap error payloads do not themselves raise. The barrier covers observed current work, not future submissions made later by unrelated callers.

## Teams

`agteam(agconfig=None, **config)` is a subclassable composition unit. It clones supplied/class-default configuration and sets extra config keywords as attributes. `setup()` runs once synchronously during construction inside the team's active context; agents constructed/forked there, or during a run, are tracked. Agent construction can inherit the team's configured LLM.

Override `run(...)` with a regular function. Class creation wraps it so each call starts a traced daemon thread and returns pending agdata immediately; calls may overlap. Returning agdata resolves its dependencies into the wrapper; returning another object wraps it under `.result`. **An exception in team.run is stored as a future exception and reraised by wait/await/field access**, unlike agent failure payloads. Base `run()` raises `NotImplementedError`.

```python
from agency import agteam, agdata

class LabelTeam(agteam):
    def setup(self):
        self.prefix = "team"

    def run(self, text):
        return agdata(label=f"{self.prefix}: {text}")

team = LabelTeam()
result = team.run("hello")
assert result.wait().label == "team: hello"
```

`team.agents` returns live tracked agents; `agteam.all()` returns live teams (weak registries). `get_config_copy()` returns a deep clone or `None`. `change_config(config)` clones it and cascades to tracked agents; future agents inherit it. Methods return `None` unless described otherwise. Team setup exceptions propagate at construction. Keep teams and their work alive until joined; use `agsync(team)` before leaving your workload.

## SIGTERM cleanup context

`sigterm_as_exit(label="agency")` yields a threading.Event. On the main thread it temporarily handles SIGTERM by setting the event and raising `SystemExit(0)`, allowing Python finally/atexit cleanup; restores the prior handler on exit. Off the main thread it yields an event without installing a handler. It does not handle SIGKILL. Use around the application entry point as in the [quickstart](../../examples/quickstart.py).

## Source signatures

[Source: agmap.py](../../agency/utils/agmap.py)

::: agency.utils.agmap.agmap

[Source: agmap.py](../../agency/utils/agmap.py)

::: agency.utils.agmap.agtask
    options:
      members: []

[Source: agsync.py](../../agency/utils/agsync.py)

::: agency.utils.agsync.agsync

[Source: agteam.py](../../agency/agteam.py)

::: agency.agteam.agteam
    options:
      members: ["__init__", "setup", "run", "change_config", "get_config_copy", "agents", "all"]

[Source: agutil.py](../../agency/utils/agutil.py)

::: agency.utils.agutil.sigterm_as_exit
