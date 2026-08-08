# Mediation interface (`agpolicy.py`)

`agpolicy` is the decision point [`agproxy_ptrace`](agharness_internal/agproxy_ptrace.md) calls on every intercepted
syscall for a harness-driven agent, and the point `agtool.dispatch_tools()` calls (when a caller
opts in via its `policy=`/`ag=` parameters — `None` by default, so every existing native call site
is unaffected) for native tool calls. Same interface, two different event sources — a syscall
trace stop for a harness-driven agent, a tool-call dispatch for a native one — so one policy
implementation can govern a mixed team of both.

```python
from agency.agpolicy import agpolicy, agdecision

class MyPolicy(agpolicy):
    def check(self, ag, event) -> agdecision:
        if event.argv and event.argv[0] == "/bin/rm":
            return agdecision.deny("rm is not allowed")
        if event.argv and event.argv[0] == "/bin/curl":
            return agdecision.rewrite(["/bin/curl", "--max-time", "10", *event.argv[1:]])
        return agdecision.allow()
```

## `agdecision`

```python
@dataclass
class agdecision:
    kind: str  # "allow" | "deny" | "rewrite"
    reason: str | None = None
    new_args: list[str] | None = None

    @staticmethod
    def allow() -> agdecision: ...
    @staticmethod
    def deny(reason: str) -> agdecision: ...
    @staticmethod
    def rewrite(new_args: list[str]) -> agdecision: ...
```

- `allow()` — the syscall/tool call proceeds unmodified.
- `deny(reason)` — the syscall is skipped and made to fail with `EPERM`; a denied tool call would
  surface `reason` back to the caller as an error, never silently substituting a fabricated
  success (see the design doc's "Design Tensions" on what mediation cannot do).
- `rewrite(new_args)` — for a syscall, `agproxy_ptrace` injects `new_args` into the tracee's
  memory and repoints its registers before resuming, so the *rewritten* command is what actually
  runs — the original arguments never execute.

## `agpolicy`

Base class; `check(self, ag, event) -> agdecision` raises `NotImplementedError` — subclass and
override it. `event` is always an `agsyscallevent` ([agproxy_ptrace.md](agharness_internal/agproxy_ptrace.md)):
for syscall-level mediation, `argv`/`envp`/`path` are populated and `tool_name`/`tool_args` are
`None`; for `agtool.dispatch_tools()`'s native retrofit, `syscall="tool_call"`,
`tool_name`/`tool_args` are populated, and `argv`/`envp`/`path` are `None` — one event type either
way, so a policy implementation never needs to special-case which mediation path it's being called
from. `rewrite` is only meaningful for the syscall path (there's no single argv-shaped string to
rewrite for a tool call's arbitrary JSON arguments) — `dispatch_tools()` treats a `rewrite`
decision as `allow` if a policy returns one for a native tool call.

## `agAllowAllPolicy`

```python
class agAllowAllPolicy(agpolicy):
    def check(self, ag, event) -> agdecision:
        return agdecision.allow()
```

Bootstrapping implementation only — allows everything unmediated. Not a real security boundary;
exists so `agproxy_ptrace`'s launch/trace loop has something to construct against before any real
policy logic is wired up.
