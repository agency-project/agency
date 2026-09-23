# Bugfixes to merge upstream

Found while building the tandem harness on top of `native_harness`/the
shared adapter base. Both bugs are general — they affect `native_harness`
and the `native`/`tandem` adapter path, not anything specific to the
tandem-harness feature itself — so they're worth a separate PR against
upstream rather than getting buried in the tandem-specific commit.

Commit in this branch: `5d429e8` — "Fix native harness bash tool's
ignored workdir/timeout, idle-timeout session loss". Touches:
`agency/native_harness/tools.py`, `agency/native_harness/react_loop.py`,
`agency/native_harness/cli.py`, `agency/harness/adapters/native.py`, plus
tests in `tests/test_native_harness_tools.py` and
`tests/harness/adapters/test_daemon_seam.py`.

## 1. The bash tool's `workdir`/`timeout` parameters were silently ignored

`_run_bash_tool` in `native_harness/tools.py` advertises both in its own
schema (`BASH_PARAMS`):

```python
BASH_PARAMS = {
    "properties": {
        "command": {...},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)..."},
        "workdir": {"type": "string", "description": "Working directory (optional)"},
    },
}
```

but the implementation never read either argument:

```python
def _run_bash_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    command = args.get("command", "") if isinstance(args, dict) else ""
    proc = subprocess.run(
        ["bash", "-c", command], capture_output=True, timeout=_BASH_TIMEOUT_S, text=True
    )
    ...
```

Every command ran in whatever the harness process's own cwd happened to
be, and always with the hardcoded 120s default, regardless of what the
model actually passed. In practice this meant a model that correctly
tried to scope a command to the repo it was given (`workdir: "/work/foo"`
with a relative command) silently ran somewhere else instead, got a
"file not found"-shaped error, and had no way to tell the difference
between "that path doesn't exist" and "the tool didn't do what I asked."

**Fix**: pass `cwd=workdir` and `timeout=<parsed args.get("timeout")>` to
`subprocess.run`. A missing `workdir` falls back to the process's own cwd
(existing behavior preserved); an invalid one surfaces a clear
`FileNotFoundError` via the existing exception handler, no special-casing
needed. Tests added: `tests/test_native_harness_tools.py::TestBash`.

**Same bug, same fix, also applied to `tandem_harness/tools.py`** (an
independent copy of the same file, per that package's own "fork of
native_harness" docstring) in this branch's other commit
(`ae2d3a1`) — not part of this upstream-facing changelog since
`tandem_harness` isn't upstream, but worth knowing the fix needs applying
in both places if upstream's `native_harness` ever forks again.

## 2. An idle-timeout retry silently discarded the whole session

`NativeAdapter.run_daemon_attempt` launches `native_harness.cli` as a
subprocess and polls a `progress.json` checkpoint file for liveness. If
that file goes stale for `_DEFAULT_TIMEOUT_S` (300s), the adapter treats
the attempt as a soft, recoverable timeout rather than a failure
(`_partial_result_from_progress`, deliberately — "a step-driven
completion signal that can arrive late is not evidence the run itself
failed").

The problem: that recovery path returned `AttemptResult(ok=True, ...)`
with **no `session_id`/`session_blob` at all**. The subprocess only ever
learns/reports its own session_id from its own stdout JSON, printed once,
right at a *clean* exit — never reached on a timeout. Separately,
`session_store.save_session(...)` (native_harness/session.py) was also
only ever called once, at that same clean-exit point — so even knowing
the right id, there was nothing checkpointed on disk to recover anyway.

Consumer-side, `engine.py`'s output-schema retry loop
(`_execute_harness`) depends entirely on `attempt.session_id`/
`attempt.session_blob_b64` to carry continuity from one retry to the
next:

```python
while True:
    attempt = self._run_attempt(prompt, ..., resume_session_id=resume_session_id, ...)
    if not attempt.ok:
        break
    if attempt.session_id:
        resume_session_id = attempt.session_id
        prior_session_blob_b64 = attempt.session_blob_b64
    ...
    prompt = self._build_retry_prompt(missing, system_instruction=prompt.system_instruction)
```

`_build_retry_prompt` deliberately sends nothing but a short "you haven't
submitted all required fields yet" reminder as the new `user_content` —
by design, since the resumed session is supposed to already carry the
real task. When an idle-timeout attempt fed back no session at all, the
next attempt had nothing to resume, `cli.py`'s `_resolve_session` minted
a brand-new session id, and the retry launched from *only* that bare
reminder — the entire prior conversation (the original task input, every
tool call and finding so far) was gone, permanently, replaced with a
generic nudge carrying zero information about what the task even was.
Confirmed end to end against a real run's own SQLite transcript: the
supervisor's own prompt dropped from dozens of accumulated messages to
exactly `[system, "you haven't submitted all required fields..."]`.

**Fix, three parts (all in this commit)**:

1. `run_daemon_attempt` decides `session_id` itself, up front —
   `resume_session_id or uuid.uuid4().hex` — instead of waiting to learn
   it from the subprocess's own stdout. Passed via `--session-id` on
   *every* attempt now (`cli.py` already treated `--session-id` and
   `--resume` identically in `_resolve_session`, so `--resume` is now
   redundant and removed).
2. `run_react_loop` (`react_loop.py`) takes a new optional
   `on_checkpoint: Callable[[list], None]` callback, invoked with the
   current `messages` right where the existing per-step `_write_progress`
   liveness checkpoint already fires — best-effort, same as that one (a
   broken save must never interrupt the actual task). `cli.py` wires this
   to `session_store.save_session(...)`, so the session is now
   checkpointed after *every* turn, not just once at a clean finish.
3. `_partial_result_from_progress` takes the now-always-known
   `scratch_dir`/`session_id`, tries reading a session file for it, and
   returns `session_id`/`session_blob` in the timeout-path
   `AttemptResult` too — previously it returned neither.

Together, a timeout-triggered retry now resumes the real, up-to-date
conversation instead of starting over blank. Verified against a real run
that hit this retry path 30+ times in a single execution: the resumed
supervisor conversation grew monotonically (23 → 77 messages) across
every single retry, with zero drops anywhere in the transcript — where
before, any one of those retries would have reset it to two bare
messages.

Tests added: `tests/harness/adapters/test_daemon_seam.py::
test_native_always_passes_a_session_id_even_when_not_resuming`,
`::test_native_idle_deadline_also_recovers_the_session_for_resume`.

### Note for whoever ports this upstream

Everything above is written against `native_harness`/`NativeAdapter`
directly, which is exactly what upstream has. The `tandem_harness`/
`TandemAdapter` side of this same branch applies the identical fix, but
that package doesn't exist upstream (yet) — safe to ignore when porting.
One difference worth carrying over deliberately: tandem's version of the
`on_checkpoint` wiring is scoped to the *supervisor's* own loop only, not
the worker's (a worker segment is stateless/ephemeral by design, so it
has no session of its own to checkpoint) — not relevant to
`native_harness`, which has only the one loop, but worth knowing if this
pattern gets reused for another multi-loop harness later.
