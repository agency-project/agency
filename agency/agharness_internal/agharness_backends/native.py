"""Native engine backend running agency's own react loop as a persistent
process INSIDE the sandbox container, instead of agskill.py's in-host-
process `execute_react`. See docs/Design_harness_integration.md's
extension unifying native and harness-driven execution under one
in-container model, and this module's own development history for why:
one launch+bridge mechanism, one credential boundary (agllm_terminus),
shared by every engine.

Two layers:
- The launch+bridge foundation (`launch_in_container_entrypoint`, `ping`):
  starting a persistent process inside an already-running, container-backed
  `agSandbox` (via `exec_detached()`), bind-mounted with the exact `agency`
  package the host is running (`agutil.agency_package_dir`), reachable over
  a Unix domain socket bridge -- the same `agharness_llm_gateway_dir()`
  mount every container-backed sandbox already carries (see agsandbox.py).
- `_NativeBackend`: the actual `agharness_backend.execute()` contract, on
  top of that foundation. LLM calls are dispatched to `agllm_terminus`'s
  `/internal/dispatch` (the same host-side, credential-holding service
  `agproxy_llm.py` uses) -- never a direct call to a real provider from
  inside the container. Built-in tools (bash/read/write/edit/glob/grep/
  webfetch/todowrite -- see `_native_in_container_entrypoint.py`'s own
  `_BUILTIN_TOOL_SCHEMAS`) run locally inside the container process, no
  bridging needed (unlike a host-driven `agtool.py` hop); resource control
  and output submission (reserve_cpu/cpu_release/daemon_release/
  submit_output/ask_human) go through the shared `agMCPServer` bridge,
  same as every other engine. Structured `output_schema` is supported via
  `submit_output`, with a bounded reprompt-retry loop (Phase 6) -- since
  the in-container process is already persistent for the whole call, a
  retry is just one more request over the same socket, not a relaunch.
  Retry-on-transient-provider-error lives in the entrypoint's own
  `_dispatch_via_terminus`, not agllm_terminus.py (see that module's
  retry-policy comment for why: it always streams and can't safely resend,
  and retrying there too would stack an uncoordinated second retry layer
  under whatever an external harness's own CLI already does). Compaction
  (`_maybe_compact`) uses the same algorithm `agllm.py`'s own
  `maybe_compact()` does, shared via `agllm_pure.py`. Pause/inbox-drain
  goes through the shared `agHarnessMessenger` bridge, checked before
  every turn -- mirrors what `execute_react()` already does in-process via
  `ag._check_pause()`/`ag._drain_inbox()`. Live UI visibility
  (`ag.terminal`/`_push_live_messages`/`_set_ui_state`/`_append_full_history`/
  `push_token_count_update_to_ui`) is reconstructed by `_LiveTranscriptPusher`,
  a background thread that polls `agllm_terminus`'s already-live per-token
  transcript while a `run_react_loop()` call is in flight, diffing against
  what was last pushed -- the only way to get `execute_react()`-equivalent
  per-step visibility out of a loop that otherwise runs opaquely inside the
  container for the whole call. Real token usage (`prev_ctx.total_input_
  tokens`/`total_output_tokens`) is accumulated from each dispatch's
  `usage` (threaded back through the entrypoint's `_dispatch_via_terminus`/
  `_run_react_loop`), not estimated -- the live poller's own per-poll
  `push_token_count_update_to_ui` call is the one place that's still an
  estimate (`agllm_pure.estimate_messages_tokens`), since no real usage
  total exists until a dispatch actually completes.

**`skill.add_tools`/`replace_tools` support:** a host-authored tool's `fn`
is cloudpickled here (host-side, where the real closure and whatever state
it captures actually live) and shipped into the container as base64 bytes,
where the entrypoint calls it against a minimal `agdata`/`agerror` stand-in
(`agdata_pure.py`) rather than the real classes (see that module's own
docstring, and `_native_in_container_entrypoint.py`'s `_make_custom_tool_
handler`, for the full mechanism). `replace_tools is not None` suppresses
the entrypoint's built-in tool set entirely (matching the deleted
`_build_toolkit()`'s own precedence: replace_tools replaces the whole set,
add_tools only ever extends it) -- this is also what makes `skill.plan_mode`
(which sets `replace_tools=[]`) work correctly now: zero custom tools plus
suppressed built-ins, same as it meant for `execute_react()`.
`cloudpickle.dumps(tool.fn)` failing (a closure capturing a live host-only
object with no picklable representation at all -- a `threading.Lock`, an
open socket) is checked **before** anything else in `execute()`: an
immediate, specific `agerror` naming the tool, no entrypoint launch
attempted. This is necessarily incomplete as a safety net, not just an
oversight: `cloudpickle.dumps`/`.loads` succeeding is not a guarantee the
closure will behave correctly once it's actually running in the container
-- a closure that captures a live host-only object that happens to *be*
structurally picklable (unlike a lock or socket) will unpickle into a
non-functional duplicate with no error at all. There is no way to detect
that mechanically; the one real requirement this places on an add_tools/
replace_tools author going forward is the same one the user who originally
scoped this feature stated directly: **user tools must only ever execute
inside the container** -- write self-contained closures, don't reach for
host-only state.

**Known gaps, not silently mishandled:**
- `reserve_gpu` isn't exposed by `agMCPServer` yet (see that module's own
  docstring for the specific gap: GPU env-var injection only reaches a
  host-driven `agtool.py`/`agsandbox_backends` hop that no in-container
  tool execution goes through, native or harness-driven).
- No `<think>`/reasoning-tag extraction: `execute_react()`'s streaming
  reassembly (`agllm.build_assistant_msg`) strips a leading `<think>...
  </think>` block into a separate `_thinking` key; the entrypoint's own
  reassembly (`_dispatch_via_terminus`) does not -- a `<think>` block, if a
  model emits one, ends up verbatim in `content`/`final_text` instead.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import socket
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle

from ... import agharness
from ...agdata import agdata, agerror
from .base import agharness_backend

if TYPE_CHECKING:
    from ...agent import agent
    from ...agcontext import agcontext
    from ...agsandbox import agSandbox
    from ...agskill import agskill

# Path *relative to* AGENCY_PACKAGE_CONTAINER_MOUNT -- kept as a plain
# string, joined at call time, rather than resolved via this module's own
# __file__ (which is a HOST path; the container's copy lives at the mount
# point instead, and the two are not related by any transformation other
# than "the same relative path under a different root").
_ENTRYPOINT_RELATIVE_PATH = (
    "agency/agharness_internal/agharness_backends/_native_in_container_entrypoint.py"
)

_BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command inside the sandbox and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}


# ---------------------------------------------------------------------------
# Wire protocol client -- must match _native_in_container_entrypoint.py's
# length-prefixed JSON framing exactly (see that module's docstring for why
# a bare recv() isn't enough once message history grows past one buffer).
# ---------------------------------------------------------------------------


def _recv_exactly(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"connection closed with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _request(sock_path: str, payload: dict, timeout_s: float) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        s.connect(sock_path)
        body = json.dumps(payload).encode("utf-8")
        s.sendall(struct.pack(">Q", len(body)) + body)
        header = _recv_exactly(s, 8)
        (length,) = struct.unpack(">Q", header)
        data = _recv_exactly(s, length)
    return json.loads(data.decode("utf-8"))


def ping(sock_path: str, timeout_s: float = 10) -> dict:
    """Health-check round-trip against a launched entrypoint -- proves the
    process is alive and reachable over the bridge."""
    return _request(sock_path, {"op": "ping"}, timeout_s)


def run_react_loop(sock_path: str, request: dict, timeout_s: float = 600) -> dict:
    """Send a `{"op": "run", ...}` request to a launched entrypoint and
    return its response -- see _native_in_container_entrypoint.py's
    `_run_react_loop` for the request/response shape."""
    return _request(sock_path, {"op": "run", **request}, timeout_s)


# ---------------------------------------------------------------------------
# Launch+bridge foundation
# ---------------------------------------------------------------------------


def launch_in_container_entrypoint(sandbox: "agSandbox", timeout_s: float = 30) -> str:
    """Start the in-container entrypoint as a persistent, detached process
    inside `sandbox`'s already-running container, and return the host-side
    path to its UDS socket once it's confirmed reachable (a real `ping`
    round-trip, not just the socket file's existence -- see `_wait_ready`).

    Not idempotent across repeated calls on the same sandbox -- callers
    should only launch once per sandbox lifetime, mirroring how a harness
    binary is launched once per skill call. `_ensure_entrypoint` below is
    the idempotent wrapper `_NativeBackend.execute()` actually uses."""
    from ...agutil import (
        AGENCY_PACKAGE_CONTAINER_MOUNT,
        ensure_python_packages_in_container,
        new_uds_path,
    )

    # `mcp` for the resource/output-submission MCP client, `html2text` for
    # the local webfetch tool's markdown conversion, `cloudpickle` for
    # loading any `skill.add_tools`/`replace_tools` closures shipped in --
    # all real client libraries the otherwise-stdlib-plus-httpx entrypoint
    # needs; see this file's own module docstring for why httpx itself
    # needs no such install (already in the base image). Always ensured
    # regardless of whether THIS call will use custom tools -- the
    # entrypoint is launched once per persistent sandbox lifetime (see
    # `_ensure_entrypoint`), and a later call on the same sandbox might.
    ensure_python_packages_in_container(sandbox, ["mcp", "html2text", "cloudpickle"], timeout_s=180)

    # Minted via new_uds_path so this socket shares the run-scoped gateway
    # directory and the sun_path budget check; the container side keeps the
    # long mount name, where no such budget applies.
    host_sock_path = Path(new_uds_path("native-entrypoint"))
    sock_name = host_sock_path.name
    container_sock_path = f"/var/run/agency_llm_gateway/{sock_name}"
    entrypoint_path = f"{AGENCY_PACKAGE_CONTAINER_MOUNT}/{_ENTRYPOINT_RELATIVE_PATH}"
    pid_path = f"/tmp/{sock_name}.pid"

    # Run by raw file path, NOT `python3 -m package.module` -- see
    # _native_in_container_entrypoint.py's module docstring: `-m` forces
    # importing every parent package first, which pulls in host-venv-only
    # dependencies (`agllm.py`'s `import openai`, etc.) the sandbox's base
    # image has no reason to carry unless `agutil.
    # ensure_python_packages_in_container` has installed them first. A raw
    # script has no relative imports to resolve, so none of that import
    # chain is touched regardless.
    #
    # `echo $$ > pid_path; exec python3 ...` (not just `python3 ...`):
    # `exec` replaces the shell with the python3 process in place, so the
    # PID captured via `$$` IS the entrypoint's own PID, not a parent shell
    # that then forks it. This matters because the entrypoint is meant to
    # persist for the sandbox's whole lifetime (see `_ensure_entrypoint`) --
    # left untracked, `agSandbox.get_live_pids()`'s auto-adopt-new-PIDs
    # behavior (agsandbox_backends/base.py) would sooner or later pick it up
    # as "background work," and `agSandbox.wait_for_processes()` (now called
    # from every engine's `execute_harness()`, not just execute_react()'s
    # old per-tool-call path) would then block on it forever, mistaking
    # deliberate persistence for an unfinished job. Releasing it as a daemon
    # PID below -- the same mechanism a skill can call on its own backgrounded
    # jobs via the `daemon_release` tool -- is the sanctioned way to tell the
    # monitoring loop "this one's supposed to keep running."
    cmd = (
        f"echo $$ > {shlex.quote(pid_path)}; "
        f"exec python3 {shlex.quote(entrypoint_path)} {shlex.quote(container_sock_path)}"
    )
    sandbox.exec_detached(cmd, workdir="/workspace")

    _wait_ready(str(host_sock_path), timeout_s=timeout_s)

    try:
        entrypoint_pid = int(sandbox.read_file(pid_path).strip())
        sandbox.release_daemon(entrypoint_pid)
    except Exception as e:
        print(f"[native] WARNING: could not release entrypoint PID from monitoring: {e}")

    return str(host_sock_path)


def _wait_ready(sock_path: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_exc: "Exception | None" = None
    while time.monotonic() < deadline:
        try:
            ping(sock_path, timeout_s=1)
            return
        except Exception as e:
            last_exc = e
            time.sleep(0.1)
    raise RuntimeError(f"in-container entrypoint at {sock_path} never became reachable: {last_exc}")


def _ensure_entrypoint(sandbox: "agSandbox") -> str:
    """Idempotent across repeated calls on the same sandbox -- reuses an
    already-launched, still-reachable entrypoint (matching a persistent
    sandbox's whole premise: the process should survive across skill calls,
    not relaunch per call) and only launches fresh if there isn't one yet
    or it stopped responding (e.g. the container itself was hibernated and
    resumed, or destroyed and recreated from a checkpoint)."""
    existing = getattr(sandbox, "_native_entrypoint_sock", None)
    if existing is not None:
        try:
            ping(existing, timeout_s=2)
            return existing
        except Exception:  # noqa: S110 - stale sockets are relaunched below
            pass  # stale -- fall through and relaunch
    sock_path = launch_in_container_entrypoint(sandbox)
    sandbox._native_entrypoint_sock = sock_path
    return sock_path


_LIVE_POLL_INTERVAL_S = 0.25


class _LiveTranscriptPusher:
    """Mirrors `execute_react()`'s per-step UI hooks for a native run, whose
    actual loop runs opaquely inside the container for the duration of one
    blocking `run_react_loop()` call. `agllm_terminus.transcript_for_token()`
    already keeps the current run's full conversation (system/history/every
    turn so far), overwritten on each dispatch (see that method's own
    docstring) -- polling it here and diffing against what was last pushed
    reconstructs the same play-by-play `execute_react()` produces natively,
    just at poll granularity instead of per-call.

    `poll_once()` is called both repeatedly from a background thread (`run()`)
    while the socket call is in flight, and once more synchronously right
    after it returns (`_NativeBackend.execute()`) to guarantee the very last
    turn isn't lost to poll-interval timing before `terminus.unregister()`
    discards the transcript. Best-effort/UI-only, matching the hooks
    themselves (agent.py wraps each in its own try/except) -- any failure
    here must never affect the actual run."""

    def __init__(self, ag: "agent", terminus, token: str, skill_name: str) -> None:
        self._ag = ag
        self._terminus = terminus
        self._token = token
        self._skill_name = skill_name
        self._last_len = 0

    def poll_once(self) -> None:
        ag = self._ag
        try:
            transcript = self._terminus.transcript_for_token(self._token)
            if not transcript or len(transcript) <= self._last_len:
                return
            new_messages = transcript[self._last_len :]
            self._last_len = len(transcript)
            for msg in new_messages:
                if ag._append_full_history:
                    ag._append_full_history(msg)
                if msg.get("role") == "assistant" and ag.terminal:
                    ag.terminal.log("LLM ✓    ", f"model={ag.llm.backend.model or '?'}")
            if ag._push_live_messages:
                ag._push_live_messages(transcript[1:])
            if ag._set_ui_state:
                ag._set_ui_state("skill", skill=self._skill_name)
            from ... import agllm_pure

            ag.push_token_count_update_to_ui(agllm_pure.estimate_messages_tokens(transcript), 0)
        except Exception:  # noqa: S110 - live UI updates are best-effort
            pass

    def run(self, stop_event: "threading.Event") -> None:
        while not stop_event.wait(_LIVE_POLL_INTERVAL_S):
            self.poll_once()


def _container_bridge_sock_path(host_sock_path: str) -> str:
    """Translate a host-side UDS path (agllm_terminus's or agmcp_server's --
    same shape, same mount, either one) to its container-side equivalent --
    same file, reached via the bind-mounted gateway dir every
    container-backed sandbox already carries (see agsandbox.py's
    `_agharness_llm_gateway` mount)."""
    return f"/var/run/agency_llm_gateway/{Path(host_sock_path).name}"


# ---------------------------------------------------------------------------
# agharness_backend.execute() contract
# ---------------------------------------------------------------------------


class _NativeBackend(agharness_backend):
    def execute(
        self,
        ag: "agent",
        prev_ctx: "agcontext",
        skill_input: agdata,
        max_steps: "int | None",
        *,
        skill: "agskill",
        extra_system: "str | None" = None,
        canonical_input=None,
    ) -> "tuple[agdata, agcontext, list[dict]]":
        # Same precedence the deleted `_build_toolkit()` used: replace_tools
        # (including plan_mode's `replace_tools=[]`) replaces the whole tool
        # set, add_tools only ever extends it -- so add_tools is ignored
        # entirely once replace_tools is set, never merged with it.
        suppress_builtins = skill.replace_tools is not None
        custom_tool_objs = skill.replace_tools if suppress_builtins else (skill.add_tools or [])

        # Fail fast, host-side, before anything else (no entrypoint launch,
        # no socket round-trip): a closure capturing a live host-only
        # object with no picklable representation at all (a
        # `threading.Lock`, an open socket -- as opposed to one that's
        # merely *wrong* once relocated, which cloudpickle can't detect;
        # see this module's own docstring) fails right here, clearly, one
        # tool at a time.
        custom_tools_payload = []
        for tool in custom_tool_objs:
            try:
                pickled = cloudpickle.dumps(tool.fn)
            except Exception as e:
                return (
                    agerror(
                        f"add_tools/replace_tools: tool {tool.name!r} could not be "
                        f"prepared for in-container execution: {type(e).__name__}: {e}"
                    ),
                    prev_ctx,
                    [],
                )
            custom_tools_payload.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "params": tool.params,
                    "fn_b64": base64.b64encode(pickled).decode(),
                }
            )

        # Structured output (Phase 6's "Outer" retry, adapted): unlike
        # claude_code.py, there is no process to relaunch -- the
        # in-container entrypoint is already a persistent process for this
        # sandbox's whole lifetime, so a retry is just one more `run_react_
        # loop()` request over the same socket, continuing the same
        # `messages` list with a reprompt appended. Cheaper than a harness
        # relaunch, not a different mechanism.
        _use_structured_output = (
            skill.output_schema is not None and skill.output_schema.raw_key() is None
        )
        output_schema_retries_left = skill.max_output_schema_retries

        sys_msg = {"role": "system", "content": skill._build_system_prompt(extra_system)}
        user_content = skill._build_user_content(skill_input)
        if _use_structured_output:
            extra = agharness.build_mcp_output_format_instruction(skill)
            if extra and isinstance(user_content, str):
                user_content = user_content + extra
        user_msg = {"role": "user", "content": user_content}

        sock_path = _ensure_entrypoint(ag.sandbox)

        from ..agllm_terminus import get_shared_terminus
        from ..agmcp_server import get_shared_mcp_server
        from ..agharness_messenger import get_shared_messenger

        terminus = get_shared_terminus(ag.agconfig)
        mcp_server = get_shared_mcp_server(ag.agconfig)
        messenger = get_shared_messenger(ag.agconfig)
        from ..agprof_ingest import get_shared_profiler_ingest
        from ...profiler import agprof

        profiler_ingest = get_shared_profiler_ingest()
        token = uuid.uuid4().hex
        profile_native_events = agprof.enabled()
        if not profile_native_events and os.environ.get("AGENCY_PROFILE"):
            print(
                "[native] WARNING: environment profiling is requested but no active "
                "profiler session can receive in-container events"
            )
        profiler_host_sock = profiler_ingest.ensure_uds_started() if profile_native_events else None
        terminus.register(token, ag)
        profiler_ingest.register(token, ag, exact_events=profile_native_events)
        mcp_server.register(token, ag, skill)
        messenger.register(token, ag)

        collected_output: dict = {}
        final_text = ""
        profiler_turn_offset = 0
        pusher = _LiveTranscriptPusher(ag, terminus, token, skill.name)
        stop_poll = threading.Event()
        poll_thread = threading.Thread(target=pusher.run, args=(stop_poll,), daemon=True)
        poll_thread.start()
        try:
            request_messages = [sys_msg] + list(prev_ctx.messages) + [user_msg]
            messages = list(request_messages)
            llm_kwargs = ag.llm.build_kwargs([], None)
            llm_kwargs.pop("model", None)
            llm_kwargs.pop("messages", None)
            while True:
                request = {
                    "token": token,
                    "terminus_sock": _container_bridge_sock_path(terminus.ensure_uds_started()),
                    "mcp_server_sock": _container_bridge_sock_path(mcp_server.ensure_uds_started()),
                    "messenger_sock": _container_bridge_sock_path(messenger.ensure_uds_started()),
                    "profiler_sock": (
                        _container_bridge_sock_path(profiler_host_sock)
                        if profiler_host_sock is not None
                        else None
                    ),
                    "model": ag.llm.backend.model or "",
                    "llm_kwargs": llm_kwargs,
                    "messages": messages,
                    "max_steps": max_steps or 20,
                    "custom_tools": custom_tools_payload,
                    "suppress_builtins": suppress_builtins,
                    "profiler_turn_offset": profiler_turn_offset,
                }
                resp = run_react_loop(sock_path, request)
                profiler_turn_offset += int(resp.get("turn_count") or 0)
                profiler_dropped_events = int(resp.get("profiler_dropped_events") or 0)
                if profiler_dropped_events:
                    agprof.annotate(profiler_dropped_events=profiler_dropped_events)
                    print(
                        "[native] WARNING: in-container profiler dropped "
                        f"{profiler_dropped_events} event(s)"
                    )
                if resp.get("status") != "done":
                    return (
                        agerror(resp.get("message", "native in-container run failed")),
                        prev_ctx,
                        [sys_msg],
                    )
                messages = resp["messages"]
                final_text = resp.get("final_text", "")
                usage = resp.get("usage") or {}
                prev_ctx.total_input_tokens += usage.get("input_tokens", 0)
                prev_ctx.total_output_tokens += usage.get("output_tokens", 0)

                if not _use_structured_output:
                    break
                # Read fresh every attempt -- agmcp_server accumulates
                # across calls for the same token, so a field submitted on
                # an earlier attempt is still there after a reprompt.
                collected_output = mcp_server.collected_output(token)
                required = set(skill.output_schema._data.keys())
                missing = sorted(required - set(collected_output.keys()))
                if not missing or output_schema_retries_left <= 0:
                    break
                output_schema_retries_left -= 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[HARNESS SYSTEM] You have not yet provided all required "
                            f"output fields. Still missing: {missing}. Call the "
                            "submit_output tool once for each of them."
                        ),
                    }
                )
        finally:
            # Stop the poller and give the terminus's live transcript one
            # last synchronous read before unregister() discards it --
            # `stop_poll.wait()`-driven polling alone could otherwise miss
            # whatever the very last dispatch produced (see
            # _LiveTranscriptPusher's own docstring).
            stop_poll.set()
            poll_thread.join(timeout=2)
            pusher.poll_once()
            terminus.unregister(token)
            profiler_ingest.unregister(token)
            mcp_server.unregister(token)
            messenger.unregister(token)

        new_since_call_start = messages[len(request_messages) :]

        if _use_structured_output:
            required = set(skill.output_schema._data.keys())
            missing = sorted(required - set(collected_output.keys()))
            if missing:
                result = agerror(
                    "structured output incomplete after "
                    f"{skill.max_output_schema_retries - output_schema_retries_left} "
                    "retry/retries -- submit_output was never called for: " + ", ".join(missing)
                )
            else:
                result = agharness.finalize_harness_result(
                    agharness.HarnessResult(submitted_fields=collected_output), skill, ag.sandbox
                )
        else:
            result = agharness.finalize_harness_result(
                agharness.HarnessResult(final_text=final_text), skill, ag.sandbox
            )

        prev_ctx.messages = list(prev_ctx.messages) + [user_msg] + new_since_call_start
        return result, prev_ctx, [sys_msg, user_msg] + new_since_call_start


__all__ = [
    "launch_in_container_entrypoint",
    "ping",
    "run_react_loop",
]
