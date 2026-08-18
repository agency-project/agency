"""Harness backend base class, shared config, and backend selection.

Mirrors `llm/base.py`'s shape: one base (`agharness_backend`)
concrete backends subclass, a shared `Fields` class so every backend reads
its tunables as plain attributes, and a `for_config()` selector with lazy,
function-local imports of the concrete backends (avoids a circular import,
same reasoning as llm).

Selection dispatches on the `engine` string itself (e.g. `ag.engine ==
"claude_code"`, set via `agent(engine=...)` -- see docs/agent.md's engine
seam) rather than a separate `provider` config field: `ag.engine` is
already the authoritative "which engine" signal (Component 4 of
docs/Design_harness_integration.md), so there is no second place a user
would need to keep in sync with it.

**`execute()` is now a real, shared TEMPLATE METHOD, not just an abstract
signature** (see the conversation that produced `agmanager_host`/
`agmanager_harness`/`native_harness` for the design this completes): every
harness-driven engine's `execute()` call shares the exact same shape --
build the user-turn prompt, run a bounded reprompt-retry loop on
incomplete structured output, capture/restore session continuity, poll
live per-turn transcript updates, build the final `(result, ctx, delta)`
return value. Only ONE thing genuinely differs per engine: how to actually
run ONE attempt -- resolving/launching the specific CLI (a plain
`sandbox.exec()` for `native_harness`, a ptrace-traced subprocess for
Claude Code/Codex/opencode/Grok), building its argv/env from
`harness_base_url`/`launch.token`, and parsing its output back into
`(final_text, usage, session_id)`. That's `_run_attempt()` -- the one hook
a migrated concrete backend implements; everything else lives here, once,
instead of five times.

**Migration status**: `native.py`'s `_NativeBackend` implements
`_run_attempt()` and inherits this template. `claude_code.py`/`codex.py`/
`opencode.py`/`grok.py` still define their OWN `execute()` (which simply
shadows this template method entirely -- ordinary Python override, no
special-casing needed here) and construct their own bridge singletons the
old way; migrating one of them means deleting its `execute()` override and
implementing `_run_attempt()` instead, exactly like `native.py` did.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...agconfig import DynamicConfigParam, _AgConfigViewBase
from ...agdata import agdata, agerror

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agent import agent
    from ...agcontext import agcontext
    from ...agskill import agskill
    from ...manager.agmanager_host import agHostAgentManager, LaunchHandle

_LIVE_POLL_INTERVAL_S = 0.25


class AgHarnessFields:
    """Every harness config field used by any backend, as
    DynamicConfigParam descriptors -- see llm/base.py's
    AgLLMBackendFields for the same pattern."""

    gateway_mode = DynamicConfigParam(
        "agharness", default="passthrough"
    )  # "passthrough" (the configured agllm backend already speaks the
    # harness's wire format, forward requests unmodified through
    # agproxy_llm) | "translate" (reshape requests/responses -- not
    # implemented yet; only the passthrough route exists so far).
    session_resume_id = DynamicConfigParam(
        "agharness", default=None
    )  # the harness's own session id from a prior run on this agent, for
    # multi-turn resume -- set by a concrete backend after its first execute().
    binary_path = DynamicConfigParam(
        "agharness", default=None
    )  # override the harness CLI's resolved path; None = look up the
    # backend-specific default binary name via PATH.
    mediation_mode = DynamicConfigParam(
        "agharness", default="auto"
    )  # "ptrace" | "native_hooks" | "auto" (ptrace if ptrace_available(),
    # else native_hooks with a logged reduced-coverage warning) -- see
    # `harness/_native_hooks.py`.

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agHarnessConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agharness fields in one call::

        cfg = agConfig(agHarnessConfig(gateway_mode="translate"))
        ag = agent(agconfig=cfg, engine="codex")

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agharness"


@dataclass
class AttemptResult:
    """What `_run_attempt()` returns for ONE harness-CLI invocation --
    everything the shared retry loop/result-building in `execute()` needs,
    normalized across every engine regardless of how different their own
    launch mechanism/output format actually is.

    `session_blob`/`session_id` are None when the engine's launch never
    reached a point where a session exists yet (e.g. a hard launch
    failure) -- `execute()` simply doesn't update `ag._harness_sessions`
    in that case, leaving whatever was there from a previous call alone."""

    ok: bool
    final_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: "str | None" = None
    session_blob: "bytes | None" = None
    error_message: str = ""


class _LiveTranscriptPusher:
    """Mirrors `execute_react()`'s per-step UI hooks while a harness CLI
    runs as one blocking call, by polling `host_manager.
    transcript_for_token()` -- the same live per-token transcript
    `agmanager_host`'s dispatch route records on every turn, regardless of
    which engine drove it. Shared here (not duplicated per backend) since
    it depends on nothing engine-specific -- only `host_manager`/a launch
    token, both already uniform across every engine."""

    def __init__(
        self, ag: "agent", host_manager: "agHostAgentManager", token: str, skill_name: str
    ) -> None:
        self._ag = ag
        self._host_manager = host_manager
        self._token = token
        self._skill_name = skill_name
        self._last_len = 0

    def poll_once(self) -> None:
        ag = self._ag
        try:
            transcript = self._host_manager.transcript_for_token(self._token)
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


class agharness_backend(AgHarnessFields):
    """One backend instance per agconfig -- drives one off-the-shelf
    harness CLI in place of agskill's native ReAct loop. Use
    `agharness_backend.for_config(engine, agconfig)` to get the right
    subclass; don't instantiate a subclass directly."""

    #: Set by each concrete subclass that implements `_run_attempt()` --
    #: the key `ag._harness_sessions` is stored/looked up under. Backends
    #: that still override `execute()` directly (not yet migrated) don't
    #: need this at all.
    engine_key: str = ""

    #: True for a backend whose own PreToolUse/PostToolUse hook reports
    #: exact per-tool timing to `/agprof/hook` (Claude Code today), so
    #: `agskill.py`'s `execute_harness()` passes `exact_tool_events=True`
    #: to `host_manager.register_launch()` -- telling `agmanager_host`'s
    #: profiler ingest to trust those hook-reported spans over its own
    #: transcript-derived estimate for the same tool calls. False (the
    #: default) for a backend with no such hook mechanism.
    uses_exact_tool_events: bool = False

    def __init__(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig.clone()

    def change_config(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig.clone()

    def get_config_copy(self) -> "agConfig":
        return self._agconfig.clone()

    def execute(
        self,
        ag: "agent",
        prev_ctx: "agcontext",
        skill_input: "agdata",
        max_steps: "int | None",
        *,
        skill: "agskill",
        extra_system: "str | None" = None,
        host_manager: "agHostAgentManager | None" = None,
        harness_base_url: "str | None" = None,
        launch: "LaunchHandle | None" = None,
    ) -> "tuple[agdata, agcontext, list[dict]]":
        """Same return contract as `agskill.execute_react()`/
        `execute_harness()`: `ctx` is the SAME `prev_ctx` object, mutated in
        place (`.messages`/`.total_input_tokens`/`.total_output_tokens`);
        `delta` is `[system_prompt_message] + every message appended since
        this call started`.

        `host_manager`/`harness_base_url`/`launch` are constructed and
        owned by `agskill.py`'s `execute_harness()` -- the host process,
        not this backend -- via `agharness.get_or_create_host_manager()`/
        `agharness.ensure_harness_bridge()`/`host_manager.register_launch()`.
        This method only ever CONSUMES them; it never constructs, caches,
        or tears any of them down. `harness_base_url` is None for a
        bare-host/chroot launch (no container to bridge into) -- whether
        that's fatal is up to each concrete `_run_attempt()` (native_harness
        requires a container; Claude Code doesn't).

        A backend not yet migrated onto `_run_attempt()` simply overrides
        THIS method entirely (ordinary Python override) and ignores all
        three, constructing its own bridge singletons the old way."""
        sys_msg = {"role": "system", "content": skill._build_system_prompt(extra_system)}

        if host_manager is None or launch is None:
            return (
                agerror(
                    f"{self.engine_key or type(self).__name__}'s harness backend requires "
                    "host_manager/launch, constructed by agskill.py's execute_harness() -- "
                    "backends no longer build their own bridge"
                ),
                prev_ctx,
                [sys_msg],
            )

        from .. import agharness

        first_prompt = agharness.build_user_turn_prompt(skill, skill_input)
        if not isinstance(first_prompt, str):
            first_prompt = json.dumps(first_prompt)
        extra = agharness.build_mcp_output_format_instruction(skill)
        if extra:
            first_prompt = first_prompt + extra

        _use_structured_output = (
            skill.output_schema is not None and skill.output_schema.raw_key() is None
        )
        output_schema_retries_left = skill.max_output_schema_retries

        prior = ag._harness_sessions.get(self.engine_key)
        resume_session_id = prior.get("session_id") if prior else None
        prior_blob_b64 = prior.get("blob_b64") if prior else None

        pusher = _LiveTranscriptPusher(ag, host_manager, launch.token, skill.name)
        stop_poll = threading.Event()
        poll_thread = threading.Thread(target=pusher.run, args=(stop_poll,), daemon=True)
        poll_thread.start()

        prompt = first_prompt
        total_input_tokens = 0
        total_output_tokens = 0
        collected_output: dict = {}
        final_text = ""
        attempt: "AttemptResult | None" = None
        try:
            # Bounded relaunch on incomplete structured output (Phase 6's
            # "Outer" layer): each attempt is a fresh, one-shot harness CLI
            # invocation, resumed via whatever session mechanism that
            # engine has from the 2nd attempt on. Only structured-output
            # skills loop at all.
            while True:
                prior_blob = base64.b64decode(prior_blob_b64) if prior_blob_b64 else None
                attempt = self._run_attempt(
                    ag,
                    host_manager,
                    harness_base_url,
                    launch,
                    skill,
                    prompt=prompt,
                    resume_session_id=resume_session_id,
                    prior_session_blob=prior_blob,
                    max_steps=max_steps,
                )
                if not attempt.ok:
                    break  # handled after the loop -- no retry on a hard launch failure

                total_input_tokens += attempt.input_tokens
                total_output_tokens += attempt.output_tokens
                final_text = attempt.final_text

                # Capture the (possibly new/updated) session blob for the
                # NEXT execute() call. Best-effort at the concrete backend
                # level (see AttemptResult's docstring); here it's just a
                # dict write.
                if attempt.session_id:
                    resume_session_id = attempt.session_id
                    if attempt.session_blob is not None:
                        ag._harness_sessions[self.engine_key] = {
                            "session_id": attempt.session_id,
                            "blob_b64": base64.b64encode(attempt.session_blob).decode(),
                        }
                prior_blob_b64 = None  # already resumed -- don't re-restore a stale prior blob

                # Read fresh every attempt -- agmanager_host accumulates
                # across calls for the same token, so a field submitted on
                # an earlier attempt is still there after a reprompt.
                collected_output = host_manager.collected_output(launch.token)
                if not _use_structured_output:
                    break
                required = set(skill.output_schema._data.keys())
                missing = sorted(required - set(collected_output.keys()))
                if not missing or output_schema_retries_left <= 0:
                    break
                output_schema_retries_left -= 1
                prompt = (
                    "[HARNESS SYSTEM] You have not yet provided all required output "
                    f"fields. Still missing: {missing}. Call the submit_output tool "
                    "once for each of them."
                )

            transcript = host_manager.transcript_for_token(launch.token)
        finally:
            stop_poll.set()
            poll_thread.join(timeout=2)
            pusher.poll_once()

        if attempt is None or not attempt.ok:
            message = attempt.error_message if attempt is not None else "no attempt was made"
            return agerror(message), prev_ctx, [sys_msg]

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
                result = agdata(**collected_output)
        else:
            out_key = skill.output_schema.raw_key() if skill.output_schema is not None else "result"
            result = agdata(**{out_key: final_text})

        prev_ctx.total_input_tokens += total_input_tokens
        prev_ctx.total_output_tokens += total_output_tokens

        if transcript:
            history = transcript[1:] if transcript[0].get("role") == "system" else transcript
            prev_ctx.messages = history
            delta = [sys_msg] + history
        else:
            # This launch never reached agmanager_host at all (a mocked
            # test double, or a launch that failed before any LLM call) --
            # fall back to the coarse shape rather than silently returning
            # an empty history.
            user_msg = {"role": "user", "content": first_prompt}
            assistant_msg = {"role": "assistant", "content": final_text}
            prev_ctx.messages = [user_msg, assistant_msg]
            delta = [sys_msg, user_msg, assistant_msg]
        return result, prev_ctx, delta

    def _run_attempt(
        self,
        ag: "agent",
        host_manager: "agHostAgentManager",
        harness_base_url: "str | None",
        launch: "LaunchHandle",
        skill: "agskill",
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        """Run ONE harness-CLI invocation and return its normalized result.
        The one thing that genuinely differs per engine -- everything else
        lives in `execute()` above, shared. A concrete backend implements
        this INSTEAD of overriding `execute()`; see `native.py`'s
        `_NativeBackend` for the reference implementation.

        Backend-specific concerns that belong HERE, not in `execute()`:
        resolving/launching the actual CLI (binary lookup, ptrace tracing,
        or a plain `sandbox.exec()`), building its argv/env from
        `harness_base_url`/`launch.token`, whether `harness_base_url is
        None` is fatal for this engine, whether `skill.add_tools`/
        `replace_tools` are supported, parsing the CLI's own output format,
        and reading back whatever this engine's own session-continuity
        file(s) contain after a successful run."""
        raise NotImplementedError

    @staticmethod
    def for_config(engine: str, agconfig: "agConfig") -> "agharness_backend":
        from .claude_code import _ClaudeCodeBackend
        from .codex import _CodexBackend
        from .grok import _GrokBackend
        from .native import _NativeBackend
        from .opencode import _OpencodeBackend

        if engine == "native":
            return _NativeBackend(agconfig)
        if engine == "opencode":
            return _OpencodeBackend(agconfig)
        if engine == "claude_code":
            return _ClaudeCodeBackend(agconfig)
        if engine == "codex":
            return _CodexBackend(agconfig)
        if engine == "grok":
            return _GrokBackend(agconfig)
        raise ValueError(
            f"Unknown harness engine {engine!r} -- set agent(engine=...) to one of "
            f"'native', 'opencode', 'claude_code', 'codex', 'grok'"
        )
