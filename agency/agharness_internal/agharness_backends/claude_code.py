"""Claude Code backend.

LLM routing: `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` are pointed at
`agproxy_llm`'s `/v1/messages` route (Anthropic Messages API,
`gateway_mode="translate"` -- see agproxy_llm.py/agproxy_llm_adapters.py),
which reshapes the request into the exact `client.chat.completions.create()`
call every other backend uses and routes it through this agent's own
configured `agConfig` backend. The host's own real Anthropic credentials
(API key, OAuth login, Bedrock env) are deliberately NOT forwarded to the
launched process -- every `claude`-driven agent's LLM traffic goes through
agency's own backend choice, not whatever this host happens to have lying
around. Everything else (isolated config home, agproxy_ptrace launch +
tracing, output-schema recovery) works the same as the opencode backend and
was verified against the real `claude` CLI (v2.1.212) during development.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ...agdata import agdata, agerror
from .base import agharness_backend

if TYPE_CHECKING:
    from ...agent import agent
    from ...agcontext import agcontext
    from ...agskill import agskill


def claude_code_available() -> bool:
    return shutil.which("claude") is not None


_BIN_CACHE_MOUNT = "/opt/agency_harness_bin"


def _resolve_binary_in_container(sandbox, binary: str) -> "str | None":
    """Find *binary* for a container-backed launch, in order: (1) already
    on the container image's own PATH -- e.g. a purpose-built image that
    bakes it in; (2) the host-side binary cache every container-backed
    sandbox has bind-mounted read-only at `_BIN_CACHE_MOUNT` (see
    `agutil.agharness_binary_cache_dir`); (3) seed that cache, on the HOST,
    from the host's own `shutil.which(binary)` -- never fetched over the
    network by Agency itself, so this never depends on knowing an install
    URL, and never requires the container to have network egress. Returns
    None only if none of the three has it."""
    import shlex

    out, rc = sandbox.exec(f"which {shlex.quote(binary)}", workdir="/")
    if rc == 0 and out.strip():
        return out.strip()

    cached_path = f"{_BIN_CACHE_MOUNT}/{binary}"
    out, rc = sandbox.exec(f"test -x {shlex.quote(cached_path)}", workdir="/")
    if rc == 0:
        return cached_path

    from ...agutil import agharness_binary_cache_dir

    cache_file = agharness_binary_cache_dir() / binary
    if not cache_file.exists():
        host_path = shutil.which(binary)
        if host_path is None:
            return None
        shutil.copy2(host_path, cache_file)
        cache_file.chmod(0o755)

    # The bind mount is a live view of the host directory, so the file
    # just written is already visible inside the container -- re-check
    # rather than assume, since the copy above could still race a
    # concurrent launch for a different agent seeding the same cache.
    out, rc = sandbox.exec(f"test -x {shlex.quote(cached_path)}", workdir="/")
    return cached_path if rc == 0 else None


# -- Native session continuity (see docs/Design_harness_history.md) ---------
#
# Claude Code's own conversation transcript, stored as
# `<config_home>/projects/<slug>/<session_id>.jsonl` where <slug> is the
# launch's cwd (== config_home, same value passed to px.launch(cwd=...))
# with every non-alphanumeric character replaced by '-'. Confirmed
# empirically against the real CLI (v2.1.220): copying that file into a
# fresh directory and resuming with --resume <session_id> from there
# recovers the exact original conversation, with real prompt-cache hits;
# without the file, --resume fails cleanly ("No conversation found").
#
# Deliberately NOT the source of truth for history -- ag.ctx stays that,
# engine-agnostic and sandbox-independent. This is a per-engine, opt-in
# optimization: extracted from and reinjected into whatever sandbox handles
# the next call, stored on the agent itself (ag._harness_sessions, and
# agent.save()/load()'s state.json), never on the sandbox's own filesystem.
_SESSION_SLUG_RE = re.compile(r"[^a-zA-Z0-9]")
_ENGINE_KEY = "claude_code"


def _session_slug(cwd: str) -> str:
    return _SESSION_SLUG_RE.sub("-", str(cwd))


def _session_path(config_home: str, session_id: str) -> str:
    return f"{config_home}/projects/{_session_slug(config_home)}/{session_id}.jsonl"


def _read_session_blob(sandbox, in_container: bool, path: str) -> "bytes | None":
    try:
        if in_container:
            return sandbox.read_file_bytes(path)
        return Path(path).read_bytes()
    except (FileNotFoundError, OSError):
        return None


def _write_session_blob(sandbox, in_container: bool, path: str, data: bytes) -> None:
    if in_container:
        sandbox.write_file_bytes(path, data)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


class _ClaudeCodeBackend(agharness_backend):
    _DEFAULT_BINARY = "claude"
    _DEFAULT_TIMEOUT_S = 600

    def execute(
        self,
        ag: "agent",
        prev_ctx: "agcontext",
        skill_input: agdata,
        max_steps: "int | None",
        *,
        skill: "agskill",
        extra_system: "str | None" = None,
    ) -> "tuple[agdata, agcontext, list[dict]]":

        from ... import agharness
        from ..agproxy_ptrace import agProxyPtrace, wire_to_sandbox

        sys_msg = {"role": "system", "content": skill._build_system_prompt(extra_system)}

        binary = self.binary_path or self._DEFAULT_BINARY
        # See docs/Design_harness_integration.md's Prerequisites: a
        # docker/podman-backed sandbox runs the harness INSIDE the
        # container (its own PID namespace, so its filesystem writes land
        # in the same workspace the rest of that agent's tools see), a
        # chroot-backed (or no) sandbox keeps the existing bare-host launch
        # -- the jail already IS a real host directory, nothing to bridge.
        in_container = agharness.is_container_backed(ag.sandbox)

        if in_container:
            resolved = _resolve_binary_in_container(ag.sandbox, binary)
        else:
            resolved = shutil.which(binary)
        if resolved is None:
            where = (
                "inside the sandbox container, in the harness binary cache "
                "(~/.cache/agency_harness_bin), or on this host's own PATH "
                "to seed that cache from"
                if in_container
                else "on the host PATH"
            )
            return agerror(f"claude binary {binary!r} not found {where}"), prev_ctx, [sys_msg]

        # A container-backed launch registers directly on the host-side
        # terminus and reaches its own LLM traffic through an in-container
        # agproxy_llm instance (Phase 2b-ii) -- no host-side agProxyLLM
        # gateway involved at all for this launch. A bare-host/chroot
        # launch keeps using the existing host-side shared gateway
        # unchanged, since there's no container boundary to relocate
        # anything across.
        if in_container:
            from ..agllm_terminus import get_shared_terminus
            from ..agproxy_llm_in_container import ensure_agproxy_llm_in_container

            gateway = None
            terminus = get_shared_terminus(ag.agconfig)
            token = uuid.uuid4().hex
            terminus.register(token, ag)
        else:
            from ..agllm_terminus import get_shared_terminus
            from ..agproxy_llm import get_shared_gateway

            gateway = get_shared_gateway(ag.agconfig)
            # Same shared singleton the in_container branch grabs directly
            # above -- agProxyLLM.register()/unregister() already delegate
            # to it internally (see agproxy_llm.py's own docstring: "the
            # registry lives entirely on the terminus now"), so this is
            # just naming a reference to state that already exists, not a
            # second registration -- needed here only to read back this
            # launch's recorded transcript after the run (Phase 5, below).
            terminus = get_shared_terminus(ag.agconfig)
            token = uuid.uuid4().hex
            gateway.register(token, ag)

        from ..agprof_ingest import get_shared_profiler_ingest

        profiler_ingest = get_shared_profiler_ingest()
        profiler_ingest.register(token, ag)

        # Shared MCP server (Phase 4): the same resource-control
        # (reserve_cpu/cpu_release/daemon_release) and output-submission
        # (submit_output) tools native's in-container loop uses, reached
        # here through Claude Code's own `--mcp-config` -- its one
        # sanctioned "additive tool" extensibility seam, per
        # docs/Design_harness_integration.md. Registered/bridged
        # unconditionally (not just for structured-output skills) since
        # resource tools are useful to every launch regardless of its
        # output shape.
        from ..agmcp_server import get_shared_mcp_server

        mcp_server = get_shared_mcp_server(ag.agconfig)
        mcp_server.register(token, ag, skill)
        mcp_relay_proc = None
        if in_container:
            from ..agproxy_ptrace_internal._in_container_launcher import start_tcp_relay

            # Same reasoning as agproxy_llm's pre-Phase-2b-ii relay: Claude
            # Code's `--mcp-config` only understands a plain `http://host:port`
            # URL, not a Unix domain socket path, so the bind-mounted UDS
            # bridge to agmcp_server needs a container-local TCP front end.
            mcp_relay_proc, mcp_relay_port = start_tcp_relay(
                ag.sandbox, mcp_server.ensure_uds_started()
            )
            mcp_base_url = f"http://127.0.0.1:{mcp_relay_port}"
        else:
            # A bare-host/chroot launch runs directly on this host, so it
            # can reach agmcp_server's own TCP listener with no bridging at
            # all -- same reasoning as the non-container LLM gateway path
            # above.
            mcp_base_url = mcp_server.start()

        if in_container:
            config_home = agharness.materialize_config_home_in_container(ag, ag.sandbox, token)
        else:
            config_home = agharness.materialize_config_home(ag, token, gateway.base_url)

        # Structured output requires collecting every required field via
        # submit_output before this call can succeed (see
        # build_mcp_output_format_instruction) -- bounded relaunch on
        # incomplete output (Phase 6's "Outer" layer) only applies here; a
        # raw-text/no-schema skill has nothing to retry on.
        _use_structured_output = (
            skill.output_schema is not None and skill.output_schema.raw_key() is None
        )
        output_schema_retries_left = skill.max_output_schema_retries

        first_prompt = agharness.build_user_turn_prompt(skill, skill_input)
        if not isinstance(first_prompt, str):
            first_prompt = json.dumps(first_prompt)
        extra = agharness.build_mcp_output_format_instruction(skill)
        if extra:
            first_prompt = first_prompt + extra

        total_input_tokens = 0
        total_output_tokens = 0

        try:
            # Resume the prior native session for this agent, if any --
            # written into THIS launch's config_home before the CLI starts,
            # never persisted on the sandbox itself (see module docstring
            # above / docs/Design_harness_history.md). Also doubles as the
            # resume target for a same-call output-schema retry below: once
            # set (here, or by a completed attempt further down), every
            # subsequent attempt in this same execute() call passes
            # `--resume` too, since the physical session file lives in this
            # same `config_home` the whole time -- no cross-directory blob
            # copy needed for a same-call retry, only for the NEXT
            # execute() call (a possibly different config_home), which is
            # what the blob capture after each attempt is for.
            resume_session_id = None
            prior = ag._harness_sessions.get(_ENGINE_KEY)
            if prior and prior.get("session_id"):
                resume_session_id = prior["session_id"]
                try:
                    blob = base64.b64decode(prior["blob_b64"])
                    _write_session_blob(
                        ag.sandbox,
                        in_container,
                        _session_path(str(config_home), resume_session_id),
                        blob,
                    )
                except Exception:
                    # Best-effort: a failure to restore the prior session
                    # blob must never block the run -- --resume will just
                    # fail its own lookup and this falls back to a fresh
                    # session (still correct, just without native memory).
                    resume_session_id = None

            # Bridge Claude Code's own PreToolUse permission check to
            # agpolicy (docs/Design_harness_integration.md): write the
            # self-contained hook script into this launch's own
            # config_home (visible to `claude` in both the host and
            # in-container case, unlike a path in this package's own
            # install location, which the container can't see), then
            # register it via `--settings`' `hooks` block -- confirmed
            # directly against the real CLI that this composes fine with
            # `--setting-sources ""` below, and that omitting `matcher`
            # hooks every tool call, not just one.
            hook_src = (Path(__file__).parent.parent / "_harness_permission_hook.py").read_bytes()
            hook_path = f"{config_home}/agpolicy_hook.py"
            if in_container:
                ag.sandbox.write_file_bytes(hook_path, hook_src)
            else:
                Path(hook_path).write_bytes(hook_src)
            hooks_settings = json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"hooks": [{"type": "command", "command": f"python3 {hook_path}"}]}
                        ]
                    }
                }
            )

            # --mcp-config -- point Claude Code's own native MCP client at
            # the shared agmcp_server bridge set up above (resource-control
            # + submit_output tools). --strict-mcp-config restricts this
            # launch to ONLY that server, ignoring any other MCP config
            # source -- redundant with the isolated config_home/cwd (no
            # `.mcp.json` lives there) but cheap, explicit insurance against
            # ever silently inheriting some other server.
            mcp_config = json.dumps(
                {
                    "mcpServers": {
                        "agency": {
                            "type": "http",
                            "url": f"{mcp_base_url}/mcp",
                            "headers": {"Authorization": f"Bearer {token}"},
                        }
                    }
                }
            )

            if in_container:
                # A harness running inside the container can't reach a
                # host-bound TCP listener the way a host-level launch does
                # (this host's rootless Docker networking didn't make that
                # reachable via either the bridge gateway IP or
                # host.docker.internal, confirmed empirically). Rather than
                # relaying a container-local port to a host-side gateway
                # (the previous design), agproxy_llm itself now runs AS an
                # in-container process (Phase 2b-ii) -- ANTHROPIC_BASE_URL
                # points directly at its own local port, no relay in
                # between. It reaches the real, credential-holding
                # agllm_terminus (registered above) over the same
                # bind-mounted Unix domain socket
                # (agutil.agharness_llm_gateway_dir) the old relay used,
                # just terminating in a different process now.
                base_url = ensure_agproxy_llm_in_container(ag.sandbox, ag.agconfig)
            else:
                base_url = gateway.base_url

            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                # Point Claude Code's own LLM traffic at agproxy_llm's
                # translated Anthropic Messages route instead of any real
                # Anthropic endpoint -- ANTHROPIC_AUTH_TOKEN sends this
                # token as a bearer `Authorization` header, which
                # `_extract_bearer_token` (agproxy_llm.py) reads to look up
                # this launch's agent. Real host credentials (API key,
                # OAuth login, Bedrock env) are deliberately NOT forwarded:
                # every claude-driven agent's LLM calls must go through
                # this agent's own configured agConfig backend, not
                # whatever this host happens to have lying around.
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_AUTH_TOKEN": token,
                # Lets agpolicy_hook.py (registered above as the
                # PreToolUse hook) reach agproxy_llm's /agpolicy/check_tool
                # route -- same base_url/token as the LLM traffic above,
                # since it's the same gateway app and the same per-run
                # bearer token identifies the same agent.
                "AGPOLICY_BASE_URL": base_url,
                "AGPOLICY_TOKEN": token,
                # Also explicitly unset so the CLI can't fall back to a
                # locally-configured Bedrock/API-key credential path.
                "CLAUDE_CODE_USE_BEDROCK": "0",
                # Relocates Claude Code's ENTIRE storage root (settings AND
                # session transcripts) into this launch's own config_home
                # instead of the real $HOME/.claude -- confirmed via strings
                # in the installed binary ("CLAUDE_CONFIG_DIR=/tmp for
                # ephemeral local writes"). This is what makes the native
                # session continuity above (_session_path/_read_session_blob/
                # _write_session_blob) work at all: without it, the session
                # file lands under whatever HOME resolves to, not
                # config_home, so it's invisible to the restore/capture
                # logic and never cleaned up by cleanup_config_home either.
                # See docs/Design_harness_history.md.
                "CLAUDE_CONFIG_DIR": str(config_home),
            }
            if in_container:
                # Deliberately does NOT forward the host's HOME: it points
                # to a path that's meaningless (or, worse, coincidentally
                # exists and means something else entirely) inside the
                # container's own filesystem. No OAuth-credential concern
                # to preserve here either, unlike the host-level case below
                # -- the container has no real Anthropic login to begin
                # with, and ANTHROPIC_AUTH_TOKEN always takes precedence
                # regardless. Left unset, so the container image's own
                # default HOME applies.
                pass
            else:
                # Deliberately does NOT override HOME: Claude Code's OAuth
                # credentials live under the real $HOME (~/.claude/
                # .credentials.json), and --setting-sources "" above is
                # already what provides the "don't inherit CLAUDE.md/settings"
                # isolation this backend needs -- overriding HOME too would
                # additionally (and unintentionally) cut off the real login,
                # forcing "Not logged in" for every run (hit and fixed during
                # development against the real CLI). This doesn't matter for
                # authentication anymore since ANTHROPIC_AUTH_TOKEN above
                # always takes precedence over OAuth login, but HOME is still
                # left alone since other CLI state may expect it.
                if "HOME" in os.environ:
                    envp["HOME"] = os.environ["HOME"]

            px = agProxyPtrace(ag.agconfig)
            policy = agharness.default_policy(ag)

            # Bounded relaunch on incomplete structured output (Phase 6's
            # "Outer" layer, docs/Design_harness_integration.md): each
            # attempt is a fresh `claude -p` process (one-shot by design),
            # resumed via `--resume` from the 2nd attempt on so the model
            # sees its own prior turn and the reprompt as one continuous
            # conversation -- the same native session-continuity mechanism
            # this backend already uses across SEPARATE execute() calls,
            # just invoked within one call here. Only structured-output
            # skills loop at all; a raw-text/no-schema skill runs once.
            prompt = first_prompt
            while True:
                argv = [
                    resolved,
                    "-p",
                    "--output-format",
                    "json",
                    "--setting-sources",
                    "",
                    "--settings",
                    hooks_settings,
                    "--mcp-config",
                    mcp_config,
                    "--strict-mcp-config",
                ]
                if resume_session_id:
                    argv += ["--resume", resume_session_id]
                argv.append(prompt)

                handle = px.launch(
                    argv,
                    envp,
                    cwd=str(config_home),
                    policy=policy,
                    ag=ag,
                    sandbox=ag.sandbox if in_container else None,
                )
                if ag.sandbox is not None:
                    wire_to_sandbox(handle, ag.sandbox)

                stdout, stderr, rc = handle.wait(timeout=self._DEFAULT_TIMEOUT_S)
                _dbg = os.environ.get("AGENCY_DEBUG_RAW_STDOUT_DUMP")
                if _dbg:
                    with open(_dbg, "a") as _f:
                        _f.write(f"rc={rc!r}\nstdout={stdout!r}\nstderr={stderr!r}\n---\n")

                if rc != 0:
                    break  # handled after the loop -- no retry on a hard launch failure

                final_text, usage, session_id = self._parse_result_json(stdout)
                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)

                # Capture the (possibly new/updated) session blob for the
                # NEXT execute() call -- MUST happen before config_home is
                # torn down in `finally` below, since that's where this
                # file physically lives for an in-container launch.
                # Best-effort: a failure here means the next call starts a
                # fresh session instead of resuming, not that this call's
                # own result is lost.
                if session_id:
                    resume_session_id = session_id
                    try:
                        blob = _read_session_blob(
                            ag.sandbox, in_container, _session_path(str(config_home), session_id)
                        )
                        if blob is not None:
                            ag._harness_sessions[_ENGINE_KEY] = {
                                "session_id": session_id,
                                "blob_b64": base64.b64encode(blob).decode(),
                            }
                    except Exception:  # noqa: S110 - session persistence is best-effort
                        pass

                # Snapshot whatever submit_output calls landed so far --
                # read fresh every attempt (agmcp_server accumulates across
                # calls for the same token, so a field submitted on attempt
                # 1 is still there after a reprompt on attempt 2).
                collected_output = mcp_server.collected_output(token)
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

            # Phase 5 (history unification): every wire-format-translating
            # engine resends its full conversation on each LLM call, so the
            # LAST dispatch agllm_terminus recorded for this token -- read
            # ONCE here, after the whole retry loop (not per-attempt) --
            # already covers every turn across every attempt, including a
            # --resume'd reprompt. MUST happen before `unregister()` below
            # discards it. None if this token's process never actually
            # reached agllm_terminus at all (e.g. a mocked test, or a launch
            # that failed before making any LLM call) -- callers fall back
            # to the coarser 2-message shape in that case.
            transcript = terminus.transcript_for_token(token)
        finally:
            if in_container:
                terminus.unregister(token)
            else:
                gateway.unregister(token)
            profiler_ingest.unregister(token)
            mcp_server.unregister(token)
            if mcp_relay_proc is not None:
                # Per-launch relay subprocess -- torn down every time. Never
                # `mcp_server.stop()` here: it's the same shared, lazily-
                # started singleton `get_shared_gateway`'s own TCP listener
                # is (never stopped per-launch either), reused by every
                # subsequent launch in this process.
                from ..agproxy_ptrace_internal._in_container_launcher import stop_tcp_relay

                stop_tcp_relay(mcp_relay_proc)
            if in_container:
                agharness.cleanup_config_home_in_container(ag.sandbox, config_home)
            else:
                agharness.cleanup_config_home(config_home)

        if rc != 0:
            return (
                agerror(f"claude exited with code {rc}: {stderr or stdout}"),
                prev_ctx,
                [sys_msg],
            )

        if _use_structured_output:
            required = set(skill.output_schema._data.keys())
            missing = sorted(required - set(collected_output.keys()))
            if missing:
                result = agerror(
                    "structured output incomplete after "
                    f"{skill.max_output_schema_retries - output_schema_retries_left} retry/retries "
                    "-- submit_output was never called for: " + ", ".join(missing)
                )
            else:
                result = agdata(**collected_output)
        else:
            out_key = skill.output_schema.raw_key() if skill.output_schema is not None else "result"
            result = agdata(**{out_key: final_text})

        prev_ctx.total_input_tokens += total_input_tokens
        prev_ctx.total_output_tokens += total_output_tokens

        if transcript:
            # Drop the leading system message the same way agskill.py's own
            # native loop drops messages[0] when building prev_ctx.messages
            # (agskill.py's `messages[1:]`) -- callers get `sys_msg` above
            # for logging/the returned delta, never inside `ctx.messages`
            # itself. This is the real turn-by-turn conversation (tool
            # calls/results included), not the old flattened 2-message
            # collapse -- retiring that collapse is the whole point of
            # Phase 5 (docs/Design_harness_integration.md).
            history = transcript[1:] if transcript[0].get("role") == "system" else transcript
            prev_ctx.messages = history
            delta = [sys_msg] + history
        else:
            # This token's launch never reached agllm_terminus at all (a
            # mocked test double, or a real launch that failed before any
            # LLM call happened) -- fall back to the coarse shape rather
            # than silently returning an empty history.
            user_msg = {"role": "user", "content": first_prompt}
            assistant_msg = {"role": "assistant", "content": final_text}
            prev_ctx.messages = [user_msg, assistant_msg]
            delta = [sys_msg, user_msg, assistant_msg]
        return result, prev_ctx, delta

    @staticmethod
    def _parse_result_json(stdout: str) -> "tuple[str, dict, str | None]":
        """Parse `claude -p --output-format json`'s single JSON result
        object -- `{"result": "...", "usage": {...}, "session_id": "...", ...}`
        (verified directly against the real CLI, v2.1.212/2.1.220, during
        development)."""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout.strip(), {}, None
        if not isinstance(payload, dict):
            return stdout.strip(), {}, None
        text = payload.get("result", "")
        usage = payload.get("usage", {}) or {}
        session_id = payload.get("session_id")
        return text, usage, session_id


__all__ = ["_ClaudeCodeBackend", "claude_code_available"]
