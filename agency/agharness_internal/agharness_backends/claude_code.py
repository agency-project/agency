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
    ) -> "tuple[agdata, agcontext, list[dict]]":
        import shlex

        from ... import agharness
        from ..agproxy_llm import get_shared_gateway
        from ..agproxy_ptrace import agProxyPtrace, wire_to_sandbox
        from ..agproxy_ptrace_internal._in_container_launcher import start_llm_relay, stop_llm_relay

        sys_msg = {"role": "system", "content": skill._build_system_prompt()}

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
                f"(~/.cache/agency_harness_bin), or on this host's own PATH "
                f"to seed that cache from"
                if in_container else "on the host PATH"
            )
            return agerror(f"claude binary {binary!r} not found {where}"), prev_ctx, [sys_msg]

        gateway = get_shared_gateway(ag.agconfig)
        token = uuid.uuid4().hex
        gateway.register(token, ag)

        if in_container:
            config_home = agharness.materialize_config_home_in_container(ag, ag.sandbox, token)
        else:
            config_home = agharness.materialize_config_home(ag, token, gateway.base_url)

        relay_proc = None
        try:
            prompt = agharness.build_user_turn_prompt(skill, skill_input)
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt)
            extra = agharness.build_output_format_instruction(skill)
            if extra:
                prompt = prompt + extra

            # Resume the prior native session for this agent, if any --
            # written into THIS launch's config_home before the CLI starts,
            # never persisted on the sandbox itself (see module docstring
            # above / docs/Design_harness_history.md).
            resume_session_id = None
            prior = ag._harness_sessions.get(_ENGINE_KEY)
            if prior and prior.get("session_id"):
                resume_session_id = prior["session_id"]
                try:
                    blob = base64.b64decode(prior["blob_b64"])
                    _write_session_blob(
                        ag.sandbox, in_container,
                        _session_path(str(config_home), resume_session_id), blob,
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

            # --setting-sources "" -- load none of the user/project/local
            # settings that would normally apply, so this run doesn't
            # inherit the caller's own Claude Code configuration (matching
            # the same isolated-config-home intent as opencode's own
            # OPENCODE_CONFIG, just via a flag here instead of a file,
            # since --settings/--setting-sources are what Claude Code
            # itself provides for this). --settings is a separate flag
            # that loads its JSON regardless of --setting-sources, which
            # is what makes the agpolicy hook above reach `claude` at all.
            argv = [
                resolved,
                "-p",
                "--output-format",
                "json",
                "--setting-sources",
                "",
                "--settings",
                hooks_settings,
            ]
            if resume_session_id:
                argv += ["--resume", resume_session_id]
            argv.append(prompt)

            if in_container:
                # A harness running inside the container can't reach the
                # gateway's TCP listener the way a host-level launch does
                # (the gateway is bound on the HOST; this host's rootless
                # Docker networking didn't make that reachable via either
                # the bridge gateway IP or host.docker.internal, confirmed
                # empirically). Bridge via a Unix domain socket instead --
                # it crosses the container boundary as the bind-mounted
                # filesystem object agsandbox already attaches to every
                # container-backed sandbox (agutil.agharness_llm_gateway_dir),
                # not a network hop, so it's unaffected by the runtime's
                # networking mode. start_llm_relay() runs a small in-
                # container process forwarding a container-local TCP port
                # to that socket; ANTHROPIC_BASE_URL points at THAT port
                # (the container's own loopback), not gateway.base_url.
                uds_path = gateway.ensure_uds_started()
                relay_proc, relay_port = start_llm_relay(ag.sandbox, uds_path)
                base_url = f"http://127.0.0.1:{relay_port}"
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
            handle = px.launch(
                argv, envp, cwd=str(config_home), policy=policy, ag=ag,
                sandbox=ag.sandbox if in_container else None,
            )
            if ag.sandbox is not None:
                wire_to_sandbox(handle, ag.sandbox)

            stdout, stderr, rc = handle.wait(timeout=self._DEFAULT_TIMEOUT_S)
            _dbg = os.environ.get("AGENCY_DEBUG_RAW_STDOUT_DUMP")
            if _dbg:
                with open(_dbg, "a") as _f:
                    _f.write(f"rc={rc!r}\nstdout={stdout!r}\nstderr={stderr!r}\n---\n")

            # Capture the (possibly new/updated) session blob for next time
            # -- MUST happen before config_home is torn down in `finally`
            # below, since that's where this file physically lives for an
            # in-container launch. Best-effort: a failure here means the
            # NEXT call starts a fresh session instead of resuming, not
            # that this call's own result is lost.
            if rc == 0:
                _, _, session_id = self._parse_result_json(stdout)
                if session_id:
                    try:
                        blob = _read_session_blob(
                            ag.sandbox, in_container, _session_path(str(config_home), session_id)
                        )
                        if blob is not None:
                            ag._harness_sessions[_ENGINE_KEY] = {
                                "session_id": session_id,
                                "blob_b64": base64.b64encode(blob).decode(),
                            }
                    except Exception:
                        pass
        finally:
            gateway.unregister(token)
            stop_llm_relay(relay_proc)
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

        final_text, usage, _session_id = self._parse_result_json(stdout)
        user_msg = {"role": "user", "content": prompt}
        assistant_msg = {"role": "assistant", "content": final_text}

        if skill.output_schema is not None and skill.output_schema.raw_key() is None:
            result, _paths = skill.output_schema.validate_and_recover(final_text, ag.sandbox)
        else:
            out_key = skill.output_schema.raw_key() if skill.output_schema is not None else "result"
            result = agdata(**{out_key: final_text})

        if usage:
            prev_ctx.total_input_tokens += usage.get("input_tokens", 0)
            prev_ctx.total_output_tokens += usage.get("output_tokens", 0)
        prev_ctx.messages = [user_msg, assistant_msg]
        return result, prev_ctx, [sys_msg, user_msg, assistant_msg]

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
