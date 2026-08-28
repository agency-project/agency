"""Claude Code harness adapter.

Builds the Claude CLI invocation, routes it through the sandbox daemon's
policy-aware runtime, and normalizes its JSON result.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from .base import AdapterRuntime, AttemptResult, agharness_backend


def claude_code_available() -> bool:
    return shutil.which("claude") is not None


_BIN_CACHE_MOUNT = "/opt/agency_harness_bin"

_DEFAULT_TIMEOUT_S = 600


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
# Deliberately NOT the source of truth for history -- ag.ctx.recent_transcript
# stays that. This is a per-engine, opt-in optimization: extracted from and
# reinjected into whatever sandbox handles the next call, stored on
# ag.ctx.harness_sessions (and agent.save()/load()'s state.json), never on
# the sandbox's own filesystem.
_SESSION_SLUG_RE = re.compile(r"[^a-zA-Z0-9]")


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

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        from .. import agharness
        from ...profiler import agprof
        from ..ptrace.supervisor import agProxyPtrace

        binary = self.binary_path or self._DEFAULT_BINARY
        # See docs/Design_harness_integration.md's Prerequisites: a
        # docker/podman-backed sandbox runs the harness INSIDE the
        # container (its own PID namespace, so its filesystem writes land
        # in the same workspace the rest of that agent's tools see), a
        # chroot-backed (or no) sandbox keeps the existing bare-host launch
        # -- the jail already IS a real host directory, nothing to bridge.
        in_container = agharness.is_container_backed(runtime.sandbox)

        if in_container:
            resolved = _resolve_binary_in_container(runtime.sandbox, binary)
        else:
            resolved = shutil.which(binary)
        if resolved is None:
            where = (
                "inside the sandbox container, in the harness binary cache "
                "(~/.cache/agency_harness_bin), or on this host's own PATH "
                "to seed that cache from"
                if in_container
                else "on PATH (the host PATH)"
            )
            return AttemptResult(
                ok=False, error_message=f"claude binary {binary!r} not found {where}"
            )

        if in_container:
            config_home = agharness.materialize_config_home_in_container(
                runtime.engine_name, runtime.sandbox, runtime.token
            )
        else:
            config_home = agharness.materialize_config_home(
                runtime.engine_name, runtime.token, runtime.harness_base_url
            )

        try:
            if resume_session_id and prior_session_blob is not None:
                _write_session_blob(
                    runtime.sandbox,
                    in_container,
                    _session_path(str(config_home), resume_session_id),
                    prior_session_blob,
                )

            # Bridge Claude Code's PreToolUse permission check to agpolicy
            # and its PreToolUse/PostToolUse boundaries to agprof: write the
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
                runtime.sandbox.write_file_bytes(hook_path, hook_src)
            else:
                Path(hook_path).write_bytes(hook_src)
            hook_command = {"hooks": [{"type": "command", "command": f"python3 {hook_path}"}]}
            hooks = {"PreToolUse": [hook_command]}
            profile_hook_events = agprof.enabled()
            if profile_hook_events:
                # Post hooks exist solely for exact profiling. Do not make
                # an unprofiled run spawn an extra Python process after
                # every tool call just to discover AGPROF_* is unset.
                hooks["PostToolUse"] = [hook_command]
                hooks["PostToolUseFailure"] = [hook_command]
            hooks_settings = json.dumps({"hooks": hooks})

            # --mcp-config -- point Claude Code's own native MCP client at
            # agmanager_harness's own "/mcp" reverse proxy (resource-control
            # + submit_output tools, same surface every engine gets).
            # --strict-mcp-config restricts this launch to ONLY that
            # server, ignoring any other MCP config source -- redundant
            # with the isolated config_home/cwd (no `.mcp.json` lives
            # there) but cheap, explicit insurance against ever silently
            # inheriting some other server.
            mcp_config = json.dumps(
                agharness.mcp_config_for(runtime.harness_base_url, runtime.token)
            )

            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                # Point Claude Code's own LLM traffic at agmanager_harness's
                # translated Anthropic Messages route instead of any real
                # Anthropic endpoint -- ANTHROPIC_AUTH_TOKEN sends this
                # token as a bearer `Authorization` header, which
                # agmanager_harness reads to authenticate against
                # agmanager_host. Real host credentials (API key, OAuth
                # login, Bedrock env) are deliberately NOT forwarded: every
                # claude-driven agent's LLM calls must go through this
                # agent's own configured agConfig backend, not whatever
                # this host happens to have lying around.
                "ANTHROPIC_BASE_URL": runtime.harness_base_url,
                "ANTHROPIC_AUTH_TOKEN": runtime.token,
                # Lets agpolicy_hook.py (registered above as the
                # PreToolUse hook) reach agmanager_harness's own
                # /agpolicy/check_tool route -- same base_url/token as the
                # LLM traffic above, since it's the same process and the
                # same per-run bearer token identifies the same agent.
                "AGPOLICY_BASE_URL": runtime.harness_base_url,
                "AGPOLICY_TOKEN": runtime.token,
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
            if profile_hook_events:
                # Off-path stays free when profiling is disabled: without
                # these variables the shared hook skips its profiler POST.
                envp["AGPROF_BASE_URL"] = runtime.harness_base_url
                envp["AGPROF_TOKEN"] = runtime.token
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

            px = agProxyPtrace(runtime.agconfig)

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
                policy=runtime.syscall_policy,
                ag=None,
            )

            stdout, stderr, rc = handle.wait(timeout=_DEFAULT_TIMEOUT_S)
            _dbg = os.environ.get("AGENCY_DEBUG_RAW_STDOUT_DUMP")
            if _dbg:
                with open(_dbg, "a") as _f:
                    _f.write(f"rc={rc!r}\nstdout={stdout!r}\nstderr={stderr!r}\n---\n")

            if rc != 0:
                return AttemptResult(
                    ok=False, error_message=f"claude exited with code {rc}: {stderr or stdout}"
                )

            final_text, usage, session_id = self._parse_result_json(stdout)
            session_blob = None
            if session_id:
                try:
                    session_blob = _read_session_blob(
                        runtime.sandbox,
                        in_container,
                        _session_path(str(config_home), session_id),
                    )
                except Exception:  # noqa: S110 - session persistence is best-effort
                    session_blob = None

            return AttemptResult(
                ok=True,
                final_text=final_text,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                session_id=session_id,
                session_blob=session_blob,
            )
        finally:
            if in_container:
                agharness.cleanup_config_home_in_container(runtime.sandbox, config_home)
            else:
                agharness.cleanup_config_home(config_home)

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
