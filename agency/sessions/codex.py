"""`agency codex`: an interactive Codex CLI session (see session.py).

Codex specifics, each checked against Codex CLI 0.162.1:

* Hooks are command hooks only, added as `-c hooks.<Event>=...` overrides,
  which Codex merges with the user's own hooks. Codex runs a non-managed hook
  only once the user trusts it, by the hash of its definition; on first use
  Codex itself asks ("Hooks need review"). The hook command is therefore the
  same in every session -- per-session endpoint and token come from the
  environment -- so one approval lasts.
* Without capture, model requests come from OpenTelemetry log events: `codex.api_request` ends
  when response headers arrive (its duration_ms covers request start to
  headers), and the following `codex.sse_event` of kind `response.completed`
  ends the response and carries its tokens. Codex records times only in the
  `event.timestamp` attribute.
* The exporter's endpoint goes in a `-c otel.exporter` override; its auth
  header comes from OTEL_EXPORTER_OTLP_LOGS_HEADERS in the environment, which
  Codex's exporter merges in. Codex does not expand `${VAR}` in configured
  OTel headers, and `-c` values are visible to every user in /proc.
* Hook and telemetry tool ids agree: PreToolUse's `tool_use_id` is the same
  `call_id` that `codex.tool_decision` and `codex.tool_result` report.
"""

from __future__ import annotations

import json
import os
import threading
import tomllib
from dataclasses import replace
from pathlib import Path

from .session import (
    TOKEN_VARIABLE,
    CliSession,
    hook_command,
    main as run_session,
    number,
    otlp_log_records,
)


HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "SubagentStart",
    "SubagentStop",
    "Interrupt",
    "Stop",
)
# Codex caps the Interrupt hook at 3 s (and warns that it clamps a longer
# one). One value for every event keeps the definitions simple; the relay
# itself gives up well within it.
_HOOK_TIMEOUT_S = 3
# Codex reports an approval prompt answered "no" (or cancelled) like this.
_REJECTED_DECISIONS = {"denied", "abort"}


class CodexSession(CliSession):
    agent_name = "codex"
    harness = "codex"
    label = "Codex"
    engine_coverage = {
        "llm": "codex_reported",
        "tools": "hook_boundaries",
        "turns": "hook_boundaries",
        "reason": (
            "Interactive Codex session: tool calls from Codex hooks once the user trusts "
            "them; model requests from its OpenTelemetry events, without prompt or "
            "response content."
        ),
    }

    def __init__(self, **kwargs) -> None:
        capture = kwargs.get("capture")
        if capture is not None and capture.native:
            kwargs["capture"] = _native_capture_config(capture, kwargs.get("cli_args", []))
        super().__init__(**kwargs)
        # conversation id -> start of its in-flight model request. Codex runs
        # one request at a time per conversation; subagents are their own.
        self._request_starts: "dict[str, float]" = {}
        self._requests_lock = threading.Lock()

    def _cli_argv(self) -> "list[str]":
        overrides = []
        handler = json.dumps(hook_command())  # a TOML basic string
        for event in HOOK_EVENTS:
            overrides += [
                "-c",
                f'hooks.{event}=[{{hooks=[{{type="command",command={handler},'
                f"timeout={_HOOK_TIMEOUT_S}}}]}}]",
            ]
        endpoint = f"http://127.0.0.1:{self.port}/v1/logs"
        overrides += [
            "-c",
            f'otel.exporter={{"otlp-http"={{endpoint="{endpoint}",protocol="json"}}}}',
        ]
        capture = []
        if self.capture is not None and self.capture.native:
            provider = {
                "name": "OpenAI (Agency capture)",
                "base_url": f"http://127.0.0.1:{self.port}/capture/v1",
                "wire_api": "responses",
                "requires_openai_auth": True,
                "supports_websockets": True,
                "env_http_headers": {"x-agency-token": TOKEN_VARIABLE},
            }
            capture = ["-c", 'model_provider="agency_native_capture"']
            for key, value in provider.items():
                # JSON strings/bools are TOML scalars; header maps need TOML '='.
                literal = (
                    "{"
                    + ",".join(f"{json.dumps(k)}={json.dumps(v)}" for k, v in value.items())
                    + "}"
                    if isinstance(value, dict)
                    else json.dumps(value)
                )
                capture += ["-c", f"model_providers.agency_native_capture.{key}={literal}"]
        elif self.capture is not None:
            capture = [
                "-c",
                'model_provider="agency_capture"',
                "-c",
                f"model={json.dumps(self.capture.model)}",
                "-c",
                'model_providers.agency_capture.name="Agency capture"',
                "-c",
                f'model_providers.agency_capture.base_url="http://127.0.0.1:{self.port}/capture/v1"',
                "-c",
                f'model_providers.agency_capture.env_key="{TOKEN_VARIABLE}"',
                "-c",
                'model_providers.agency_capture.wire_api="responses"',
                "-c",
                "model_providers.agency_capture.requires_openai_auth=false",
                "-c",
                "model_providers.agency_capture.supports_websockets=false",
            ]
        # A shared daemon would run the actual tools outside this session's
        # tracing and workload scope. Keep the native TUI and local server
        # together; this does not change approval or sandbox settings.
        return [self.binary, "--no-daemon", *overrides, *self.cli_args, *capture]

    def _cli_environment(self, env: dict) -> None:
        exporter = _configured_otel_exporter(env)
        if exporter not in (None, "none"):
            print(
                "[agency] WARNING: your Codex config exports telemetry; this session's "
                "Codex log events are sent to Agency instead.",
                flush=True,
            )
        env.update(
            OTEL_EXPORTER_OTLP_LOGS_HEADERS=f"x-agency-token={self.host_token}",
        )

    def handle_log_records(self, body: dict) -> None:
        for record in otlp_log_records(body, "codex."):
            name = record["name"]
            attributes = record["attributes"]
            at = record["time"]
            if name == "conversation_starts" and attributes.get("model"):
                self._register(self._main, model=attributes["model"])
            elif name == "api_request":
                self._api_request(attributes, at)
            elif name == "sse_event":
                self._stream_event(attributes, at)
            elif name == "tool_decision":
                if str(attributes.get("decision", "")).lower() in _REJECTED_DECISIONS:
                    source = attributes.get("source") or "user"
                    self._tool_finished(attributes.get("call_id"), error=f"Rejected ({source})")
            elif name == "tool_result" and str(attributes.get("success")).lower() == "false":
                # A failing command still reaches PostToolUse; this closes
                # only a call that no hook completed.
                self._tool_finished(
                    attributes.get("call_id"),
                    error=str(attributes.get("output") or "Tool failed"),
                    duration_ms=number(attributes.get("duration_ms")),
                )

    def _api_request(self, attributes: dict, headers_at: float) -> None:
        started_at = headers_at - (number(attributes.get("duration_ms")) or 0) / 1000
        status = number(attributes.get("http.response.status_code"))
        error = attributes.get("error.message") or attributes.get("error")
        if error or (status is not None and status >= 400):
            self._record_model_request(
                started_at,
                headers_at,
                model=attributes.get("model"),
                actor=self._known_subagent(attributes.get("conversation.id")),
                error=str(error or f"HTTP {int(status)}"),
                details={"attempt": number(attributes.get("attempt"))},
            )
            return
        with self._requests_lock:
            self._request_starts[attributes.get("conversation.id")] = started_at

    def _stream_event(self, attributes: dict, at: float) -> None:
        kind = attributes.get("event.kind")
        if kind not in {"response.completed", "response.failed", "response.incomplete"}:
            return
        with self._requests_lock:
            started_at = self._request_starts.pop(attributes.get("conversation.id"), None)
        failed = kind != "response.completed"
        self._record_model_request(
            at if started_at is None else started_at,
            at,
            model=attributes.get("model"),
            # A Codex subagent's conversation id is its agent_id.
            actor=self._known_subagent(attributes.get("conversation.id")),
            input_tokens=attributes.get("input_token_count"),
            output_tokens=attributes.get("output_token_count"),
            error=f"Codex reported {kind}" if failed else None,
            details={
                "ttft_ms": number(attributes.get("ttft_ms")),
                "cached_tokens": number(attributes.get("cached_token_count")),
                "reasoning_tokens": number(attributes.get("reasoning_token_count")),
                "start": "request_sent" if started_at is not None else "unavailable",
            },
        )


def _configured_otel_exporter(env: dict) -> "str | None":
    """The exporter the user's own Codex config selects, if any."""
    home = Path(env.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        config = tomllib.loads((home / "config.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    exporter = (config.get("otel") or {}).get("exporter")
    if isinstance(exporter, dict):
        return next(iter(exporter), None)
    return exporter


def _native_capture_config(capture, cli_args):
    """Resolve only routing inputs; never open Codex's credential store.

    Codex still resolves its own models, permissions and other configuration.
    Refuse other providers instead of silently changing their authentication.
    """
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")

    def read(path):
        try:
            return tomllib.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"cannot resolve Codex capture routing from {path}") from exc

    config = read(home / "config.toml")
    overrides, profile = {}, config.get("profile")

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    args = iter(cli_args)
    for arg in args:
        if arg == "--":
            break
        if arg == "--oss" or arg.startswith("--local-provider"):
            raise ValueError(
                "--capture-native requires the OpenAI provider; omit --oss/--local-provider"
            )
        if arg in {"-c", "--config"}:
            value = next(args, "")
        elif arg.startswith("--config="):
            value = arg.split("=", 1)[1]
        elif arg.startswith("-c"):
            value = arg[2:]
        elif arg in {"-p", "--profile"}:
            profile = next(args, "")
            continue
        elif arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]
            continue
        elif arg.startswith("-p"):
            profile = arg[2:]
            continue
        else:
            continue
        try:
            try:
                parsed = tomllib.loads(value)
            except tomllib.TOMLDecodeError:
                key, literal = value.split("=", 1)
                parsed = tomllib.loads(key + "=" + json.dumps(literal))
            merge(overrides, parsed)
        except (tomllib.TOMLDecodeError, ValueError) as exc:
            raise ValueError("--capture-native requires valid TOML -c/--config overrides") from exc
    profile = overrides.get("profile", profile)
    merge(config, overrides)
    if profile:
        if not isinstance(profile, str) or Path(profile).name != profile:
            raise ValueError("invalid Codex capture profile")
        merge(config, config.get("profiles", {}).get(profile, {}))
        merge(config, read(home / f"{profile}.config.toml"))
    merge(config, overrides)
    if config.get("model_provider", "openai") != "openai":
        raise ValueError("--capture-native currently requires model_provider='openai'")
    if "agency_native_capture" in config.get("model_providers", {}):
        raise ValueError("agency_native_capture is reserved for this capture session")
    if (
        config.get("chatgpt_base_url", "https://chatgpt.com/backend-api/").rstrip("/")
        != "https://chatgpt.com/backend-api"
    ):
        raise ValueError("--capture-native does not yet support custom ChatGPT service routing")
    base = config.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
    # An explicit built-in provider base overrides the native default for either
    # auth type. Preserve it; never route its credential to a guessed destination.
    return replace(capture, api_base_url=base, chatgpt_base_url=base) if base else capture


def main(argv=None) -> int:
    return run_session(CodexSession, "codex", argv)
