"""`agency claude`: an interactive Claude Code session (see session.py).

Claude Code specifics:

* Hooks are HTTP hooks in a `--settings` file, which Claude Code merges with
  the user's own settings. The token reaches the header by environment
  interpolation, so it never appears in a command line.
* Without capture, model requests come from the OpenTelemetry `api_request`/`api_error` log
  events, which carry the request's measured duration, tokens and cost.
* A rejected or cancelled tool fires no completion hook; its `tool_decision`
  or `tool_result` log event closes it instead.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .session import (
    HOOK_TIMEOUT_S,
    TOKEN_VARIABLE,
    CliSession,
    hook_command,
    main as run_session,
    number,
)
from .session import otlp_log_records

SETTINGS_FILE = "claude_settings.json"

# Every hook Agency observes. The full set is subscribed without matchers.
HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PostToolUseFailure",
    "Notification",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "StopFailure",
    "SessionEnd",
)
_USER_OTEL_LOG_VARIABLES = (
    "OTEL_LOGS_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
)


def _additional_settings(cli_args, cwd):
    """Compose only Claude's singular --settings option; leave other args alone."""
    remaining, source = [], None
    index = 0
    while index < len(cli_args):
        argument = cli_args[index]
        if argument == "--":
            remaining.extend(cli_args[index:])
            break
        if argument == "--settings":
            index += 1
            if index == len(cli_args):
                raise ValueError("Claude Code --settings requires a file path or JSON object")
            source = cli_args[index]
        elif argument.startswith("--settings="):
            source = argument.split("=", 1)[1]
        else:
            remaining.append(argument)
        index += 1
    if source is None:
        return {}, remaining

    def invalid_constant(_value):
        raise ValueError("non-finite values are not JSON")

    try:
        text = source if source.lstrip().startswith("{") else (Path(cwd) / source).read_text()
        settings = json.loads(text, parse_constant=invalid_constant)
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")
    except (OSError, ValueError):
        raise ValueError("Cannot load Claude Code --settings as a JSON object") from None
    return settings, remaining


class ClaudeCodeSession(CliSession):
    agent_name = "claude"
    harness = "claude_code"
    label = "Claude Code"
    engine_coverage = {
        "llm": "claude_code_reported",
        "tools": "hook_boundaries",
        "turns": "hook_boundaries",
        "reason": (
            "Interactive Claude Code session: tool calls from Claude Code hooks; model "
            "requests from its OpenTelemetry events, placed by their reported duration, "
            "without prompt or response content."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._claude_settings, self._claude_cli_args = _additional_settings(self.cli_args, self.cwd)
        arguments = self._claude_cli_args
        if "--" in arguments:
            arguments = arguments[: arguments.index("--")]
        hooks_disabled = (
            any(flag in arguments for flag in ("--bare", "--safe-mode"))
            or any(
                os.environ.get(name, "").lower() in {"1", "true"}
                for name in ("CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_SAFE_MODE")
            )
            or self._claude_settings.get("disableAllHooks") is True
        )
        self.engine_coverage = dict(type(self).engine_coverage)
        if hooks_disabled:
            if self.policy.tool_hooks or self.policy.default_to_deny:
                raise ValueError("Claude Code tool policy requires native hooks to be enabled")
            self.engine_coverage.update(
                tools="unavailable",
                turns="unavailable",
                reason="Native Claude Code hooks are explicitly disabled; hook-based tool and turn observation is unavailable.",
            )

    def _prepare(self) -> None:
        hook = {
            "type": "http",
            "url": f"http://127.0.0.1:{self.port}/hooks",
            "timeout": HOOK_TIMEOUT_S,
            # Interpolated by Claude Code from its own environment, so the
            # token never appears in a command line other users can read.
            "headers": {"Authorization": f"Bearer ${TOKEN_VARIABLE}"},
            "allowedEnvVars": [TOKEN_VARIABLE],
        }
        agency_hooks = {event: [{"hooks": [hook]}] for event in HOOK_EVENTS}
        if self.policy.tool_hooks or self.policy.default_to_deny:
            # Command hooks can fail closed on relay outages (exit 2).
            # HTTP hook errors alone do not guarantee a native denial.
            agency_hooks["PreToolUse"] = [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": hook_command(),
                            "timeout": HOOK_TIMEOUT_S,
                        }
                    ]
                }
            ]
        settings = copy.deepcopy(self._claude_settings)
        hooks = settings.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            raise ValueError("Claude Code --settings hooks must be an object")
        for event, entries in agency_hooks.items():
            existing = hooks.get(event, [])
            if not isinstance(existing, list):
                raise ValueError(f"Claude Code --settings hooks.{event} must be a list")
            hooks[event] = [*existing, *entries]
        self._write_private_json(SETTINGS_FILE, settings)

    def _cli_argv(self) -> "list[str]":
        return [
            self.binary,
            "--settings",
            str(self.run_dir / SETTINGS_FILE),
            *self._claude_cli_args,
        ]

    def _cli_environment(self, env: dict) -> None:
        if self.capture is not None:
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            env.update(
                ANTHROPIC_BASE_URL=f"http://127.0.0.1:{self.port}/capture",
                ANTHROPIC_AUTH_TOKEN=self.host_token,
                ANTHROPIC_MODEL=self.capture.model,
                ANTHROPIC_DEFAULT_OPUS_MODEL=self.capture.model,
                ANTHROPIC_DEFAULT_SONNET_MODEL=self.capture.model,
                ANTHROPIC_DEFAULT_HAIKU_MODEL=self.capture.model,
                CLAUDE_CODE_SUBAGENT_MODEL=self.capture.model,
                CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                CLAUDE_CODE_USE_BEDROCK="0",
                CLAUDE_CODE_USE_VERTEX="0",
                CLAUDE_CODE_USE_FOUNDRY="0",
            )
        overridden = [name for name in _USER_OTEL_LOG_VARIABLES if name in env]
        if overridden:
            print(
                f"[agency] WARNING: {', '.join(overridden)} is set; this session's Claude Code "
                "log events are sent to Agency instead.",
                flush=True,
            )
        env.update(
            CLAUDE_CODE_ENABLE_TELEMETRY="1",
            OTEL_LOGS_EXPORTER="otlp",
            OTEL_EXPORTER_OTLP_LOGS_PROTOCOL="http/json",
            OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=f"http://127.0.0.1:{self.port}/v1/logs",
            OTEL_EXPORTER_OTLP_LOGS_HEADERS=f"x-agency-token={self.host_token}",
            # The receiver parses plain JSON; a user-wide gzip setting must not
            # apply to this one exporter.
            OTEL_EXPORTER_OTLP_LOGS_COMPRESSION="none",
            OTEL_LOGS_EXPORT_INTERVAL="1000",
        )
        # Enabling telemetry must not start a metrics exporter the user never
        # asked for (it would post to a default endpoint nobody serves).
        env.setdefault("OTEL_METRICS_EXPORTER", "none")

    def handle_log_records(self, body: dict) -> None:
        for record in otlp_log_records(body, "claude_code."):
            name = record["name"]
            attributes = record["attributes"]
            if name in {"api_request", "api_error"}:
                self._record_api_request(
                    attributes,
                    record["time"],
                    error=str(attributes.get("error") or "API error")
                    if name == "api_error"
                    else None,
                )
            elif name == "tool_decision" and attributes.get("decision") == "reject":
                source = attributes.get("source") or "user"
                self._tool_finished(attributes.get("tool_use_id"), error=f"Rejected ({source})")
                if source in {"user_reject", "user_abort"}:
                    # Claude ends the turn and waits for the user, but fires
                    # no Stop hook for a turn the user interrupted.
                    self._set_state("agent_idle")
            elif name == "tool_result" and str(attributes.get("success")) == "false":
                # PostToolUseFailure closes every failure except a running tool
                # the user cancelled (Escape), which fires no hook at all. So a
                # failure still pending here was cancelled, which also ends
                # the turn without a Stop hook.
                reported = attributes.get("error") or attributes.get("error_type") or "failure"
                cancelled = self._tool_finished(
                    attributes.get("tool_use_id"),
                    error=f"Cancelled by the user (Claude Code reported {reported})",
                    duration_ms=number(attributes.get("duration_ms")),
                )
                if cancelled:
                    self._set_state("agent_idle")

    def _record_api_request(self, attributes: dict, ended_at: float, error=None) -> None:
        # Claude Code reports a request once it has finished, with its own
        # measured duration, so the start is end minus duration.
        duration_ms = number(attributes.get("duration_ms"))
        input_tokens = sum(
            number(attributes.get(key)) or 0
            for key in ("input_tokens", "cache_read_tokens", "cache_creation_tokens")
        )
        # A subagent's requests carry its type (`agent.name`) but no id, and
        # may arrive after its SubagentStop; see _subagent_running_at().
        subagent_type = attributes.get("agent.name")
        actor = self._subagent_running_at(subagent_type, ended_at) if subagent_type else None
        attribution = {}
        if subagent_type:
            attribution["attribution"] = (
                "subagent_type_and_time" if actor else "subagent_ambiguous_kept_on_parent"
            )
            attribution["subagent_type"] = subagent_type
        self._record_model_request(
            ended_at - (duration_ms or 0) / 1000,
            ended_at,
            model=attributes.get("model"),
            actor=actor,
            input_tokens=input_tokens,
            output_tokens=attributes.get("output_tokens"),
            request_id=attributes.get("request_id"),
            error=error,
            details={
                "duration_ms": duration_ms,
                "cost_usd": number(attributes.get("cost_usd")),
                "query_source": attributes.get("query_source"),
                **attribution,
            },
        )


def main(argv=None) -> int:
    return run_session(ClaudeCodeSession, "claude", argv)
