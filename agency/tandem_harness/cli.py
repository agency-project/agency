"""CLI entrypoint for the standalone tandem harness -- the whole point of
this package: `PYTHONPATH=<repo>/agency python3 -m tandem_harness.cli -p
"<prompt>" --supervisor-model <name> --worker-model <name> ...` (see
`__init__.py`'s docstring for why NOT `agency.tandem_harness`) runs a
complete tandem session and exits, exactly the same one-shot shape
`claude -p ...`/`codex ...`/`native_harness.cli` already have, so a future
agency backend can launch and read this exactly like `claude_code.py`
launches and reads the real `claude` binary today -- and a person can run
the identical command from their own bash prompt with no agency involved
at all.

**Two models, one endpoint by default**: `--supervisor-model`/
`--worker-model` are both dispatched through the SAME resolved LLM
endpoint (bridged or standalone -- see below) unless
`--supervisor-llm-base-url`/`--supervisor-llm-api-key` are given to point
the supervisor at a genuinely different provider. This is what lets one
litellm-routed bridge serve both a large supervisor model and a small
worker model with no extra plumbing in the common case.

**Two configurations, same code path** (see `llm_client.py`'s docstring):
- Bridged (launched by agency): `--bridge-base-url`/`--bridge-token` point
  at this run's own `agmanager_harness` instance. That one (base_url,
  token) pair configures LLM dispatch, per-tool policy checks, and
  pause/inbox check-in all at once -- mirroring how Claude Code's single
  `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` pair already serves all three
  through that same process.
- Standalone: `--llm-base-url`/`--llm-api-key` (or `OPENAI_BASE_URL`/
  `OPENAI_API_KEY`) point at a real provider directly; no policy checks, no
  check-in, no compaction context-limit lookup (all silently no-op --
  agency-side control-plane concerns with no meaning outside agency).

Output: a single JSON object on stdout, `{"result", "usage", "session_id"}`
-- the same shape `claude -p --output-format json` produces, which
`claude_code.py`'s `_parse_result_json` already parses. `usage` additionally
breaks totals down into `supervisor_input_tokens`/`supervisor_output_tokens`/
`worker_input_tokens`/`worker_output_tokens`/`segment_count` -- the whole
point of this harness is measuring that split. No streaming output format:
live per-turn visibility, when bridged, comes from agency polling
`agmanager_host`'s own live transcript for this run's token (the same
transcript used by the host UI), not from anything this CLI prints."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .bridge_client import BridgeClient
from .llm_client import LLMClient
from .mcp_client import McpToolset
from .tandem_loop import SUPERVISOR_SYSTEM, DEFAULT_SEGMENT_STEP_CAP, run_tandem_loop
from . import session as session_store

_DEFAULT_SESSION_DIR = os.path.join(os.path.expanduser("~"), ".tandem_harness", "sessions")
_DEFAULT_OFFLOAD_DIR = "./long_tool_call_outputs"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tandem_harness")
    p.add_argument("-p", "--prompt", required=True, help="The task prompt for this turn.")
    p.add_argument("--supervisor-model", required=True)
    p.add_argument("--worker-model", required=True)
    p.add_argument(
        "--segment-step-cap",
        type=int,
        default=DEFAULT_SEGMENT_STEP_CAP,
        help="Max tool calls a worker segment may make before it must report back to the "
        "supervisor (default: %(default)s).",
    )
    p.add_argument(
        "--system",
        default=None,
        help="Supervisor system prompt override (only used for a fresh session; defaults to "
        "tandem_loop.SUPERVISOR_SYSTEM).",
    )
    p.add_argument("--max-steps", type=int, default=4096, help="Max supervisor turns (orders).")
    p.add_argument(
        "--output-format", choices=["json"], default="json", help="Only 'json' is supported today."
    )

    p.add_argument("--session-id", default=None)
    p.add_argument("--resume", dest="resume_id", default=None, metavar="SESSION_ID")
    p.add_argument("--session-dir", default=_DEFAULT_SESSION_DIR)

    p.add_argument("--bridge-base-url", default=None)
    p.add_argument("--bridge-token", default=None)

    p.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"))

    p.add_argument(
        "--supervisor-llm-base-url",
        default=None,
        help="Overrides the endpoint for the supervisor model only. Defaults to whatever the "
        "worker resolves to (bridged or standalone) -- set this only if the supervisor needs a "
        "genuinely different provider.",
    )
    p.add_argument("--supervisor-llm-api-key", default=None)

    p.add_argument(
        "--mcp-config",
        default=None,
        help='JSON, same shape as Claude Code\'s own flag: {"mcpServers": {"name": '
        '{"type": "http", "url": "...", "headers": {...}}}} -- the worker gets every '
        "configured tool; the supervisor additionally gets submit_output directly (see "
        "tandem_loop.py).",
    )
    p.add_argument("--offload-dir", default=_DEFAULT_OFFLOAD_DIR)
    p.add_argument(
        "--progress-file",
        default=None,
        help="Path to checkpoint per-step progress to, for a bridged caller to poll for "
        "liveness and recover a partial answer from if it gives up waiting.",
    )
    return p


def _resolve_llm_endpoint(args: argparse.Namespace) -> "tuple[str, str]":
    if args.bridge_base_url:
        return args.bridge_base_url, args.bridge_token or ""
    if not args.llm_base_url or not args.llm_api_key:
        raise SystemExit(
            "no LLM endpoint configured: pass --bridge-base-url/--bridge-token (agency-managed "
            "run), or --llm-base-url/--llm-api-key (or set OPENAI_BASE_URL/OPENAI_API_KEY)"
        )
    return args.llm_base_url, args.llm_api_key


def _resolve_supervisor_llm(
    args: argparse.Namespace, worker_llm: "LLMClient", worker_base_url: str, worker_api_key: str
) -> "LLMClient":
    base_url = args.supervisor_llm_base_url or worker_base_url
    api_key = args.supervisor_llm_api_key or worker_api_key
    if base_url == worker_base_url and api_key == worker_api_key:
        return worker_llm  # same endpoint -- share the connection, just dispatch a different model
    return LLMClient(base_url, api_key)


def _resolve_session(args: argparse.Namespace) -> "tuple[str, list]":
    session_id = args.resume_id or args.session_id or session_store.new_session_id()
    prior = session_store.load_session(args.session_dir, session_id)
    if prior is not None:
        return session_id, prior
    # See tandem_loop.py's matching worker-side tag: a zero-host-changes
    # stopgap so a human reading the bridged live transcript can tell
    # supervisor turns from worker-segment turns on sight. Only applied to
    # the default prompt -- an explicit --system override is left untouched.
    system = args.system or f"[TANDEM SUPERVISOR]\n{SUPERVISOR_SYSTEM}"
    return session_id, [{"role": "system", "content": system}]


def main(argv: "list[str] | None" = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    llm_base_url, llm_api_key = _resolve_llm_endpoint(args)
    worker_llm = LLMClient(llm_base_url, llm_api_key)
    supervisor_llm = _resolve_supervisor_llm(args, worker_llm, llm_base_url, llm_api_key)

    bridge = (
        BridgeClient(args.bridge_base_url, args.bridge_token)
        if args.bridge_base_url and args.bridge_token
        else None
    )
    worker_context_limit = bridge.context_limit() if bridge is not None else None

    mcp_config = json.loads(args.mcp_config) if args.mcp_config else None
    mcp = McpToolset(mcp_config) if mcp_config else None

    session_id, messages = _resolve_session(args)
    messages = messages + [{"role": "user", "content": args.prompt}]

    result = run_tandem_loop(
        messages,
        args.supervisor_model,
        args.worker_model,
        supervisor_llm,
        worker_llm,
        mcp=mcp,
        bridge=bridge,
        worker_context_limit=worker_context_limit,
        segment_step_cap=args.segment_step_cap,
        max_segments=args.max_steps,
        offload_dir=args.offload_dir,
        progress_path=args.progress_file,
    )

    if result.status != "done":
        print(json.dumps({"error": result.message}), file=sys.stderr)
        return 1

    session_store.save_session(args.session_dir, session_id, args.supervisor_model, result.messages)
    print(
        json.dumps(
            {
                "result": result.final_text,
                "usage": {
                    "input_tokens": result.supervisor_input_tokens + result.worker_input_tokens,
                    "output_tokens": result.supervisor_output_tokens + result.worker_output_tokens,
                    "supervisor_input_tokens": result.supervisor_input_tokens,
                    "supervisor_output_tokens": result.supervisor_output_tokens,
                    "worker_input_tokens": result.worker_input_tokens,
                    "worker_output_tokens": result.worker_output_tokens,
                    "segment_count": result.segment_count,
                },
                "session_id": session_id,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
