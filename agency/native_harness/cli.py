"""CLI entrypoint for the standalone native harness -- the whole point of
this package: `PYTHONPATH=<repo>/agency python3 -m native_harness.cli -p
"<prompt>" --model <name> ...` (see `__init__.py`'s docstring for why NOT
`agency.native_harness`) runs a complete ReAct session and exits, exactly
the same one-shot shape `claude -p ...`/`codex ...` already have, so a
future agency backend can launch and read this exactly like
`claude_code.py` launches and reads the real `claude` binary today -- and a
person can run
the identical command from their own bash prompt with no agency involved
at all.

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
`claude_code.py`'s `_parse_result_json` already parses. No streaming
output format: live per-turn visibility, when bridged, comes from agency
polling `agmanager_host`'s own live transcript for this run's token (the
same transcript used by the host UI), not from anything this CLI prints."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .bridge_client import BridgeClient
from .llm_client import LLMClient
from .mcp_client import McpToolset
from .react_loop import run_react_loop
from . import session as session_store
from . import react_loop
from . import tools

_DEFAULT_SESSION_DIR = os.path.join(os.path.expanduser("~"), ".native_harness", "sessions")
_DEFAULT_OFFLOAD_DIR = "./long_tool_call_outputs"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="native_harness")
    p.add_argument("-p", "--prompt", required=True, help="The task prompt for this turn.")
    p.add_argument("--model", required=True)
    p.add_argument("--system", default=None, help="System prompt (only used for a fresh session).")
    p.add_argument("--max-steps", type=int, default=4096)
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
        "--mcp-config",
        default=None,
        help='JSON, same shape as Claude Code\'s own flag: {"mcpServers": {"name": '
        '{"type": "http", "url": "...", "headers": {...}}}}',
    )
    p.add_argument("--offload-dir", default=_DEFAULT_OFFLOAD_DIR)
    p.add_argument(
        "--canonicalize-run-output",
        action="store_true",
        help="Compact large program/test-run output from bash (repeated blocks, head/tail, errors).",
    )
    p.add_argument(
        "--code-read-tools",
        action="store_true",
        help="Offer read(symbol=...) and an optional `want` on bash/read/grep.",
    )
    p.add_argument("--want-required", action="store_true", help="Make `want` required on bash.")
    p.add_argument(
        "--code-read-hint",
        action="store_true",
        help="Start a fresh session with a system note pointing at read(symbol=...) and `want`.",
    )
    p.add_argument(
        "--compact-code-reads",
        action="store_true",
        help="Drop content-free lines (license header, blank runs) from bash file reads, with markers.",
    )
    p.add_argument(
        "--compact-tables",
        action="store_true",
        help="Print box-drawn result tables from bash as plain `a | b` rows and drop progress bars.",
    )
    p.add_argument(
        "--end-on-submit",
        action="store_true",
        help="End the run once submit_output reports every output field in, without a closing turn.",
    )
    p.add_argument(
        "--drop-tools",
        default="",
        help="Comma-separated tool names to leave out of the tool list.",
    )
    p.add_argument(
        "--submit-with-check",
        action="store_true",
        help="Tell the agent it may submit its outputs in the same turn as its final check.",
    )
    p.add_argument(
        "--compact-tests",
        action="store_true",
        help="Cut test-runner output from bash to failures, summaries and the last lines.",
    )
    p.add_argument(
        "--compact-search",
        action="store_true",
        help="Group grep hits by file and drop licence headers from bash output.",
    )
    p.add_argument(
        "--elide-large",
        type=int,
        default=0,
        help="Show bash/read outputs above this many tokens as outlines, expandable with show_elided (0 = off).",
    )
    p.add_argument(
        "--cache-keepalive-s",
        type=int,
        default=0,
        help="While tools run, re-send the last request with max 1 output token once this many seconds "
        "have passed since the cache was last used (0 = off).",
    )
    p.add_argument("--selector-model", default=None, help="Model that picks lines of elided outputs to show in full.")
    p.add_argument("--selector-llm-base-url", default=None)
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


def _resolve_session(args: argparse.Namespace) -> "tuple[str, list]":
    session_id = args.resume_id or args.session_id or session_store.new_session_id()
    prior = session_store.load_session(args.session_dir, session_id)
    if prior is not None:
        return session_id, prior
    messages = [{"role": "system", "content": args.system}] if args.system else []
    return session_id, messages


def main(argv: "list[str] | None" = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    tools.CANONICALIZE_RUN_OUTPUT = args.canonicalize_run_output
    tools.COMPACT_TABLES = args.compact_tables
    tools.COMPACT_TESTS = args.compact_tests
    react_loop.END_ON_SUBMIT = args.end_on_submit
    react_loop.DROP_TOOLS = frozenset(t for t in args.drop_tools.split(",") if t)
    if args.code_read_tools or args.want_required or args.code_read_hint:
        tools.enable_code_read_tools(want_required=args.want_required)
    tools.COMPACT_CODE_READS = args.compact_code_reads
    tools.elide.COMPACT_SEARCH = args.compact_search
    tools.elide.ELIDE_MIN_TOKENS = args.elide_large
    if args.elide_large:
        tools.enable_show_elided()
    react_loop.CACHE_KEEPALIVE_S = args.cache_keepalive_s

    llm_base_url, llm_api_key = _resolve_llm_endpoint(args)
    llm = LLMClient(llm_base_url, llm_api_key)
    if args.selector_model and args.selector_llm_base_url:
        selector = LLMClient(args.selector_llm_base_url, llm_api_key)
        tools.elide.SELECTOR = lambda prompt: selector.dispatch(
            args.selector_model, [{"role": "user", "content": prompt}], internal_kind="selector"
        )

    bridge = (
        BridgeClient(args.bridge_base_url, args.bridge_token)
        if args.bridge_base_url and args.bridge_token
        else None
    )
    context_limit = bridge.context_limit() if bridge is not None else None

    mcp_config = json.loads(args.mcp_config) if args.mcp_config else None
    mcp = McpToolset(mcp_config) if mcp_config else None

    session_id, messages = _resolve_session(args)
    if args.submit_with_check and not any(m.get("role") != "system" for m in messages):
        messages = messages + [{"role": "system", "content": react_loop.SUBMIT_WITH_CHECK}]
    if args.code_read_hint and not any(m.get("role") != "system" for m in messages):
        messages = [{"role": "system", "content": tools.CODE_READ_HINT}] + messages
    messages = messages + [{"role": "user", "content": args.prompt}]

    result = run_react_loop(
        messages,
        args.model,
        llm,
        mcp=mcp,
        bridge=bridge,
        context_limit=context_limit,
        max_steps=args.max_steps,
        offload_dir=args.offload_dir,
        progress_path=args.progress_file,
        # Checkpoints the session after every turn, not just on a clean
        # finish -- so a host that kills this process mid-run (an idle
        # timeout, say) still has a resumable session on disk, not just
        # whatever `--resume`/`--session-id` it started with.
        on_checkpoint=lambda msgs: session_store.save_session(
            args.session_dir, session_id, args.model, msgs
        ),
    )

    if result.status != "done":
        print(json.dumps({"error": result.message}), file=sys.stderr)
        return 1

    session_store.save_session(args.session_dir, session_id, args.model, result.messages)
    print(
        json.dumps(
            {
                "result": result.final_text,
                "usage": {
                    "input_tokens": result.total_input_tokens,
                    "output_tokens": result.total_output_tokens,
                },
                "session_id": session_id,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
