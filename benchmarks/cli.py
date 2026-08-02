"""CLI entry point: run a BenchmarkProvider's tasks through AgencyBackend.

Usage:
    uv run python run-benchmark --provider swe-bench \
        --tasks path/to/instances.jsonl --output runs/report.json

    uv run python run-benchmark --provider terminal-bench \
        --tasks path/to/tasks_dir --limit 10 --output runs/report.json

There is a single agent backend (Agency). Which `ExecutionEnvironment` it
runs a task's agent inside — a bind-mounted host workspace, or a container
image the task itself defines — is chosen from `--provider`, since that's
what determines the shape of a task's files (see `_select_environment`
below).

LLM config comes from environment variables — set one of:
    ANTHROPIC_API_KEY                     talk to Claude directly via api.anthropic.com
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY   any OpenAI-compatible endpoint (vLLM, etc.)

For `--provider swe-bench`, a predictions file in the format the official
SWE-bench evaluation harness expects is also written (default
`<output_dir>/predictions.jsonl`) — this run only produces patches, it does
not determine whether an issue was actually resolved. Run the official
harness against that file for that.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agency.agconfig import agConfig
from agency.agllm_backends import agAnthropicBackendConfig, agVLLMBackendConfig

from .backends import AgencyBackend
from .base import ExecutionEnvironment
from .environments import ContainerImageEnvironment, HostWorkspaceEnvironment
from .providers import SWEBenchProvider, TerminalBenchProvider
from .runner import Runner

ENVIRONMENTS = {
    "swe-bench": HostWorkspaceEnvironment,
    "terminal-bench": ContainerImageEnvironment,
}


def _select_environment(provider_name: str) -> ExecutionEnvironment:
    try:
        return ENVIRONMENTS[provider_name]()
    except KeyError:
        raise ValueError(f"no execution environment registered for provider {provider_name!r}")


def _build_llm_config() -> tuple[agConfig, str]:
    """Returns (agConfig, model_name_or_path) — the latter is a plain string
    identifying which model produced a run's results, for recording
    alongside exported predictions."""
    model = os.environ.get("LLM_MODEL", "")
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("LLM_BASE_URL"):
        model = model or "claude-sonnet-5"
        cfg = agConfig(
            agAnthropicBackendConfig(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
        )
        return cfg, model
    cfg = agConfig(
        agVLLMBackendConfig(
            base_url=os.environ.get("LLM_BASE_URL"),
            model=model,
            api_key=os.environ.get("LLM_API_KEY", ""),
        )
    )
    return cfg, (model or "vllm-endpoint")


def _build_provider(args: argparse.Namespace):
    if args.provider == "swe-bench":
        return SWEBenchProvider(tasks_path=args.tasks, cache_dir=args.cache_dir)
    return TerminalBenchProvider(tasks_dir=args.tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a benchmark provider's tasks through the Agency agent backend."
    )
    parser.add_argument("--provider", required=True, choices=sorted(ENVIRONMENTS))
    parser.add_argument(
        "--tasks",
        required=True,
        type=Path,
        help="JSONL file (swe-bench) or task directory (terminal-bench)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the JSON report (a .txt summary is written alongside it)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".benchmark_cache"),
        help="Local cache dir for cloned repos (swe-bench only)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Where to write official-SWE-bench-harness-format predictions "
            "(swe-bench only; default <output_dir>/predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Identifier recorded in exported predictions (default: derived from LLM env vars)",
    )
    args = parser.parse_args()

    cfg, model_name_or_path = _build_llm_config()
    if args.model_name_or_path:
        model_name_or_path = args.model_name_or_path

    provider = _build_provider(args)
    environment = _select_environment(args.provider)
    backend = AgencyBackend(agconfig=cfg, environment=environment)

    tasks = provider.load_tasks(limit=args.limit)
    print(f"Loaded {len(tasks)} task(s) from {args.provider!r}")

    def _report_progress(record: dict) -> None:
        status = "ok" if record["completed"] else "FAILED"
        passed = record["metrics"].get("passed")
        passed_str = "n/a" if passed is None else ("pass" if passed else "fail")
        print(
            f"  [{status}] {record['task_id']}  passed={passed_str}  ({record['elapsed_s']:.1f}s)"
        )

    runner = Runner(provider, backend, output_dir=args.output.parent)
    report = runner.run(tasks, on_result=_report_progress)
    Runner.save_report(report, args.output)

    print()
    print(args.output.with_suffix(".txt").read_text())
    print(f"Full report: {args.output}")

    if isinstance(provider, SWEBenchProvider):
        predictions_path = args.predictions or (args.output.parent / "predictions.jsonl")
        SWEBenchProvider.write_predictions(report["results"], predictions_path, model_name_or_path)
        print(
            f"\nSWE-bench predictions (official-harness format): {predictions_path}\n"
            "This run only produced patches — resolution status is unknown until "
            "the official SWE-bench evaluation harness is run against that file."
        )


if __name__ == "__main__":
    main()
