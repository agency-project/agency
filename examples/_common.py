"""Shared runtime setup for the numbered tutorials.

The tutorials default to Codex running on an OpenAI model, but the harness and
model backend remain independent. Change either with environment variables:

    AGENCY_HARNESS=codex
    AGENCY_LLM_PROVIDER=openai
    AGENCY_LLM_MODEL=gpt-5.6-luna
    OPENAI_API_KEY=...

The API key is read only when a tutorial starts. Importing tutorial modules is
therefore safe in documentation tooling and the unit test suite.
"""

from __future__ import annotations

import os
import gc
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agency import agent, sigterm_as_exit
from agency.orchestrator import peek_orchestrator
from agency.configs.agconfig import (
    agentconfig,
    agconfig,
    llmconfig,
    orchestratorconfig,
    sandboxconfig,
)


DEFAULT_HARNESS = "codex"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_WAIT_TIMEOUT_SECONDS = 300

# Profiling gets its own explicit lesson. Keeping process-lifetime profiling
# off everywhere else makes the tutorial suite faster and its artifacts easier
# to understand.
os.environ.setdefault("AGENCY_PROFILE", "0")


def example_wait_timeout() -> float:
    """Return the per-result wait bound, allowing slower backends to opt up."""
    return float(
        os.environ.get("AGENCY_EXAMPLE_WAIT_TIMEOUT_SECONDS", DEFAULT_WAIT_TIMEOUT_SECONDS)
    )


def tutorial_config(
    example_name: str,
    *,
    harness: str | None = None,
    max_concurrent_engines: int | None = None,
    require_llm: bool = True,
) -> tuple[agconfig, Path]:
    """Build one current, namespaced config and a unique artifact directory."""
    provider = os.environ.get("AGENCY_LLM_PROVIDER", "openai")
    model = os.environ.get("AGENCY_LLM_MODEL", DEFAULT_MODEL)
    selected_harness = harness or os.environ.get("AGENCY_HARNESS", DEFAULT_HARNESS)

    llm_kwargs: dict[str, object] = {
        "provider": provider,
        "model": model,
        # Luna's Chat Completions endpoint requires "none" when function
        # tools are present. Responses-based harnesses can still opt into a
        # higher value through the environment.
        "reasoning_effort": os.environ.get("AGENCY_REASONING_EFFORT", "none"),
        "max_completion_tokens": int(os.environ.get("AGENCY_MAX_COMPLETION_TOKENS", "4096")),
    }
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and require_llm:
            raise SystemExit("OPENAI_API_KEY is required for AGENCY_LLM_PROVIDER=openai")
        if api_key:
            llm_kwargs["api_key"] = api_key
        if os.environ.get("OPENAI_BASE_URL"):
            llm_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and require_llm:
            raise SystemExit("ANTHROPIC_API_KEY is required for AGENCY_LLM_PROVIDER=anthropic")
        if api_key:
            llm_kwargs["api_key"] = api_key
        llm_kwargs.pop("reasoning_effort", None)
    elif provider == "vllm":
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url and require_llm:
            raise SystemExit("LLM_BASE_URL is required for AGENCY_LLM_PROVIDER=vllm")
        if base_url:
            llm_kwargs["base_url"] = base_url
        llm_kwargs["api_key"] = os.environ.get("LLM_API_KEY", "")
        llm_kwargs["model"] = os.environ.get("LLM_MODEL", model)
        llm_kwargs.pop("reasoning_effort", None)
    elif provider == "bedrock":
        llm_kwargs["region"] = os.environ.get("AWS_REGION", "us-east-2")
        llm_kwargs.pop("reasoning_effort", None)
    else:
        raise SystemExit(f"Unsupported AGENCY_LLM_PROVIDER: {provider!r}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = Path(os.environ.get("AGENCY_EXAMPLE_RUN_ROOT", "runs/tutorials"))
    run_dir = (root / f"{timestamp}_{example_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = agconfig(
        llmconfig(**llm_kwargs),
        agentconfig(
            harness=selected_harness,
            log_dir=str(run_dir / "logs"),
            output_dir=str(run_dir / "agent_output"),
        ),
        sandboxconfig(
            base_image=os.environ.get("AGENCY_SANDBOX_IMAGE", "docker.io/library/python:3.12-slim")
        ),
        orchestratorconfig(
            max_concurrent_engines=max_concurrent_engines,
            db_path=str(run_dir / "agency.sqlite3"),
        ),
    )
    return cfg, run_dir


def close_sandboxes(agents: Iterable[agent]) -> None:
    """Eager cleanup keeps a full tutorial-suite run from accumulating containers."""
    for current in agents:
        if current.sandbox is not None:
            current.sandbox.destroy()
            current.sandbox = None


def run_example(main) -> None:
    """Give standalone scripts graceful SIGTERM handling and deterministic shutdown."""
    try:
        with sigterm_as_exit("agency-example"):
            main()
    finally:
        # Agent/engine references can form a cycle. Collect while their
        # per-agent loggers are still open so best-effort teardown remains
        # quiet and records its final events before orchestrator shutdown.
        gc.collect()
        orchestrator = peek_orchestrator()
        if orchestrator is not None:
            orchestrator.flush()
            orchestrator.shutdown(timeout_s=120)
