"""Live compatibility probe for first-party OpenAI and Anthropic models.

Each model runs the same short task through Agency's native in-container
ReAct harness.  The task exercises both a built-in tool call and structured
output recovery, rather than accepting a plain text completion as success.

Run every model::

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
        uv run python examples/native_model_compatibility.py

Run one or more models by display name or API ID::

    uv run python examples/native_model_compatibility.py \
        --model "5.6 luna" --model claude-sonnet-5
"""

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agency import agent, agdata, agskill
from agency.configs.agconfig import agconfig as agconfig_cls


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    provider: str
    api_model: str


MODELS = (
    ModelSpec("5.6 luna", "openai", "gpt-5.6-luna"),
    ModelSpec("5.6 terra", "openai", "gpt-5.6-terra"),
    ModelSpec("5.6 sol", "openai", "gpt-5.6-sol"),
    ModelSpec("Sonnet 5", "anthropic", "claude-sonnet-5"),
    ModelSpec("Opus 5", "anthropic", "claude-opus-5"),
    ModelSpec("Fable 5", "anthropic", "claude-fable-5"),
)

PROBE_TEXT = "native-harness-ok"


def _config_for(spec: ModelSpec) -> agconfig_cls:
    if spec.provider == "openai":
        return agconfig_cls(
            provider="openai",
            model=spec.api_model,
            api_key=os.environ["OPENAI_API_KEY"],
            max_completion_tokens=2048,
            reasoning_effort="none",
        )
    return agconfig_cls(
        provider="anthropic",
        model=spec.api_model,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_completion_tokens=2048,
    )


def _probe_skill() -> agskill:
    return agskill(
        name="native_model_compatibility",
        system_prompt=(
            "Use the bash tool exactly once to run the command supplied by the user. "
            "Then submit structured output. Do not infer or fabricate command output."
        ),
        input_schema=agdata(command=str),
        output_schema=agdata(status=str, proof=str),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _host_output_root() -> Path:
    configured = os.environ.get("AGENCY_COMPAT_OUTPUT_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).parent.parent / "runs"


def _select_models(requested: list[str]) -> list[ModelSpec]:
    if not requested:
        return list(MODELS)
    by_name = {key.lower(): spec for spec in MODELS for key in (spec.display_name, spec.api_model)}
    unknown = [name for name in requested if name.lower() not in by_name]
    if unknown:
        valid = ", ".join(spec.display_name for spec in MODELS)
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}. Valid names: {valid}")
    return [by_name[name.lower()] for name in requested]


def _run_one(spec: ModelSpec, run_root: Path) -> tuple[bool, str]:
    model_dir = run_root / _slug(spec.api_model)
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.log_dir = model_dir / "logs"
    agent.output_dir = None

    ag = agent(agconfig=_config_for(spec), harness="native")
    try:
        result = ag.run(
            _probe_skill(),
            agdata(command=f"printf {PROBE_TEXT}"),
        ).to_dict()
        if "error" in result:
            return False, str(result["error"])
        ok = bool(result.get("status")) and result.get("proof") == PROBE_TEXT
        if ok:
            return True, f"status={result['status']!r}, proof={result['proof']!r}"
        return False, f"unexpected structured output: {result!r}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if ag.sandbox is not None:
            ag.sandbox.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Display name or exact API model ID. Repeat to run several; omit to run all.",
    )
    args = parser.parse_args()
    selected = _select_models(args.model)

    missing = sorted(
        {
            "OPENAI_API_KEY" if spec.provider == "openai" else "ANTHROPIC_API_KEY"
            for spec in selected
            if not os.environ.get(
                "OPENAI_API_KEY" if spec.provider == "openai" else "ANTHROPIC_API_KEY"
            )
        }
    )
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing)}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = _host_output_root() / f"{timestamp}_native_model_compatibility"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_root}")

    failures = 0
    for spec in selected:
        print(f"\n[{spec.display_name}] {spec.api_model}")
        ok, detail = _run_one(spec, run_root)
        print(f"{'PASS' if ok else 'FAIL'}: {detail}")
        failures += not ok

    print(f"\nSummary: {len(selected) - failures}/{len(selected)} passed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
