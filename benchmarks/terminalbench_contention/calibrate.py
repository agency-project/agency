"""Offline calibration of the prompt-token estimator against recorded usage.

Reads Agency data-logger databases (read-only, immutable) from earlier runs,
rebuilds each recorded request exactly as Agency's OpenAI-compatible backend
puts it on the wire, and compares estimates with the ``prompt_tokens`` the model
reported. No model request is made.

    .venv/bin/python -m benchmarks.terminalbench_contention.calibrate \
        --source artifacts/cpu-baselines/<root>/runs --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from pathlib import Path

from .tokens import EstimatorConfig, TokenEstimator, completion_cap, legacy_estimate


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def recorded_exchanges(db_path: Path):
    """Yield (agency_request, usage) for every successful recorded exchange."""
    con = _open_readonly(db_path)
    try:
        labels = [
            row[0]
            for row in con.execute(
                "select call_label from exchanges where type='llm_block' order by timestamp"
            )
        ]
        for label in labels:
            chain = {}
            for kind in ("prompt", "response"):
                chain[kind] = [
                    json.loads(payload)
                    for (payload,) in con.execute(
                        "select b.payload from exchange_chain c join blocks b on b.hash=c.hash "
                        "where c.call_label=? and c.kind=? order by c.seq",
                        (label, kind),
                    )
                ]
            usage = None
            for block in chain["response"]:
                if block.get("type") != "metadata":
                    continue
                if isinstance(block.get("usage"), dict):
                    usage = block["usage"]
                fragments = block.get("data")
                for fragment in fragments if isinstance(fragments, list) else [fragments]:
                    if isinstance(fragment, dict) and isinstance(fragment.get("usage"), dict):
                        usage = fragment["usage"]
            if not usage or not usage.get("prompt_tokens"):
                continue
            messages, tools, extra = [], [], {}
            previous_index = None
            for block in chain["prompt"]:
                role = block.get("role")
                if role == "tool_schema":
                    if block.get("type") == "tool_definition":
                        tools.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "description": block.get("description", ""),
                                    "parameters": block.get("parameters"),
                                },
                            }
                        )
                    elif block.get("type") == "tool_choice":
                        extra["tool_choice"] = block.get("value")
                    continue
                body = {k: v for k, v in block.items() if k != "role"}
                index = body.get("index", 0)
                # Blocks were flattened per message; an index that does not
                # advance (or a new role) starts the next message.
                if (
                    not messages
                    or messages[-1]["role"] != role
                    or previous_index is None
                    or index <= previous_index
                ):
                    messages.append({"role": role, "blocks": []})
                messages[-1]["blocks"].append(body)
                previous_index = index
            yield {"messages": messages, "tools": tools or None, **extra}, usage
    finally:
        con.close()


def _backend(model: str, max_completion_tokens: int):
    from agency.configs.agconfig import agconfig, llmconfig
    from agency.llm.agllm import agllm

    return agllm.for_config(
        agconfig(
            llmconfig(
                provider="openai",
                model=model,
                base_url="http://calibration.invalid/v1",
                reasoning_effort="none",
                max_completion_tokens=max_completion_tokens,
            )
        )
    )


def _quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def summarize(ratios):
    return dict(
        n=len(ratios),
        mean=statistics.fmean(ratios),
        median=statistics.median(ratios),
        p05=_quantile(ratios, 0.05),
        p95=_quantile(ratios, 0.95),
        minimum=min(ratios),
        maximum=max(ratios),
        fraction_under=sum(r < 1.0 for r in ratios) / len(ratios),
    )


def calibrate(databases, *, model="gpt-6-luna", max_completion_tokens=4096, limit=None):
    backend = _backend(model, max_completion_tokens)
    variants = {
        "typescript_raw": EstimatorConfig(safety_factor=1.0, safety_tokens=0),
        "json_raw": EstimatorConfig(tool_schema_mode="json", safety_factor=1.0, safety_tokens=0),
        "default": EstimatorConfig(),
    }
    estimators = {name: TokenEstimator(cfg) for name, cfg in variants.items()}
    ratios = {name: [] for name in [*variants, "legacy_bytes_over_2"]}
    totals = dict(
        requests=0,
        reported_prompt=0,
        reported_completion=0,
        reported_cached=0,
        legacy_charge=0,
        default_prompt_estimate=0,
        default_charge=0,
        reserved_completion=0,
    )
    for path in databases:
        try:
            exchanges = list(recorded_exchanges(Path(path)))
        except sqlite3.Error:
            continue
        for request, usage in exchanges:
            wire = backend._format_context_agency_to_backend(request)
            wire = {**wire, "stream": True, "stream_options": {"include_usage": True}}
            body = json.dumps(wire, separators=(",", ":"), ensure_ascii=False).encode()
            reported = int(usage["prompt_tokens"])
            for name, estimator in estimators.items():
                estimate = estimator.estimate(wire)
                value = estimate.raw_tokens if name.endswith("_raw") else estimate.estimated_tokens
                ratios[name].append(value / reported)
                if name == "default":
                    totals["default_prompt_estimate"] += estimate.estimated_tokens
                    totals["default_charge"] += estimate.estimated_tokens + completion_cap(
                        wire, max_completion_tokens
                    )
            legacy = legacy_estimate(body, wire)
            ratios["legacy_bytes_over_2"].append(
                (legacy - completion_cap(wire, max_completion_tokens)) / reported
            )
            details = usage.get("prompt_tokens_details") or {}
            totals["requests"] += 1
            totals["reported_prompt"] += reported
            totals["reported_completion"] += int(usage.get("completion_tokens") or 0)
            totals["reported_cached"] += int(details.get("cached_tokens") or 0)
            totals["legacy_charge"] += legacy
            totals["reserved_completion"] += completion_cap(wire, max_completion_tokens)
            if limit and totals["requests"] >= limit:
                break
        if limit and totals["requests"] >= limit:
            break
    if not totals["requests"]:
        raise SystemExit("No recorded exchanges with reported usage were found")
    return dict(
        model=model,
        max_completion_tokens=max_completion_tokens,
        databases=len(databases),
        totals=totals,
        prompt_estimate_over_reported={k: summarize(v) for k, v in ratios.items()},
        estimator_default=EstimatorConfig().to_dict(),
        note=(
            "Ratios are estimated prompt tokens / model-reported prompt_tokens per request. "
            "fraction_under is the share of requests whose estimate was below reported usage."
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, action="append")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    databases = sorted(
        p for source in args.source for p in source.rglob("agent_*_data.sqlite3") if p.is_file()
    )
    result = calibrate(
        databases,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        limit=args.limit,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
