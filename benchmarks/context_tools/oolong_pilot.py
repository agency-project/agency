"""Oolong pilot: does letting the model edit its own context help under a tight budget?

Each question runs in a fresh agent whose context is capped (``llmconfig.context_limit``;
the native harness compacts its history at 90% of it). The Oolong document goes in as
an input, which Agency offloads to a file in the sandbox, so the model has to read it
in pieces. Three arms:

* ``A`` -- the harness alone;
* ``B`` -- plus ``context_tools()`` (the model may edit its own context);
* ``C`` -- B plus a short note in the task on how to use them.

Run on a host set up with ``agency setup-host`` (see docs/setup-host.md)::

    sudo .venv/bin/agency run benchmarks/context_tools/oolong_pilot.py \\
        --data ~/bench/oolong_val_65536.jsonl --n 4 --arms A,B,C --out ~/bench/pilot

Writes one JSON line per run to ``<out>/results.jsonl`` and prints a summary per arm.
Scoring follows Oolong (arXiv 2511.02817 sec. 2.3): exact match for labels, users, dates
and comparisons; ``0.75 ** |error|`` for numbers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import random
import re
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from agency import Agent, agconfig, agdata, agrawstring, agskill, context_tools
from agency.agpolicy import agpolicy
from agency.configs.agconfig import agentconfig, llmconfig, orchestratorconfig, sandboxconfig

# $ per million tokens (Anthropic first-party rates).
PRICES = {
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.2},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.2},
}

TASK = (
    "Answer a question about a dataset. The dataset is the `document` input, which is "
    "too large to read in one go; read it from the sandbox in pieces with shell "
    "commands. The question is the `question` input. Work out the exact answer from the "
    "document alone -- no outside data, and do not guess or estimate -- and end your "
    "answer with the final-answer line in exactly the format the question asks for."
)
TIPS = (
    " Your context window is small (about {limit} tokens) and older turns get "
    "summarized away when it fills up. You can edit your own context: once you have "
    "taken what you need from a tool output (for example, running per-label tallies "
    "for a chunk), call list_context and use edit_context to replace that output with "
    "a one-line note of your tallies so far. Keep your context small this way."
)

_ANSWER_PREFIX = {
    "ANSWER_TYPE.LABEL": "label:",
    "ANSWER_TYPE.USER": "user:",
    "ANSWER_TYPE.DATE": "date:",
}
_COMPARISONS = ("more common than", "less common than", "same frequency as")


# -- scoring ---------------------------------------------------------------------------


def gold_value(row: dict):
    answer = row["answer"]
    if "datetime.date" in answer:
        year, month, day = (int(x) for x in re.findall(r"\d+", answer)[:3])
        return dt.date(year, month, day)
    value = json.loads(answer.replace("'", '"')) if answer.startswith("[") else answer
    return value[0] if isinstance(value, list) else value


def extract(text: str, answer_type: str) -> str:
    """The value after the last final-answer prefix the question asked for."""
    prefix = _ANSWER_PREFIX.get(answer_type, "answer:")
    lower = text.lower()
    at = lower.rfind(prefix)
    if at < 0:
        lines = [line for line in text.strip().splitlines() if line.strip()]
        return lines[-1].strip() if lines else ""
    value = (
        text[at + len(prefix) :].strip().splitlines()[0] if text[at + len(prefix) :].strip() else ""
    )
    return value.strip().strip("*`'\".[] ")


def score(row: dict, answer_text: str) -> "tuple[float, str]":
    answer_type = row["answer_type"]
    predicted = extract(answer_text or "", answer_type)
    gold = gold_value(row)
    if answer_type == "ANSWER_TYPE.NUMERIC":
        numbers = re.findall(r"-?\d+", predicted.replace(",", ""))
        return (0.75 ** abs(int(numbers[0]) - int(gold)) if numbers else 0.0), predicted
    if answer_type == "ANSWER_TYPE.COMPARISON":
        found = [c for c in _COMPARISONS if c in predicted.lower()]
        return (1.0 if found and found[0] == str(gold).lower() else 0.0), predicted
    if answer_type == "ANSWER_TYPE.DATE":
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
            try:
                return (
                    1.0 if dt.datetime.strptime(predicted, fmt).date() == gold else 0.0
                ), predicted
            except ValueError:
                continue
        return 0.0, predicted
    return (1.0 if predicted.lower() == str(gold).lower() else 0.0), predicted


def relative_error(row: dict, predicted: str) -> "float | None":
    """|predicted - gold| / gold for a numeric answer (1.0 when there is no number)."""
    if row["answer_type"] != "ANSWER_TYPE.NUMERIC":
        return None
    gold = int(gold_value(row))
    numbers = re.findall(r"-?\d+", (predicted or "").replace(",", ""))
    if not numbers:
        return 1.0
    return abs(int(numbers[0]) - gold) / max(gold, 1)


# -- what a run cost and did, from the agent's own data log ------------------------------


def run_stats(db_path: Path, model: str) -> dict:
    stats = {
        "llm_calls": 0,
        "compactions": 0,
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "peak_prompt_tokens": 0,
        "tool_calls": {},
        "requests_with_model_patches": 0,
    }
    if not db_path.exists():
        return stats
    connection = sqlite3.connect(db_path)

    def payload(digest):
        row = connection.execute(
            "SELECT payload FROM message_block_lookup WHERE hash = ?", (digest,)
        ).fetchone()
        return json.loads(row[0]) if row else {}

    for (label,) in connection.execute("SELECT call_label FROM exchanges ORDER BY id").fetchall():
        stats["llm_calls"] += 1
        for kind, digest in connection.execute(
            "SELECT kind, hash FROM exchange_chain WHERE call_label = ? ORDER BY kind, seq",
            (label,),
        ):
            block = payload(digest)
            if kind == "prompt" and block.get("type") == "request_extra":
                if (block.get("value") or {}).get("agency_internal_kind") == "compaction":
                    stats["compactions"] += 1
            if kind != "response":
                continue
            if block.get("type") == "tool_use":
                name = str(block.get("name") or "").rsplit("__", 1)[-1]
                stats["tool_calls"][name] = stats["tool_calls"].get(name, 0) + 1
            if block.get("type") == "metadata":
                for item in block.get("data") or []:
                    usage = item.get("usage") if isinstance(item, dict) else None
                    if not usage:
                        continue
                    fresh = usage.get("input_tokens") or 0
                    read = usage.get("cache_read_input_tokens") or 0
                    write = usage.get("cache_creation_input_tokens") or 0
                    stats["input_tokens"] += fresh
                    stats["cache_read_tokens"] += read
                    stats["cache_write_tokens"] += write
                    stats["output_tokens"] += usage.get("output_tokens") or 0
                    stats["peak_prompt_tokens"] = max(
                        stats["peak_prompt_tokens"], fresh + read + write
                    )
        patched = connection.execute(
            "SELECT 1 FROM message_patch_lookup WHERE call_label = ? AND origin = 'model' LIMIT 1",
            (label,),
        ).fetchone()
        stats["requests_with_model_patches"] += 1 if patched else 0
    price = PRICES.get(model)
    if price:
        stats["cost_usd"] = round(
            (
                stats["input_tokens"] * price["input"]
                + stats["output_tokens"] * price["output"]
                + stats["cache_write_tokens"] * price["cache_write"]
                + stats["cache_read_tokens"] * price["cache_read"]
            )
            / 1e6,
            4,
        )
    return stats


# -- one run ---------------------------------------------------------------------------


def _no_web(_tool_input: dict) -> "tuple[bool, str]":
    return False, "web access is disabled for this task; use only the document"


# The Oolong messages come from public labelled datasets (SMS Spam Collection, TREC):
# with web access a model can download the labels instead of classifying. So the
# sandbox has no network (--network=none) and the web tools are denied outright.
NO_WEB_POLICY = agpolicy(tool_hooks={"webfetch": _no_web, "websearch": _no_web})


def make_skill(arm: str, limit: int) -> agskill:
    prompt = TASK + (TIPS.format(limit=limit) if arm == "C" else "")
    return agskill(
        name=f"oolong_{arm}",
        prompt=prompt,
        input_schema=agdata(document=agrawstring, question=agrawstring),
        output_schema=agdata(answer=agrawstring),
        add_host_mcp_tools=context_tools() if arm in ("B", "C") else None,
        policy=NO_WEB_POLICY,
    )


def make_config(args, log_dir: Path) -> agconfig:
    llm = {"provider": "anthropic", "model": args.model, "context_limit": args.context_limit}
    if os.environ.get("ANTHROPIC_API_KEY"):
        llm["api_key"] = os.environ["ANTHROPIC_API_KEY"]
    return agconfig(
        llmconfig(**llm),
        agentconfig(harness=args.harness, log_dir=str(log_dir), output_dir=str(log_dir / "out")),
        sandboxconfig(base_image=args.image, flags=["--network=none"]),
        orchestratorconfig(
            max_concurrent_engines=args.concurrency, db_path=str(args.out / "agency.sqlite3")
        ),
    )


def run_one(args, row: dict, arm: str) -> dict:
    name = f"{arm}_{row['id']}"
    log_dir = args.out / "logs"
    cfg = make_config(args, log_dir)
    record = {
        "arm": arm,
        "id": row["id"],
        "context_len": row["context_len"],
        "dataset": row["dataset"],
        "task": row["task"],
        "answer_type": row["answer_type"],
        "gold": str(gold_value(row)),
    }
    started = time.monotonic()
    agent = None
    try:
        agent = Agent(name, agconfig=cfg)
        result = agent.run(
            make_skill(arm, args.context_limit),
            agdata(document=row["context_window_text"], question=row["question"]),
        )
        result.wait(timeout=args.timeout)
        output = result.to_dict()
        if "error" in output:
            record["error"] = str(output["error"])[:500]
            answer = ""
        else:
            answer = str(output.get("answer", ""))
        record["score"], record["predicted"] = score(row, answer)
        record["relative_error"] = relative_error(row, record["predicted"])
        record["answer_tail"] = answer[-300:]
    except Exception as exc:  # a failed run is a data point, not a crash
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        record["traceback"] = traceback.format_exc()[-1500:]
        record["score"], record["predicted"] = 0.0, ""
        record["relative_error"] = relative_error(row, "")
    finally:
        record["seconds"] = round(time.monotonic() - started, 1)
        if agent is not None:
            try:
                agent.data_logger.flush()
            except Exception as exc:
                record.setdefault("cleanup_errors", []).append(f"flush: {exc}")
            if agent.sandbox is not None:
                try:
                    agent.sandbox.destroy()
                except Exception as exc:
                    record.setdefault("cleanup_errors", []).append(f"destroy: {exc}")
                agent.sandbox = None
    db_name = f"agent_{name}_0000_data.sqlite3"
    matches = sorted(log_dir.glob(f"*{name}*_data.sqlite3")) or [log_dir / db_name]
    record.update(run_stats(matches[0], args.model))
    return record


# -- main ------------------------------------------------------------------------------


def summarize(records: "list[dict]") -> str:
    lines = [
        "| arm | runs | errors | score | rel. error | cost $/run | output tok/run | peak prompt | "
        "compactions/run | edit_context/run | requests w/ model edits |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in sorted({r["arm"] for r in records}):
        rows = [r for r in records if r["arm"] == arm]
        n = len(rows)

        def mean(key, rows=rows, n=n):
            return sum(r.get(key) or 0 for r in rows) / n

        edits = sum((r.get("tool_calls") or {}).get("edit_context", 0) for r in rows) / n
        numeric = [r["relative_error"] for r in rows if r.get("relative_error") is not None]
        rel = f"{sum(numeric) / len(numeric):.3f}" if numeric else "-"
        lines.append(
            f"| {arm} | {n} | {sum('error' in r for r in rows)} | {mean('score'):.3f} | {rel} | "
            f"{mean('cost_usd'):.3f} | {mean('output_tokens'):,.0f} | "
            f"{max(r.get('peak_prompt_tokens') or 0 for r in rows):,} | "
            f"{mean('compactions'):.1f} | {edits:.1f} | {mean('requests_with_model_patches'):.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4, help="questions (each runs once per arm)")
    parser.add_argument("--offset", type=int, default=0, help="skip this many shuffled questions")
    parser.add_argument("--ids", default="", help="comma-separated question ids (overrides --n)")
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--harness", default="native")
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", default="docker.io/library/python:3.12-slim")
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.data.expanduser().open()]
    if args.ids:
        wanted = {int(i) for i in args.ids.split(",")}
        rows = [row for row in rows if row["id"] in wanted]
    else:
        random.Random(args.seed).shuffle(rows)
        rows = rows[args.offset : args.offset + args.n]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    jobs = [(row, arm) for row in rows for arm in arms]
    print(f"{len(jobs)} runs: {len(rows)} questions x arms {arms}, budget {args.context_limit}")

    results_path = args.out / "results.jsonl"
    lock = threading.Lock()
    records: "list[dict]" = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, args, row, arm): (row["id"], arm) for row, arm in jobs}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
                with results_path.open("a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            print(
                f"[{len(records)}/{len(jobs)}] {record['arm']} q{record['id']} "
                f"score={record['score']:.2f} rel_err={record.get('relative_error')} cost=${record.get('cost_usd', 0):.3f} "
                f"edits={record['tool_calls'].get('edit_context', 0)} "
                f"compactions={record['compactions']} {record.get('error', '')[:120]}",
                flush=True,
            )
    summary = summarize(records)
    (args.out / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
