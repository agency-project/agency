"""Explicit commands keep planning/analysis separate from live side effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native tool annotation experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Offline validation and immutable randomized assignment"
    )
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    prepare = commands.add_parser(
        "prepare", help="Explicit dataset preparation (network and filesystem writes)"
    )
    prepare.add_argument("--suite", choices=("swebench", "terminalbench"), required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--lite", action="store_true")
    run = commands.add_parser("run", help="Explicit model-backed execution in fresh environments")
    run.add_argument("--plan", required=True)
    run.add_argument("--resume-partial", action="store_true")
    run.add_argument(
        "--limit",
        type=int,
        help="Stop after this many new assignments; remaining trials stay missing",
    )
    evaluate = commands.add_parser("evaluate", help="Explicit official/fixture verification")
    evaluate.add_argument("--plan", required=True)
    analyze = commands.add_parser("analyze", help="Offline analysis of saved artifacts")
    analyze.add_argument("--plan", required=True)
    analyze.add_argument("--pricing", help="Explicit versioned pricing JSON; no rates are inferred")
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze.add_argument(
        "--tokenizer-file",
        help="Optional local tokenizer JSON (tokenizers library); never downloaded",
    )
    review = commands.add_parser("review-export", help="Offline blinded human review packet")
    review.add_argument("--plan", required=True)
    review.add_argument("--output", required=True)
    power = commands.add_parser("power", help="Approximate task-cluster sample-size planning")
    power.add_argument("--margin", type=float, required=True)
    power.add_argument("--task-difference-sd", type=float)
    power.add_argument(
        "--pilot-analysis", help="Use saved task SD for an explicit suite/model/contrast"
    )
    power.add_argument("--suite")
    power.add_argument("--model-id")
    power.add_argument("--active", default="purpose")
    power.add_argument("--control", default="schema_only")
    power.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.command == "plan":
        from .planning import save_plan

        manifest = save_plan(read_json(args.config), args.output)
        print(
            json.dumps(
                {"run_counts": manifest["run_counts"], "prerequisites": manifest["prerequisites"]},
                indent=2,
            )
        )
    elif args.command == "prepare":
        from .adapters import prepare

        print(prepare(args.suite, args.output, lite=args.lite))
    elif args.command == "run":
        from .runner import run

        print(
            json.dumps(
                {
                    "new_assignments": run(
                        args.plan, resume_partial=args.resume_partial, limit=args.limit
                    )
                }
            )
        )
    elif args.command == "evaluate":
        from .runner import evaluate

        evaluate(args.plan)
    elif args.command in ("analyze", "review-export"):
        from .runner import rows
        from .analysis import export_report, review_export

        manifest, records = rows(args.plan)
        if args.command == "analyze":
            export_report(
                args.plan,
                manifest,
                records,
                bootstrap_samples=args.bootstrap_samples,
                pricing=read_json(args.pricing) if args.pricing else None,
                tokenizer_file=args.tokenizer_file,
            )
            print(Path(args.plan) / "report.md")
        else:
            review_export(args.output, records, seed=manifest["config"].get("seed", 0))
            print(Path(args.output) / "raw-blinded.json")
    else:
        from .analysis import required_tasks

        sd = args.task_difference_sd
        if args.pilot_analysis:
            saved = read_json(args.pilot_analysis)
            matches = [
                contrast
                for group in saved["groups"]
                if group["suite"] == args.suite and group["model_id"] == args.model_id
                for contrast in group["contrasts"]
                if contrast["active"] == args.active and contrast["control"] == args.control
            ]
            if len(matches) != 1 or matches[0]["task_difference_sd"] is None:
                parser.error(
                    "Pilot SD unavailable; specify suite/model/contrast with at least two task clusters"
                )
            sd = matches[0]["task_difference_sd"]
        if sd is None:
            parser.error("Supply hypothetical --task-difference-sd or a saved pilot analysis")
        print(
            json.dumps(
                {
                    "approximate_required_task_clusters": required_tasks(
                        margin=args.margin, task_difference_sd=sd, power=args.power
                    ),
                    "equivalence_claim": False,
                }
            )
        )


if __name__ == "__main__":
    main()
