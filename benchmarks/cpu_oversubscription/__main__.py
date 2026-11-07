"""CLI deliberately has no command that runs both conditions."""

import argparse
import json
import sys
from pathlib import Path

from .config import digest, load_config, safe_json, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "run"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        if name == "plan":
            cmd.add_argument(
                "--topology", type=Path, help="Saved JSON rows for offline CPU inspection"
            )
        if name == "run":
            cmd.add_argument("--out", type=Path, required=True)
            cmd.add_argument("--trial-id", required=True)
    commands.add_parser(
        "isolation-script",
        help="Print runtime-only system changes for operator review; never apply them",
    )
    cmd = commands.add_parser("fetch")
    cmd.add_argument("--out", type=Path, required=True)
    cmd.add_argument("--revision", required=True)
    cmd = commands.add_parser("select")
    cmd.add_argument("--dataset-dir", type=Path, required=True)
    cmd.add_argument(
        "--images",
        type=Path,
        help="Omit to generate an image-map template with real sampled instance IDs",
    )
    cmd.add_argument("--repo", required=True)
    cmd.add_argument("--version", required=True)
    cmd.add_argument("--seed", type=int, default=1729)
    cmd.add_argument("--replicated", action="store_true")
    cmd.add_argument("--out", type=Path, required=True)
    for name in ("evaluate", "recover"):
        cmd = commands.add_parser(name)
        cmd.add_argument("directory", type=Path)
    cmd = commands.add_parser("compare")
    cmd.add_argument("directories", nargs="+", type=Path)
    cmd.add_argument("--out", type=Path, required=True)
    for name in ("_worker", "_watchdog"):
        cmd = commands.add_parser(name, help=argparse.SUPPRESS)
        cmd.add_argument("directory", type=Path)
        cmd.add_argument("number", type=int)
    args = parser.parse_args()
    if args.command in ("_worker", "_watchdog"):
        from .linux import require_linux

        require_linux()
        try:
            if args.command == "_worker":
                from .backend import worker

                worker(args.directory, args.number)
            else:
                from .runner import watchdog

                watchdog(args.directory, args.number)
        except BaseException as error:
            safe_json(
                args.directory / f"{args.command}-error-{args.number}.json",
                {"error": type(error).__name__, "message": str(error)},
            )
            return 1
        return 0
    if args.command in ("plan", "preflight", "run"):
        cfg = load_config(args.config)
        manifest = validate_manifest(json.loads(Path(cfg["manifest"]).read_text()))
        if args.command == "plan":
            result = {
                "condition": cfg["condition"],
                "assigned": 30,
                "agents": 6,
                "condition_timeout_seconds": cfg["condition_timeout_seconds"],
                "manifest_hash": digest(manifest),
                "model": cfg["model"],
                "note": "Offline plan only; use Linux preflight to verify actual affinity",
            }
            if args.topology:
                from .linux import layout_from_rows

                result["layout"] = layout_from_rows(
                    json.loads(args.topology.read_text()), cfg["condition"]
                )
        elif args.command == "preflight":
            from .linux import experiment_lock, preflight

            with experiment_lock():
                layout, provenance = preflight(cfg, manifest)
                result = {
                    "layout": layout,
                    "provenance": provenance,
                    "note": "No agents/model calls started; image execution verified during run preparation",
                }
        else:
            from .runner import run

            result = run(cfg, args.out, args.trial_id)
    elif args.command == "isolation-script":
        from .linux import isolation_script, topology

        print(isolation_script(topology("DCB")))
        return 0
    elif args.command == "fetch":
        from .workload import fetch

        fetch(args.out, args.revision)
        result = {"dataset_dir": str(args.out)}
    elif args.command == "select":
        from .workload import select

        result = select(
            args.dataset_dir,
            args.images,
            args.repo,
            args.version,
            args.seed,
            args.replicated,
            args.out,
        )
    elif args.command in ("evaluate", "compare"):
        from .report import compare, evaluate

        result = (
            evaluate(args.directory)
            if args.command == "evaluate"
            else compare(args.directories, args.out)
        )
    else:
        from .runner import recover

        result = recover(args.directory)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        # Validation errors contain no transport secrets. Runtime exceptions may,
        # so expose only their type unless they originate from explicit guards.
        if isinstance(error, (ValueError, RuntimeError)):
            from benchmarks.tool_annotation_effect.common import native

            print(native("annotations").redact(str(error)), file=sys.stderr)
        else:
            print(type(error).__name__ + "; inspect the private run artifacts", file=sys.stderr)
        sys.exit(1)
