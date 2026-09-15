"""Run two real Codex invocations across a filesystem-only hibernation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import agency

from benchmarks.checkpoint_size_microbenchmark.runner import configure


SYSTEM = """Execute only the exact functions.exec JavaScript supplied in the input. Do not change the command. After it succeeds, copy its complete output verbatim into summary and finish."""


def prompt(marker: str, create: bool):
    operation = (
        "printf persisted-session >/testbed/.agency-cow-pty-state; "
        if create
        else 'test "$(cat /testbed/.agency-cow-pty-state)" = persisted-session; '
    )
    command = (
        operation + f"echo RUN={marker}; " + "ps -eo pid,ppid,sid,tty,comm,args | grep '[c]odex'"
    )
    return (
        "Submit this exact JavaScript:\n\n"
        "let result = await tools.exec_command({"
        + "cmd: "
        + json.dumps(command)
        + ", yield_time_ms: 30000, max_output_tokens: 2000}); text(result.output);"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = configure(args.root, args.key_file)
    cfg.agent.log_dir = str(args.output / "logs")
    learner = agency.Agent("cow_new_pty_validation", agconfig=cfg)
    skill = agency.agskill(
        name="cow_new_pty_validation",
        prompt=SYSTEM,
        input_schema=agency.agdata(instruction=str),
        output_schema=agency.agdata(summary=agency.agrawstring),
    )
    records = []
    try:
        with agency.agprof.session(args.output / "profile", sample_hz=5, sample_gpu=False):
            for marker, create in (("one", True), ("two", False)):
                result = learner.run(skill, agency.agdata(instruction=prompt(marker, create)))
                result.wait()
                payload = result.to_dict()
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                checkpoint = learner.sandbox._backend._checkpointer.latest
                records.append(
                    {
                        "run": marker,
                        "result": payload,
                        "checkpoint": checkpoint.reference,
                        "generation": checkpoint.stats["checkpoint_generation"],
                    }
                )
    finally:
        learner.sandbox.destroy()
        agency.get_orchestrator().shutdown()
    args.output.joinpath("result.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
