"""Run diagnostics, the balanced cohort, official grading, and offline reporting on the target host."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

from settings import validate_host

ROOT = Path(__file__).resolve().parent


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, check=True)


def main(cohort=None, diagnostics_only=False, skip_analysis=False):
    validate_host(ROOT)
    config = ROOT / "configuration.json"
    if not config.exists():
        raise RuntimeError("Run fetch_dataset.py and prepare_ec2.py first")
    gate_path = ROOT / "diagnostic_gate.json"
    if cohort is not None or not gate_path.exists():
        cohort = cohort or "diagnostics-initial"
        if not cohort.startswith("diagnostics"):
            raise ValueError("Diagnostic cohort names must begin with diagnostics")
        run("run_experiment.py", cohort)
        run("grade.py", cohort)
        (ROOT / "preparation_complete.json").write_text(
            json.dumps(
                {
                    "configuration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "validated_unix_s": time.time(),
                    "diagnostic_cohort": cohort,
                },
                indent=2,
            )
            + "\n"
        )
        run("check_diagnostics.py", cohort)
    gate = json.loads(gate_path.read_text())
    if not gate["passed"]:
        raise RuntimeError("Diagnostic gate did not pass")
    for name, expected in gate["script_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Infrastructure changed after diagnostics; use a fresh --cohort")
    if diagnostics_only:
        return
    run("run_experiment.py", "measured")
    run("grade.py", "batch")
    results = [json.loads(path.read_text()) for path in ROOT.glob("attempts/*/*/summary.json")]
    latest = {}
    for result in results:
        if (
            result["run_id"] not in latest
            or result["attempt"] > latest[result["run_id"]]["attempt"]
        ):
            latest[result["run_id"]] = result
    if len(latest) != json.loads(config.read_text())["executions"]:
        raise RuntimeError("Measured cohort is incomplete; no final report will be claimed")
    for result in latest.values():
        report = json.loads((ROOT / result["directory"] / "official-report.json").read_text())
        if not isinstance(report.get("resolved"), bool) or report.get("grading_error"):
            raise RuntimeError("Evaluator failure requires an audit; preserve the original report")
    (ROOT / "grading_complete.json").write_text(json.dumps({"unix_s": time.time()}) + "\n")
    scripts = (
        (
            "analysis.py",
            "audit_results.py",
            "report.py",
            "analysis_secondary.py",
            "export_results.py",
            "validate_results.py",
        )
        if not skip_analysis
        else ("export_results.py",)
    )
    for script in scripts:
        run(script)
    (ROOT / "pipeline_complete.json").write_text(
        json.dumps(
            {
                "unix_s": time.time(),
                "executions": len(latest),
                "analysis_complete": not skip_analysis,
                "evidence_archive_ready": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", help="Fresh diagnostic label after an infrastructure change")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Export after grading; analyze later in a pinned local environment",
    )
    arguments = parser.parse_args()
    main(arguments.cohort, arguments.diagnostics_only, arguments.skip_analysis)
