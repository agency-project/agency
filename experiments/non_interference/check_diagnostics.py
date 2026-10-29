"""Require native request fidelity, matching starting prompts, usage, and resource limits."""

from pathlib import Path
import difflib
import hashlib
import json
import sys
import time

import fidelity

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main(prefix):
    assert (ROOT / "preparation_complete.json").exists(), "Final setup validation has not completed"
    tasks = read(ROOT / "task_manifest.json")["tasks"][:2]
    checks = []
    for i, task in enumerate(tasks):
        directories = {}
        auxiliary = {}
        for arm in ("standalone", "agency"):
            runs = sorted(
                (ROOT / "diagnostics" / f"{prefix}-t{i + 1}-{arm}").glob("*/summary.json")
            )
            assert runs, f"Diagnostic missing: {arm}, {task['instance_id']}"
            result = read(runs[-1])
            directory = runs[-1].parent
            directories[arm] = directory
            assert result["status"] in ("completed", "failed", "timeout"), result.get("error")
            assert result["provider"]["model_requests"] > 0
            calls = result["provider"]["calls"]
            titles = [
                value
                for _, value in fidelity.requests(directory)
                if not value.get("tools")
                and "The session content is provided inside <session> tags." in json.dumps(value)
            ]
            assert titles, "Session-title model calls must not be suppressed in either arm"
            auxiliary[arm] = fidelity.normalized_initial_request(titles[0])
            assert all(c["model"] == "claude-haiku-5-5" for c in calls)
            assert sum(c["usage_valid"] for c in calls) > 0
            assert (directory / "baseline-import.txt").exists()
            report = read(directory / "official-report.json")
            assert report.get("resolved") is not None and not report.get("grading_error"), report
            if arm == "agency":
                preserved = fidelity.preservation(directory)
                response = fidelity.response_preservation(directory)
                checks.append(
                    {
                        "task": task["instance_id"],
                        "request_fidelity": preserved,
                        "response_fidelity": response,
                    }
                )
                assert preserved["all_preserved"], preserved
                assert response["all_preserved"], response
        s, a = directories["standalone"], directories["agency"]
        assert read(s / "resources.json") == read(a / "resources.json")
        assert read(s / "environment.json") == read(a / "environment.json")
        assert (s / "submitted-prompt.txt").read_bytes() == (
            a / "submitted-prompt.txt"
        ).read_bytes()
        assert auxiliary["standalone"] == auxiliary["agency"], "Auxiliary title settings differ"
        initial = [
            fidelity.normalized_initial_request(fidelity.first_task_request(d)) for d in (s, a)
        ]
        if initial[0] != initial[1]:
            diff = "\n".join(
                difflib.unified_diff(
                    json.dumps(initial[0], indent=2, sort_keys=True).splitlines(),
                    json.dumps(initial[1], indent=2, sort_keys=True).splitlines(),
                    fromfile="standalone",
                    tofile="agency",
                )
            )
            (ROOT / f"diagnostic-initial-{i + 1}.diff").write_text(diff)
            raise AssertionError(
                "Initial model-visible settings differ; inspect diagnostic-initial diff"
            )
    samples = [
        json.loads(line) for line in (ROOT / "resource_samples.jsonl").read_text().splitlines()
    ]
    assert samples and max(len(s["containers"]) for s in samples) == 4
    assert max(s["cpu_quota_total"] for s in samples) <= 8
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and (
            p.suffix == ".py"
            or str(p.relative_to(ROOT)) in ("support/harness-python", "launcher/claude")
        )
        and not any(
            x in p.parts for x in ("src", "diagnostics", "attempts", "private", "binary", "staging")
        )
    ]
    value = {
        "passed": True,
        "checked_unix_s": time.time(),
        "cohort": prefix,
        "checks": checks,
        "max_observed_containers": max(len(s["containers"]) for s in samples),
        "max_observed_cpu_quota": max(s["cpu_quota_total"] for s in samples),
        "script_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "caveat": "Provider failures remain possible and are retained. Passing gate requires usage collection and native fidelity, never task success or favorable costs.",
    }
    (ROOT / "diagnostic_gate.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "diagnostics-v2")
