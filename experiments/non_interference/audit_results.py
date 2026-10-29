"""Audit all recorded runs without dropping failures or altering measured data."""

from pathlib import Path
from collections import Counter
import json
import sys

import fidelity


def read(path):
    return json.loads(path.read_text())


def main(root):
    config = read(root / "configuration.json")
    manifest = read(root / "task_manifest.json")
    frozen = {t["instance_id"]: t for t in manifest["tasks"]}
    latest = {}
    for path in (root / "attempts").glob("*/*/summary.json"):
        result = read(path)
        if (
            result["run_id"] not in latest
            or result["attempt"] > latest[result["run_id"]][0]["attempt"]
        ):
            latest[result["run_id"]] = (result, path.parent)
    runs = []
    response_ids = []
    cache_headers = []
    response_models = set()
    initial = {task_id: {arm: set() for arm in ("standalone", "agency")} for task_id in frozen}
    for result, directory in latest.values():
        item = {"run_id": result["run_id"], "status": result["status"], "arm": result["arm"]}
        calls = result.get("provider", {}).get("calls", [])
        item["requests_missing_completion_metadata"] = [
            path.name
            for path in (directory / "provider").glob("*.request.json")
            if not path.with_name(path.name.replace("request.json", "metadata.json")).exists()
        ]
        metadata_count = sum(
            read(path)["path"].split("?")[0].endswith("/messages")
            for path in (directory / "provider").glob("*.metadata.json")
        )
        item["summary_request_count_matches"] = metadata_count == result.get("provider", {}).get(
            "model_requests", 0
        )
        item["request_models_match"] = all(c.get("model") == config["model"] for c in calls)
        for call in calls:
            response_models.update(m for m in call.get("response_models", []) if m)
            headers = {
                k: v
                for k, v in call.get("headers", {}).items()
                if k in ("x-litellm-cache-key", "x-litellm-cache-hit")
            }
            if headers:
                cache_headers.append({"run_id": result["run_id"], "headers": headers})
        for path in (directory / "provider").glob("*.response.body"):
            with path.open(errors="replace") as source:
                for line in source:
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except ValueError:
                        continue
                    if event.get("type") == "message_start":
                        identifier = event.get("message", {}).get("id")
                        if identifier:
                            response_ids.append(identifier)
                        break
        resources = (
            read(directory / "resources.json") if (directory / "resources.json").exists() else {}
        )
        environment = (
            read(directory / "environment.json")
            if (directory / "environment.json").exists()
            else {}
        )
        item["resources_match"] = (
            resources.get("NanoCpus") == 2000000000
            and resources.get("Memory") == resources.get("MemorySwap") == 4294967296
        )
        item["starting_image_match"] = (
            environment.get("image_id") == config["images"][result["instance_id"]]["image_id"]
        )
        item["base_commit_match"] = (
            environment.get("base_commit") == frozen[result["instance_id"]]["base_commit"]
        )
        first = fidelity.first_task_request(directory)
        if first:
            initial[result["instance_id"]][result["arm"]].add(
                fidelity.digest(fidelity.normalized_initial_request(first))
            )
        if result["arm"] == "agency" and result.get("provider", {}).get("model_requests"):
            item["request_preservation"] = fidelity.preservation(directory)
            item["response_preservation"] = fidelity.response_preservation(directory)
        runs.append(item)
    initial_rows = {
        task_id: {
            "standalone_variants": len(arms["standalone"]),
            "agency_variants": len(arms["agency"]),
            "one_identical_initial_request": len(arms["standalone"] | arms["agency"]) == 1
            and all(arms.values()),
            "standalone_hashes": sorted(arms["standalone"]),
            "agency_hashes": sorted(arms["agency"]),
        }
        for task_id, arms in initial.items()
    }
    counts = Counter(response_ids)
    output = {
        "recorded_executions": len(runs),
        "runs": runs,
        "initial_requests": initial_rows,
        "request_model_mismatch_runs": [r["run_id"] for r in runs if not r["request_models_match"]],
        "recording_coverage_mismatch_runs": [
            r["run_id"]
            for r in runs
            if r["requests_missing_completion_metadata"] or not r["summary_request_count_matches"]
        ],
        "response_model_ids": sorted(response_models),
        "response_cache_evidence": {
            "response_messages": len(response_ids),
            "unique_message_ids": len(counts),
            "repeated_message_ids": {key: n for key, n in counts.items() if n > 1},
            "router_cache_headers": cache_headers,
            "limitation": "Absent hit headers and unique IDs are empirical evidence, not administrative proof of disabled response caching.",
        },
        "request_mismatch_runs": [
            r["run_id"]
            for r in runs
            if r.get("request_preservation") and not r["request_preservation"]["all_preserved"]
        ],
        "response_mismatch_runs": [
            r["run_id"]
            for r in runs
            if r.get("response_preservation") and not r["response_preservation"]["all_preserved"]
        ],
        "resource_or_starting_state_mismatch_runs": [
            r["run_id"]
            for r in runs
            if not all(
                r[k] for k in ("resources_match", "starting_image_match", "base_commit_match")
            )
        ],
        "scope": "Request JSON semantics and native beta/version headers; response message_start/message_delta control fields. Failed/cancelled streams may lack a native response. No runs are excluded on audit results.",
    }
    (root / "fidelity_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in ("runs", "initial_requests")}, indent=2
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent)
