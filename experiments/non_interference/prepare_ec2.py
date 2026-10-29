"""Read EC2 identity, pin cached images/pricing, and verify the Haiku endpoint."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import urllib.request

from worker import credential, docker, write, ROOT, MODEL
from settings import DEFAULT_ROUTER, host_settings, validate_host


def catalog(route):
    request = urllib.request.Request(
        host_settings(ROOT).get("router", DEFAULT_ROUTER).rstrip("/") + route,
        headers={
            "Authorization": "Bearer " + credential(),
            "User-Agent": "agency-non-interference/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def main(pull_images=False):
    if (ROOT / "configuration.json").exists():
        raise SystemExit("Study already prepared; refusing to change its execution pins")
    execution_host, identity = validate_host(ROOT)
    host = host_settings(ROOT, required=True)
    for field in ("runtime", "grader", "claude_binary"):
        if not Path(host[field]).is_absolute() or not Path(host[field]).exists():
            raise RuntimeError("Supply an existing absolute host path for " + field)
    for name in ("agency-python", "agency-harness-venv"):
        assert (Path(host["runtime"]) / name).is_dir(), "Missing container runtime: " + name
    assert (ROOT / "src/agency").is_dir(), "Stage the frozen Agency source under src/ first"
    manifest = json.loads((ROOT / "task_manifest.json").read_text())
    freeze = json.loads((ROOT / "freeze_metadata.json").read_text())
    assert (
        hashlib.sha256((ROOT / "task_manifest.json").read_bytes()).hexdigest()
        == freeze["manifest_sha256"]
    )
    info = next(row for row in catalog("/model/info")["data"] if row["model_name"] == MODEL)[
        "model_info"
    ]
    pricing = {
        k: v
        for k, v in info.items()
        if v is not None
        and (
            "cost" in k
            or k
            in (
                "key",
                "litellm_provider",
                "max_input_tokens",
                "max_output_tokens",
                "id",
                "tpm",
                "rpm",
            )
        )
    }
    write(
        ROOT / "pricing.json",
        {
            "model": MODEL,
            "source": "LiteLLM /model/info",
            "rates": pricing,
            "units": "USD per token",
            "reference_policy": "cache-neutral: all uncached+cache-read+cache-write input at base input rate; output at output rate; >100k prompt threshold applied per request",
            "public_pricing_source": "https://platform.claude.com/docs/en/models/haiku-5-5/overview",
            "billing_note": "Router prices are account-specific. x-litellm-response-cost is router-reported, not an audited provider invoice.",
        },
    )
    images = {}
    image_pins = json.loads((ROOT / "image_pins.json").read_text())
    for task in manifest["tasks"]:
        name = image_pins[task["instance_id"]]["reference"]
        if pull_images:
            docker("pull", name, timeout=1800)
        image = json.loads(docker("image", "inspect", name))[0]
        images[task["instance_id"]] = {
            "source_tag": name,
            "image_id": image["Id"],
            "repo_digests": image.get("RepoDigests", []),
        }
    import pyarrow.parquet as pq

    dataset = ROOT / "evidence/swe-bench-verified.parquet"
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest["source_sha256"]
    rows = pq.read_table(dataset).to_pylist()
    ordered = [
        next(row for row in rows if row["instance_id"] == t["instance_id"])
        for t in manifest["tasks"]
    ]
    for row, frozen_task in zip(ordered, manifest["tasks"]):
        canonical = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == frozen_task["dataset_row_sha256"]
    write(
        ROOT / "public_tasks.json",
        [
            {
                k: row[k]
                for k in ("instance_id", "repo", "base_commit", "problem_statement", "version")
            }
            for row in ordered
        ],
    )
    (ROOT / "private").mkdir(exist_ok=True)
    write(ROOT / "private/tasks.json", ordered)
    (ROOT / "binary").mkdir(exist_ok=True)
    binary_source = Path(host["claude_binary"])
    assert (
        hashlib.sha256(binary_source.read_bytes()).hexdigest()
        == "2d894a4b1bfaaa9aab72aeff7803e3c73bc2510f279a171f6b8f59f72d17d077"
    ), "Claude binary differs from the frozen version"
    shutil.copy2(binary_source, ROOT / "binary/2.1.299")
    environment = {
        "ec2": {
            k: identity.get(k)
            for k in (
                "instanceId",
                "instanceType",
                "region",
                "imageId",
                "availabilityZone",
                "architecture",
            )
        },
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "kernel": platform.release(),
        "os_release": Path("/etc/os-release").read_text(),
        "meminfo": Path("/proc/meminfo").read_text(),
        "docker": docker("--version").strip(),
        "agency_commit": freeze["agency_source_commit"],
        "agency_version": "0.1.0",
        "claude_code_version": subprocess.check_output(
            [str(ROOT / "binary/2.1.299"), "--version"], text=True
        ).strip(),
        "claude_binary_sha256": hashlib.sha256((ROOT / "binary/2.1.299").read_bytes()).hexdigest(),
        "model": MODEL,
        "router_upstream_model": info.get("key"),
        "source_files": {
            str(p.relative_to(ROOT / "src")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/agency").rglob("*")
            if p.is_file()
        },
        "packages": {
            x: importlib.metadata.version(x) for x in ("anthropic", "httpx", "numpy", "pyarrow")
        },
        "execution_host": execution_host,
    }
    grader_source = Path(host["grader"])
    assert (grader_source / "swebench/harness/run_evaluation.py").is_file(), (
        "Missing official evaluator source"
    )
    write(
        ROOT / "grader_metadata.json",
        {
            "source_commit": None,
            "source_files": {
                str(path.relative_to(grader_source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (grader_source / "swebench").rglob("*.py")
            },
        },
    )
    write(ROOT / "ec2_environment.json", environment)
    bridge = json.loads(docker("network", "inspect", "bridge"))[0]
    config = {
        "model": MODEL,
        "router": host.get("router", DEFAULT_ROUTER).rstrip("/"),
        "routing": "LiteLLM for both configurations",
        "execution_host": execution_host,
        "manifest_sha256": freeze["manifest_sha256"],
        "freeze_commit": freeze["freeze_commit"],
        "repetitions": 10,
        "tasks": 5,
        "executions": 100,
        "max_concurrency": 4,
        "workers_per_arm": 2,
        "cpu_per_execution": 2,
        "cpu_total_max": 8,
        "memory_per_execution_bytes": 4294967296,
        "memory_headroom_required_bytes": 20 * 1024**3,
        "timeout_s": 900,
        "max_turns": None,
        "schedule_seed": 2026102801,
        "bootstrap_seed": 2026102802,
        "bootstrap_resamples": 20000,
        "model_settings": {
            "effort": "medium",
            "max_output_tokens": 32000,
            "temperature": None,
            "autocompact": 900000,
            "subagents": False,
        },
        "interventions": {
            x: False
            for x in (
                "batching",
                "context_gc",
                "semantic_compression",
                "search_substitution",
                "dependency_scoping",
                "schema_offload",
                "tool_offload",
                "session_title_suppression",
                "approximate_local_token_counter",
            )
        },
        "native_compaction": "Claude Code's own compaction kept identical in both configurations; Agency context edits disabled",
        "prompt": "Same rendered Agency skill wrapper in both arms; session path/device/session IDs necessarily differ",
        "cache_policy": "Preserve native cache_control identically; request HTTP Cache-Control:no-cache from LiteLLM, whose response-cache policy is not administratively verified. Provider prompt caches remain shared; record all usage. Primary: cache-neutral reference cost.",
        "admission": {
            "requests_per_minute": 60,
            "tokens_per_minute": 3000000,
            "max_inflight": 4,
            "reservation": "UTF-8 request bytes plus max_tokens, sliding 60s window in process-shared SQLite; no refund; text-only pilot",
        },
        "retry_policy": "Claude Code owns provider retries identically. Recorder and Agency SDK/host do not retry. One additional attempt only for verified pre-inference infrastructure failure with zero recorded LLM requests. Preserve all attempts. No retries for agent failures, provider failures, or timeouts.",
        "runtime": host["runtime"],
        "docker_gateway": bridge["IPAM"]["Config"][0]["Gateway"],
        "images": images,
        "grader": host["grader"],
        "model_pin_limit": "Provider exposes undated model identifier only; account catalog ID and upstream model recorded; immutable provider weights cannot be independently verified.",
    }
    write(ROOT / "configuration.json", config)
    print(
        json.dumps(
            {"ready": True, "model": MODEL, "ec2": environment["ec2"], "images": len(images)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pull-images", action="store_true", help="Pull immutable image digests before preparation"
    )
    main(parser.parse_args().pull_images)
