"""The frozen task manifest and the eligibility rules behind it."""

import copy
import json

import pytest

from benchmarks.terminalbench_contention import corpus


def dataset_task(directory="alpha", **changes):
    task = dict(
        valid=True,
        directory=directory,
        name=f"terminal-bench/{directory}",
        digest="sha256:" + "a" * 64,
        instruction_sha256="b" * 64,
        environment_sha256="c" * 64,
        tests_sha256="d" * 64,
        metadata=dict(category="Software"),
        steps=0,
        environment=dict(
            cpus=2,
            memory_mb=4096,
            storage_mb=10240,
            gpus=0,
            os="linux",
            docker_image="org/img:tag@sha256:" + "1" * 64,
            compose_files=[],
            mcp_servers=[],
            skills_dir=None,
            healthcheck=False,
            network_mode="public",
        ),
        agent=dict(timeout_sec=28800.0, user=None),
        verifier=dict(
            timeout_sec=300.0,
            environment_mode="separate",
            env_keys=[],
            collect_hooks=0,
            test_script=True,
            separate_definition=dict(
                docker_image="org/ver:tag@sha256:" + "2" * 64,
                cpus=2,
                memory_mb=4096,
                bundled_tests=True,
            ),
        ),
        artifacts=["/app/out"],
    )
    for dotted, value in changes.items():
        node = task
        parts = dotted.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return task


def test_frozen_manifest_is_valid_and_complete():
    manifest = corpus.load_manifest()
    tasks = manifest["tasks"]
    assert manifest["task_count"] == len(tasks) == 24
    assert manifest["dataset"] == "terminal-bench/terminal-bench@4.0.0"
    assert len({t["task_id"] for t in tasks}) == 24
    assert {t["directory"] for t in tasks} == set(corpus.SELECTED)
    for task in tasks:
        assert task["task_id"].startswith("terminal-bench/") and task["version"] == "4.0.0"
        assert task["digest"].startswith("sha256:") and len(task["digest"]) == 71
        for kind in ("environment", "verifier"):
            assert task[kind]["image"].startswith(
                "docker.io/harborframework/terminal-bench@sha256:"
            )
        assert task["environment"]["gpus"] == 0 and not task["environment"]["privileged"]
        assert task["environment"]["memory_mb"] <= corpus.MAX_TASK_MEMORY_MB
        assert task["environment"]["cpus"] <= corpus.MAX_TASK_CPUS
        assert task["environment"]["external_services"] == []
        assert task["verifier"]["mode"] == "separate" and task["artifacts"]
        assert task["expected_resources"]["workload"] in corpus.WORKLOADS
        assert task["selection_rationale"] and task["language"]


def test_corpus_is_a_mixture_not_a_cpu_heavy_selection():
    manifest = corpus.load_manifest()
    workload = manifest["composition"]["by_workload"]
    assert sum(workload.values()) == 24 and all(workload[w] >= 4 for w in corpus.WORKLOADS)
    assert workload["computation-heavy"] <= 12
    assert len(manifest["composition"]["by_category"]) >= 6
    assert len({t["language"] for t in manifest["tasks"]}) >= 8


def test_manifest_documents_every_other_dataset_task():
    selection = corpus.load_manifest()["selection"]
    assert len(selection["excluded"]) + len(selection["eligible_not_selected"]) + 24 == 66
    assert all(entry["reasons"] for entry in selection["excluded"])
    reasons = " ".join(r for entry in selection["excluded"] for r in entry["reasons"])
    for phrase in ("requires a GPU", "multi-container", "collect hooks"):
        assert phrase in reasons


def test_tampering_with_the_manifest_is_detected(tmp_path):
    manifest = json.loads(corpus.TASKS_MANIFEST.read_text())
    path = tmp_path / "tasks.json"
    for mutate in (
        lambda m: m["tasks"][0].__setitem__("digest", "sha256:" + "0" * 64),
        lambda m: m["tasks"][3]["environment"].__setitem__(
            "image", "docker.io/harborframework/terminal-bench@sha256:" + "f" * 64
        ),
        lambda m: m["tasks"].pop(),
        lambda m: m["tasks"].append(copy.deepcopy(m["tasks"][0])),
    ):
        changed = copy.deepcopy(manifest)
        mutate(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(corpus.ManifestError):
            corpus.load_manifest(path)
    floating = copy.deepcopy(manifest)
    floating["tasks"][0]["environment"]["image"] = "docker.io/org/img:latest"
    floating["corpus_sha256"] = corpus.sha256_json(
        [
            (t["task_id"], t["digest"], t["environment"]["image"], t["verifier"]["image"])
            for t in floating["tasks"]
        ]
    )
    path.write_text(json.dumps(floating))
    with pytest.raises(corpus.ManifestError, match="not digest-pinned"):
        corpus.load_manifest(path)
    path.write_text("{")
    with pytest.raises(corpus.ManifestError):
        corpus.load_manifest(path)


@pytest.mark.parametrize(
    "change,phrase",
    [
        ({"environment__gpus": 1}, "GPU"),
        ({"environment__compose_files": ["docker-compose.yaml"]}, "multi-container"),
        ({"verifier__collect_hooks": 1}, "collect hooks"),
        ({"environment__mcp_servers": ["x"]}, "MCP"),
        ({"environment__memory_mb": 16384}, "memory"),
        ({"environment__cpus": 8}, "CPUs"),
        ({"environment__docker_image": "org/img:latest"}, "digest-pinned"),
        ({"environment__docker_image": None}, "digest-pinned"),
        ({"verifier__environment_mode": "shared"}, "separate-mode"),
        ({"verifier__env_keys": ["OPENAI_API_KEY"]}, "environment variables"),
        ({"agent__user": "agent"}, "non-root"),
        ({"environment__network_mode": "allowlist"}, "network policy"),
        ({"steps": 2}, "multi-step"),
        ({"environment__healthcheck": True}, "healthcheck"),
    ],
)
def test_eligibility_rules(change, phrase):
    assert corpus.exclusion_reasons(dataset_task()) == []
    reasons = corpus.exclusion_reasons(dataset_task(**change))
    assert any(phrase in reason for reason in reasons), reasons


def test_images_without_python_on_the_harness_path_are_excluded():
    """Agency's Codex hooks run `python3`; an image without it cannot acknowledge prompts."""
    assert corpus.exclusion_reasons(dataset_task(), None, {"alpha": True}) == []
    assert corpus.exclusion_reasons(dataset_task(), None, {"alpha": None}) == []  # not inspected
    reasons = corpus.exclusion_reasons(dataset_task(), None, {"alpha": False})
    assert len(reasons) == 1 and "python3" in reasons[0] and corpus.HARNESS_PATH in reasons[0]
    excluded = {e["task_id"]: e["reasons"] for e in corpus.load_manifest()["selection"]["excluded"]}
    for name in ("gsea-proteomics", "react-lead-form"):
        assert any("python3" in reason for reason in excluded[f"terminal-bench/{name}"])
        assert name not in corpus.SELECTED


def test_non_root_image_user_and_unparsable_tasks_are_excluded():
    assert any(
        "non-root" in r for r in corpus.exclusion_reasons(dataset_task(), {"alpha": "agent"})
    )
    assert corpus.exclusion_reasons(dataset_task(), {"alpha": ""}) == []
    assert corpus.exclusion_reasons(dict(valid=False, error="boom"))[0].startswith("Harbor")


def test_pinned_reference_drops_the_tag_and_keeps_the_digest():
    digest = "sha256:" + "9" * 64
    assert corpus.pinned_reference(f"harborframework/terminal-bench:x-env-1@{digest}") == (
        f"docker.io/harborframework/terminal-bench@{digest}"
    )
    assert corpus.pinned_reference(f"ghcr.io/org/img:tag@{digest}") == f"ghcr.io/org/img@{digest}"


def test_verify_against_dataset_reports_drift():
    manifest = corpus.load_manifest()
    dataset = dict(
        tasks=[
            dataset_task(
                t["directory"],
                digest=t["digest"],
                instruction_sha256=t["instruction_sha256"],
                environment_sha256=t["environment_sha256"],
                tests_sha256=t["tests_sha256"],
            )
            for t in manifest["tasks"]
        ]
    )
    assert corpus.verify_against_dataset(manifest, dataset) == []
    dataset["tasks"][0]["digest"] = "sha256:" + "0" * 64
    dataset["tasks"][1]["environment"]["gpus"] = 1
    removed = dataset["tasks"].pop()
    problems = corpus.verify_against_dataset(manifest, dataset)
    assert len(problems) == 3 and any(removed["name"] in p for p in problems)
