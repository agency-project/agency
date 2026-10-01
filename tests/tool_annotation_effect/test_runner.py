import copy
import json

import pytest

from benchmarks.tool_annotation_effect.common import atomic_json, read_json
from benchmarks.tool_annotation_effect.planning import build_plan, save_plan, load_plan, validate
from benchmarks.tool_annotation_effect.runner import run, evaluate, rows


def config():
    return {
        "profile": "custom",
        "harness": "native",
        "seed": 17,
        "task_sampling_seed": 3,
        "repetitions": 2,
        "models": [
            {
                "id": "fake",
                "provider": "openai",
                "model": "fake",
                "base_url_env": "TEST_URL",
                "api_key_env": "TEST_KEY",
            }
        ],
        "budgets": {"max_steps": 5, "timeout_s": 30},
        "context_limit": None,
        "cache_policy": "unknown",
        "suites": {"rag": {"sample_size": 2}},
    }


def fake_executor(task, trial, model, config, directory):
    return {
        "final_text": json.dumps({"answer": task["gold"], "evidence": task["evidence"]}),
        "events": [],
        "failure": None,
        "metrics": {"input_tokens": None, "output_tokens": None},
    }


def test_schedule_is_deterministic_and_each_block_contains_all_arms():
    a, b = build_plan(config()), build_plan(config())
    assert a["schedule"] == b["schedule"]
    assert len(a["schedule"]) == 16
    blocks = {}
    for trial in a["schedule"]:
        blocks.setdefault(trial["block_id"], []).append(trial["arm"])
    assert all(
        set(arms) == {"baseline", "schema_only", "purpose", "purpose_workstreams"}
        for arms in blocks.values()
    )
    changed = config()
    changed["seed"] += 1
    assert a["schedule"] != build_plan(changed)["schedule"]


def test_manifest_is_immutable_and_tampering_is_detected(tmp_path):
    save_plan(config(), tmp_path)
    save_plan(config(), tmp_path)
    manifest = read_json(tmp_path / "manifest.json")
    manifest["config"]["seed"] = 1
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="modified"):
        load_plan(tmp_path)


def test_completed_trials_are_not_overwritten_and_evaluation_is_idempotent(tmp_path):
    save_plan(config(), tmp_path)
    assert run(tmp_path, executor=fake_executor) == 16
    assert run(tmp_path, executor=lambda *args: pytest.fail("must not rerun")) == 0
    evaluate(tmp_path)
    evaluate(tmp_path, evaluator=lambda *args: pytest.fail("must not reevaluate"))
    manifest, records = rows(tmp_path)
    assert len(records) == len(manifest["schedule"])
    assert all(row["success"] for row in records)
    assert all(row["input_tokens"] is None for row in records)


def test_partial_recovery_preserves_previous_attempt(tmp_path):
    manifest = save_plan(config(), tmp_path)
    trial_dir = tmp_path / "trials" / manifest["schedule"][0]["trial_id"]
    first = trial_dir / "attempt-001"
    first.mkdir(parents=True)
    status = {"state": "running", "attempt": first.name}
    atomic_json(first / "status.json", status)
    atomic_json(trial_dir / "status.json", status)
    (first / "partial.txt").write_text("retained")
    with pytest.raises(RuntimeError, match="Partial attempt"):
        run(tmp_path, executor=fake_executor)
    run(tmp_path, resume_partial=True, executor=fake_executor, limit=1)
    assert (first / "partial.txt").read_text() == "retained"
    assert read_json(first / "status.json")["state"] == "interrupted"
    assert (trial_dir / "attempt-002" / "execution.json").exists()


def test_infrastructure_failure_is_terminal_not_selectively_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "private provider credential")
    save_plan(config(), tmp_path)

    def fail(*args):
        raise RuntimeError("private provider credential must not be logged")

    run(tmp_path, executor=fail)
    assert run(tmp_path, executor=fake_executor) == 0
    evaluate(tmp_path)
    _, records = rows(tmp_path)
    assert all(row["failure"] == "infrastructure" and row["success"] is None for row in records)
    assert "private provider credential" not in "\n".join(
        path.read_text() for path in tmp_path.rglob("*.json")
    )


@pytest.mark.parametrize(
    "update",
    [
        {"harness": "claude_code"},
        {"repetitions": 0},
        {"models": []},
        {"profile": "expanded", "repetitions": 3},
        {"contention": True},
        {"api_key": "secret"},
    ],
)
def test_invalid_or_unsupported_configuration_fails_clearly(update):
    cfg = copy.deepcopy(config())
    cfg.update(update)
    with pytest.raises(ValueError):
        validate(cfg)
