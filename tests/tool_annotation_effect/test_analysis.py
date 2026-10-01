import copy
import json

from benchmarks.tool_annotation_effect.analysis import (
    analyze,
    compare_actions,
    normalize,
    cluster_interval,
    required_tasks,
    estimate_cost,
    review_export,
)
from benchmarks.tool_annotation_effect.execution import trace_metrics
from benchmarks.tool_annotation_effect.fixtures import (
    local_tasks,
    CorpusTools,
    evaluate_rag,
    differential_inputs,
    evaluate_tandem,
)


def event(tool="read", args=None, label="purpose", event_id="1", agent_id="agent"):
    return {
        "kind": "tool_annotation",
        "event_id": event_id,
        "tool_name": tool,
        "arguments": args or {"file_path": "/workspace/a"},
        "annotation": {"raw": {"purpose": label}, "status": "valid"},
        "agent_id": agent_id,
        "timestamp_ns": 1,
        "exchange": 0,
        "model_tool_call_id": "m1",
        "batch_size": 1,
        "execution": "serial",
    }


def test_normalization_ignores_annotations_and_preserves_compound_commands():
    a = event("bash", {"command": "cd src && pytest; echo done", "_agency": {"purpose": "a"}})
    b = copy.deepcopy(a)
    b["arguments"]["_agency"] = {"purpose": "different", "workstream_ids": ["different"]}
    b["annotation"]["raw"] = b["arguments"]["_agency"]
    assert normalize([a]) == normalize([b])
    assert normalize([a])[0]["arguments"]["command"] == "cd src && pytest; echo done"
    assert normalize([a])[0]["validation"] is True
    assert normalize([event("custom")])[0]["operation"] == "unknown"


def test_action_comparisons_match_known_distances():
    a = normalize([event("read"), event("write", event_id="2")])
    b = normalize([event("read")])
    comparison = compare_actions(a, b)
    assert comparison["tool_lcs_distance"] == 0.5
    assert comparison["operation_edit_distance"] == 0.5
    assert comparison["target_overlap"] == 1
    assert comparison["action_count_difference"] == 1
    assert compare_actions([], [])["tool_lcs_distance"] == 0


def test_call_and_result_correlation_and_denial():
    events = [
        event(),
        {
            "kind": "tool_admission",
            "event_id": "1",
            "call_id": "agency1",
            "decision": {"decision": "deny"},
        },
        {"kind": "tool_result", "event_id": "1", "category": "error"},
    ]
    action = normalize(events)[0]
    assert action["call_id"] == "agency1"
    assert action["result_category"] == "denied"


def test_missing_usage_stays_unknown_and_workstream_reuse_is_counted():
    events = [
        {"kind": "model_exchange", "response": {"usage": None}},
        {
            "kind": "model_exchange",
            "response": {"usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        },
        event(),
    ]
    assert trace_metrics(events)["input_tokens"] is None
    for ids in [["a", "shared"], ["b", "shared"]]:
        annotation_event = event()
        annotation_event["annotation"]["raw"]["workstream_ids"] = ids
        events.append(annotation_event)
    assert trace_metrics(events)["workstream_reuses"] == 1


def test_analysis_known_success_difference_and_all_baseline_comparators():
    rows = []
    for task in range(3):
        for repetition in range(3):
            for arm in ("baseline", "schema_only", "purpose", "purpose_workstreams"):
                actions = normalize([event("read")] if arm == "baseline" else [event("write")])
                rows.append(
                    {
                        "trial_id": f"{task}-{repetition}-{arm}",
                        "suite": "rag",
                        "model_id": "m",
                        "task_id": str(task),
                        "repetition": repetition,
                        "block_id": f"{task}-{repetition}",
                        "arm": arm,
                        "attempt": "attempt-001",
                        "state": "completed",
                        "success": arm != "baseline",
                        "actions": actions,
                        "events": [event()],
                        "input_tokens": None,
                        "tool_calls": len(actions),
                    }
                )
    report = analyze({"schedule": rows}, rows, bootstrap_samples=100)
    group = report["groups"][0]
    contrast = next(
        c for c in group["contrasts"] if c["active"] == "purpose" and c["control"] == "baseline"
    )
    assert contrast["assigned_confirmed_success_difference"]["mean"] == 1
    assert contrast["assigned_confirmed_success_difference"]["interval"] == [1, 1]
    baseline_pair = next(
        c for c in group["trajectory_pairs"] if c["arms"] == ("baseline", "baseline")
    )
    assert baseline_pair["pairs"] == 9
    cross = next(c for c in group["trajectory_pairs"] if c["arms"] == ("baseline", "purpose"))
    assert cross["pairs"] == 27
    assert cross["excess_over_baseline"]["operation_edit_distance"]["mean"] == 1
    assert group["arms"]["baseline"]["metrics"]["input_tokens"]["mean"] is None
    assert report["equivalence_claim"] is False


def test_infrastructure_and_missing_assignments_have_bounds_and_sensitivity():
    base = {
        "suite": "rag",
        "model_id": "m",
        "task_id": "t",
        "block_id": "b",
        "actions": [],
        "events": [],
    }
    rows = [
        {**base, "trial_id": "a", "arm": "baseline", "success": None, "failure": "infrastructure"},
        {**base, "trial_id": "b", "arm": "purpose", "success": True, "failure": None},
    ]
    report = analyze({"schedule": rows}, rows, bootstrap_samples=10)
    group = report["groups"][0]
    assert group["arms"]["baseline"]["success_bounds"] == [0, 1]
    contrast = next(
        c for c in group["contrasts"] if c["active"] == "purpose" and c["control"] == "baseline"
    )
    assert contrast["complete_case_sensitivity"]["mean"] is None
    assert contrast["missing_pairs"] == 1


def test_power_and_bootstrap_use_tasks_not_trials():
    assert required_tasks(margin=0.1, task_difference_sd=0.3) >= 70
    assert cluster_interval({"one": 1})["interval"] is None
    assert cluster_interval({"a": 0, "b": 1}, seed=3) == cluster_interval({"a": 0, "b": 1}, seed=3)


def test_explicit_pricing_does_not_use_annotation_token_estimates():
    pricing = {
        "version": "test",
        "currency": "USD",
        "models": {"m": {"input_per_million": 1, "output_per_million": 2}},
    }
    row = {
        "model_id": "m",
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "annotation_token_estimate": 100,
    }
    assert estimate_cost(row, pricing) == 3
    assert estimate_cost({**row, "input_tokens": None}, pricing) is None


def test_blinded_review_separates_raw_actions_labels_and_arm_key(tmp_path):
    rows = [
        {
            "trial_id": "trial",
            "arm": "purpose",
            "actions": normalize([event()]),
            "events": [event()],
        }
    ]
    review_export(tmp_path, rows)
    raw = (tmp_path / "raw-blinded.json").read_text()
    assert "purpose" not in raw and "trial" not in raw
    assert "purpose" in (tmp_path / "labels-phase-two.json").read_text()


def test_rag_fixtures_cover_stale_documents_and_supported_multihop_answers():
    tasks = local_tasks("rag")
    assert len(tasks) == 5
    corpus = CorpusTools(tasks[0]["corpus"])
    assert "atlas-2024" in corpus.retrieve({"query": "Atlas"})["document_ids"]
    assert corpus.read({"document_id": "atlas-2026"})["version"] == 2026
    for task in tasks:
        text = json.dumps({"answer": task["gold"], "evidence": task["evidence"]})
        assert evaluate_rag(task, text)["success"]
        assert not evaluate_rag(task, json.dumps({"answer": task["gold"], "evidence": []}))[
            "success"
        ]


def test_migration_inputs_are_seeded_and_within_defined_domains():
    for task in local_tasks("migration"):
        assert differential_inputs(task, 42) == differential_inputs(task, 42)
        assert all(
            task["domain"][0] <= x <= task["domain"][1] for x in differential_inputs(task, 42)
        )


def test_tandem_checks_independent_final_artifact():
    for task in local_tasks("tandem"):
        answer = {
            "left": sum(task["left"]),
            "right": sum(task["right"]),
            "total": sum(task["left"] + task["right"]),
        }
        assert evaluate_tandem(task, json.dumps(answer))["success"]
        answer["total"] += 1
        assert not evaluate_tandem(task, json.dumps(answer))["success"]


def test_annotation_token_estimator_uses_supplied_tokenizer():
    from benchmarks.tool_annotation_effect.tokenization import annotation_tokens

    events = [event()]
    strings = []

    def tokenizer(text):
        strings.append(text)
        return [1, 2, 3]

    assert annotation_tokens(events, tokenizer) == 3
    assert "purpose" in strings[0]


def test_failed_dispatch_usage_is_unknown_even_when_prior_dispatch_reported_usage():
    events = [
        {"kind": "model_dispatch_complete", "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        {"kind": "model_dispatch_complete", "usage": None, "finish_reason": "error"},
    ]
    assert trace_metrics(events)["input_tokens"] is None
    assert trace_metrics(events)["missing_input_usage"] == 1


def test_rag_public_prompt_defines_fields_without_exposing_gold_values():
    from benchmarks.tool_annotation_effect.fixtures import rag_prompt

    task = local_tasks("rag")[1]
    prompt = rag_prompt(task)
    assert "contact" in prompt
    assert "Mira" not in prompt
    assert "atlas-2026" not in prompt


def test_rag_rejects_malformed_evidence_without_crashing():
    task = local_tasks("rag")[0]
    text = json.dumps({"answer": task["gold"], "evidence": [{"bad": "value"}]})
    assert evaluate_rag(task, text)["success"] is False


def test_repair_cycles_keep_interleaved_agent_roles_separate():
    from benchmarks.tool_annotation_effect.analysis import repair_cycles

    actions = normalize(
        [
            event("read", agent_id="left"),
            event("write", event_id="2", agent_id="right"),
            event("write", event_id="3", agent_id="left"),
            event("bash", {"command": "pytest"}, event_id="4", agent_id="left"),
        ]
    )
    actions[0]["result_category"] = "error"
    assert repair_cycles(actions) == 1
    assert repair_cycles(actions[:2]) == 0
