"""Configuration parsing and the experimental plan (methodology invariants)."""

import json

import pytest

from benchmarks.terminalbench_contention import config as C
from benchmarks.terminalbench_contention import corpus

TASKS = [f"org/task-{i:02d}" for i in range(24)]


def test_defaults_describe_the_requested_experiment():
    config = C.load()
    assert config["conditions"] == {"LSB-6": 6, "LSB-12": 12, "LSB-24": 24}
    assert config["repetitions"] == 3 and config["benchmark"]["tasks"] == 24
    assert config["benchmark"]["dataset"] == "terminal-bench/terminal-bench@4.0.0"
    assert config["cpu"]["experimental_physical_cores"] == 6
    assert config["cpu"]["reserved_physical_cores"] == 2
    assert config["cpu"]["scheduler"] == "linux-default"
    assert config["api"]["tokens_per_window"] == 1_200_000
    assert config["model"]["model"] == "gpt-6-luna" and config["model"]["route"] == "litellm"
    assert config["agent"]["harness"] == "codex"
    json.dumps(config)  # fully serializable


def test_overrides_from_file_and_assignments(tmp_path):
    path = tmp_path / "override.json"
    path.write_text(
        json.dumps({"api": {"tokens_per_window": 2_000_000}, "timeouts": {"task_seconds": 900}})
    )
    config = C.load(
        path,
        [
            "api.requests_per_window=5000",
            "cpu.smt_policy=primary",
            "api.estimator.safety_factor=1.2",
            "sandbox.network=none",
        ],
    )
    assert config["api"]["tokens_per_window"] == 2_000_000
    assert config["api"]["requests_per_window"] == 5000
    assert config["api"]["estimator"]["safety_factor"] == 1.2
    assert config["timeouts"]["task_seconds"] == 900
    assert config["cpu"]["smt_policy"] == "primary" and config["sandbox"]["network"] == "none"
    assert C.api_policy(config).tokens_per_window == 2_000_000
    assert C.fingerprint(config) != C.fingerprint(C.load())
    assert C.fingerprint(C.load()) == C.fingerprint(C.load())


@pytest.mark.parametrize(
    "assignment",
    [
        "api.tokens_per_windows=5",
        "nosuch.key=1",
        "cpu.smt_policy=sometimes",
        "cpu.scheduler=custom",
        "repetitions=0",
        "timeouts.task_seconds=-5",
        "api.reconciliation=guess",
        "sandbox.engine=docker",
        "agent.harness=native",
        'conditions={"X": 0}',
        "api.admission_order=fifo",
        "justakey",
    ],
)
def test_invalid_configuration_is_rejected(assignment):
    with pytest.raises(C.ConfigError):
        C.load(assignments=[assignment])


def test_unreadable_configuration_file_is_a_configuration_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(C.ConfigError):
        C.load(bad)
    with pytest.raises(C.ConfigError):
        C.load(tmp_path / "missing.json")


def test_plan_has_nine_runs_and_every_run_covers_the_same_corpus_once():
    config = C.load()
    plan = C.build_plan(config, TASKS)
    assert len(plan) == 9
    assert sorted((e["condition"], e["repetition"]) for e in plan) == sorted(
        (c, r) for c in config["conditions"] for r in (1, 2, 3)
    )
    for entry in plan:
        assert sorted(entry["launch_order"]) == sorted(TASKS)  # same 24 tasks, each once
        assert len(set(entry["launch_order"])) == 24 == entry["tasks"]
        assert entry["agents"] == config["conditions"][entry["condition"]]
        assert entry["task_timeout_seconds"] == config["timeouts"]["task_seconds"]
        assert entry["run_label"] == f"{entry['condition']}-r{entry['repetition']}"
    assert [e["index"] for e in plan] == list(range(1, 10))


def test_orderings_are_matched_within_a_repetition_and_differ_between_repetitions():
    config = C.load()
    plan = C.build_plan(config, TASKS)
    by_repetition = {}
    for entry in plan:
        by_repetition.setdefault(entry["repetition"], []).append(entry)
    orders = []
    for repetition, entries in by_repetition.items():
        assert len({tuple(e["launch_order"]) for e in entries}) == 1
        assert len({e["order_seed"] for e in entries}) == 1
        assert entries[0]["order_seed"] == C.repetition_seed(config, repetition)
        orders.append(tuple(entries[0]["launch_order"]))
    assert len(set(orders)) == 3
    assert all(order != tuple(sorted(TASKS)) for order in orders)


def test_ordering_is_reproducible_and_independent_of_input_order():
    config = C.load()
    first = C.build_plan(config, TASKS)
    second = C.build_plan(C.load(), list(reversed(TASKS)))
    assert first == second
    other_seed = C.build_plan(C.load(assignments=["seed=7"]), TASKS)
    assert [e["launch_order"] for e in other_seed] != [e["launch_order"] for e in first]
    assert C.launch_order(TASKS, 123) == C.launch_order(sorted(TASKS, reverse=True), 123)


def test_condition_order_rotates_so_none_is_always_first():
    config = C.load()
    sequences = [C.condition_sequence(config, r) for r in (1, 2, 3)]
    assert sequences[0] == ["LSB-6", "LSB-12", "LSB-24"]
    for position in range(3):
        assert {s[position] for s in sequences} == {"LSB-6", "LSB-12", "LSB-24"}


def test_plan_entry_lookup_and_condition_resolution():
    config = C.load()
    entry = C.plan_entry(config, TASKS, "LSB-12", 2)
    assert entry["agents"] == 12 and entry["repetition"] == 2
    assert C.condition_for_agents(config, 24) == "LSB-24"
    for bad in (
        lambda: C.plan_entry(config, TASKS, "LSB-12", 4),
        lambda: C.plan_entry(config, TASKS, "LSB-7", 1),
        lambda: C.condition_for_agents(config, 7),
    ):
        with pytest.raises(C.ConfigError):
            bad()


def test_plan_rejects_a_wrong_or_duplicated_corpus():
    config = C.load()
    with pytest.raises(C.ConfigError):
        C.build_plan(config, TASKS[:23])
    with pytest.raises(C.ConfigError):
        C.build_plan(config, TASKS[:23] + TASKS[:1])


def test_run_time_limit_scales_with_waves_and_can_be_fixed():
    config = C.load()
    per_wave = config["timeouts"]["task_seconds"] + config["timeouts"]["setup_seconds"]
    assert C.run_seconds(config, 6) == 4 * per_wave + 600
    assert C.run_seconds(config, 24) == 1 * per_wave + 600
    assert C.run_seconds(C.load(assignments=["timeouts.run_seconds=99"]), 6) == 99


def test_same_cpu_and_api_policy_apply_to_every_condition():
    config = C.load()
    plan = C.build_plan(config, TASKS)
    # Conditions differ only in the concurrency limit; there is one policy object.
    varying = {
        key
        for entry in plan
        for key in entry
        if len({json.dumps(e[key], sort_keys=True) for e in plan}) > 1
    }
    assert varying == {
        "index",
        "run_label",
        "condition",
        "agents",
        "repetition",
        "order_seed",
        "launch_order",
        "run_timeout_seconds",
    }
    assert "cpu" not in config["conditions"] and isinstance(config["api"], dict)


def test_the_frozen_manifest_matches_the_plan():
    manifest = corpus.load_manifest()
    plan = C.build_plan(C.load(), corpus.task_ids(manifest))
    assert all(len(entry["launch_order"]) == 24 for entry in plan)
