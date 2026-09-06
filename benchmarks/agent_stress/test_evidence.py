"""Preparation-only evidence checks. Never executes the child workload."""

import json
import shlex

import pytest

from benchmarks.agent_stress.run import peak, receipts, replay


def test_half_open_overlap_excludes_zero_length_intervals():
    assert peak([(1, 2), (2, 3), (2, 2)]) == 1
    assert peak([(1, 4), (2, 3), (2.5, 5)]) == 3
    assert peak([]) == 0


def test_replay_contains_valid_child_source_and_restarts_per_backend(tmp_path):
    from agency.configs.agconfig import agconfig, llmconfig
    from agency.llm.mock import _MockBackend

    db = tmp_path / "replay.sqlite3"
    source = replay(db, 0.01)
    compile(source, "<generated child>", "exec")  # syntax only, never executed
    cfg = agconfig(llmconfig(provider="mock", replay_db_path=str(db), timing_mode="instant"))
    backend = _MockBackend(cfg)
    blocks = backend.dispatch({})["message"]["blocks"]
    argv = shlex.split(json.loads(blocks[0]["arguments"])["command"])
    assert argv == ["python3", "-c", source]
    final = backend.dispatch({})["message"]["blocks"][0]
    assert json.loads(final["text"])["receipt"] == "/workspace/stress_receipt.json"
    with pytest.raises(RuntimeError, match="exhausted"):
        backend.dispatch({})
    assert next(_MockBackend(cfg).dispatch_stream({}))["block_type"] == "tool_use"


def test_receipt_parser_handles_native_tool_result_encoding_and_missing_evidence():
    receipt = {"start": 1, "end": 2, "affinity": [4]}
    payload = {"result": json.dumps({"output": "STRESS_RECEIPT=" + json.dumps(receipt) + "\n"})}
    assert list(receipts(payload)) == [receipt]
    assert list(receipts({"result": '{"error":"tool denied"}'})) == []


def test_execution_requires_explicit_flag_before_preflight(monkeypatch, tmp_path):
    from benchmarks.agent_stress import run

    out = tmp_path / "must-not-exist"
    monkeypatch.setattr("sys.argv", ["run.py", "--layout", "not-read.json", "--out", str(out)])

    def forbidden_command(_):
        raise AssertionError("Preparation must not contact the runtime")

    monkeypatch.setattr(run, "command", forbidden_command)
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert not out.exists()
