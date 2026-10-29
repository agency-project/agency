"""Offline portability checks: no models, Docker workers, or cloud resources are launched."""

import hashlib
import json
from pathlib import Path

import pytest

import fetch_dataset
import settings
from recording_proxy import summarize


def test_environment_key_does_not_require_old_hosts_credentials_file(tmp_path, monkeypatch):
    (tmp_path / "host.local.json").write_text(
        json.dumps({"credentials_file": "/missing/private.env"})
    )
    monkeypatch.setenv("LITE_LLM_KEY", "synthetic-fixture-key")
    assert settings.credential(tmp_path) == "synthetic-fixture-key"


def test_private_credential_file_is_parsed_without_executing_shell(tmp_path, monkeypatch):
    monkeypatch.delenv("LITE_LLM_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    private = tmp_path / "private.env"
    private.write_text('export LITE_LLM_KEY="synthetic-$(never-execute)"\n')
    (tmp_path / "host.local.json").write_text(json.dumps({"credentials_file": str(private)}))
    assert settings.credential(tmp_path) == "synthetic-$(never-execute)"


def test_machine_guard_rejects_laptop_and_wrong_remote_hostname(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        settings.validate_host(tmp_path)
    monkeypatch.setattr(settings.platform, "system", lambda: "Linux")
    monkeypatch.setattr(settings.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(settings.platform, "node", lambda: "actual-remote")
    (tmp_path / "host.local.json").write_text(
        json.dumps({"host_kind": "linux", "expected_hostname": "another-remote"})
    )
    with pytest.raises(RuntimeError, match="hostname"):
        settings.validate_host(tmp_path)
    (tmp_path / "host.local.json").write_text(
        json.dumps({"host_kind": "linux", "expected_hostname": "actual-remote"})
    )
    assert settings.validate_host(tmp_path)[0] == {"kind": "linux", "hostname": "actual-remote"}


def test_ec2_guard_does_not_fall_back_to_linux_on_wrong_identity(tmp_path, monkeypatch):
    import io

    monkeypatch.setattr(settings.platform, "system", lambda: "Linux")
    monkeypatch.setattr(settings.platform, "machine", lambda: "x86_64")
    (tmp_path / "host.local.json").write_text(
        json.dumps({"host_kind": "ec2", "expected_instance_id": "i-fixture-expected"})
    )
    replies = iter((b"fixture-imds-token", b'{"instanceId":"i-fixture-wrong"}'))
    monkeypatch.setattr(
        settings.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(next(replies))
    )
    with pytest.raises(RuntimeError, match="instance ID"):
        settings.validate_host(tmp_path)


def test_dataset_mismatch_fails_without_replacing_existing_data(tmp_path):
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"synthetic dataset")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_dataset.verify(path, hashlib.sha256(b"different").hexdigest())
    assert path.read_bytes() == b"synthetic dataset"


def test_recorder_marks_usage_less_502_unknown_even_with_successful_response(tmp_path):
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "message_stop", "usage": {"output_tokens": 2}},
    ]
    for number, status in enumerate((200, 502)):
        prefix = tmp_path / f"{number:04d}"
        Path(str(prefix) + ".request.json").write_text(json.dumps({"model": "claude-haiku-5-5"}))
        Path(str(prefix) + ".metadata.json").write_text(
            json.dumps({"path": "/v1/messages", "status": status, "forwarded_upstream": True})
        )
        Path(str(prefix) + ".response.body").write_text(
            "\n".join("data: " + json.dumps(e) for e in events)
            if status == 200
            else '{"error":"fixture gateway failure"}'
        )
    result = summarize(tmp_path)
    assert not result["usage_complete"]
    assert result["usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_export_omits_live_pipeline_log_and_private_settings(tmp_path, monkeypatch):
    import export_results

    monkeypatch.setattr(export_results, "ROOT", tmp_path)
    monkeypatch.setenv("LITE_LLM_KEY", "synthetic-fixture-secret")
    (tmp_path / "pipeline.log").write_text("live orchestrator output")
    (tmp_path / ".env").write_text("LITE_LLM_KEY=synthetic-fixture-secret")
    (tmp_path / "host.local.json").write_text("{}")
    (tmp_path / "runs.jsonl").write_text('{"fixture":true}\n')
    export_results.main()
    names = json.loads((tmp_path / "export_manifest.json").read_text())["files"]
    assert "runs.jsonl" in names
    assert all(name not in names for name in ("pipeline.log", ".env", "host.local.json"))


def test_pipeline_keeps_failed_setup_attempt_but_grades_latest_allowed_attempt(
    tmp_path, monkeypatch
):
    import pipeline

    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "validate_host", lambda _: None)
    (tmp_path / "configuration.json").write_text("{}")
    (tmp_path / "diagnostic_gate.json").write_text('{"passed":true,"script_sha256":{}}')
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda script, *args: calls.append((script, args)))
    for number in range(100):
        run_id = f"synthetic-{number}"
        for attempt in (1, 2) if number == 0 else (1,):
            directory = tmp_path / "attempts" / run_id / str(attempt)
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "directory": str(directory.relative_to(tmp_path)),
                    }
                )
            )
            (directory / "official-report.json").write_text(
                json.dumps(
                    {
                        "resolved": None if number == 0 and attempt == 1 else True,
                    }
                )
            )
    pipeline.main()
    assert (tmp_path / "grading_complete.json").exists()
    assert len(list(tmp_path.glob("attempts/*/*/summary.json"))) == 101
    assert calls[0] == ("run_experiment.py", ("measured",))
    assert calls[-1] == ("validate_results.py", ())
