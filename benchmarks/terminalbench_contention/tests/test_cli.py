"""Command-line behaviour: dry runs launch nothing, errors map to exit codes."""

import json

import pytest

from benchmarks.terminalbench_contention import cli, runner
from benchmarks.terminalbench_contention.util import read_json


@pytest.fixture
def root(tmp_path):
    return ["--set", f"paths.artifact_root={tmp_path / 'artifacts'}"], tmp_path / "artifacts"


@pytest.fixture
def no_launch(monkeypatch):
    """Any attempt to build or start a run fails the test."""

    def forbidden(*args, **kwargs):
        raise AssertionError("a run was started")

    monkeypatch.setattr(runner, "build_runner", forbidden)
    monkeypatch.setattr(cli, "start_run", forbidden)


def test_plan_is_a_dry_run(root, no_launch, capsys):
    arguments, artifacts = root
    assert cli.main(["plan", *arguments]) == runner.EXIT_OK
    output = capsys.readouterr().out
    assert "9 runs over 24 tasks" in output
    for label in ("LSB-6-r1", "LSB-12-r2", "LSB-24-r3"):
        assert label in output
    assert "scripts/agency-bench run --agents 12 --rep 2" in output
    assert not artifacts.exists()  # nothing written, nothing launched


def test_plan_json_and_write(root, no_launch, capsys):
    arguments, artifacts = root
    assert cli.main(["plan", "--json", "--write", *arguments]) == runner.EXIT_OK
    text = capsys.readouterr().out
    document = json.loads(text[text.index("{") :])
    assert document["dry_run"] is True and len(document["runs"]) == 9
    assert len(document["tasks"]) == 24 and document["conditions"]["LSB-24"] == 24
    assert all(len(run["launch_order"]) == 24 for run in document["runs"])
    assert {run["command"] for run in document["runs"]} == {
        f"scripts/agency-bench run --agents {a} --rep {r}" for a in (6, 12, 24) for r in (1, 2, 3)
    }
    assert (
        read_json(artifacts / "manifests" / "plan.json")["corpus_sha256"]
        == document["corpus_sha256"]
    )
    assert not (artifacts / "runs").exists()
    assert document["api_policy"]["tokens_per_window"] == 1_200_000


def test_matrix_without_execute_only_lists_commands(root, no_launch, capsys):
    arguments, artifacts = root
    assert cli.main(["matrix", *arguments]) == runner.EXIT_OK
    output = capsys.readouterr().out
    assert "NOT started" in output and output.count("agency-bench run --agents") == 9
    assert not artifacts.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--rep", "1"],
        ["run", "--agents", "6", "--condition", "LSB-6", "--rep", "1"],
        ["run", "--agents", "7", "--rep", "1"],
        ["run", "--agents", "6", "--rep", "4"],
        ["run", "--agents", "6", "--rep", "0"],
        ["run", "--condition", "LSB-9", "--rep", "1"],
        ["plan", "--set", "api.nonexistent=1"],
        ["plan", "--set", "cpu.scheduler=custom"],
        ["grade", "--run", "no-such-run"],
    ],
)
def test_usage_and_configuration_errors_exit_2(root, no_launch, arguments, capsys):
    extra, _ = root
    assert cli.main([*arguments, *extra]) == runner.EXIT_USAGE
    assert capsys.readouterr().out.strip()  # a message explains why


def test_missing_required_arguments_exit_2(capsys):
    for arguments in (["run", "--agents", "6"], [], ["nosuchcommand"]):
        with pytest.raises(SystemExit) as raised:
            cli.main(arguments)
        assert raised.value.code == 2


def test_run_refuses_to_start_in_an_unprepared_environment(root, no_launch, capsys):
    arguments, artifacts = root
    assert cli.main(["run", "--agents", "6", "--rep", "1", *arguments]) == runner.EXIT_PRECONDITION
    output = capsys.readouterr().out
    assert "environment prepared" in output and "nothing was started" in output
    assert not (artifacts / "runs").exists()


def test_matrix_execute_checks_preconditions_first(root, no_launch, capsys):
    arguments, artifacts = root
    assert cli.main(["matrix", "--execute", *arguments]) == runner.EXIT_PRECONDITION
    assert not (artifacts / "runs").exists()


def test_validate_reports_and_persists_its_findings(root, no_launch, capsys):
    arguments, artifacts = root
    assert cli.main(["validate", *arguments]) == runner.EXIT_PRECONDITION
    output = capsys.readouterr().out
    assert "[FAIL] environment prepared" in output and "validation FAILED" in output
    report = read_json(artifacts / "validation" / "preflight.json")
    assert report["ok"] is False and len(report["checks"]) > 15
    assert all(set(check) == {"name", "ok", "detail", "optional"} for check in report["checks"])
    assert sum(check["optional"] for check in report["checks"]) == 1
    environment = read_json(artifacts / "manifests" / "environment.json")
    assert environment["kernel"] and "cpu_layout" in environment
    assert read_json(artifacts / "manifests" / "tasks.json")["task_count"] == 24
    assert read_json(artifacts / "manifests" / "benchmark.json")["dataset"].endswith("@4.0.0")


def test_report_lists_every_attempt_including_failed_ones(root, capsys):
    arguments, artifacts = root
    for label, attempt, status in (
        ("LSB-6-r1", "attempt-01", "aborted"),
        ("LSB-6-r1", "attempt-02", "completed"),
    ):
        directory = artifacts / "runs" / label / attempt
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(
                dict(
                    run_label=label,
                    condition="LSB-6",
                    repetition=1,
                    agents=6,
                    status=status,
                    exit_code=0,
                    makespan_seconds=12.5,
                    counts=dict(completed=24, passed=3),
                )
            )
        )
    assert cli.main(["report", "--json", *arguments]) == runner.EXIT_OK
    text = capsys.readouterr().out
    document = json.loads(text[text.index("{") :])
    assert [row["status"] for row in document["runs"]] == ["aborted", "completed"]
    assert read_json(artifacts / "reports" / "runs.json")["runs"][1]["makespan_seconds"] == 12.5


def test_help_documents_commands_and_exit_codes(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    for word in (
        "prepare",
        "validate",
        "plan",
        "run",
        "matrix",
        "grade",
        "report",
        "smoke",
        "cleanup",
        "Exit codes",
    ):
        assert word in text


def test_operator_command_lines():
    assert cli.run_command_line(dict(agents=24, repetition=3)) == (
        "scripts/agency-bench run --agents 24 --rep 3"
    )
