"""Lesson 9: orchestrator state, data logging, and explicit profiling."""

from __future__ import annotations

from agency import (
    Agent,
    ExecutionScheduler,
    GlobalAgentOrchestrator,
    OrchestratorSnapshot,
    agDataLogger,
    agdata,
    agprof,
    agskill,
    get_orchestrator,
)

from _common import close_sandboxes, run_example, tutorial_config


def main() -> None:
    cfg, run_dir = tutorial_config("09_observability")
    profile_dir = run_dir / "profile"
    report = agskill(
        name="observable_run",
        system_prompt="Return the supplied state in a short sentence.",
        input_schema=agdata(state=str),
        output_schema=agdata(message=str),
    )

    with agprof.session(
        profile_dir,
        sample_hz=5,
        sample_gpu=False,
        auto_functions=False,
    ):
        learner = Agent("observable", agconfig=cfg)
        learner.record_state("preparing", skill=report.name)
        result = learner.run(report, agdata(state="scheduled and observable"))
        print(f"agent result: {result.message}")
        learner.record_state("complete", skill=report.name)

        orchestrator = get_orchestrator(cfg)
        assert isinstance(orchestrator, GlobalAgentOrchestrator)
        assert isinstance(orchestrator.scheduler, ExecutionScheduler)
        assert isinstance(orchestrator.data_logger, agDataLogger)
        snapshot = orchestrator.snapshot()
        assert isinstance(snapshot, OrchestratorSnapshot)
        print(
            "orchestrator snapshot: "
            f"submitted={snapshot.submitted_total}, completed={snapshot.completed_total}, "
            f"running={snapshot.running_count}"
        )
        orchestrator.flush()

    metrics = agprof.summary_metrics()
    records = agprof.profile_records()
    assert metrics is not None and records
    print(agprof.summary_table(row_limit=8))
    print(f"profile records: {len(records)}")
    print(f"artifacts: {run_dir}")
    close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
