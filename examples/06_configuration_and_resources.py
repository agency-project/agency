"""Lesson 6: change configuration, mount a directory, and inspect resources."""

from agency import Agent, agdata, agskill, get_orchestrator

from _common import close_sandboxes, run_example, tutorial_config


def main() -> None:
    cfg, run_dir = tutorial_config("06_configuration_and_resources")
    changed = cfg.clone().update(llm={"temperature": 0.2})
    print(f"original temperature: {cfg.llm.temperature}")
    print(f"clone temperature: {changed.llm.temperature}")
    print(f"config namespaces: {', '.join(cfg.safe_snapshot())}")

    mounted = run_dir / "mounted"
    mounted.mkdir()
    cfg.sandbox.add_mount("tutorial-data", mounted, "/tutorial-data")
    learner = Agent("configured", agconfig=cfg, harness="native")
    skill = agskill(
        name="inspect_resources",
        prompt=(
            "Write hello to /tutorial-data/note.txt. "
            "Call get_current_resources and return the resource totals."
            "\nAfter submitting all output fields, finish with a short confirmation."
        ),
        input_schema=agdata(request=str),
        output_schema=agdata(total_cpus=int, total_memory_mb=int, total_gpus=int),
    )
    try:
        result = learner.run(skill, agdata(request="Inspect this sandbox."))
        print(f"resource totals: {result.to_dict()}")
        print(f"mounted file: {(mounted / 'note.txt').read_text()}")

        updated = learner.get_config_copy()
        updated.llm.max_completion_tokens = 2048
        learner.change_config(updated)
        print(f"updated token budget: {learner.get_config_copy().llm.max_completion_tokens}")
        print(f"shared resource pool: {get_orchestrator(cfg).agresource_pool!r}")
        print(f"artifacts: {run_dir}")
    finally:
        close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
