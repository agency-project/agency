"""Lesson 5: deterministic tasks, agent fan-out, fan-in, and teams."""

from __future__ import annotations

from agency import Agent, agdata, agmap, agskill, agsync, agtask, agteam

from _common import close_sandboxes, run_example, tutorial_config


RESEARCH = agskill(
    name="research_one",
    system_prompt="Return one short practical fact about the supplied topic.",
    input_schema=agdata(topic=str),
    output_schema=agdata(fact=str),
)

MERGE = agskill(
    name="merge_research",
    system_prompt="Combine the supplied research results into one concise paragraph.",
    input_schema=agdata(items=list),
    output_schema=agdata(report=str),
)


class ResearchTeam(agteam):
    def setup(self) -> None:
        self.writer = Agent("team-writer")

    def run(self) -> agdata:
        return self.writer.run(RESEARCH, agdata(topic=self.topic))


def main() -> None:
    tasks = agmap(lambda number: number * number, [2, 3, 4], is_asynchronous=True)
    assert all(isinstance(task, agtask) for task in tasks)
    agsync(tasks)
    print(f"agmap via agsync: {[task.result for task in tasks]}")

    more_tasks = agmap(lambda number: number + 1, [4, 5], is_asynchronous=True)
    agdata.wait_all(more_tasks)
    print(f"agmap via agdata.wait_all: {[task.result for task in more_tasks]}")

    cfg, run_dir = tutorial_config("05_parallel_workflows", max_concurrent_engines=4)
    parent = Agent("coordinator", agconfig=cfg)
    workers = [Agent.fork(parent, f"research-{index}") for index in range(3)]
    pending = [
        worker.run(RESEARCH, agdata(topic=topic))
        for worker, topic in zip(workers, ["sandboxes", "schedulers", "typed data"])
    ]
    merged = parent.run(MERGE, agdata(items=pending))
    print(f"fork fan-in: {merged.report}")

    team = ResearchTeam(agconfig=cfg, topic="process-wide orchestration")
    team_result = team.run()
    assert team_result.is_pending()
    agsync(team)
    print(f"team result: {team_result.fact}")
    print(f"team registry: {len(agteam.all())}; team agents: {len(team.agents)}")

    copied = team.get_config_copy()
    assert copied is not None
    copied.llm.max_completion_tokens = 2048
    team.change_config(copied)
    print(f"team config update: max tokens={team.get_config_copy().llm.max_completion_tokens}")
    print(f"artifacts: {run_dir}")

    close_sandboxes([parent, *workers, *team.agents])


if __name__ == "__main__":
    run_example(main)
