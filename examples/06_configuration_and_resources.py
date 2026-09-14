"""Lesson 6: the complete namespaced config and shared resource pool."""

from __future__ import annotations

from agency import Agent, agResourcePool, agSandbox, agconfig, agdata, agskill, get_orchestrator
from agency.configs.agconfig import (
    agentconfig,
    dataloggerconfig,
    harnessadapterconfig,
    hostserverconfig,
    llmconfig,
    orchestratorconfig,
    ptraceconfig,
    resourcesconfig,
    sandboxconfig,
    schemaconfig,
    skillconfig,
    toolconfig,
)

from _common import close_sandboxes, run_example, tutorial_config


def main() -> None:
    cfg, run_dir = tutorial_config("06_configuration_and_resources")

    # agconfig recognizes namespace objects by type. Supplying every one here
    # is a compact catalog of the complete configuration surface.
    catalog = agconfig(
        llmconfig(provider="openai", model="example", api_key="redact-me"),
        sandboxconfig(),
        orchestratorconfig(),
        resourcesconfig(),
        agentconfig(),
        schemaconfig(),
        skillconfig(),
        toolconfig(),
        harnessadapterconfig(),
        ptraceconfig(),
        dataloggerconfig(),
        hostserverconfig(),
    )
    snapshot = catalog.safe_snapshot()
    assert set(snapshot) == {
        "llm",
        "sandbox",
        "orchestrator",
        "resources",
        "agent",
        "schema",
        "skill",
        "tool",
        "harness_adapter",
        "ptrace",
        "data_logger",
        "host_server",
    }
    assert "api_key" not in snapshot["llm"]
    changed = catalog.clone().update(llm={"temperature": 0.2}, tool={"timeout_s": 60})
    assert catalog.llm.temperature is None and changed.llm.temperature == 0.2
    print(f"config namespaces: {', '.join(snapshot)}")
    print("safe snapshot: API key redacted")
    print("clone isolation: original temperature=None, clone temperature=0.2")

    mounted = run_dir / "mounted"
    mounted.mkdir()
    cfg.sandbox.add_mount("tutorial-data", mounted, "/tutorial-data")
    # Resource inspection is a native host service, so this lesson explicitly
    # selects the native harness. The mount is exercised deterministically below.
    learner = Agent("configured", agconfig=cfg, harness="native")
    inspect_resources = agskill(
        name="inspect_resources",
        prompt="Call get_current_resources and return its resource totals.",
        input_schema=agdata(request=str),
        output_schema=agdata(total_cpus=int, total_memory_mb=int, total_gpus=int),
    )
    resources = learner.run(inspect_resources, agdata(request="Inspect this sandbox."))
    mounted_sandbox = agSandbox("configured-mount", agconfig=cfg)
    try:
        _, return_code = mounted_sandbox.exec("printf MOUNTED > /tutorial-data/from-agent.txt")
        assert return_code == 0
        assert (mounted / "from-agent.txt").read_text().strip() == "MOUNTED"
        print(f"mount round-trip: host={mounted}, sandbox=/tutorial-data, content=MOUNTED")
    finally:
        mounted_sandbox.destroy()
    print(
        f"resource totals: cpu={resources.total_cpus}, memory_mb={resources.total_memory_mb}, gpu={resources.total_gpus}"
    )

    agent_cfg = learner.get_config_copy()
    agent_cfg.llm.max_completion_tokens = 2048
    learner.change_config(agent_cfg)
    assert learner.get_config_copy().llm.max_completion_tokens == 2048
    print("agent config update: max_completion_tokens=2048")

    orchestrator = get_orchestrator(cfg)
    pool = orchestrator.agresource_pool
    assert isinstance(pool, agResourcePool)
    pool_cfg = pool.get_config_copy()
    pool.change_config(pool_cfg)
    print(f"shared resource pool: {pool!r}")
    print(
        "resource interpretation: total_cpus is host capacity; "
        f"idle_cpus={pool.agconfig.resources.idle_cpus} is each idle sandbox's CPU limit"
    )
    print(
        f"agent output paths: host={learner.output_path}, sandbox={learner.container_output_path}"
    )
    print(f"artifacts: {run_dir}")

    close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
