import json

from agency.native_harness import sysstats, tools


def test_footer_reports_load_pressure_and_memory():
    snap = {"cpus": 16, "load1": 34.12, "host_cpu_psi": 61.7, "sandbox_cpu_psi": 70.2,
            "host_memory_psi": 0.0, "sandbox_memory_psi": None, "host_io_psi": None, "sandbox_io_psi": None,
            "mem_available_gb": 2.14}
    assert sysstats.footer(41.3, snap) == (
        "[system: 41 s; host load 34.1 on 16 CPUs; CPU pressure 62% (sandbox 70%); memory pressure 0%; "
        "2.1 GB memory free]"
    )


def test_snapshot_reads_this_host():
    snap = sysstats.snapshot()
    assert snap["cpus"] >= 1 and "load1" in snap


def test_bash_results_carry_the_footer_only_when_enabled(monkeypatch):
    monkeypatch.setattr(tools, "SYSTEM_STATS", False)
    assert "[system:" not in json.loads(tools._run_bash_tool(json.dumps({"command": "echo hi"})))["output"]
    monkeypatch.setattr(tools, "SYSTEM_STATS", True)
    out = json.loads(tools._run_bash_tool(json.dumps({"command": "echo hi"})))["output"]
    assert out.startswith("hi\n") and "[system:" in out
    err = json.loads(tools._run_bash_tool(json.dumps({"command": "sleep 3", "timeout": 1})))["error"]
    assert err.startswith("command timed out after 1s\n[system: 1")


def test_assess_levels():
    base = {"cpus": 16, "load1": 4.0, "host_cpu_psi": 2.0}
    assert sysstats.assess(base) == "idle"
    assert sysstats.assess({**base, "host_cpu_psi": 15.0}) == "busy"
    assert sysstats.assess({**base, "load1": 34.0}) == "heavily loaded"
    assert sysstats.assess({**base, "host_cpu_psi": 61.0}) == "heavily loaded"


def test_explained_footer_says_what_numbers_mean_and_what_to_do():
    snap = {"cpus": 16, "load1": 34.12, "host_cpu_psi": 61.7, "sandbox_cpu_psi": 70.2,
            "host_memory_psi": None, "sandbox_memory_psi": None, "host_io_psi": None, "sandbox_io_psi": None,
            "mem_available_gb": 2.14}
    out = sysstats.explained_footer(41.3, snap)
    assert out.startswith("[system load while this command ran (41 s wall time): machine is HEAVILY LOADED\n")
    assert "2.1 runnable tasks per CPU" in out and "stalled waiting for CPU 62% of the last 10 s (70% inside" in out
    assert "rather than the code" in out and out.endswith("]")
    assert "memory pressure" not in out


def test_bash_results_carry_the_explained_footer_when_enabled(monkeypatch):
    monkeypatch.setattr(tools, "SYSTEM_STATS", True)
    monkeypatch.setattr(tools, "SYSTEM_STATS_EXPLAINED", True)
    out = json.loads(tools._run_bash_tool(json.dumps({"command": "echo hi"})))["output"]
    assert out.startswith("hi\n") and "\n[system load while this command ran (" in out


def test_assess_counts_sandbox_pressure_when_the_host_looks_idle():
    assert sysstats.assess({"cpus": 16, "load1": 1.0, "host_cpu_psi": 3.0, "sandbox_cpu_psi": 70.0}) == "heavily loaded"
