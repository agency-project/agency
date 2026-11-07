"""What "fresh" means for each run, and what it does not mean.

Different caches are controlled by different mechanisms. This module states the
strategy per cache class (persisted in every run's ``metadata.json``) and
records the host's cache-related state when a run starts. It performs no
privileged or host-wide cleanup.
"""

from __future__ import annotations

from pathlib import Path

from .sampler import MEMINFO_KEYS, parse_flat_keyed, parse_meminfo


def strategy(config: dict) -> dict:
    return dict(
        application_state=dict(
            scope="Codex sessions and history, Agency agent state, harness daemon state",
            control="Every task gets a new Agency agent in a new container with new log, "
            "output and profile directories; Codex's home lives inside that container; no "
            "session is resumed and the sandbox is not persistent.",
            verified="runner refuses an existing run/agent directory; worker records the "
            "container ID; per-agent writable mounts are checked to be unique",
            fresh_per="task",
        ),
        api_admission_ledger=dict(
            control="A new ledger directory per run; creation fails if it is not empty.",
            fresh_per="run",
        ),
        build_and_dependency_caches=dict(
            scope="compiler outputs, pip/npm/cargo caches, anything an agent builds",
            control="They live only in the container's own writable layer, which is destroyed "
            "with the container. No writable volume or cache directory is shared between "
            "agents or runs (every shared mount is read-only).",
            verified="isolation.json lists each container's writable mounts; summary.json "
            "reports cross_agent_writable_mounts",
            fresh_per="task",
        ),
        writable_filesystem=dict(
            control="Fresh overlay upper layer per container from a digest-pinned image; the "
            "image ID is compared with the prepared image before the agent starts.",
            fresh_per="task",
        ),
        container_image_layers=dict(
            control="Read-only image layers are pulled once during preparation and shared by "
            "all runs. They are identical inputs, not reused outputs.",
            fresh_per="never (immutable, digest-pinned)",
        ),
        host_page_cache=dict(
            control="NOT controlled. A new container does not make the kernel's page cache "
            "cold: image files read by an earlier run may still be cached. The runner does "
            "not drop caches (that needs root and affects the whole host).",
            mitigation="Condition order is rotated across repetitions so no condition "
            "always runs first; page-cache size at run start is recorded in "
            "metadata.host_state_at_start and page-in/reclaim counters are sampled.",
            optional_operator_step="For a cold start, reboot or run "
            "`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before each run and apply it "
            "to every run alike.",
            guaranteed_cold=False,
        ),
        network_and_provider=dict(
            control="NOT controlled by the runner. Provider-side prompt caching and any "
            "gateway response cache are outside this host.",
            observed="cached prompt tokens reported by the model are recorded per request; "
            "each task prompt carries a unique Agency run marker, so identical-request "
            "response caching cannot hit across runs",
            requires_live_validation=True,
            sandbox_network=config["sandbox"]["network"],
            note="With sandbox.network='default' agents can reach package indexes; those "
            "downloads are not cached locally and vary with the network.",
        ),
    )


def host_state() -> dict:
    """Cache- and memory-related host facts at the moment a run starts."""
    state: dict = {}
    try:
        state["meminfo_bytes"] = parse_meminfo(Path("/proc/meminfo").read_text(), MEMINFO_KEYS)
    except OSError:
        state["meminfo_bytes"] = None
    try:
        state["vmstat"] = parse_flat_keyed(
            Path("/proc/vmstat").read_text(),
            {
                "pgpgin",
                "pgpgout",
                "pswpin",
                "pswpout",
                "pgmajfault",
                "oom_kill",
                "workingset_refault_file",
            },
        )
    except OSError:
        state["vmstat"] = None
    for name, path in (
        ("uptime_seconds", "/proc/uptime"),
        ("loadavg", "/proc/loadavg"),
        ("transparent_hugepage", "/sys/kernel/mm/transparent_hugepage/enabled"),
        ("swappiness", "/proc/sys/vm/swappiness"),
        ("smt_active", "/sys/devices/system/cpu/smt/active"),
        ("kernel", "/proc/sys/kernel/osrelease"),
    ):
        try:
            state[name] = Path(path).read_text().strip()
        except OSError:
            state[name] = None
    governors = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"):
        try:
            governors.add(path.read_text().strip())
        except OSError:
            continue
    state["cpufreq_governors"] = sorted(governors)
    state["page_cache_cold"] = "unknown (not controlled)"
    return state
