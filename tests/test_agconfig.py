"""Tests for agconfig — the one-level-namespaced config object that replaced
the old flat agconfig (and, before that, the tiered descriptor system). Every
tunable field now lives on one of twelve small per-domain namespace
dataclasses (llmconfig, sandboxconfig, ...), each inheriting clone()/
update()/safe_snapshot() from the shared confignamespace base. This file
tests: type-dispatched construction (no keyword-argument form at all),
per-namespace typo protection via __slots__, clone()/update()/safe_snapshot()
at both the namespace level and the top agconfig level, the sandbox
namespace's add_mount()/remove_mount() helpers, and the handful of
process-wide module constants that this restructuring left untouched."""

import json
import sys

import pytest

from agency.configs.agconfig import (
    agconfig,
    agentconfig,
    confignamespace,
    llmconfig,
    ptraceconfig,
    sandboxconfig,
)

# `agency.configs.__init__` does `from .agconfig import agconfig`, which
# overwrites the `agconfig` attribute on the `agency.configs` package with
# the class -- so `import agency.configs.agconfig as m` resolves `m` to the
# class too (dotted-import binding is attribute lookup on the parent
# package, not a sys.modules lookup). Go through sys.modules to get the
# actual module object and its constants.
agconfig_module = sys.modules["agency.configs.agconfig"]


# ---------------------------------------------------------------------------
# Construction: zero-arg defaults, type-dispatched positional namespaces,
# no keyword-argument form at all
# ---------------------------------------------------------------------------


def test_no_args_gives_every_namespace_fresh_defaults():
    cfg = agconfig()
    assert cfg.llm.model == ""
    assert cfg.llm.provider is None
    assert cfg.sandbox.backend == "auto"
    assert cfg.sandbox.mounts == {}
    assert cfg.agent.harness == "native"


def test_positional_namespaces_are_dispatched_by_type():
    cfg = agconfig(
        llmconfig(model="x", api_key="k"),
        sandboxconfig(backend="docker"),
    )
    assert cfg.llm.model == "x"
    assert cfg.llm.api_key == "k"
    assert cfg.sandbox.backend == "docker"
    # Untouched namespaces still get fresh defaults.
    assert cfg.agent.harness == "native"


def test_positional_namespaces_dispatch_regardless_of_argument_order():
    cfg = agconfig(
        sandboxconfig(backend="podman"),
        llmconfig(model="y"),
    )
    assert cfg.llm.model == "y"
    assert cfg.sandbox.backend == "podman"


def test_unrecognized_positional_type_raises_type_error():
    with pytest.raises(TypeError):
        agconfig(object())


def test_duplicate_namespace_type_raises_type_error():
    with pytest.raises(TypeError):
        agconfig(llmconfig(), llmconfig())


def test_no_keyword_argument_form_at_all():
    with pytest.raises(TypeError):
        agconfig(llm=llmconfig(model="x"))


# ---------------------------------------------------------------------------
# Unknown field on a namespace's own constructor
# ---------------------------------------------------------------------------


def test_unknown_field_on_namespace_constructor_raises_type_error():
    with pytest.raises(TypeError):
        llmconfig(nonexistent_field=1)


# ---------------------------------------------------------------------------
# Typo protection via __slots__, at both levels
# ---------------------------------------------------------------------------


def test_setting_unknown_attribute_on_namespace_instance_raises_attribute_error():
    cfg = agconfig()
    with pytest.raises(AttributeError):
        cfg.llm.nonexistent_field = 1


def test_setting_unknown_attribute_on_agconfig_itself_raises_attribute_error():
    cfg = agconfig()
    with pytest.raises(AttributeError):
        cfg.nonexistent_field = 1


# ---------------------------------------------------------------------------
# clone() independence, at both the namespace level and the top level
# ---------------------------------------------------------------------------


def test_namespace_clone_returns_independent_copy():
    llm = llmconfig(model="original")
    llm2 = llm.clone()
    assert llm2 is not llm
    llm2.model = "mutated"
    assert llm.model == "original"
    assert llm2.model == "mutated"


def test_top_level_clone_deep_copies_every_namespace():
    cfg = agconfig(llmconfig(model="original"))
    cfg2 = cfg.clone()
    assert cfg2 is not cfg
    assert cfg2.llm is not cfg.llm
    cfg2.llm.model = "mutated"
    assert cfg.llm.model == "original"
    assert cfg2.llm.model == "mutated"


def test_top_level_clone_mutable_field_does_not_leak_either_direction():
    cfg = agconfig()
    cfg.sandbox.add_mount("data", "/host/data", "/container/data")
    cfg2 = cfg.clone()

    cfg2.sandbox.add_mount("extra", "/host/extra", "/container/extra")
    assert "extra" not in cfg.sandbox.mounts
    assert "extra" in cfg2.sandbox.mounts

    cfg.sandbox.add_mount("more", "/host/more", "/container/more")
    assert "more" not in cfg2.sandbox.mounts

    assert cfg.sandbox.mounts is not cfg2.sandbox.mounts


# ---------------------------------------------------------------------------
# update(), at both the top level and the namespace level
# ---------------------------------------------------------------------------


def test_top_level_update_merges_into_named_namespaces():
    cfg = agconfig()
    result = cfg.update(
        llm={"model": "x", "temperature": 0.7},
        sandbox={"backend": "podman"},
    )
    assert cfg.llm.model == "x"
    assert cfg.llm.temperature == 0.7
    assert cfg.sandbox.backend == "podman"
    assert result is cfg


def test_top_level_update_unknown_namespace_raises_type_error_naming_it():
    cfg = agconfig()
    with pytest.raises(TypeError, match="nonexistent"):
        cfg.update(nonexistent={"a": 1})


def test_namespace_update_sets_multiple_fields_at_once():
    cfg = agconfig()
    result = cfg.llm.update(model="x", temperature=0.5)
    assert cfg.llm.model == "x"
    assert cfg.llm.temperature == 0.5
    assert result is cfg.llm


def test_namespace_update_unknown_field_raises_type_error_naming_it():
    cfg = agconfig()
    with pytest.raises(TypeError, match="bogus_field"):
        cfg.llm.update(bogus_field=1)


def test_namespace_update_unknown_field_does_not_partially_apply():
    cfg = agconfig()
    with pytest.raises(TypeError):
        cfg.llm.update(model="should-not-stick", bogus_field=1)
    assert cfg.llm.model == ""


# ---------------------------------------------------------------------------
# safe_snapshot() — nested at the top level, secret-redacted within llm
# ---------------------------------------------------------------------------


def test_top_level_safe_snapshot_is_nested_one_key_per_namespace():
    cfg = agconfig()
    snap = cfg.safe_snapshot()
    assert isinstance(snap, dict)
    for name in (
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
    ):
        assert name in snap
        assert isinstance(snap[name], dict)


def test_top_level_safe_snapshot_reflects_namespace_field_values():
    cfg = agconfig(llmconfig(model="m"), sandboxconfig(backend="docker"))
    snap = cfg.safe_snapshot()
    assert snap["llm"]["model"] == "m"
    assert snap["sandbox"]["backend"] == "docker"


def test_safe_snapshot_omits_secret_fields_and_their_values_entirely():
    cfg = agconfig(
        llmconfig(
            api_key="api-key-secret-value",
            aws_access_key="aws-access-secret-value",
            aws_secret_key="aws-secret-key-secret-value",
            aws_session_token="aws-session-token-secret-value",
            model="m",
        )
    )
    snap = cfg.safe_snapshot()
    llm_snap = snap["llm"]

    for secret_field in ("api_key", "aws_access_key", "aws_secret_key", "aws_session_token"):
        assert secret_field not in llm_snap

    secret_values = [
        "api-key-secret-value",
        "aws-access-secret-value",
        "aws-secret-key-secret-value",
        "aws-session-token-secret-value",
    ]
    assert not any(v in secret_values for v in llm_snap.values())
    assert llm_snap["model"] == "m"

    whole_snapshot_text = json.dumps(snap)
    for secret_value in secret_values:
        assert secret_value not in whole_snapshot_text


def test_namespace_safe_snapshot_directly_is_also_secret_free():
    llm = llmconfig(
        api_key="api-key-secret-value", aws_secret_key="aws-secret-key-secret-value", model="m"
    )
    snap = llm.safe_snapshot()
    assert "api_key" not in snap
    assert "aws_secret_key" not in snap
    assert snap["model"] == "m"


def test_safe_snapshot_omits_non_json_safe_values():
    cfg = agconfig()
    cfg.llm.timing_fn = lambda: None
    snap = cfg.safe_snapshot()
    assert "timing_fn" not in snap["llm"]


def test_safe_snapshot_excludes_private_fields_at_namespace_level():
    cfg = agconfig()
    for namespace_name in ("llm", "sandbox", "agent"):
        namespace_snap = getattr(cfg, namespace_name).safe_snapshot()
        assert not any(name.startswith("_") for name in namespace_snap)


def test_whole_safe_snapshot_is_json_serializable():
    cfg = agconfig(
        llmconfig(api_key="k", model="m"),
        sandboxconfig(backend="docker"),
    )
    cfg.sandbox.add_mount("data", "/host/data", "/container/data")
    snap = cfg.safe_snapshot()
    serialized = json.dumps(snap)
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# add_mount() / remove_mount() on the sandbox namespace
# ---------------------------------------------------------------------------


def test_add_mount_default_mode_is_rw():
    cfg = agconfig()
    result = cfg.sandbox.add_mount("data", "/host/data", "/container/data")
    assert cfg.sandbox.mounts["data"] == ("/host/data", "/container/data", "rw")
    assert result is cfg.sandbox


def test_add_mount_explicit_mode():
    cfg = agconfig()
    cfg.sandbox.add_mount("data", "/host/data", "/container/data", mode="ro")
    assert cfg.sandbox.mounts["data"] == ("/host/data", "/container/data", "ro")


def test_add_mount_stringifies_host_path():
    from pathlib import Path

    cfg = agconfig()
    cfg.sandbox.add_mount("data", Path("/host/data"), "/container/data")
    host_path, container_path, mode = cfg.sandbox.mounts["data"]
    assert host_path == "/host/data"
    assert isinstance(host_path, str)


def test_remove_mount_removes_existing_entry():
    cfg = agconfig()
    cfg.sandbox.add_mount("data", "/host/data", "/container/data")
    result = cfg.sandbox.remove_mount("data")
    assert "data" not in cfg.sandbox.mounts
    assert result is cfg.sandbox


def test_remove_mount_missing_name_is_a_no_op():
    cfg = agconfig()
    result = cfg.sandbox.remove_mount("does-not-exist")
    assert cfg.sandbox.mounts == {}
    assert result is cfg.sandbox


# ---------------------------------------------------------------------------
# confignamespace base class: shared, not redefined per-namespace
# ---------------------------------------------------------------------------


def test_sandboxconfig_inherits_confignamespace():
    assert issubclass(sandboxconfig, confignamespace)
    assert issubclass(agentconfig, confignamespace)
    # clone/update/safe_snapshot are the base class's own methods, not
    # per-namespace redefinitions.
    assert sandboxconfig.clone is confignamespace.clone
    assert sandboxconfig.update is confignamespace.update
    assert sandboxconfig.safe_snapshot is confignamespace.safe_snapshot
    assert agentconfig.clone is confignamespace.clone
    assert agentconfig.update is confignamespace.update
    assert agentconfig.safe_snapshot is confignamespace.safe_snapshot


# ---------------------------------------------------------------------------
# Process-wide module-level constants — unchanged by this restructuring
# ---------------------------------------------------------------------------


def test_module_level_constants_exist_with_expected_types():
    assert isinstance(agconfig_module.DOCKER_SEMAPHORE_LIMIT, int)
    assert isinstance(agconfig_module.MIN_CPUS, float)
    assert isinstance(agconfig_module.MIN_MEMORY_MB, int)
    assert isinstance(agconfig_module.GPU_DETECT_TIMEOUT_S, (int, float))
    assert isinstance(agconfig_module.SYSCTL_DETECT_TIMEOUT_S, (int, float))
    assert isinstance(agconfig_module.MEMORY_DETECT_FALLBACK_MB, int)
    assert isinstance(agconfig_module.MARKER_MB, int)
    assert isinstance(agconfig_module.IDLE_CHECK_INTERVAL_S, (int, float))


def test_module_level_constants_are_not_fields_on_agconfig_or_any_namespace():
    cfg = agconfig()
    names = (
        "DOCKER_SEMAPHORE_LIMIT",
        "MIN_CPUS",
        "MIN_MEMORY_MB",
        "GPU_DETECT_TIMEOUT_S",
        "SYSCTL_DETECT_TIMEOUT_S",
        "MEMORY_DETECT_FALLBACK_MB",
        "MARKER_MB",
        "IDLE_CHECK_INTERVAL_S",
    )
    for name in names:
        assert not hasattr(cfg, name)
        assert not hasattr(cfg.sandbox, name)
        assert not hasattr(cfg.llm, name)


# ---------------------------------------------------------------------------
# ptraceconfig.syscalls default
# ---------------------------------------------------------------------------


def test_ptraceconfig_default_syscalls_include_network_destinations():
    """connect/bind/sendto are decoded with real address/port by the tracer
    (agency/harness/ptrace/_tracer_loop.py's _resolve_syscall_args) -- on by
    default so a caller gets network-destination visibility without having
    to know that decode table exists. execve/execveat stay too: process
    launches are the original, still-needed default."""
    cfg = ptraceconfig()
    assert set(cfg.syscalls) == {"execve", "execveat", "connect", "bind", "sendto"}
