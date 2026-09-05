"""Tests for agconfig — the one flat, typed config dataclass that replaced
the old tiered descriptor system (GlobalConfigParam/StaticConfigParam/
DynamicConfigParam, owner-scoped *Fields/*Config classes). This file tests
the new class's own behavior only: construction/defaults, typo protection,
clone()/update(), safe_snapshot()'s JSON-safety and secret redaction, mount
helpers, and the handful of process-wide module constants it still exposes."""

import sys

import pytest

from agency.configs.agconfig import agconfig

# `agency.configs.__init__` does `from .agconfig import agconfig`, which
# overwrites the `agconfig` attribute on the `agency.configs` package with
# the class -- so `import agency.configs.agconfig as m` resolves `m` to the
# class too (dotted-import binding is attribute lookup on the parent
# package, not a sys.modules lookup). Go through sys.modules to get the
# actual module object and its constants.
agconfig_module = sys.modules["agency.configs.agconfig"]


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


def test_no_args_produces_documented_defaults():
    cfg = agconfig()
    assert cfg.provider is None
    assert cfg.model == ""
    assert cfg.backend == "auto"
    assert cfg.harness == "native"
    assert cfg.api_key is None
    assert cfg.base_image == "agency-sandbox:latest"
    assert cfg.mounts == {}
    assert cfg.persistent is False


def test_kwargs_set_the_given_fields():
    cfg = agconfig(provider="anthropic", model="claude-sonnet-5", temperature=0.5)
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-5"
    assert cfg.temperature == 0.5
    # Untouched fields keep their defaults.
    assert cfg.backend == "auto"


# ---------------------------------------------------------------------------
# Unknown-kwarg rejection at construction and attribute typo protection
# ---------------------------------------------------------------------------


def test_unknown_constructor_kwarg_raises_type_error():
    with pytest.raises(TypeError):
        agconfig(nonexistent_field=1)


def test_setting_unknown_attribute_on_instance_raises_attribute_error():
    cfg = agconfig()
    with pytest.raises(AttributeError):
        cfg.nonexistent_field = 1


def test_typo_on_known_field_name_still_raises():
    cfg = agconfig()
    with pytest.raises(AttributeError):
        cfg.modle = "typo"


# ---------------------------------------------------------------------------
# clone() independence
# ---------------------------------------------------------------------------


def test_clone_returns_a_different_but_equal_object():
    cfg = agconfig(model="m", temperature=0.7)
    cfg2 = cfg.clone()
    assert cfg2 is not cfg
    assert cfg2.model == cfg.model
    assert cfg2.temperature == cfg.temperature


def test_clone_mutation_of_scalar_field_does_not_affect_original():
    cfg = agconfig(model="original")
    cfg2 = cfg.clone()
    cfg2.model = "mutated"
    assert cfg.model == "original"
    assert cfg2.model == "mutated"


def test_clone_mutation_of_mounts_dict_does_not_affect_original():
    cfg = agconfig()
    cfg.add_mount("data", "/host/data", "/container/data")
    cfg2 = cfg.clone()
    cfg2.add_mount("extra", "/host/extra", "/container/extra")
    assert "extra" not in cfg.mounts
    assert "extra" in cfg2.mounts
    # Original mount present in both, but the dicts are independent objects.
    assert cfg.mounts["data"] == cfg2.mounts["data"]
    assert cfg.mounts is not cfg2.mounts


def test_clone_mutation_on_original_does_not_affect_clone():
    cfg = agconfig()
    cfg.add_mount("data", "/host/data", "/container/data")
    cfg2 = cfg.clone()
    cfg.add_mount("more", "/host/more", "/container/more")
    assert "more" not in cfg2.mounts


# ---------------------------------------------------------------------------
# update() partial merge
# ---------------------------------------------------------------------------


def test_update_sets_multiple_fields_at_once():
    cfg = agconfig()
    result = cfg.update(model="m", temperature=0.9, backend="docker")
    assert cfg.model == "m"
    assert cfg.temperature == 0.9
    assert cfg.backend == "docker"
    assert result is cfg


def test_update_unknown_field_raises_type_error_naming_it():
    cfg = agconfig()
    with pytest.raises(TypeError, match="bogus_field"):
        cfg.update(bogus_field=1)


def test_update_unknown_field_does_not_partially_apply():
    cfg = agconfig()
    with pytest.raises(TypeError):
        cfg.update(model="should-not-stick", bogus_field=1)
    assert cfg.model == ""


def test_update_multiple_unknown_fields_names_all_of_them():
    cfg = agconfig()
    with pytest.raises(TypeError) as excinfo:
        cfg.update(bogus_one=1, bogus_two=2)
    message = str(excinfo.value)
    assert "bogus_one" in message
    assert "bogus_two" in message


# ---------------------------------------------------------------------------
# safe_snapshot() — JSON-safety and secret redaction
# ---------------------------------------------------------------------------


def test_safe_snapshot_returns_a_flat_dict():
    cfg = agconfig(model="m")
    snap = cfg.safe_snapshot()
    assert isinstance(snap, dict)
    assert snap["model"] == "m"


def test_safe_snapshot_omits_secret_fields_and_their_values_entirely():
    cfg = agconfig(
        api_key="k",
        aws_access_key="ak",
        aws_secret_key="sk-secret",
        aws_session_token="tok-secret",
    )
    snap = cfg.safe_snapshot()

    for secret_field in ("api_key", "aws_access_key", "aws_secret_key", "aws_session_token"):
        assert secret_field not in snap

    secret_values = ["k", "ak", "sk-secret", "tok-secret"]
    assert not any(k in secret_values for k in snap.keys())
    assert not any(v in secret_values for v in snap.values())


def test_safe_snapshot_omits_secrets_even_when_none():
    cfg = agconfig()
    snap = cfg.safe_snapshot()
    for secret_field in ("api_key", "aws_access_key", "aws_secret_key", "aws_session_token"):
        assert secret_field not in snap


def test_safe_snapshot_omits_non_json_safe_values():
    cfg = agconfig()
    cfg.timing_fn = lambda: None
    snap = cfg.safe_snapshot()
    assert "timing_fn" not in snap


def test_safe_snapshot_includes_normal_json_safe_fields():
    cfg = agconfig(model="claude-sonnet-5", temperature=0.3, backend="docker")
    snap = cfg.safe_snapshot()
    assert snap["model"] == "claude-sonnet-5"
    assert snap["temperature"] == 0.3
    assert snap["backend"] == "docker"


def test_safe_snapshot_keeps_json_safe_containers():
    cfg = agconfig()
    cfg.add_mount("data", "/host/data", "/container/data")
    snap = cfg.safe_snapshot()
    assert snap["mounts"] == {"data": ("/host/data", "/container/data", "rw")}


def test_safe_snapshot_excludes_private_fields():
    cfg = agconfig()
    snap = cfg.safe_snapshot()
    assert not any(name.startswith("_") for name in snap)


# ---------------------------------------------------------------------------
# add_mount() / remove_mount()
# ---------------------------------------------------------------------------


def test_add_mount_default_mode_is_rw():
    cfg = agconfig()
    result = cfg.add_mount("data", "/host/data", "/container/data")
    assert cfg.mounts["data"] == ("/host/data", "/container/data", "rw")
    assert result is cfg


def test_add_mount_explicit_mode():
    cfg = agconfig()
    cfg.add_mount("data", "/host/data", "/container/data", mode="ro")
    assert cfg.mounts["data"] == ("/host/data", "/container/data", "ro")


def test_add_mount_stringifies_host_path():
    from pathlib import Path

    cfg = agconfig()
    cfg.add_mount("data", Path("/host/data"), "/container/data")
    host_path, container_path, mode = cfg.mounts["data"]
    assert host_path == "/host/data"
    assert isinstance(host_path, str)


def test_remove_mount_removes_existing_entry():
    cfg = agconfig()
    cfg.add_mount("data", "/host/data", "/container/data")
    result = cfg.remove_mount("data")
    assert "data" not in cfg.mounts
    assert result is cfg


def test_remove_mount_missing_name_is_a_no_op():
    cfg = agconfig()
    result = cfg.remove_mount("does-not-exist")
    assert cfg.mounts == {}
    assert result is cfg


# ---------------------------------------------------------------------------
# Process-wide module-level constants
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


def test_module_level_constants_are_not_fields_on_agconfig():
    cfg = agconfig()
    for name in (
        "DOCKER_SEMAPHORE_LIMIT",
        "MIN_CPUS",
        "MIN_MEMORY_MB",
        "GPU_DETECT_TIMEOUT_S",
        "SYSCTL_DETECT_TIMEOUT_S",
        "MEMORY_DETECT_FALLBACK_MB",
        "MARKER_MB",
        "IDLE_CHECK_INTERVAL_S",
    ):
        assert not hasattr(cfg, name)
