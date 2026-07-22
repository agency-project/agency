"""Tests for agconfig — tiered override store, ConfigParam descriptors, registry, GLOBAL, _OwnerView.

Includes agLLMBackendConfig/agllm_backend tests (formerly test_agllmconfig.py
-- moved here since it's the same config system, just exercised through a
real framework owner instead of test-only ones)."""

import httpx
import pytest
from unittest.mock import MagicMock, patch

from agency.agconfig import (
    agConfig,
    _ConfigParam,
    GlobalConfigParam,
    StaticConfigParam,
    DynamicConfigParam,
    _OwnerView,
    _AgConfigViewBase,
)
from agency.agllm import agllm
from agency.agllm_backends import (
    agllm_backend,
    AgLLMBackendFields,
    agLLMBackendConfig,
    agVLLMBackendConfig,
    agOpenAIBackendConfig,
    agAnthropicBackendConfig,
    agBedrockBackendConfig,
)
from agency.agllm_backends.openai import _OpenAICompatibleBackend
from agency.agllm_backends.bedrock import (
    _OpenAICompatibleBedrockBackend,
    _AnthropicAWSBackend,
    _AnthropicBedrockBackend,
)
from agency.agllm_backends.anthropic import _AnthropicBackend
from agency.agsandbox import _AgSandboxFields, agSandboxConfig
from agency.agsandbox_backends import AgSandboxBackendFields
from agency.agllm import _AgLLMFields, agLLMConfig
from agency.agtool import _AgToolFields, agToolConfig
from agency.agresources import _AgResourcePoolFields, agResourcePoolConfig
from agency.agschema import _AgSchemaFields, agSchemaConfig
from agency.agutil import _AgUtilFields, agUtilConfig
from agency.aglog import agLogConfig
from agency.agent import agAgentConfig
from agency.agskill import agSkillConfig


def _cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


# ---------------------------------------------------------------------------
# Test-only owners, registered once at import time. Unique owner names keep
# these isolated from real framework owners (agllm, agtool, ...) and from
# each other, so no cross-test or cross-file interference through the
# process-wide FIELD_REGISTRY / GLOBAL singleton.
# ---------------------------------------------------------------------------


class _OwnerA:
    # Each GlobalConfigParam field below is touched by exactly one test --
    # tier-1 fields lock process-wide on first read, so reusing a field
    # across tests would make results depend on test execution order.
    tier1_lifecycle = GlobalConfigParam("test_agconfig_a", default=1)
    tier1_shared = GlobalConfigParam("test_agconfig_a", default=2)
    tier1_ownerview_route = GlobalConfigParam("test_agconfig_a", default=3)
    tier2_field = StaticConfigParam("test_agconfig_a", default="static-default")
    tier3_field = DynamicConfigParam("test_agconfig_a", default="dynamic-default")
    shared_name = DynamicConfigParam("test_agconfig_a", default="a-default")

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig


class _OwnerB:
    """A second owner with a field of the same name as _OwnerA's, to prove
    no collision/ambiguity across owners."""

    shared_name = DynamicConfigParam("test_agconfig_b", default="b-default")

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig


# ---------------------------------------------------------------------------
# Storage primitives: get / get_static / set / clone
# ---------------------------------------------------------------------------


def test_get_returns_default_and_never_locks():
    cfg = agConfig()
    assert cfg.get("owner", "field", "default") == "default"
    cfg.set("owner", "field", "value")
    assert cfg.get("owner", "field", "default") == "value"
    # A plain get() never locks -- set() afterward still succeeds.
    cfg.set("owner", "field", "value2")
    assert cfg.get("owner", "field", "default") == "value2"


def test_get_static_locks_and_blocks_further_set():
    cfg = agConfig()
    cfg.set("owner", "field", "initial")
    assert cfg.get_static("owner", "field", "default") == "initial"
    with pytest.raises(ValueError, match="already read as static"):
        cfg.set("owner", "field", "later")


def test_set_before_get_static_is_unrestricted():
    cfg = agConfig()
    cfg.set("owner", "field", "one")
    cfg.set("owner", "field", "two")
    assert cfg.get_static("owner", "field", "default") == "two"


def test_clone_carries_data_but_not_lock_history():
    cfg = agConfig()
    cfg.set("owner", "field", "value")
    cfg.get_static("owner", "field", "default")  # locks on cfg
    with pytest.raises(ValueError):
        cfg.set("owner", "field", "blocked")

    clone = cfg.clone()
    assert clone.get("owner", "field", "default") == "value"
    clone.set("owner", "field", "overridden")  # no lock history on the clone
    assert clone.get("owner", "field", "default") == "overridden"
    # The original is untouched by the clone's write.
    assert cfg.get("owner", "field", "default") == "value"


# ---------------------------------------------------------------------------
# dynamic_snapshot() — the webui config editor's data source
# ---------------------------------------------------------------------------


def test_dynamic_snapshot_includes_dynamic_field_with_default():
    cfg = agConfig()
    snap = cfg.dynamic_snapshot()
    assert snap["test_agconfig_a"]["tier3_field"] == "dynamic-default"


def test_dynamic_snapshot_reflects_override():
    cfg = agConfig({"test_agconfig_a": {"tier3_field": "overridden"}})
    snap = cfg.dynamic_snapshot()
    assert snap["test_agconfig_a"]["tier3_field"] == "overridden"


def test_dynamic_snapshot_excludes_static_and_global_fields():
    """Static fields cache per-instance at first read, and Global fields
    always read agConfig.GLOBAL regardless of this agconfig -- editing either
    through a later change_config() call has no observable effect, so
    dynamic_snapshot() must not offer them as if they were live-editable."""
    cfg = agConfig({"test_agconfig_a": {"tier2_field": "x", "tier1_lifecycle": 99}})
    snap = cfg.dynamic_snapshot()
    assert "tier2_field" not in snap.get("test_agconfig_a", {})
    assert "tier1_lifecycle" not in snap.get("test_agconfig_a", {})


def test_dynamic_snapshot_skips_non_json_safe_values():
    cfg = agConfig({"test_agconfig_a": {"tier3_field": object()}})
    snap = cfg.dynamic_snapshot()
    assert "tier3_field" not in snap.get("test_agconfig_a", {})


def test_dynamic_snapshot_keeps_json_safe_containers():
    cfg = agConfig({"test_agconfig_a": {"tier3_field": {"a": [1, 2, "x"], "b": None}}})
    snap = cfg.dynamic_snapshot()
    assert snap["test_agconfig_a"]["tier3_field"] == {"a": [1, 2, "x"], "b": None}


def test_dynamic_snapshot_two_owners_do_not_collide():
    cfg = agConfig(
        {
            "test_agconfig_a": {"shared_name": "from-a"},
            "test_agconfig_b": {"shared_name": "from-b"},
        }
    )
    snap = cfg.dynamic_snapshot()
    assert snap["test_agconfig_a"]["shared_name"] == "from-a"
    assert snap["test_agconfig_b"]["shared_name"] == "from-b"


def test_dict_based_init():
    cfg = agConfig({"owner": {"field": "value", "other": 1}})
    assert cfg.get("owner", "field") == "value"
    assert cfg.get("owner", "other") == 1
    assert cfg.get("owner", "missing", "fallback") == "fallback"


def test_dict_based_init_does_not_alias_input():
    source = {"owner": {"field": "value"}}
    cfg = agConfig(source)
    source["owner"]["field"] = "mutated-after-construction"
    assert cfg.get("owner", "field") == "value"


# ---------------------------------------------------------------------------
# Registration: __set_name__ populates FIELD_REGISTRY purely by import,
# no instance of the owning class required.
# ---------------------------------------------------------------------------


def test_fields_registered_without_constructing_any_instance():
    assert ("test_agconfig_a", "tier1_lifecycle") in agConfig.FIELD_REGISTRY
    assert ("test_agconfig_a", "tier2_field") in agConfig.FIELD_REGISTRY
    assert ("test_agconfig_a", "tier3_field") in agConfig.FIELD_REGISTRY
    assert ("test_agconfig_b", "shared_name") in agConfig.FIELD_REGISTRY
    knob = agConfig.FIELD_REGISTRY[("test_agconfig_a", "tier3_field")]
    assert isinstance(knob, DynamicConfigParam)
    assert knob.owner == "test_agconfig_a"
    assert knob.name == "tier3_field"
    assert knob.default == "dynamic-default"


def test_duplicate_registration_raises_at_class_definition_time():
    with pytest.raises(ValueError, match="already registered"):

        class _Colliding:
            tier2_field = StaticConfigParam("test_agconfig_a", default="oops")


def test_class_level_descriptor_access_returns_descriptor_itself():
    # Accessing a ConfigParam on the class (not an instance) returns the
    # descriptor object, not a resolved value -- callers needing the default
    # use `.default` on it (see agsandbox.py's agSandboxConfig.base_image).
    assert isinstance(_OwnerA.tier3_field, DynamicConfigParam)
    assert _OwnerA.tier3_field.default == "dynamic-default"


# ---------------------------------------------------------------------------
# Tier 1: GlobalConfigParam
# ---------------------------------------------------------------------------


def test_global_write_before_read_then_locks_after_read():
    # A GlobalConfigParam ignores whichever agconfig the instance holds --
    # every read/write routes through agConfig.GLOBAL regardless, so writing
    # via cfg.set(...) directly would silently do nothing observable here;
    # the write must go through the descriptor itself (or agConfig.GLOBAL).
    cfg = agConfig()
    obj = _OwnerA(agconfig=cfg)
    obj.tier1_lifecycle = 8  # descriptor-level write, before any read -- succeeds
    assert obj.tier1_lifecycle == 8  # read locks it, process-wide
    with pytest.raises(ValueError, match="tier-1 \\(global\\)"):
        obj.tier1_lifecycle = 16
    # A second, unrelated instance is also blocked -- same shared GLOBAL.
    other = _OwnerA()
    with pytest.raises(ValueError, match="tier-1 \\(global\\)"):
        other.tier1_lifecycle = 99


def test_global_shared_across_instances_regardless_of_agconfig():
    obj_with_cfg = _OwnerA(agconfig=agConfig())
    obj_with_cfg.tier1_shared = 42  # descriptor-level write, routes to GLOBAL
    obj_without_cfg = _OwnerA()  # no agconfig at all
    obj_other_cfg = _OwnerA(agconfig=agConfig())  # different, empty agconfig
    assert obj_with_cfg.tier1_shared == 42
    assert obj_without_cfg.tier1_shared == 42
    assert obj_other_cfg.tier1_shared == 42


def test_all_three_tiers_are_configparam_subclasses():
    assert issubclass(GlobalConfigParam, _ConfigParam)
    assert issubclass(StaticConfigParam, _ConfigParam)
    assert issubclass(DynamicConfigParam, _ConfigParam)


# ---------------------------------------------------------------------------
# Tier 2: StaticConfigParam
# ---------------------------------------------------------------------------


def test_static_resolves_default_with_no_agconfig():
    obj = _OwnerA()
    assert obj.tier2_field == "static-default"


def test_static_resolves_once_and_caches_per_instance():
    cfg = agConfig()
    cfg.set("test_agconfig_a", "tier2_field", "first-read")
    obj = _OwnerA(agconfig=cfg)
    assert obj.tier2_field == "first-read"
    # Underlying agconfig is now locked (get_static locked it on first read);
    # the instance's own cached value is unaffected either way.
    assert obj.tier2_field == "first-read"
    with pytest.raises(ValueError, match="already read as static"):
        cfg.set("test_agconfig_a", "tier2_field", "too-late")


def test_static_descriptor_set_always_raises():
    obj = _OwnerA(agconfig=agConfig())
    with pytest.raises(AttributeError, match="fixed once at construction"):
        obj.tier2_field = "nope"


def test_static_different_instances_can_resolve_different_values():
    cfg1 = agConfig({"test_agconfig_a": {"tier2_field": "value-1"}})
    cfg2 = agConfig({"test_agconfig_a": {"tier2_field": "value-2"}})
    obj1 = _OwnerA(agconfig=cfg1)
    obj2 = _OwnerA(agconfig=cfg2)
    assert obj1.tier2_field == "value-1"
    assert obj2.tier2_field == "value-2"


# ---------------------------------------------------------------------------
# Tier 3: DynamicConfigParam
# ---------------------------------------------------------------------------


def test_dynamic_live_read_reflects_later_writes():
    cfg = agConfig()
    obj = _OwnerA(agconfig=cfg)
    assert obj.tier3_field == "dynamic-default"
    cfg.set("test_agconfig_a", "tier3_field", "updated")
    assert obj.tier3_field == "updated"


def test_dynamic_settable_through_descriptor():
    cfg = agConfig()
    obj = _OwnerA(agconfig=cfg)
    obj.tier3_field = "written-via-instance"
    assert cfg.get("test_agconfig_a", "tier3_field") == "written-via-instance"
    assert obj.tier3_field == "written-via-instance"


def test_dynamic_without_agconfig_reads_default_and_set_raises():
    obj = _OwnerA()  # no agconfig
    assert obj.tier3_field == "dynamic-default"
    with pytest.raises(AttributeError, match="no agconfig"):
        obj.tier3_field = "x"


# ---------------------------------------------------------------------------
# _OwnerView / cfg.owner.field nested syntax
# ---------------------------------------------------------------------------


def test_ownerview_returned_for_known_owner():
    cfg = agConfig()
    view = cfg.test_agconfig_a
    assert isinstance(view, _OwnerView)


def test_ownerview_unknown_owner_raises():
    cfg = agConfig()
    with pytest.raises(AttributeError, match="no owner"):
        cfg.not_a_real_owner


def test_ownerview_unknown_field_raises_on_get_and_set():
    cfg = agConfig()
    with pytest.raises(AttributeError, match="no registered field"):
        cfg.test_agconfig_a.totally_made_up
    with pytest.raises(AttributeError, match="no registered field"):
        cfg.test_agconfig_a.totally_made_up = 1


def test_ownerview_dynamic_field_roundtrip():
    cfg = agConfig()
    cfg.test_agconfig_a.tier3_field = "via-nested-syntax"
    assert cfg.test_agconfig_a.tier3_field == "via-nested-syntax"
    assert cfg.get("test_agconfig_a", "tier3_field") == "via-nested-syntax"


def test_ownerview_static_field_write_bypasses_descriptor_raise():
    # cfg.owner.field = X talks to the underlying store directly, not through
    # the descriptor's __set__ -- this is what makes pre-configuration (before
    # any real instance exists) possible for tier-2 fields.
    cfg = agConfig()
    cfg.test_agconfig_a.tier2_field = "pre-configured"
    obj = _OwnerA(agconfig=cfg)
    assert obj.tier2_field == "pre-configured"


def test_ownerview_disambiguates_same_field_name_across_owners():
    cfg = agConfig()
    cfg.test_agconfig_a.shared_name = "from-a"
    cfg.test_agconfig_b.shared_name = "from-b"
    assert cfg.test_agconfig_a.shared_name == "from-a"
    assert cfg.test_agconfig_b.shared_name == "from-b"
    obj_a = _OwnerA(agconfig=cfg)
    obj_b = _OwnerB(agconfig=cfg)
    assert obj_a.shared_name == "from-a"
    assert obj_b.shared_name == "from-b"


def test_ownerview_global_field_write_routes_to_global_not_cfg():
    cfg = agConfig()
    cfg.test_agconfig_a.tier1_ownerview_route = 77
    # Routed straight to GLOBAL -- cfg itself never stored it.
    assert agConfig.GLOBAL.get("test_agconfig_a", "tier1_ownerview_route") == 77
    assert (
        cfg.get("test_agconfig_a", "tier1_ownerview_route", "unset-on-cfg-itself")
        == "unset-on-cfg-itself"
    )
    # Reading through the view (on this or any other agConfig instance)
    # also routes to GLOBAL and sees the same value.
    assert cfg.test_agconfig_a.tier1_ownerview_route == 77
    assert agConfig().test_agconfig_a.tier1_ownerview_route == 77


# ---------------------------------------------------------------------------
# _AgConfigViewBase -- generic per-owner "*Config" view (agLLMBackendConfig,
# agAgentConfig, agSandboxConfig, ...). Dedicated test-only owner/fields so
# these don't collide with real framework owners or with _OwnerA/_OwnerB above.
# ---------------------------------------------------------------------------


class _ViewOwnerFields:
    v_global = GlobalConfigParam("test_agconfig_view_owner", default="g-default")
    v_static = StaticConfigParam("test_agconfig_view_owner", default="s-default")
    v_dynamic = DynamicConfigParam("test_agconfig_view_owner", default="d-default")


class _ViewOwnerConfig(_AgConfigViewBase):
    _OWNER = "test_agconfig_view_owner"


class _OtherViewOwnerFields:
    other_field = DynamicConfigParam("test_agconfig_other_view_owner", default=None)


class _OtherViewOwnerConfig(_AgConfigViewBase):
    _OWNER = "test_agconfig_other_view_owner"


def test_view_creates_fresh_agconfig_when_none_given():
    view = _ViewOwnerConfig()
    assert isinstance(view.agconfig, agConfig)


def test_view_wraps_given_agconfig_instead_of_copying():
    cfg = agConfig()
    view = _ViewOwnerConfig(cfg)
    assert view.agconfig is cfg


def test_view_constructor_kwargs_apply_immediately():
    view = _ViewOwnerConfig(v_dynamic="set-at-construction")
    assert view.agconfig.get("test_agconfig_view_owner", "v_dynamic") == "set-at-construction"


def test_view_update_sets_dynamic_field_on_wrapped_agconfig():
    cfg = agConfig()
    _ViewOwnerConfig(cfg).update(v_dynamic="via-update")
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "via-update"


def test_view_update_returns_self_for_chaining():
    view = _ViewOwnerConfig()
    assert view.update(v_dynamic="x") is view


def test_view_update_routes_global_field_to_agconfig_global_not_wrapped_cfg():
    cfg = agConfig()
    _ViewOwnerConfig(cfg).update(v_global="routed-to-global")
    assert agConfig.GLOBAL.get("test_agconfig_view_owner", "v_global") == "routed-to-global"
    # The wrapped, non-GLOBAL agConfig never stored it.
    assert cfg.get("test_agconfig_view_owner", "v_global", "absent") == "absent"


def test_view_update_static_field_before_any_read_is_unrestricted():
    cfg = agConfig()
    _ViewOwnerConfig(cfg).update(v_static="pre-configured")
    assert cfg.get_static("test_agconfig_view_owner", "v_static") == "pre-configured"


def test_view_update_unknown_field_raises_type_error_naming_the_field():
    with pytest.raises(TypeError, match="bogus"):
        _ViewOwnerConfig().update(bogus=1)


def test_view_update_unknown_field_error_names_the_subclass():
    with pytest.raises(TypeError, match="_ViewOwnerConfig"):
        _ViewOwnerConfig().update(bogus=1)


def test_view_rejects_unknown_field_before_applying_any_valid_ones():
    """A mix of valid + unknown fields must apply none of them, not a partial write."""
    cfg = agConfig()
    with pytest.raises(TypeError):
        _ViewOwnerConfig(cfg).update(v_dynamic="should-not-stick", bogus=1)
    assert cfg.get("test_agconfig_view_owner", "v_dynamic", "absent") == "absent"


def test_two_different_view_subclasses_do_not_collide_on_same_agconfig():
    cfg = agConfig()
    _ViewOwnerConfig(cfg).update(v_dynamic="from-view-a")
    _OtherViewOwnerConfig(cfg).update(other_field="from-view-b")
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "from-view-a"
    assert cfg.get("test_agconfig_other_view_owner", "other_field") == "from-view-b"


# ---------------------------------------------------------------------------
# agConfig(*sources) -- variadic merge constructor
# ---------------------------------------------------------------------------


def test_agconfig_no_args_is_empty():
    cfg = agConfig()
    assert cfg.data == {}


def test_agconfig_single_dict_source_backward_compatible():
    cfg = agConfig({"test_agconfig_view_owner": {"v_dynamic": "value"}})
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "value"


def test_agconfig_merges_from_another_agconfig_instance():
    src = agConfig({"test_agconfig_view_owner": {"v_dynamic": "from-src"}})
    cfg = agConfig(src)
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "from-src"
    assert cfg is not src


def test_agconfig_merges_from_a_view():
    view = _ViewOwnerConfig(v_dynamic="from-view")
    cfg = agConfig(view)
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "from-view"
    assert cfg is not view.agconfig


def test_agconfig_merges_two_views_of_different_owners():
    cfg = agConfig(
        _ViewOwnerConfig(v_dynamic="llm-ish"),
        _OtherViewOwnerConfig(other_field="sandbox-ish"),
    )
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "llm-ish"
    assert cfg.get("test_agconfig_other_view_owner", "other_field") == "sandbox-ish"


def test_agconfig_later_source_wins_on_conflicting_field():
    cfg = agConfig(
        {"test_agconfig_view_owner": {"v_dynamic": "first"}},
        {"test_agconfig_view_owner": {"v_dynamic": "second"}},
    )
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "second"


def test_agconfig_later_source_only_overwrites_its_own_fields():
    cfg = agConfig(
        {"test_agconfig_view_owner": {"v_dynamic": "keep-me", "v_static": "also-keep"}},
        {"test_agconfig_view_owner": {"v_dynamic": "overwritten"}},
    )
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "overwritten"
    assert cfg.get("test_agconfig_view_owner", "v_static") == "also-keep"


def test_agconfig_none_sources_are_skipped():
    cfg = agConfig(None, {"test_agconfig_view_owner": {"v_dynamic": "value"}}, None)
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "value"


def test_agconfig_invalid_source_type_raises_type_error():
    with pytest.raises(TypeError, match="int"):
        agConfig(42)


def test_agconfig_does_not_alias_view_agconfig_data():
    view = _ViewOwnerConfig(v_dynamic="original")
    cfg = agConfig(view)
    cfg.set("test_agconfig_view_owner", "v_dynamic", "mutated-on-merged-copy")
    # The view's own agConfig is untouched by mutating the merged copy.
    assert view.agconfig.get("test_agconfig_view_owner", "v_dynamic") == "original"


def test_clone_still_works_under_variadic_constructor():
    cfg = agConfig()
    cfg.set("test_agconfig_view_owner", "v_dynamic", "value")
    cfg.get_static("test_agconfig_view_owner", "v_dynamic")  # lock it
    clone = cfg.clone()
    clone.set("test_agconfig_view_owner", "v_dynamic", "overridden")  # no lock history
    assert clone.get("test_agconfig_view_owner", "v_dynamic") == "overridden"


# ---------------------------------------------------------------------------
# _AgConfigViewBase._ALLOWED_FIELDS -- restricting a view to a subset of its
# owner's registered fields (test-only owner, mirroring the mechanism the
# real agVLLMBackendConfig/agAnthropicBackendConfig/... classes use).
# ---------------------------------------------------------------------------


class _RestrictedViewOwnerConfig(_ViewOwnerConfig):
    _ALLOWED_FIELDS = frozenset({"v_dynamic"})


def test_allowed_fields_accepts_a_listed_field():
    cfg = agConfig(_RestrictedViewOwnerConfig(v_dynamic="ok"))
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "ok"


def test_allowed_fields_rejects_a_registered_but_unlisted_field():
    # v_static IS registered under this owner (via _ViewOwnerFields) -- just
    # not in this subclass's allowlist.
    with pytest.raises(TypeError, match="v_static"):
        _RestrictedViewOwnerConfig(v_static="nope")


def test_allowed_fields_rejection_is_all_or_nothing():
    cfg = agConfig()
    with pytest.raises(TypeError):
        _RestrictedViewOwnerConfig(cfg).update(v_dynamic="should-not-stick", v_static="not-allowed")
    assert cfg.get("test_agconfig_view_owner", "v_dynamic", "absent") == "absent"


def test_unrestricted_parent_view_still_accepts_everything():
    # _ViewOwnerConfig (the parent, no _ALLOWED_FIELDS) is unaffected by its
    # restricted subclass.
    cfg = agConfig(_ViewOwnerConfig(v_static="fine", v_dynamic="also-fine"))
    assert cfg.get("test_agconfig_view_owner", "v_static") == "fine"
    assert cfg.get("test_agconfig_view_owner", "v_dynamic") == "also-fine"


# ---------------------------------------------------------------------------
# agLLMBackendConfig / agllm_backend -- real framework usage of the view
# system above (formerly test_agllmconfig.py).
# ---------------------------------------------------------------------------


class TestAgLLMBackendFieldDescriptors:
    def test_model_is_dynamic_config_param(self):
        assert isinstance(AgLLMBackendFields.__dict__["model"], DynamicConfigParam)

    def test_api_key_is_dynamic_config_param(self):
        assert isinstance(AgLLMBackendFields.__dict__["api_key"], DynamicConfigParam)

    def test_model_listing_timeout_is_global_config_param(self):
        assert isinstance(
            AgLLMBackendFields.__dict__["model_listing_timeout_seconds"], GlobalConfigParam
        )

    def test_default_max_tokens_is_global_config_param(self):
        assert isinstance(AgLLMBackendFields.__dict__["default_max_tokens"], GlobalConfigParam)

    def test_fields_registered_under_agllm_backend_owner(self):
        assert ("agllm_backend", "model") in agConfig.FIELD_REGISTRY
        assert ("agllm_backend", "api_key") in agConfig.FIELD_REGISTRY
        assert ("agllm_backend", "workspace_id") in agConfig.FIELD_REGISTRY

    def test_bare_instance_has_no_agconfig_and_reads_defaults(self):
        """Without a private agconfig, DynamicConfigParam falls back to its
        declared default -- lets AgLLMBackendFields() be used as a throwaway
        (e.g. to read the GlobalConfigParam tunables)."""
        bare = AgLLMBackendFields()
        assert bare.model == ""
        assert bare.api_key is None


class TestAgLLMBackendConfig:
    """The canonical construction form is always agConfig(agLLMBackendConfig(...)),
    never agLLMBackendConfig(...).agconfig directly -- see docs/agconfig.md."""

    def test_agconfig_property_returns_agconfig_instance(self):
        cfg = agConfig(agLLMBackendConfig(model="m"))
        assert isinstance(cfg, agConfig)

    def test_fields_readable_via_owner_view(self):
        cfg = agConfig(agLLMBackendConfig(model="m", api_key="k", base_url="http://x"))
        assert cfg.agllm_backend.model == "m"
        assert cfg.agllm_backend.api_key == "k"
        assert cfg.agllm_backend.base_url == "http://x"

    def test_result_usable_directly_by_agllm(self):
        cfg = agConfig(agLLMBackendConfig(model="m", api_key="k"))
        llm = agllm(cfg, context_limit=128_000)
        assert llm.backend.model == "m"
        assert llm.backend.api_key == "k"

    def test_unknown_field_raises_type_error(self):
        with pytest.raises(TypeError, match="bogus_field"):
            agLLMBackendConfig(model="m", bogus_field=1)

    def test_no_fields_returns_empty_but_valid_agconfig(self):
        cfg = agConfig(agLLMBackendConfig())
        assert isinstance(cfg, agConfig)
        assert cfg.get("agllm_backend", "model") is None

    def test_equivalent_to_raw_nested_dict(self):
        via_helper = agConfig(agLLMBackendConfig(model="m", api_key="k"))
        via_dict = agConfig({"agllm_backend": {"model": "m", "api_key": "k"}})
        assert via_helper.data == via_dict.data

    def test_each_call_returns_independent_agconfig(self):
        cfg_a = agConfig(agLLMBackendConfig(model="a"))
        cfg_b = agConfig(agLLMBackendConfig(model="b"))
        assert cfg_a.agllm_backend.model == "a"
        assert cfg_b.agllm_backend.model == "b"

    def test_update_can_target_an_existing_agconfig(self):
        cfg = agConfig()
        result = agLLMBackendConfig(cfg).update(model="m", api_key="k")
        assert result.agconfig is cfg
        assert cfg.agllm_backend.model == "m"

    def test_composes_with_agconfig_merge_constructor(self):
        merged = agConfig(agLLMBackendConfig(model="m", api_key="k"))
        assert merged.agllm_backend.model == "m"
        assert merged.agllm_backend.api_key == "k"


class TestBackendOwnsPrivateAgConfig:
    def test_backend_is_agllmbackendfields(self):
        backend = agllm_backend.for_config(_cfg(model="m"))
        assert isinstance(backend, AgLLMBackendFields)

    def test_backend_clones_the_given_agconfig(self):
        """The backend's own agconfig is independent of the caller's --
        mutating cfg afterward must not affect an already-built backend."""
        cfg = _cfg(model="m")
        backend = agllm_backend.for_config(cfg)
        assert backend._agconfig is not cfg
        assert backend.model == "m"
        cfg.agllm_backend.model = "changed"
        assert backend.model == "m"

    def test_config_values_readable_as_attributes(self):
        backend = agllm_backend.for_config(
            _cfg(model="gpt-x", api_key="k", temperature=0.5, top_k=40)
        )
        assert backend.model == "gpt-x"
        assert backend.api_key == "k"
        assert backend.temperature == 0.5
        assert backend.top_k == 40

    def test_unset_fields_default_to_none(self):
        backend = agllm_backend.for_config(_cfg(model="m"))
        assert backend.workspace_id is None
        assert backend.extra_body is None

    def test_as_dict_reflects_config(self):
        cfg = _cfg(model="m", api_key="k")
        backend = agllm_backend.for_config(cfg)
        assert backend.as_dict() == {"model": "m", "api_key": "k"}

    def test_two_backend_instances_over_separate_agconfigs_have_independent_values(self):
        """Each backend is given its own agConfig -- values on one instance
        must never leak into another, even though the DynamicConfigParam
        descriptors are shared class attributes."""
        a = agllm_backend.for_config(_cfg(model="model-a", api_key="key-a"))
        b = agllm_backend.for_config(_cfg(model="model-b", api_key="key-b"))
        assert a.model == "model-a"
        assert b.model == "model-b"
        assert a.api_key == "key-a"
        assert b.api_key == "key-b"
        assert a._agconfig is not b._agconfig

    def test_setting_attribute_on_one_instance_does_not_affect_another(self):
        a = agllm_backend.for_config(_cfg(model="model-a"))
        b = agllm_backend.for_config(_cfg(model="model-b"))
        a.model = "changed"
        assert a.model == "changed"
        assert b.model == "model-b"


class TestGlobalTunables:
    """Read-only: model_listing_timeout_seconds/default_max_tokens are
    GlobalConfigParam, a process-wide singleton shared with every other test
    file in this run -- asserting anything beyond their still-default value
    (nothing else in the suite writes to them) would make pass/fail depend
    on collection order."""

    def test_default_max_tokens_readable_from_any_backend_instance(self):
        backend = agllm_backend.for_config(_cfg(model="m"))
        assert backend.default_max_tokens == 128000

    def test_model_listing_timeout_readable_from_any_backend_instance(self):
        backend = agllm_backend.for_config(_cfg(model="m"))
        assert backend.model_listing_timeout_seconds == 10.0


class TestForConfigDispatch:
    def test_plain_config_returns_openai_compatible(self):
        assert isinstance(agllm_backend.for_config(_cfg(model="m")), _OpenAICompatibleBackend)

    def test_bedrock_provider_non_anthropic_model(self):
        backend = agllm_backend.for_config(
            _cfg(provider="bedrock", model="nvidia.x", region="us-east-2")
        )
        assert isinstance(backend, _OpenAICompatibleBedrockBackend)

    def test_bedrock_provider_anthropic_model(self):
        backend = agllm_backend.for_config(
            _cfg(provider="bedrock", model="us.anthropic.claude-sonnet-5", region="us-east-2")
        )
        assert isinstance(backend, _AnthropicBedrockBackend)

    def test_anthropic_provider(self):
        assert isinstance(
            agllm_backend.for_config(_cfg(provider="anthropic", model="claude-sonnet-5")),
            _AnthropicBackend,
        )

    def test_anthropic_aws_provider(self):
        assert isinstance(
            agllm_backend.for_config(_cfg(provider="anthropicAWS", model="claude-sonnet-5")),
            _AnthropicAWSBackend,
        )


class TestAttributeBackedClientConstruction:
    def test_openai_compatible_make_client_uses_attributes(self):
        backend = _OpenAICompatibleBackend(_cfg(api_key="k", base_url="http://x/v1"))
        with patch("agency.agllm_backends.openai.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="k", base_url="http://x/v1", timeout=httpx.Timeout(5.0)
        )

    def test_anthropic_backend_uses_api_key_attribute(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-x"))
        mock_sdk = MagicMock()
        with patch("agency.agllm_backends.anthropic._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        mock_sdk.Anthropic.assert_called_once_with(api_key="sk-ant-x", timeout=httpx.Timeout(5.0))

    def test_anthropic_aws_backend_uses_credential_attributes(self):
        backend = _AnthropicAWSBackend(
            _cfg(
                api_key="aws-api-key",
                region="us-east-2",
                workspace_id="wrkspc_test",
            )
        )
        mock_sdk = MagicMock()
        mock_sdk.AnthropicAWS = MagicMock()
        with patch("agency.agllm_backends.bedrock._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(30.0))
        mock_sdk.AnthropicAWS.assert_called_once_with(
            timeout=httpx.Timeout(30.0),
            api_key="aws-api-key",
            aws_region="us-east-2",
            workspace_id="wrkspc_test",
        )


class TestAgllmUsesAgConfig:
    def test_agllm_reads_backend_from_agconfig(self):
        llm = agllm(_cfg(model="gpt-x"), context_limit=128_000)
        assert llm.backend.model == "gpt-x"
        assert isinstance(llm.backend, _OpenAICompatibleBackend)

    def test_agllm_backend_attributes_reflect_config(self):
        llm = agllm(
            _cfg(model="claude-sonnet-5", temperature=0.3, provider="anthropic"),
            context_limit=128_000,
        )
        assert llm.backend.model == "claude-sonnet-5"
        assert llm.backend.temperature == 0.3

    def test_build_kwargs_unaffected(self):
        llm = agllm(_cfg(model="claude-sonnet-5", temperature=0.3), context_limit=128_000)
        kw = llm.build_kwargs([{"role": "user", "content": "hi"}])
        assert kw["temperature"] == 0.3
        assert kw["model"] == "claude-sonnet-5"


# ---------------------------------------------------------------------------
# The four provider-specific agllm_backend *Config classes -- provider
# auto-set (non-overridable), _ALLOWED_FIELDS restriction, and for_config()
# dispatch actually landing on the matching backend.
# ---------------------------------------------------------------------------


class TestProviderBackendConfigClasses:
    def test_vllm_config_fixes_provider(self):
        cfg = agConfig(agVLLMBackendConfig(model="m"))
        assert cfg.agllm_backend.provider == "vllm"

    def test_openai_config_fixes_provider(self):
        cfg = agConfig(agOpenAIBackendConfig(model="m"))
        assert cfg.agllm_backend.provider == "openai"

    def test_anthropic_config_fixes_provider(self):
        cfg = agConfig(agAnthropicBackendConfig(model="m"))
        assert cfg.agllm_backend.provider == "anthropic"

    def test_bedrock_config_fixes_provider(self):
        cfg = agConfig(agBedrockBackendConfig(model="m"))
        assert cfg.agllm_backend.provider == "bedrock"

    def test_provider_cannot_be_passed_as_a_field(self):
        """provider isn't in any provider-specific class's _ALLOWED_FIELDS --
        the class name is what fixes it, not a keyword the caller controls."""
        with pytest.raises(TypeError, match="provider"):
            agVLLMBackendConfig(provider="bedrock")

    def test_vllm_config_accepts_sampling_extensions(self):
        cfg = agConfig(
            agVLLMBackendConfig(model="m", top_k=40, repetition_penalty=1.1, guided_json="{}")
        )
        assert cfg.agllm_backend.top_k == 40
        assert cfg.agllm_backend.repetition_penalty == 1.1

    def test_openai_config_rejects_vllm_only_sampling_extensions(self):
        with pytest.raises(TypeError, match="top_k"):
            agOpenAIBackendConfig(model="m", top_k=40)

    def test_anthropic_config_rejects_fields_its_adapter_silently_drops(self):
        for bad_field in ("frequency_penalty", "presence_penalty", "n", "stop", "logprobs", "seed"):
            with pytest.raises(TypeError, match=bad_field):
                agAnthropicBackendConfig(model="m", **{bad_field: 1})

    def test_anthropic_config_accepts_the_fields_its_adapter_uses(self):
        cfg = agConfig(
            agAnthropicBackendConfig(
                model="claude-sonnet-5",
                api_key="k",
                workspace_id="w",
                temperature=0.5,
                max_completion_tokens=1000,
            )
        )
        assert cfg.agllm_backend.temperature == 0.5
        assert cfg.agllm_backend.workspace_id == "w"

    def test_bedrock_config_accepts_region_and_full_generation_surface(self):
        cfg = agConfig(
            agBedrockBackendConfig(model="minimax.minimax-m2", region="us-east-1", top_k=40)
        )
        assert cfg.agllm_backend.region == "us-east-1"
        assert cfg.agllm_backend.top_k == 40

    def test_vllm_config_routes_to_openai_compatible_backend(self):
        cfg = agConfig(agVLLMBackendConfig(model="m", base_url="http://localhost:8000/v1"))
        assert isinstance(agllm_backend.for_config(cfg), _OpenAICompatibleBackend)

    def test_vllm_config_without_base_url_raises(self):
        cfg = agConfig(agVLLMBackendConfig(model="m"))
        with pytest.raises(ValueError, match="base_url"):
            agllm_backend.for_config(cfg)

    def test_openai_config_routes_to_openai_compatible_backend(self):
        cfg = agConfig(agOpenAIBackendConfig(model="m"))
        assert isinstance(agllm_backend.for_config(cfg), _OpenAICompatibleBackend)

    def test_anthropic_config_routes_to_anthropic_backend(self):
        cfg = agConfig(agAnthropicBackendConfig(model="claude-sonnet-5"))
        assert isinstance(agllm_backend.for_config(cfg), _AnthropicBackend)

    def test_bedrock_config_routes_to_openai_compatible_bedrock_for_non_anthropic_model(self):
        cfg = agConfig(agBedrockBackendConfig(model="minimax.minimax-m2", region="us-east-1"))
        assert isinstance(agllm_backend.for_config(cfg), _OpenAICompatibleBedrockBackend)

    def test_bedrock_config_routes_to_anthropic_bedrock_for_anthropic_model(self):
        cfg = agConfig(
            agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5", region="us-east-1")
        )
        assert isinstance(agllm_backend.for_config(cfg), _AnthropicBedrockBackend)

    def test_two_provider_configs_composed_stay_independent(self):
        """Each is a separate agConfig unless explicitly merged -- picking
        agVLLMBackendConfig for one agent and agAnthropicBackendConfig for
        another must never cross-contaminate."""
        cfg_a = agConfig(agVLLMBackendConfig(model="local-model"))
        cfg_b = agConfig(agAnthropicBackendConfig(model="claude-sonnet-5"))
        assert cfg_a.agllm_backend.provider == "vllm"
        assert cfg_b.agllm_backend.provider == "anthropic"
        assert cfg_a.agllm_backend.model == "local-model"
        assert cfg_b.agllm_backend.model == "claude-sonnet-5"


# ---------------------------------------------------------------------------
# Dynamic-field sweep across every owner in the framework that has at least
# one DynamicConfigParam -- creation via agConfig(agXXXConfig(...)), reading
# through the owner view, a live update on an existing agConfig, and
# composing alongside a different owner in one agConfig(...) call. Dynamic
# fields never lock, so these are safe regardless of what other test files
# in this run have already touched.
# ---------------------------------------------------------------------------

_DYNAMIC_OWNER_CASES = [
    (agLogConfig, "aglog", "dump_content_truncate_len", 999),
    (agAgentConfig, "agent", "checkpoint_save_timeout_s", 111),
    (agSkillConfig, "agskill", "react_max_steps", 7),
    (agLLMConfig, "agllm", "max_retries", 1),
    (agSchemaConfig, "agschema", "input_offload_chars", 123),
    (agToolConfig, "agtool", "timeout_s", 45),
    (agLLMBackendConfig, "agllm_backend", "model", "sweep-model"),
    (agResourcePoolConfig, "agResourcePool", "idle_cpus", 2.0),
]


@pytest.mark.parametrize("config_cls,owner,field,value", _DYNAMIC_OWNER_CASES)
class TestDynamicFieldSweepAcrossOwners:
    def test_view_creates_agconfig_with_field_set(self, config_cls, owner, field, value):
        cfg = agConfig(config_cls(**{field: value}))
        assert cfg.get(owner, field) == value

    def test_field_readable_via_owner_view(self, config_cls, owner, field, value):
        cfg = agConfig(config_cls(**{field: value}))
        assert getattr(getattr(cfg, owner), field) == value

    def test_live_update_on_an_existing_agconfig(self, config_cls, owner, field, value):
        cfg = agConfig(config_cls())
        assert cfg.get(owner, field) != value
        config_cls(cfg).update(**{field: value})
        assert cfg.get(owner, field) == value

    def test_composes_alongside_a_different_owner_in_one_call(
        self, config_cls, owner, field, value
    ):
        cfg = agConfig(config_cls(**{field: value}), agUtilConfig())
        assert cfg.get(owner, field) == value


# ---------------------------------------------------------------------------
# Static field (agSandbox.base_image is the only one in the framework) --
# fresh agConfig per case, since Static locks per-instance, not process-wide.
# ---------------------------------------------------------------------------


class TestStaticFieldAcrossOwners:
    def test_base_image_settable_before_any_read(self):
        cfg = agConfig(agSandboxConfig(base_image="custom-image:latest"))
        assert cfg.get("agSandbox", "base_image") == "custom-image:latest"

    def test_base_image_resolves_once_and_locks(self):
        cfg = agConfig()
        cfg.set("agSandbox", "base_image", "first-read")
        assert cfg.get_static("agSandbox", "base_image") == "first-read"
        with pytest.raises(ValueError, match="already read as static"):
            agSandboxConfig(cfg).update(base_image="too-late")

    def test_clone_unblocks_a_static_field_change(self):
        cfg = agConfig(agSandboxConfig(base_image="v1"))
        cfg.get_static("agSandbox", "base_image")  # simulate a sandbox having resolved it
        cfg2 = cfg.clone()
        agSandboxConfig(cfg2).update(base_image="v2")
        assert cfg2.get("agSandbox", "base_image") == "v2"
        assert cfg.get("agSandbox", "base_image") == "v1"  # original untouched


# ---------------------------------------------------------------------------
# Global fields across owners -- registration + read-consistency only (never
# a specific-value assertion): GlobalConfigParam is a process-wide singleton
# shared with every other test file in this run, so a value assertion here
# would make pass/fail depend on collection order across files.
# ---------------------------------------------------------------------------

_GLOBAL_OWNER_CASES = [
    (_AgLLMFields, "agllm", "call_max_concurrency"),
    (_AgToolFields, "agtool", "pool_max_workers"),
    (AgSandboxBackendFields, "agsandbox_backend", "docker_semaphore_limit"),
    (_AgResourcePoolFields, "agResourcePool", "memory_detect_fallback_mb"),
    (_AgUtilFields, "agutil", "idle_check_interval_s"),
    (AgLLMBackendFields, "agllm_backend", "default_max_tokens"),
]


@pytest.mark.parametrize("fields_cls,owner,field", _GLOBAL_OWNER_CASES)
class TestGlobalFieldSweepAcrossOwners:
    def test_registered_as_global_config_param(self, fields_cls, owner, field):
        assert isinstance(agConfig.FIELD_REGISTRY[(owner, field)], GlobalConfigParam)

    def test_descriptor_read_matches_agconfig_global_directly(self, fields_cls, owner, field):
        knob = agConfig.FIELD_REGISTRY[(owner, field)]
        bare = fields_cls()
        assert getattr(bare, field) == agConfig.GLOBAL.get(owner, field, knob.default)


# ---------------------------------------------------------------------------
# Constant-inlining changes: fields that used to freeze a plain class
# constant into their `default=` (a no-op-on-reassignment trap -- see
# docs/agconfig.md's "The three tiers") now inline the literal directly.
# Other code needing the same value reads it off the descriptor's `.default`
# instead of a separate constant -- verify that accessor still resolves to
# the expected value for every field this session's refactor touched.
# ---------------------------------------------------------------------------


def test_sandbox_base_image_default_accessor():
    assert _AgSandboxFields.base_image.default == "agency-sandbox:latest"


def test_llm_default_context_limit_default_accessor():
    assert _AgLLMFields.default_context_limit.default == 200_000


def test_llm_tail_turns_default_accessor():
    assert _AgLLMFields.tail_turns.default == 3


def test_tool_timeout_s_default_accessor():
    assert _AgToolFields.timeout_s.default == 1800


def test_tool_output_offload_chars_default_accessor():
    assert _AgToolFields.output_offload_chars.default == 40_000


def test_resource_pool_memory_detect_fallback_mb_default_accessor():
    assert _AgResourcePoolFields.memory_detect_fallback_mb.default == 4096


def test_schema_input_offload_chars_default_accessor():
    assert _AgSchemaFields.input_offload_chars.default == 40_000


def test_default_accessor_matches_actual_bare_instance_read():
    """The .default accessor and a real (unconfigured) instance's read must
    always agree -- that's the whole point of reading .default instead of a
    separate constant that could drift from it."""
    assert _AgToolFields().timeout_s == _AgToolFields.timeout_s.default
    assert _AgSchemaFields().input_offload_chars == _AgSchemaFields.input_offload_chars.default
    assert _AgLLMFields().tail_turns == _AgLLMFields.tail_turns.default
