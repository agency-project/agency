"""Tests for agconfig — tiered override store, ConfigParam descriptors, registry, GLOBAL, _OwnerView."""
import pytest

from agency.agconfig import (
    agConfig,
    _ConfigParam,
    GlobalConfigParam,
    StaticConfigParam,
    DynamicConfigParam,
    _OwnerView,
)


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
    tier1_lifecycle       = GlobalConfigParam("test_agconfig_a", default=1)
    tier1_shared          = GlobalConfigParam("test_agconfig_a", default=2)
    tier1_ownerview_route = GlobalConfigParam("test_agconfig_a", default=3)
    tier2_field           = StaticConfigParam("test_agconfig_a", default="static-default")
    tier3_field           = DynamicConfigParam("test_agconfig_a", default="dynamic-default")
    shared_name           = DynamicConfigParam("test_agconfig_a", default="a-default")

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
    assert cfg.get("test_agconfig_a", "tier1_ownerview_route", "unset-on-cfg-itself") == "unset-on-cfg-itself"
    # Reading through the view (on this or any other agConfig instance)
    # also routes to GLOBAL and sees the same value.
    assert cfg.test_agconfig_a.tier1_ownerview_route == 77
    assert agConfig().test_agconfig_a.tier1_ownerview_route == 77
