"""Toy example demonstrating a descriptor-based agconfig: each field is
declared as a real class attribute (a descriptor), so linters/mypy/IDE
autocomplete see it just like any other field -- yet at the use site it's
plain ``self.xxx`` access, with no resolve_config()/get_static()/set()
visible anywhere near the call site.

Tier 1 is *not* a separate mechanism here -- it's tier 2's exact same
get_static()/set() logic, just pointed at one shared DemoConfig.GLOBAL
instance instead of each object's own. Same lock, same raise-on-write-
after-read, zero new code; "global" falls out of "everyone points at the
same object" rather than being its own special case.

Pre-configuring a field (before any Foobar exists) doesn't need a
throwaway Foobar() either: _ConfigParam.__set_name__ registers every field
into DemoConfig.FIELD_REGISTRY purely by importing this module -- no
instance required -- and cfg.foobar.max_workers = 8 looks the field up in
that registry and dispatches to the right tier automatically.

Standalone: reimplements a minimal DemoConfig here rather than importing
the real agency.agconfig, so this can be verified without touching the
actual package (which a live pipeline may currently depend on).

Run directly: python3 foobar_config_demo.py
"""
from __future__ import annotations
import threading
from typing import Any, ClassVar


class DemoConfig:
    """A plain, generic, class-agnostic key-value store. Exactly four
    storage methods total -- no separate tier-1 machinery. DemoConfig.GLOBAL
    (set right after the class body, below) is what makes tier 1 possible:
    it's just a DemoConfig instance like any other, referenced by every
    GlobalConfigParam instead of each object's own.

    DemoConfig itself never hard-codes any class's fields -- FIELD_REGISTRY
    starts empty and is populated purely by whichever consuming files (e.g.
    Foobar, below) happen to get imported; see _ConfigParam.__set_name__."""

    GLOBAL: ClassVar["DemoConfig"]  # assigned once, right after the class body

    # (owner, name) -> the descriptor instance that owns that field. Populated
    # entirely by _ConfigParam.__set_name__ at class-body-execution time -- no
    # instance of the owning class (Foobar, agllm, ...) is ever constructed
    # just to make its fields discoverable.
    FIELD_REGISTRY: ClassVar[dict[tuple[str, str], "_ConfigParam"]] = {}
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, data: dict | None = None) -> None:
        self.data: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (data or {}).items()}
        self._locked_keys: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def get(self, owner: str, name: str, default: Any = None) -> Any:
        return self.data.get(owner, {}).get(name, default)

    def get_static(self, owner: str, name: str, default: Any = None) -> Any:
        with self._lock:
            self._locked_keys.add((owner, name))
            return self.data.get(owner, {}).get(name, default)

    def set(self, owner: str, name: str, value: Any) -> None:
        with self._lock:
            if (owner, name) in self._locked_keys:
                raise ValueError(f"{owner}.{name} was already read as static; clone() first to change it")
            self.data.setdefault(owner, {})[name] = value

    def clone(self) -> "DemoConfig":
        return DemoConfig(self.data)

    def __getattr__(self, name: str) -> "_OwnerView":
        """cfg.foobar -- "foobar" isn't a real attribute; it's recognized as
        a known owner (something has registered at least one field under
        that name) and returns a small view scoped to (self, "foobar").
        From there, cfg.foobar.mode = "careful" dispatches to the right
        tier automatically -- and unlike a bare cfg.mode, "foobar" being
        explicit means two different owners can still both have a field of
        the same name without any ambiguity about which one you meant.
        """
        owners = {owner for owner, _n in DemoConfig.FIELD_REGISTRY}
        if name in owners:
            return _OwnerView(self, name)
        raise AttributeError(f"DemoConfig has no owner {name!r} registered")


DemoConfig.GLOBAL = DemoConfig()


# ---------------------------------------------------------------------------
# The actual ask: three descriptor classes, one per tier. Assigned as real
# class attributes on the *consuming* class (Foobar below) -- DemoConfig
# itself never becomes class-aware, so nothing changes about how it
# propagates from parent objects to children.
# ---------------------------------------------------------------------------

class _ConfigParam:
    def __init__(self, owner: str, default: Any) -> None:
        self.owner = owner
        self.default = default
        self.name: str | None = None

    def __set_name__(self, objtype: type, name: str) -> None:
        self.name = name  # PEP 487: told our own attribute name automatically
        key = (self.owner, name)
        with DemoConfig._registry_lock:
            if key in DemoConfig.FIELD_REGISTRY:
                raise ValueError(
                    f"config field {self.owner}.{name} is already registered "
                    f"(by {DemoConfig.FIELD_REGISTRY[key]!r}); field names must be unique per owner"
                )
            DemoConfig.FIELD_REGISTRY[key] = self


class GlobalConfigParam(_ConfigParam):
    """Tier 1: process-wide. This is *not* separate machinery from tier 2 --
    it's the exact same get_static()/set() on DemoConfig, just always
    pointed at the one shared DemoConfig.GLOBAL instance instead of
    whichever agconfig a particular object was given. Locking, and raising
    on a write after the first read, come for free from get_static()/set()
    -- no tier-1-specific logic exists anywhere.

    Configure it the same way you'd configure any DemoConfig: call .set()
    on the relevant instance -- here, that instance happens to be
    DemoConfig.GLOBAL: ``DemoConfig.GLOBAL.set(owner, name, value)``.
    Instance-level assignment (``some_foobar.max_workers = 8``) still
    raises, same as a tier-2 field would after being read once -- it's
    just that tier-1 fields are considered "already read" the moment
    *anyone, anywhere* has read them, since they all share GLOBAL."""

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return DemoConfig.GLOBAL.get_static(self.owner, self.name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        try:
            DemoConfig.GLOBAL.set(self.owner, self.name, value)
        except ValueError:
            # DemoConfig.set()'s message ("clone() first") is tier-2 advice --
            # cloning doesn't help here, since every GlobalConfigParam always points
            # at the literal DemoConfig.GLOBAL, never at a clone of it.
            raise ValueError(
                f"{self.owner}.{self.name} is a tier-1 (global) field already read somewhere in "
                f"this process; it can't be changed now -- there's no per-instance override to "
                f"fall back to, since every instance shares the same value"
            ) from None


class StaticConfigParam(_ConfigParam):
    """Tier 2: resolved once per instance, on first access, then cached.
    Also read-only through the descriptor -- construct a new instance
    (with a different or cloned agconfig) to get a different value."""

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        cache = obj.__dict__.setdefault("_static_cache", {})
        if self.name not in cache:
            agconfig = getattr(obj, "_agconfig", None)
            cache[self.name] = (
                agconfig.get_static(self.owner, self.name, self.default) if agconfig is not None else self.default
            )
        return cache[self.name]

    def __set__(self, obj: Any, value: Any) -> None:
        raise AttributeError(
            f"{self.owner}.{self.name} is a tier-2 field, fixed once at construction; "
            f"construct a new instance with a different (or cloned) agconfig instead"
        )


class DynamicConfigParam(_ConfigParam):
    """Tier 3: live -- re-read from agconfig on every access. Settable:
    writing goes straight through to the shared agconfig, so it's visible
    to every other instance/read sharing that same agconfig immediately."""

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        agconfig = getattr(obj, "_agconfig", None)
        if agconfig is None:
            return self.default
        return agconfig.get(self.owner, self.name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        agconfig = getattr(obj, "_agconfig", None)
        if agconfig is None:
            raise AttributeError(f"{self.owner}.{self.name} can't be set -- this instance has no agconfig")
        agconfig.set(self.owner, self.name, value)


class _OwnerView:
    """Returned by DemoConfig.<owner> (e.g. cfg.foobar) -- lets that owner's
    fields be read/written as plain nested attribute access, dispatching to
    the right tier exactly like DemoConfig.configure() does. Not a
    descriptor itself; a small bound object holding which DemoConfig
    instance and which owner it's scoped to."""

    def __init__(self, agconfig: "DemoConfig", owner: str) -> None:
        object.__setattr__(self, "_agconfig", agconfig)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name: str) -> Any:
        knob = DemoConfig.FIELD_REGISTRY.get((self._owner, name))
        if knob is None:
            raise AttributeError(f"{self._owner} has no registered field {name!r}")
        if isinstance(knob, GlobalConfigParam):
            return DemoConfig.GLOBAL.get_static(self._owner, name, knob.default)
        if isinstance(knob, StaticConfigParam):
            return self._agconfig.get_static(self._owner, name, knob.default)
        return self._agconfig.get(self._owner, name, knob.default)  # DynamicConfigParam

    def __setattr__(self, name: str, value: Any) -> None:
        knob = DemoConfig.FIELD_REGISTRY.get((self._owner, name))
        if knob is None:
            raise AttributeError(f"{self._owner} has no registered field {name!r}")
        if isinstance(knob, GlobalConfigParam):
            DemoConfig.GLOBAL.set(self._owner, name, value)  # same shared target regardless of which agconfig this is
        else:
            self._agconfig.set(self._owner, name, value)


# ---------------------------------------------------------------------------
# Foobar: all three tiers declared as plain class attributes. No
# resolve_config()/get_static()/set_class_default() anywhere near the
# actual use sites (describe(), or any real caller) -- just self.xxx.
# ---------------------------------------------------------------------------
class Foobar:
    max_workers = GlobalConfigParam("foobar", default=4)
    mode        = StaticConfigParam("foobar", default="fast")
    retry_count = DynamicConfigParam("foobar", default=3)

    def __init__(self, agconfig: "DemoConfig | None" = None) -> None:
        self._agconfig = agconfig

    def describe(self) -> str:
        return f"mode={self.mode}  retry_count={self.retry_count}  pool_size={self.max_workers}"


class Bazqux:
    """A second, unrelated class -- also has a field called "mode", to prove
    cfg.foobar.mode and cfg.bazqux.mode never collide despite the same name,
    since the owner is explicit at every access."""
    mode = DynamicConfigParam("bazqux", default="quiet")

    def __init__(self, agconfig: "DemoConfig | None" = None) -> None:
        self._agconfig = agconfig


if __name__ == "__main__":
    print("=== A linter/IDE sees these as real attributes ===")
    print("hasattr(Foobar, 'mode'):", hasattr(Foobar, "mode"))
    print("[n for n in vars(Foobar) if not n.startswith('_')]:",
          [n for n in vars(Foobar) if not n.startswith("_")])

    print()
    print("=== FIELD_REGISTRY was populated purely by importing this module -- no Foobar() constructed yet ===")
    print("DemoConfig.FIELD_REGISTRY keys:", list(DemoConfig.FIELD_REGISTRY.keys()))

    print()
    print("=== Pre-configuring via cfg.owner.field -- no throwaway Foobar(), no raw .set()/.configure() ===")
    cfg_a = DemoConfig()
    cfg_a.foobar.max_workers = 8   # tier 1 -- routed to GLOBAL automatically, cfg_a itself never stores it
    cfg_a.foobar.mode = "careful"  # tier 2 -- stored in cfg_a.data["foobar"]
    cfg_a.foobar.retry_count = 10  # tier 3 -- also cfg_a.data["foobar"], no return-value threading needed
    print("cfg_a.data:", cfg_a.data)
    print("DemoConfig.GLOBAL.data:", DemoConfig.GLOBAL.data)  # max_workers landed here instead, as expected

    print()
    print("=== Disambiguation: two owners, same field name, no collision ===")
    cfg_a.bazqux.mode = "loud"
    print("cfg_a.foobar.mode:", cfg_a.foobar.mode, " cfg_a.bazqux.mode:", cfg_a.bazqux.mode)

    print()
    print("=== Tier 1: self.max_workers -- tier 2's write-once/lock-on-read, pointed at DemoConfig.GLOBAL ===")
    probe = Foobar()
    print("probe.max_workers:", probe.max_workers)  # this read is what locks it now -- process-wide,
                                                      # since every GlobalConfigParam shares the same DemoConfig.GLOBAL.
    try:
        cfg_a.foobar.max_workers = 16  # too late -- probe already read it above
        print("UNEXPECTED: did not raise")
    except ValueError as e:
        print("cfg_a.foobar.max_workers = ... raised as expected:", e)
    another_probe = Foobar()
    try:
        another_probe.max_workers = 99  # a DIFFERENT instance -- still raises, same shared global
        print("UNEXPECTED: did not raise")
    except ValueError as e:
        print("write via a second, unrelated instance ALSO raised (same shared global):", e)

    print()
    print("=== Tier 2: self.mode (static, resolved once at construction, read-only via descriptor) ===")
    a = Foobar(agconfig=cfg_a)
    print("a.mode:", a.mode)
    try:
        cfg_a.foobar.mode = "reckless"  # a already consumed "mode" from cfg_a
        print("UNEXPECTED: did not raise")
    except ValueError as e:
        print("agconfig-level lock raised as expected:", e)
    try:
        a.mode = "reckless"  # descriptor-level guard, independent of the agconfig lock above
        print("UNEXPECTED: did not raise")
    except AttributeError as e:
        print("descriptor-level guard raised as expected:", e)
    cfg_a_clone = cfg_a.clone()
    cfg_a_clone.foobar.mode = "reckless"
    a3 = Foobar(agconfig=cfg_a_clone)
    print("a3.mode (cloned + overridden):", a3.mode)

    print()
    print("=== Tier 3: self.retry_count (dynamic, live, settable) ===")
    print("a.retry_count (default via cfg_a):", a.retry_count)
    a.retry_count = 10  # writes straight through to cfg_a
    print("a.retry_count (after a.retry_count = 10):", a.retry_count)
    cfg_a.foobar.retry_count = 99  # setting via cfg_a.foobar works too, same effect
    print("a.retry_count (after cfg_a.foobar.retry_count = 99):", a.retry_count)
    try:
        probe.retry_count = 5  # probe has no agconfig at all
        print("UNEXPECTED: did not raise")
    except AttributeError as e:
        print("set on agconfig-less instance raised as expected:", e)

    print()
    print("=== cfg.owner for an unregistered owner, and a registered owner's unregistered field ===")
    try:
        cfg_a.not_a_real_owner.anything = 1
        print("UNEXPECTED: did not raise")
    except AttributeError as e:
        print("unregistered owner raised as expected:", e)
    try:
        cfg_a.foobar.totally_made_up = 1
        print("UNEXPECTED: did not raise")
    except AttributeError as e:
        print("unregistered field raised as expected:", e)

    print()
    print("=== Registration collision: same (owner, name) declared twice ===")
    try:
        class _Colliding:
            mode = StaticConfigParam("foobar", default="oops")  # "foobar"."mode" already registered by Foobar.mode
        print("UNEXPECTED: did not raise")
    except ValueError as e:
        print("raised as expected:", e)

    print()
    print("=== describe() ===")
    print("a.describe():", a.describe())
