from __future__ import annotations

import threading
from typing import Any, ClassVar


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


class agConfig:
    """Nested override store: ``data[owner][param] = value``.

    Passed as ``agconfig=`` to any framework class's constructor. A parent
    object forwards its own ``agconfig`` to every child object it creates
    (e.g. ``agent`` -> its ``agllm`` and ``agSandbox``), so one ``agConfig``
    built at the top of a script flows to everything it spawns.

    Each consuming class declares its tunable fields as real class
    attributes -- ``GlobalConfigParam``/``StaticConfigParam``/
    ``DynamicConfigParam`` below -- so they're visible to linters/IDEs and
    read at the call site as plain ``self.xxx``, not through any method on
    this class. This class itself is just the storage + registry:

    1. Global (``GlobalConfigParam``) -- process-wide, not scoped to any one
       ``agConfig`` instance. Locked on first read.
    2. Static (``StaticConfigParam``) -- resolved once by the consumer at
       construction, because the underlying resource is physically fixed
       once created. ``get_static`` locks that key *on this instance*
       against further ``set()`` calls.
    3. Dynamic (``DynamicConfigParam``) -- read fresh on every access via
       ``get()``; a later ``set()`` is visible on the next read. Never
       locked.

    To pre-configure a field before any consuming instance exists, use
    nested attribute access: ``cfg.agllm.max_retries = 5``. ``agllm`` is
    recognized because something has registered a field under that owner
    name (see ``FIELD_REGISTRY``); the returned ``_OwnerView`` dispatches
    the read/write to the right tier automatically.

    For setting several fields on one owner at once (instead of one
    ``cfg.<owner>.<field> = value`` line per field), each owner has a
    ``*Config`` view subclassing ``_AgConfigViewBase`` below (e.g.
    ``agLLMBackendConfig``, ``agAgentConfig`` -- see the owning module for
    each). Passing several of these views (or plain dicts, or other
    ``agConfig`` instances) straight to ``agConfig(...)`` merges them into
    one config in a single call. Class-specific vocabulary that doesn't fit
    a flat field (like ``agSandbox``'s ``add_mount``) belongs on that
    owner's view instead -- see ``agSandboxConfig`` in ``agsandbox.py``.
    """

    GLOBAL: ClassVar["agConfig"]  # assigned once, right after the class body

    # (owner, name) -> the descriptor instance that owns that field. Populated
    # entirely by _ConfigParam.__set_name__ at class-body-execution time --
    # no instance of the owning class (agllm, agtool, ...) is ever
    # constructed just to make its fields discoverable.
    FIELD_REGISTRY: ClassVar[dict[tuple[str, str], "_ConfigParam"]] = {}
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self, *sources: "agConfig | dict[str, dict[str, Any]] | _AgConfigViewBase"
    ) -> None:
        """Each source contributes its data, later sources winning on a
        conflicting (owner, name) -- same semantics as ``{**a, **b}``. A
        source may be a plain nested dict (the original single-argument
        form, e.g. ``agConfig({"agllm_backend": {...}})``), another
        ``agConfig`` (e.g. ``.clone()``'s ``agConfig(self.data)``), or any
        ``*Config`` view (``agLLMBackendConfig``, ``agSandboxConfig``, ...)
        -- anything exposing an ``.agconfig`` property. This lets several
        owners' worth of config be assembled in one call::

            cfg = agConfig(
                agLLMBackendConfig(model="...", api_key="..."),
                agSandboxConfig().add_mount("out", path, "/agent_output"),
            )
        """
        self.data: dict[str, dict[str, Any]] = {}
        self._locked_keys: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        for src in sources:
            if src is None:
                continue
            if isinstance(src, dict):
                src_data = src
            elif isinstance(src, agConfig):
                src_data = src.data
            elif hasattr(src, "agconfig"):
                src_data = src.agconfig.data
            else:
                raise TypeError(
                    f"agConfig() cannot merge a {type(src).__name__} -- pass a dict, "
                    f"an agConfig, or a *Config view (something with an .agconfig property)"
                )
            for owner, fields in src_data.items():
                self.data.setdefault(owner, {}).update(fields)

    def get(self, owner: str, name: str, default: Any = None) -> Any:
        """Tier 3 (dynamic): live read, never locks."""
        return self.data.get(owner, {}).get(name, default)

    def get_static(self, owner: str, name: str, default: Any = None) -> Any:
        """Tier 2 (static): locks this key on this instance against further
        ``set()`` calls -- the consumer is expected to cache the result
        rather than re-read it."""
        with self._lock:
            self._locked_keys.add((owner, name))
            return self.data.get(owner, {}).get(name, default)

    def set(self, owner: str, name: str, value: Any) -> "agConfig":
        with self._lock:
            if (owner, name) in self._locked_keys:
                raise ValueError(
                    f"{owner}.{name} was already read as static on this "
                    f"agConfig; clone() it first to change it for new objects"
                )
            self.data.setdefault(owner, {})[name] = value
        return self

    def dynamic_snapshot(self) -> "dict[str, dict[str, Any]]":
        """Return {owner: {field: value}} for every registered
        DynamicConfigParam field, using this agConfig's current effective
        value (its override, or the field's default).

        Dynamic is the only tier where a value written here and later
        applied via ``agent.change_config()`` has any observable effect at
        runtime: Static fields are cached per-instance on first read (a new
        agconfig doesn't invalidate that cache), and Global fields always
        read ``agConfig.GLOBAL`` regardless of which agconfig an object
        holds. So this is what's meaningful to expose as "live-editable
        configuration" (e.g. the webui's config editor).

        Non-JSON-safe values (anything but str/int/float/bool/None, or a
        list/dict composed only of those) are skipped -- they can't cross
        the wire to the browser anyway.
        """
        result: dict[str, dict[str, Any]] = {}
        for (owner, name), knob in agConfig.FIELD_REGISTRY.items():
            if not isinstance(knob, DynamicConfigParam):
                continue
            value = self.get(owner, name, knob.default)
            if not _is_json_safe(value):
                continue
            result.setdefault(owner, {})[name] = value
        return result

    def clone(self) -> "agConfig":
        """Return a fresh agConfig with the same current data but no lock
        history. Use this to layer a per-instance override (e.g. one
        agent's own output-dir mount) onto a shared/propagated config
        without touching what other in-flight objects already consumed."""
        return agConfig(self.data)

    def __getattr__(self, name: str) -> "_OwnerView":
        owners = {owner for owner, _n in agConfig.FIELD_REGISTRY}
        if name in owners:
            return _OwnerView(self, name)
        raise AttributeError(f"agConfig has no owner {name!r} registered")


agConfig.GLOBAL = agConfig()


# ---------------------------------------------------------------------------
# Three descriptor classes, one per tier. Assigned as real class attributes
# on the *consuming* class (agllm, agtool, ...) -- agConfig itself never
# becomes class-aware, so nothing changes about how it propagates from
# parent objects to children.
# ---------------------------------------------------------------------------


class _ConfigParam:
    def __init__(self, owner: str, default: Any) -> None:
        self.owner = owner
        self.default = default
        self.name: str | None = None

    def __set_name__(self, objtype: type, name: str) -> None:
        self.name = name  # PEP 487: told our own attribute name automatically
        key = (self.owner, name)
        with agConfig._registry_lock:
            if key in agConfig.FIELD_REGISTRY:
                raise ValueError(
                    f"config field {self.owner}.{name} is already registered "
                    f"(by {agConfig.FIELD_REGISTRY[key]!r}); field names must be unique per owner"
                )
            agConfig.FIELD_REGISTRY[key] = self


class GlobalConfigParam(_ConfigParam):
    """Tier 1: process-wide. This is *not* separate machinery from tier 2 --
    it's the exact same get_static()/set() on agConfig, just always pointed
    at the one shared agConfig.GLOBAL instance instead of whichever agconfig
    a particular object was given. Locking, and raising on a write after the
    first read, come for free from get_static()/set() -- no tier-1-specific
    logic exists anywhere."""

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return agConfig.GLOBAL.get_static(self.owner, self.name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        try:
            agConfig.GLOBAL.set(self.owner, self.name, value)
        except ValueError:
            # agConfig.set()'s message ("clone() first") is tier-2 advice --
            # cloning doesn't help here, since every GlobalConfigParam always
            # points at the literal agConfig.GLOBAL, never at a clone of it.
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
                agconfig.get_static(self.owner, self.name, self.default)
                if agconfig is not None
                else self.default
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
            raise AttributeError(
                f"{self.owner}.{self.name} can't be set -- this instance has no agconfig"
            )
        agconfig.set(self.owner, self.name, value)


class _OwnerView:
    """Returned by agConfig.<owner> (e.g. cfg.agllm) -- lets that owner's
    fields be read/written as plain nested attribute access, dispatching to
    the right tier automatically. Not a descriptor itself; a small bound
    object holding which agConfig instance and which owner it's scoped to."""

    def __init__(self, agconfig: "agConfig", owner: str) -> None:
        object.__setattr__(self, "_agconfig", agconfig)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name: str) -> Any:
        knob = agConfig.FIELD_REGISTRY.get((self._owner, name))
        if knob is None:
            raise AttributeError(f"{self._owner} has no registered field {name!r}")
        if isinstance(knob, GlobalConfigParam):
            return agConfig.GLOBAL.get_static(self._owner, name, knob.default)
        if isinstance(knob, StaticConfigParam):
            return self._agconfig.get_static(self._owner, name, knob.default)
        return self._agconfig.get(self._owner, name, knob.default)  # DynamicConfigParam

    def __setattr__(self, name: str, value: Any) -> None:
        knob = agConfig.FIELD_REGISTRY.get((self._owner, name))
        if knob is None:
            raise AttributeError(f"{self._owner} has no registered field {name!r}")
        if isinstance(knob, GlobalConfigParam):
            agConfig.GLOBAL.set(
                self._owner, name, value
            )  # same shared target regardless of which agconfig this is
        else:
            self._agconfig.set(self._owner, name, value)


class _AgConfigViewBase:
    """Base for a ``*Config`` view scoped to one owner (e.g. ``agLLMBackendConfig``
    for ``"agllm_backend"``, ``agAgentConfig`` for ``"agent"``). Each subclass
    just sets ``_OWNER``; this class provides the shared construction,
    ``.agconfig`` handoff, and multi-field ``update()`` -- so setting several
    fields on one owner doesn't need one ``cfg.<owner>.<field> = value`` line
    each::

        cfg = agConfig(agLLMBackendConfig(model="...", api_key="...", base_url="..."))
        ag = agent(agconfig=cfg)

    Not a descriptor and not related to ``_OwnerView`` above (which dispatches
    live attribute reads/writes on an *existing* agConfig for a class that
    already has one) -- this is a standalone builder usable before any
    consuming instance, or its agConfig, exists yet. Wraps an existing
    ``agConfig`` if given one (mutating it in place, so it composes with other
    owners' views on the same object regardless of order), or creates a fresh
    one. Multiple views (or plain dicts) can also be merged in one call via
    ``agConfig(view_a, view_b, ...)``.

    Only covers flat fields registered as a ``ConfigParam`` under ``_OWNER``.
    Structural, non-flat vocabulary (like ``agSandboxConfig``'s ``add_mount``,
    which read-modify-writes a dict rather than overwriting one field) is
    layered on top by the subclass -- see ``agSandboxConfig`` in ``agsandbox.py``.
    """

    _OWNER: ClassVar[str]  # set by each subclass
    # Narrows update()/__init__ to a subset of _OWNER's registered fields --
    # e.g. a backend-specific agLLMBackendConfig subclass that only exposes
    # the fields its backend actually reads, so passing one it silently
    # ignores raises immediately instead of failing later at the API call.
    # None (the default) means every field registered under _OWNER is
    # allowed -- unchanged behavior for every pre-existing *Config view.
    _ALLOWED_FIELDS: "ClassVar[frozenset[str] | None]" = None

    def __init__(self, agconfig: "agConfig | None" = None, **fields: Any) -> None:
        self._agconfig = agconfig if agconfig is not None else agConfig()
        if fields:
            self.update(**fields)

    @property
    def agconfig(self) -> "agConfig":
        """The underlying, propagatable agConfig -- pass this onward, not the view."""
        return self._agconfig

    def update(self, **fields: Any) -> "_AgConfigViewBase":
        # Same per-field tier dispatch as _OwnerView.__setattr__: a
        # GlobalConfigParam always targets agConfig.GLOBAL, regardless of
        # which agConfig this view wraps -- writing it onto self._agconfig
        # instead would silently create a dead override nothing ever reads,
        # since GlobalConfigParam.__get__ only ever consults agConfig.GLOBAL.
        known = {
            name: knob
            for (owner, name), knob in agConfig.FIELD_REGISTRY.items()
            if owner == self._OWNER
        }
        if self._ALLOWED_FIELDS is not None:
            known = {name: knob for name, knob in known.items() if name in self._ALLOWED_FIELDS}
        unknown = set(fields) - set(known)
        if unknown:
            raise TypeError(f"{type(self).__name__} has no field(s) {sorted(unknown)}")
        for name, value in fields.items():
            if isinstance(known[name], GlobalConfigParam):
                agConfig.GLOBAL.set(self._OWNER, name, value)
            else:
                self._agconfig.set(self._OWNER, name, value)
        return self
