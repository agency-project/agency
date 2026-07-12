from __future__ import annotations

import threading

from .agutil import _b36_suffix

_NOUNS = [
    "alex",
    "andy",
    "arch",
    "bake",
    "bale",
    "band",
    "bart",
    "base",
    "beam",
    "bear",
    "beef",
    "bell",
    "bill",
    "bird",
    "blue",
    "boat",
    "bolt",
    "bond",
    "bone",
    "bonk",
    "book",
    "boss",
    "brim",
    "buzz",
    "byte",
    "cage",
    "cake",
    "cane",
    "cant",
    "cape",
    "cart",
    "cask",
    "cave",
    "chip",
    "clam",
    "clay",
    "coal",
    "coil",
    "coin",
    "colt",
    "cord",
    "core",
    "corn",
    "cove",
    "crab",
    "crag",
    "crow",
    "dale",
    "dart",
    "deer",
    "dome",
    "dove",
    "down",
    "drum",
    "duck",
    "dune",
    "dust",
    "east",
    "edge",
    "evil",
    "fang",
    "fast",
    "fate",
    "fawn",
    "felt",
    "fern",
    "film",
    "fire",
    "fish",
    "fist",
    "flat",
    "flaw",
    "flux",
    "foam",
    "font",
    "fork",
    "frog",
    "fuse",
    "gale",
    "game",
    "gate",
    "gear",
    "glen",
    "greg",
    "grip",
    "gust",
    "hail",
    "hare",
    "hawk",
    "haze",
    "hemp",
    "hill",
    "hind",
    "hole",
    "hoop",
    "hull",
    "ibex",
    "ivan",
    "jake",
    "jane",
    "joey",
    "juke",
    "kite",
    "kodo",
    "ksen",
    "lake",
    "land",
    "lard",
    "lash",
    "lava",
    "leaf",
    "lego",
    "lily",
    "lion",
    "lord",
    "love",
    "lynx",
    "made",
    "many",
    "mark",
    "mean",
    "mert",
    "mess",
    "meta",
    "mick",
    "mill",
    "mink",
    "moba",
    "moon",
    "moth",
    "mule",
    "must",
    "nail",
    "nate",
    "next",
    "node",
    "onix",
    "pain",
    "park",
    "peat",
    "pier",
    "pike",
    "pile",
    "pine",
    "plug",
    "pony",
    "pool",
    "pork",
    "puma",
    "rain",
    "rate",
    "real",
    "reef",
    "rest",
    "rice",
    "road",
    "rock",
    "roll",
    "rope",
    "rust",
    "sage",
    "salt",
    "sand",
    "seal",
    "shot",
    "silk",
    "slag",
    "snow",
    "soda",
    "soil",
    "sold",
    "song",
    "spam",
    "star",
    "surf",
    "swan",
    "tack",
    "tail",
    "tide",
    "tire",
    "toad",
    "tony",
    "tool",
    "tree",
    "tuna",
    "turf",
    "vast",
    "vent",
    "vine",
    "wake",
    "ward",
    "wasp",
    "well",
    "wick",
    "wind",
    "wire",
    "wolf",
    "wood",
    "yang",
    "zinc",
]


class agname(str):
    """A unique agent name string, allocated from a shared global registry.

    Subclasses ``str``, so every instance IS a string and can be used
    anywhere a plain string is expected — f-strings, dict keys, path
    components, equality checks, etc.

    Class-level state is shared across all agents in a process.
    """

    _noun_index: int = 0
    _noun_counters: dict[str, int] = {}
    _allocated: set[str] = set()
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, value: str) -> "agname":
        return super().__new__(cls, value)

    @classmethod
    def claim_unique_agname(cls, name: str) -> "agname":
        """Claim an exact, fully-formed name as in-use, raising if already taken.

        Used when restoring an agent from saved state where the name is known.
        """
        with cls._lock:
            if name in cls._allocated:
                raise ValueError(f"agname {name!r} is already in use by another agent")
            cls._allocated.add(name)
        return cls(name)

    @classmethod
    def allocate_agname(cls, name: str | None = None) -> "agname":
        """Return a unique name in the form <base>_XXXX and mark it in-use.

        If ``name`` is given, it is used as the base (e.g. ``"Worker"`` →
        ``"Worker_0000"``).  If ``name`` is ``None``, the next noun from the
        built-in ``_NOUNS`` list is chosen automatically, cycling back to the
        start after the last entry.

        XXXX is a 4-character base-36 suffix (1 679 616 unique values per base).
        """
        with cls._lock:
            if name is None:
                name = _NOUNS[cls._noun_index % len(_NOUNS)]
                cls._noun_index += 1
            n = cls._noun_counters.get(name, 0)
            cls._noun_counters[name] = n + 1
            full = f"{name}_{_b36_suffix(n)}"
            cls._allocated.add(full)
        return cls(full)
