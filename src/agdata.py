import json


class agdata:
    """Generic data container with JSON/dict serialization."""

    def __init__(self, **data):
        object.__setattr__(self, "_data", data)

    def to_dict(self) -> dict:
        return dict(self._data)

    def to_json(self) -> str:
        return json.dumps(self._data)

    @classmethod
    def from_dict(cls, d: dict) -> "agdata":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "agdata":
        return cls(**json.loads(s))

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value):
        object.__getattribute__(self, "_data")[name] = value

    def __repr__(self) -> str:
        return f"agdata({self._data!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, agdata):
            return self._data == other._data
        return NotImplemented
