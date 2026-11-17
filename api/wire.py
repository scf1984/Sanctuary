"""The one encoding of `api.values` as JSON, shared by the service and every remote client (#36).

**One codec, used by both ends.** Gate 1 on #35 warned that a stream beside requests means "two
places a message shape is defined", and that the failure is a field added to one and not the
other. There is no shape defined here at all: a dataclass is encoded field by field from its own
definition and tagged with its type's name, so the shape is whatever `api.values` says it is,
read the same way on both sides.

- A frozen dataclass becomes an object holding its fields and a `"$type"`.
- A `Column` becomes its dtype, its shape and its bytes as base64. That is the cheapest text form
  of the packed bytes the owner chose on #35: about 2.1 MB a frame at 100,000 animals, against
  1.6 MB raw.
- A tuple becomes a JSON array and decodes back to a tuple, because every sequence in
  `api.values` is a tuple, and a list would compare unequal to the value it came from.

Only the types listed in `_TYPES` decode. Anything else is refused (§8.7). A decoder that built
whatever class a message named would be a remote-code door on a public repository's server.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from typing import Any

from api import values

_TYPES: dict[str, type] = {
    name: getattr(values, name)
    for name in values.__all__
    if isinstance(getattr(values, name), type) and dataclasses.is_dataclass(getattr(values, name))
}


def encode(value: Any) -> Any:
    """`value` as JSON-compatible Python: dicts, lists, strings, numbers, booleans and None."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, values.Column):
        return {
            "$type": "Column",
            "dtype": value.dtype,
            "shape": list(value.shape),
            "data": base64.b64encode(value.data).decode("ascii"),
        }
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _TYPES.get(name) is not type(value):
            raise TypeError(f"{name} is not a type the boundary carries")
        encoded = {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
        encoded["$type"] = name
        return encoded
    raise TypeError(f"cannot encode a {type(value).__module__}.{type(value).__name__}")


def decode(data: Any) -> Any:
    """The value `encode` produced `data` from."""
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return tuple(decode(item) for item in data)
    if isinstance(data, dict):
        name = data.get("$type")
        if name == "Column":
            return values.Column(
                dtype=data["dtype"],
                shape=tuple(int(n) for n in data["shape"]),
                data=base64.b64decode(data["data"]),
            )
        cls = _TYPES.get(name) if isinstance(name, str) else None
        if cls is None:
            raise ValueError(f"not a type the boundary carries: {name!r}")
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: decode(item) for key, item in data.items() if key in fields})
    raise ValueError(f"cannot decode a {type(data).__name__}")


def dumps(value: Any) -> bytes:
    """`value` as UTF-8 JSON. Refuses NaN and infinity, which `JSON.parse` in a browser rejects,
    so a value that cannot reach the browser client (#292) fails here, loudly, rather than there."""
    return json.dumps(encode(value), allow_nan=False, separators=(",", ":")).encode("utf-8")


def loads(payload: bytes) -> Any:
    return decode(json.loads(payload))
