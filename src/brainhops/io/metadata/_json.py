"""
The vocabulary as a flat JSON object: the codec that BIDS sidecars
(`bids`) and the JSON stores of formats (x5 node `Metadata`, Zarr
attributes) share.

A vocabulary field is stored under its *sidecar key*: its BIDS key
(`"RepetitionTime"`), or, when BIDS has none, its name in `CamelCase`
(`display_range` as `"DisplayRange"`). Every other key of the object is
`extra`. `GeneratedBy` is the BIDS list of objects (`Name`, `Version`,
`Description`, `CodeURL`), and `channels` a list of objects with the
`CamelCase` names of the `Channel` fields. Times are ISO 8601 strings.
An encoding direction is its BIDS string (`"j-"`), or, when it has none,
an object (`Vector`, `Space`). A known term (a `Space`, an `Intent`,
...) is its string, a `data_unit` its unit symbol (`"ms"`, `"a.u."`), a
`data_type` its `numpy` name (`"int16"`).
"""

__all__ = [
    "decode_object",
    "encode_changes",
    "encode_extra",
    "from_json",
    "jsonable",
    "sidecar_key",
    "to_json",
]

# stdlib
import datetime
import enum

# externals
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel.metadata import (
    BIDS_KEYS,
    Channel,
    ConversionReport,
    EncodingDirection,
    GeneratedBy,
)
from brainhops.datamodel.units import Unit


def sidecar_key(name: str) -> str:
    """The key of a vocabulary field in a JSON object: its BIDS key, or
    its name in `CamelCase` when BIDS has none."""
    return BIDS_KEYS.get(name) or _camel(name)


def decode_object(
    obj: tx.Mapping[str, tx.Any], names: tx.Iterable[str]
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
    """
    Split a JSON object into the values of the vocabulary fields `names`
    (read from their sidecar keys) and the other keys, as they are.
    """
    fields = {sidecar_key(name): name for name in names}
    values: tx.Dict[str, tx.Any] = {}
    others: tx.Dict[str, tx.Any] = {}
    for key, value in obj.items():
        name = fields.get(key)
        if name is None:
            others[key] = value
        else:
            values[name] = from_json(name, value)
    return values, others


def encode_changes(
    obj: tx.Dict[str, tx.Any], changed: tx.Mapping[str, tx.Any]
) -> None:
    """Write changed vocabulary values into a JSON object, in place,
    under their sidecar keys; a `None` removes the key."""
    for name, value in changed.items():
        key = sidecar_key(name)
        if value is None:
            obj.pop(key, None)
        else:
            obj[key] = to_json(name, value)


def encode_extra(
    obj: tx.Dict[str, tx.Any],
    diff: tx.Mapping[str, tx.Any],
    *,
    report: ConversionReport,
    reserved: tx.Collection[str] = (),
) -> None:
    """
    Apply an `extra` diff (see `FileBasedMetadata.changed_fields`) to a
    JSON object, in place: a `None` removes the key, any other value is
    written as JSON. A key in `reserved` (one the format keeps for its
    own use) is not written, and is reported as lost.
    """
    # Not `FileBasedMetadata`'s diff helper: this one also writes JSON
    # and refuses the keys a format reserves, two things the data model
    # knows nothing of.
    for key, value in diff.items():
        if key in reserved:
            report.lost[f"extra[{key!r}]"] = value
        elif value is None:
            obj.pop(key, None)
        else:
            obj[key] = to_json("extra", value)


def from_json(name: str, value: tx.Any) -> tx.Any:
    """The value of the vocabulary field `name` from its JSON form."""
    if value is None:
        return None
    if name == "generated_by":
        entries = value if isinstance(value, list) else [value]
        return tuple(
            GeneratedBy(
                **{
                    _GENERATED_BY_KEYS[k]: v
                    for k, v in entry.items()
                    if k in _GENERATED_BY_KEYS
                }
            )
            for entry in entries
        )
    if name == "channels":
        return tuple(
            Channel(
                **{
                    _CHANNEL_KEYS[k]: v
                    for k, v in entry.items()
                    if k in _CHANNEL_KEYS
                }
            )
            for entry in value
        )
    if name in _TIMES and isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            return value
    if name in ("history", "sources") and isinstance(value, str):
        return (value,)
    return value


def to_json(name: str, value: tx.Any) -> tx.Any:
    """The JSON form of a value of the vocabulary field `name` (or of
    an `extra` value): `GeneratedBy` and `Channel` entries as objects
    with BIDS-style keys, an encoding direction as its BIDS string (or,
    when BIDS cannot write it, an object `Vector`/`Space`)."""
    if name == "generated_by":
        return [
            {
                key: getattr(entry, attr)
                for key, attr in _GENERATED_BY_KEYS.items()
                if getattr(entry, attr) is not None
            }
            for entry in value
        ]
    if name == "channels":
        return [
            {
                key: jsonable(getattr(entry, attr))
                for key, attr in _CHANNEL_KEYS.items()
                if getattr(entry, attr) is not None
            }
            for entry in value
        ]
    if isinstance(value, EncodingDirection):
        bids = value.to_bids()
        if bids is not None:
            return bids
        out = {"Vector": list(value.vector)}
        if value.space is not None:
            out["Space"] = str(value.space)
        return out
    return jsonable(value)


def jsonable(value: tx.Any) -> tx.Any:
    """A value made JSON-serialisable: times as ISO 8601 strings, terms
    as their strings, a data type as its name, a unit as its symbol,
    tuples and arrays as lists."""
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return jsonable(value.value)
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, Unit):
        # The symbol (`"a.u."`, `"mm / s"`) parses back to the same unit.
        return value.symbol
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------

_GENERATED_BY_KEYS = {
    "Name": "name",
    "Version": "version",
    "Description": "description",
    "CodeURL": "code_url",
}

_CHANNEL_KEYS = {
    "Name": "name",
    "Color": "color",
    "DisplayRange": "display_range",
    "Unit": "unit",
}

_TIMES = ("creation_time", "acquisition_time")


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))
