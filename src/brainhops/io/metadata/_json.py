"""
The vocabulary as a flat JSON object: the codec that BIDS sidecars
(`_bids`) and the JSON stores of formats (the `Metadata` of an X5 node)
share.

A vocabulary field is stored under its *sidecar key*: its BIDS key
(`"RepetitionTime"`), or, when BIDS has none, its name in `CamelCase`
(`display_range` as `"DisplayRange"`). Every other key of the object is
`extra`. `GeneratedBy` is the BIDS list of objects (`Name`, `Version`,
`Description`, `CodeURL`), and `channels` a list of objects with the
`CamelCase` names of the `Channel` fields. Times are ISO 8601 strings.
An encoding direction is its BIDS string (`"j-"`), or, when it has none,
an object (`Vector`, `Space`). A known term (a `SpaceEnum`, an `IntentEnum`,
...) is its string, a `data_unit` its unit symbol (`"ms"`, `"a.u."`), a
`dtype` its `numpy` name (`"int16"`).
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

from brainhops.datamodel.metadata import (
    Channel,
    ConversionReport,
    EncodingDirection,
    GeneratedBy,
)

# internals
from brainhops.datamodel.metadata._internal_api import BIDS_KEYS
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.units import Unit


def sidecar_key(name: str) -> str:
    """
    The key of a vocabulary field in a JSON object.

    Parameters
    ----------
    name : str
        The name of the vocabulary field.

    Returns
    -------
    str
        The BIDS key of the field, or its name in `CamelCase` when BIDS
        has no key for it.
    """
    return BIDS_KEYS.get(name) or _camel(name)


def decode_object(
    obj: tx.Mapping[str, tx.Any], names: tx.Iterable[str]
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
    """
    Split a JSON object into vocabulary values and other keys.

    Parameters
    ----------
    obj : mapping
        The JSON object.
    names : iterable of str
        The vocabulary fields to read, each from its key (see
        `sidecar_key`).

    Returns
    -------
    values : dict
        Field name to decoded value, for the fields found in `obj`.
    others : dict
        The other keys of `obj`, with their values as they are.
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
    obj: tx.Dict[str, tx.Any],
    changed: tx.Mapping[str, tx.Any],
    *,
    report: ConversionReport,
) -> None:
    """
    Write changed vocabulary values into a JSON object, in place.

    A value that JSON cannot hold (an encoding direction in a coordinate
    system without a name) is not written: its key is removed, so that
    the object does not keep a stale value, and the value is reported as
    lost.

    Parameters
    ----------
    obj : dict
        The JSON object to edit.
    changed : mapping
        Field name to new value. Each value is written under the key of
        its field (see `sidecar_key`), and a `None` value removes the key.
    report : ConversionReport
        The report to fill with the values that were not written.
    """
    for name, value in changed.items():
        key = sidecar_key(name)
        if value is None:
            obj.pop(key, None)
            continue
        try:
            obj[key] = to_json(name, value)
        except ValueError:
            obj.pop(key, None)
            report.lost[name] = value


def encode_extra(
    obj: tx.Dict[str, tx.Any],
    extra: tx.Mapping[str, tx.Any],
    names: tx.Iterable[str],
    *,
    report: ConversionReport,
    reserved: tx.Collection[str] = (),
) -> None:
    """
    Replace the free-form keys of a JSON object with `extra`, in place.

    The free-form keys of the object are those that hold no vocabulary
    field, which `decode_object` decodes as `extra`. They are removed,
    and each key of `extra` is written as JSON. A key that the format
    keeps for its own use, or that holds a vocabulary field, is not
    written, and is reported as lost.

    Parameters
    ----------
    obj : dict
        The JSON object to edit.
    extra : mapping
        The new free-form keys and their values.
    names : iterable of str
        The vocabulary fields that the object stores, whose keys are not
        free-form (see `sidecar_key`).
    report : ConversionReport
        The report to fill with the keys that were refused.
    reserved : collection of str, optional
        The keys the format keeps for its own use.
    """
    vocabulary = {sidecar_key(name) for name in names}
    for key in [key for key in obj if key not in vocabulary]:
        if key not in reserved:
            del obj[key]
    for key, value in extra.items():
        if key in reserved or key in vocabulary:
            report.lost[f"extra[{key!r}]"] = value
        else:
            obj[key] = to_json("extra", value)


def from_json(name: str, value: tx.Any) -> tx.Any:
    """
    Decode the JSON form of a vocabulary value.

    Parameters
    ----------
    name : str
        The name of the vocabulary field.
    value : object
        The value, as JSON holds it.

    Returns
    -------
    object
        The value, ready to be assigned to the field, which converts it
        further.
    """
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
    """
    Encode a vocabulary value, or a value of `extra`, as JSON.

    `GeneratedBy` and `Channel` entries become objects with keys in the
    BIDS style. An encoding direction becomes its BIDS string, or, when
    BIDS cannot write it, an object with the keys `Vector` and `Space`;
    a space that is a `CoordinateSystem` is written as its name.

    Parameters
    ----------
    name : str
        The name of the vocabulary field, or `"extra"`.
    value : object
        The value to encode.

    Returns
    -------
    object
        A value that can be serialised to JSON.

    Raises
    ------
    ValueError
        If `value` is an encoding direction in a coordinate system
        without a name, which JSON cannot hold.
    """
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
            out["Space"] = _space_label(value.space)
        return out
    return jsonable(value)


def jsonable(value: tx.Any) -> tx.Any:
    """
    Make a value serialisable to JSON.

    Times become ISO 8601 strings, known terms become their strings, a
    data type becomes its name, a unit becomes its symbol, and tuples and
    arrays become lists.

    Parameters
    ----------
    value : object
        The value to convert.

    Returns
    -------
    object
        A value that can be serialised to JSON.
    """
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


def _space_label(space: tx.Any) -> str:
    """The string a JSON object holds for the space of a direction: a
    label as it is, a coordinate system by its name."""
    if not isinstance(space, CoordinateSystem):
        return str(space)
    if not space.name:
        raise ValueError(
            "A direction in a coordinate system without a name cannot be "
            "written: JSON names the space of a direction."
        )
    return space.name
