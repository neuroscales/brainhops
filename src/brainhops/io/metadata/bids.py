"""
The BIDS JSON sidecar codec.

A sidecar is a flat JSON object. A key that is the BIDS key of a
vocabulary field (`"RepetitionTime"` for `repetition_time`, see the
`Bids(...)` annotation of each field) fills that field, and the units
already match (seconds, degrees, tesla). A vocabulary field that BIDS
has no key for is written under its own name in `CamelCase`
(`display_range` as `"DisplayRange"`), so that a sidecar written by
brainhops reads back whole. Every other key lands in `extra`, and
`extra` is written back key by key.

`GeneratedBy` is the BIDS list of objects (`Name`, `Version`,
`Description`, `CodeURL`), and `channels` a list of objects with the
`CamelCase` names of [`Channel`][brainhops.datamodel.metadata.Channel]
fields. Times are ISO 8601 strings. An encoding direction is its BIDS
string (`"j-"`); one BIDS cannot write (along no voxel axis) is reported
as lost by `to_bids`, and written as an object (`Vector`, `Space`) in the
JSON stores of other formats (x5, Zarr), which read it back. A known
term (a `Space`, an `Intent`, ...) is its string, a `data_unit` its unit
symbol (`"ms"`, `"a.u."`), a `data_type` its `numpy` name (`"int16"`).
The fields of the
[`DiffusionMetadata`][brainhops.datamodel.metadata.DiffusionMetadata]
group are not sidecar keys (BIDS keeps them in `.bval`/`.bvec` files, in
voxel axes), so `to_bids` reports them as lost.
"""

__all__ = ["from_bids", "to_bids"]

# stdlib
import datetime
import enum
import json
import os

# externals
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.datamodel.metadata import (
    BIDS_KEYS,
    GROUPS,
    UNSUPPORTED,
    VOCABULARY,
    Channel,
    ConversionReport,
    DiffusionMetadata,
    EncodingDirection,
    GeneratedBy,
    LossPolicy,
    Metadata,
    apply_loss_policy,
)
from brainhops.datamodel.units import Unit

# Not sidecar keys: BIDS stores them as `.bval`/`.bvec` files.
_NOT_IN_SIDECAR = GROUPS[DiffusionMetadata]

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


def sidecar_key(name: str) -> str:
    """The sidecar key of a vocabulary field: its BIDS key, or its name
    in `CamelCase` when BIDS has none."""
    return BIDS_KEYS.get(name) or _camel(name)


def _keys() -> tx.Dict[str, str]:
    """Sidecar key -> vocabulary field name."""
    return {
        sidecar_key(name): name
        for name in VOCABULARY
        if name not in _NOT_IN_SIDECAR
    }


def _read(source: tx.Any) -> tx.Dict[str, tx.Any]:
    if isinstance(source, tx.Mapping):
        return dict(source)
    if isinstance(source, str) and source.lstrip().startswith("{"):
        return json.loads(source)
    if isinstance(source, (str, os.PathLike, path.PathLike)):
        with path.Path(source).open("r") as f:
            return json.load(f)
    if hasattr(source, "read"):
        return json.load(source)
    raise TypeError(
        f"A BIDS sidecar is a mapping, a JSON string, a path or an open "
        f"file, not {type(source).__name__}."
    )


def _from_json(name: str, value: tx.Any) -> tx.Any:
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


def _to_json(name: str, value: tx.Any) -> tx.Any:
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
                key: _jsonable(getattr(entry, attr))
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
    return _jsonable(value)


def _jsonable(value: tx.Any) -> tx.Any:
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, Unit):
        # The symbol (`"a.u."`, `"mm / s"`) parses back to the same unit.
        return value.symbol
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def from_bids(source: tx.Any) -> Metadata:
    """
    Read a BIDS JSON sidecar into
    [`Metadata`][brainhops.datamodel.metadata.Metadata].

    Parameters
    ----------
    source : mapping | str | PathLike | file
        The sidecar, as a decoded JSON object, a JSON string, a path, or
        an open file.

    Returns
    -------
    Metadata
        The vocabulary fields the sidecar names, and its other keys in
        `extra`.
    """
    sidecar = _read(source)
    keys = _keys()
    values: tx.Dict[str, tx.Any] = {}
    extra: tx.Dict[str, tx.Any] = {}
    for key, value in sidecar.items():
        name = keys.get(key)
        if name is None:
            extra[key] = value
        else:
            values[name] = _from_json(name, value)
    return Metadata(extra=extra, **values)


def to_bids(
    metadata: Metadata,
    *,
    on_loss: tx.Optional[LossPolicy] = None,
) -> tx.Dict[str, tx.Any]:
    """
    Write metadata as a BIDS JSON sidecar.

    Parameters
    ----------
    metadata : Metadata
        The metadata to write; only its vocabulary and `extra` are used.
    on_loss : {"ignore", "warn", "raise"}, optional
        What to do with the fields a sidecar cannot hold (the diffusion
        fields, an encoding direction along no voxel axis). Defaults to
        the policy in effect.

    Returns
    -------
    dict
        A JSON-serialisable sidecar. A key of `extra` that collides with
        a vocabulary key is overridden by the vocabulary value.
    """
    report = ConversionReport(source=metadata.format, target="bids")
    sidecar: tx.Dict[str, tx.Any] = {}
    extra = metadata.extra
    if extra and extra is not UNSUPPORTED:
        sidecar.update(_jsonable(dict(extra)))
    for name in VOCABULARY:
        value = getattr(metadata, name, None)
        if value is None or value is UNSUPPORTED:
            continue
        if name in _NOT_IN_SIDECAR or (
            isinstance(value, EncodingDirection) and value.to_bids() is None
        ):
            report.lost[name] = value
            continue
        sidecar[sidecar_key(name)] = _to_json(name, value)
    apply_loss_policy(report, on_loss, stacklevel=3)
    return sidecar
