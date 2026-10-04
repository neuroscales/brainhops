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
import json
import os

# externals
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.datamodel.metadata import (
    GROUPS,
    UNSUPPORTED,
    VOCABULARY,
    ConversionReport,
    DiffusionMetadata,
    EncodingDirection,
    LossPolicy,
    Metadata,
    apply_loss_policy,
)

from ._json import decode_object, jsonable, sidecar_key, to_json


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
    values, extra = decode_object(_read(source), _SIDECAR_FIELDS)
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
        sidecar.update(jsonable(dict(extra)))
    for name in VOCABULARY:
        value = getattr(metadata, name, None)
        if value is None or value is UNSUPPORTED:
            continue
        if name not in _SIDECAR_FIELDS or (
            isinstance(value, EncodingDirection) and value.to_bids() is None
        ):
            report.lost[name] = value
            continue
        sidecar[sidecar_key(name)] = to_json(name, value)
    apply_loss_policy(report, on_loss, stacklevel=3)
    return sidecar


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------

# Not the diffusion fields: BIDS stores them as `.bval`/`.bvec` files.
_SIDECAR_FIELDS = tuple(
    name for name in VOCABULARY if name not in GROUPS[DiffusionMetadata]
)


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
