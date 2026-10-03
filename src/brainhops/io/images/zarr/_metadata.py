"""
The metadata of Zarr images: `ZarrMetadata` for a plain array, and
`OmeZarrMetadata` for an OME-Zarr multiscale pyramid.

**Plain Zarr.** A plain array has no metadata convention, only free-form
attributes. The vocabulary is stored as a BIDS-style sidecar (the keys of
[`Metadata.to_bids`][brainhops.datamodel.metadata.Metadata.to_bids]) under
the array attribute `"brainhops"`, and `extra` maps to the other
attributes of the array. Everything but the diffusion fields (which are
not sidecar keys) is supported. The record (`raw`) is the dict of the
array's attributes.

**OME-Zarr.** The record is an `OmeZarrRecord`: the typed `abczarr`
multiscale (normalised to 0.6), the `omero` block as JSON, and the group
attributes that are not OME metadata. What the vocabulary covers:

| Field | Record |
|---|---|
| `name` | multiscale `name` |
| `channels` | `omero.channels`: `label`, `color`, `window.start`/`end` |
| `display_range` | `omero.channels[*].window.start`/`end` |
| `extra` | the other group attributes |

An omero color (`RRGGBB`) reads as an RGBA string (`RRGGBBFF`). The
display range is read when every channel shares it, and written to every
channel. A window is required for each channel: when nothing gives one,
it is the range of the data type (0 to 1 for floats). OME-Zarr has no
unit for the values, so `data_unit` is unsupported, and the unit of a
channel is dropped (approximated).

There is one metadata object per pyramid; each level holds a derived copy
(see `OmeZarrImage`).
"""

__all__ = ["OmeZarrMetadata", "OmeZarrRecord", "ZarrMetadata"]

# stdlib
import copy

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr

# internals
from brainhops.datamodel.metadata import (
    _VOCABULARY,
    Channel,
    ConversionReport,
    FormatMetadata,
)
from brainhops.io.metadata.bids import _from_json, _to_json, sidecar_key

# The array attribute that holds the vocabulary of a plain Zarr image.
BRAINHOPS_KEY = "brainhops"

# Not sidecar keys (see `brainhops.io.metadata.bids`).
_NOT_IN_SIDECAR = ("diffusion_bvalues", "diffusion_bvectors")

# Group attributes that belong to the OME metadata (0.5 and later nest it
# under "ome"; 0.4 writes it at the top level).
OME_KEYS = frozenset(
    {
        "ome",
        "multiscales",
        "omero",
        "image-label",
        "labels",
        "plate",
        "well",
    }
)


def _apply_extra(
    attrs: tx.Dict[str, tx.Any],
    diff: tx.Mapping[str, tx.Any],
    reserved: tx.Collection[str],
    report: ConversionReport,
) -> None:
    for key, value in diff.items():
        if key in reserved:
            report.lost[f"extra[{key!r}]"] = value
        elif value is None:
            attrs.pop(key, None)
        else:
            attrs[key] = _to_json("extra", value)


# ----------------------------------------------------------------------
#   PLAIN ZARR
# ----------------------------------------------------------------------


class ZarrMetadata(
    FormatMetadata,
    on={"format": "zarr"},
    supports=tuple(
        name for name in _VOCABULARY if name not in _NOT_IN_SIDECAR
    ),
):
    """
    The metadata of a plain Zarr array: the vocabulary as a sidecar under
    the attribute `"brainhops"`, and `extra` as the other attributes.
    Its record (`raw`) is the dict of the array's attributes.
    """

    format: tx.Annotated[tx.Literal["zarr"], tx.Doc("Always `'zarr'`.")] = (
        "zarr"
    )

    raw: tx.Annotated[
        tx.Optional[tx.Dict[str, tx.Any]],
        tx.Doc("The attributes of the array that was read, as JSON."),
        NoRepr(),
        NoEq(),
    ] = None

    @classmethod
    def _default_raw(cls) -> tx.Dict[str, tx.Any]:
        return {}

    @classmethod
    def _decode(
        cls, raw: tx.Optional[tx.Mapping[str, tx.Any]], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        if not raw:
            return {}
        out: tx.Dict[str, tx.Any] = {}
        block = raw.get(BRAINHOPS_KEY)
        if isinstance(block, tx.Mapping):
            for name in cls.vocabulary_fields:
                if name in cls.unsupported_fields:
                    continue
                key = sidecar_key(name)
                if key in block:
                    out[name] = _from_json(name, block[key])
        extra = {k: v for k, v in raw.items() if k != BRAINHOPS_KEY}
        if extra:
            out["extra"] = extra
        return out

    def _encode(
        self,
        raw: tx.Dict[str, tx.Any],
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> tx.Dict[str, tx.Any]:
        block = dict(raw.get(BRAINHOPS_KEY) or {})
        for name, value in changed.items():
            if name == "extra":
                continue
            key = sidecar_key(name)
            if value is None:
                block.pop(key, None)
            else:
                block[key] = _to_json(name, value)
        if "extra" in changed:
            _apply_extra(raw, changed["extra"], (BRAINHOPS_KEY,), report)
        if block:
            raw[BRAINHOPS_KEY] = block
        else:
            raw.pop(BRAINHOPS_KEY, None)
        return raw


# ----------------------------------------------------------------------
#   OME-ZARR
# ----------------------------------------------------------------------


class OmeZarrRecord:
    """
    The record of an OME-Zarr pyramid: its typed `abczarr` multiscale
    (normalised to 0.6), its `omero` block as JSON, and the group
    attributes that are not OME metadata.
    """

    __slots__ = ("multiscale", "omero", "attrs")

    def __init__(
        self,
        multiscale: tx.Any = None,
        omero: tx.Optional[tx.Dict[str, tx.Any]] = None,
        attrs: tx.Optional[tx.Dict[str, tx.Any]] = None,
    ) -> None:
        self.multiscale = multiscale
        self.omero = omero
        self.attrs = dict(attrs or {})

    def __deepcopy__(self, memo: tx.Dict) -> "OmeZarrRecord":
        # The typed multiscale is immutable: it is shared.
        return OmeZarrRecord(
            self.multiscale,
            copy.deepcopy(self.omero, memo),
            copy.deepcopy(self.attrs, memo),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OmeZarrRecord):
            return NotImplemented
        return (
            _json(self.multiscale) == _json(other.multiscale)
            and self.omero == other.omero
            and self.attrs == other.attrs
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        name = getattr(self.multiscale, "name", None)
        return (
            f"OmeZarrRecord(multiscale={name!r}, "
            f"omero={self.omero is not None}, attrs={sorted(self.attrs)})"
        )

    @classmethod
    def from_attributes(
        cls, multiscale: tx.Any, attrs: tx.Mapping[str, tx.Any]
    ) -> "OmeZarrRecord":
        """The record of a group, from its multiscale and its attributes
        (as JSON)."""
        attrs = dict(attrs)
        block = attrs.get("ome")
        holder = block if isinstance(block, tx.Mapping) else attrs
        omero = holder.get("omero")
        return cls(
            multiscale,
            copy.deepcopy(dict(omero))
            if isinstance(omero, tx.Mapping)
            else None,
            {k: v for k, v in attrs.items() if k not in OME_KEYS},
        )


def _json(obj: tx.Any) -> tx.Any:
    return None if obj is None else obj.to_json()


def _color_in(color: tx.Any) -> tx.Optional[str]:
    """An omero color (`RRGGBB`) as an RGBA hex string."""
    if not isinstance(color, str):
        return None
    color = color.lstrip("#").upper()
    if len(color) == 6:
        return color + "FF"
    return color or None


def _color_out(
    color: tx.Optional[str], default: str, report: ConversionReport
) -> str:
    """An RGBA hex string as an omero color (`RRGGBB`)."""
    if not color:
        return default
    color = color.lstrip("#").upper()
    if len(color) == 8:
        if color[6:] != "FF":
            report.approximated["channels"] = "alpha of the colors dropped"
        color = color[:6]
    return color


def _default_window(
    image: tx.Any, report: tx.Optional[ConversionReport] = None
) -> tx.Dict[str, float]:
    """
    The display window of a channel nothing says anything about.

    OME-Zarr requires one. The window (`start`, `end`) is the range of
    the values of the smallest level (cheap to read), and the allowed
    range (`min`, `max`) that of the data type (of the values, for
    floats); with no data, both are the range of the data type (0..1
    for floats). It is reported as approximated in `report`, under
    `"channels"`: nothing in the metadata gave it.
    """
    dtype = data = None
    try:
        dtype = np.dtype(image.images[0].data.dtype)
        data = np.asarray(image.images[-1].data)
    except Exception:
        pass
    if dtype is not None and dtype.kind in "iub":
        info = np.iinfo(np.uint8 if dtype.kind == "b" else dtype)
        lo, hi = float(info.min), float(info.max)
    else:
        lo, hi = 0.0, 1.0
    start, end, where = lo, hi, "the range of the data type"
    if data is not None and data.size:
        values = data[np.isfinite(data)] if data.dtype.kind == "f" else data
        if values.size:
            start, end = float(values.min()), float(values.max())
            where = "the range of the values of the smallest level"
            if dtype is None or dtype.kind not in "iub":
                lo, hi = start, end
    if report is not None:
        report.approximated["channels"] = (
            f"display window not given, written as {where} "
            f"({start:g}..{end:g})"
        )
    return {"min": lo, "max": hi, "start": start, "end": end}


def _channel_count(image: tx.Any) -> int:
    """The size of the channel axis of a pyramid (1 without one)."""
    try:
        shape = image.images[0].data.shape
        axes = image._write_axes(len(shape))
        for axis, size in zip(axes, shape):
            if getattr(axis, "type", None) == "channel":
                return int(size)
    except Exception:
        pass
    return 1


def _window(
    base: tx.Mapping[str, tx.Any],
    display_range: tx.Optional[tx.Tuple[float, float]],
    image: tx.Any,
    report: ConversionReport,
) -> tx.Dict[str, float]:
    if base:
        window = dict(base)
    else:
        # A display range gives the window: only invented without one.
        invented = report if display_range is None else None
        window = _default_window(image, invented)
    if display_range is not None:
        start, end = (float(v) for v in display_range)
        window["start"], window["end"] = start, end
        window["min"] = min(float(window.get("min", start)), start)
        window["max"] = max(float(window.get("max", end)), end)
    return window


class OmeZarrMetadata(
    FormatMetadata,
    on={"format": "ome-zarr"},
    supports=("name", "channels", "display_range", "extra"),
):
    """
    The metadata of an OME-Zarr multiscale pyramid; its record is an
    `OmeZarrRecord` (the multiscale, `omero` and the other group
    attributes).

    `multiscale` and `omero` are the parts of the record under their
    familiar names.
    """

    format: tx.Annotated[
        tx.Literal["ome-zarr"], tx.Doc("Always `'ome-zarr'`.")
    ] = "ome-zarr"

    raw: tx.Annotated[
        tx.Optional[OmeZarrRecord],
        tx.Doc(
            """
            The record of the pyramid that was read: its typed multiscale
            (normalised to OME-NGFF 0.6), its `omero` block and its other
            group attributes. The levels and the coordinate
            transformations are rewritten from the data model on save.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    @property
    def multiscale(self) -> tx.Any:
        """The typed `abczarr` multiscale of the record."""
        return None if self.raw is None else self.raw.multiscale

    @property
    def omero(self) -> tx.Optional[tx.Dict[str, tx.Any]]:
        """The `omero` block of the record, as JSON."""
        return None if self.raw is None else self.raw.omero

    # --- hooks --------------------------------------------------------

    @classmethod
    def _default_raw(cls) -> OmeZarrRecord:
        return OmeZarrRecord()

    @classmethod
    def _decode(
        cls, raw: tx.Optional[OmeZarrRecord], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        if raw is None:
            return {}
        out: tx.Dict[str, tx.Any] = {}
        name = getattr(raw.multiscale, "name", None)
        out["name"] = name if isinstance(name, str) and name else None
        entries = (raw.omero or {}).get("channels") or []
        channels = []
        for entry in entries:
            window = entry.get("window") or {}
            start, end = window.get("start"), window.get("end")
            channels.append(
                Channel(
                    name=entry.get("label"),
                    color=_color_in(entry.get("color")),
                    display_range=(
                        (float(start), float(end))
                        if start is not None and end is not None
                        else None
                    ),
                )
            )
        if channels:
            out["channels"] = tuple(channels)
            ranges = {c.display_range for c in channels}
            if len(ranges) == 1:
                out["display_range"] = ranges.pop()
        if raw.attrs:
            out["extra"] = dict(raw.attrs)
        return out

    def _encode(
        self,
        raw: OmeZarrRecord,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> OmeZarrRecord:
        if "name" in changed and raw.multiscale is not None:
            block = raw.multiscale.to_json()
            if changed["name"] is None:
                block.pop("name", None)
            else:
                block["name"] = changed["name"]
            raw.multiscale = type(raw.multiscale).from_json(block)
        if "channels" in changed or "display_range" in changed:
            self._encode_omero(raw, changed, image, report)
        if "extra" in changed:
            _apply_extra(raw.attrs, changed["extra"], OME_KEYS, report)
        return raw

    def _derive_raw(
        self,
        raw: tx.Optional[OmeZarrRecord],
        *,
        grid_changed: bool,
        volumes: tx.Optional[tx.Sequence[int]],
    ) -> tx.Optional[OmeZarrRecord]:
        return copy.deepcopy(raw)

    def _encode_omero(
        self,
        raw: OmeZarrRecord,
        changed: tx.Dict[str, tx.Any],
        image: tx.Any,
        report: ConversionReport,
    ) -> None:
        omero = dict(raw.omero or {})
        entries = [dict(e) for e in omero.get("channels") or []]
        common = self.display_range or None
        if "channels" in changed:
            channels = changed["channels"]
            if channels is None:
                omero.pop("channels", None)
                raw.omero = omero if omero else None
                return
            new = []
            for index, channel in enumerate(channels):
                base = entries[index] if index < len(entries) else {}
                entry = dict(base)
                if channel.name is None:
                    entry.pop("label", None)
                else:
                    entry["label"] = channel.name
                entry["color"] = _color_out(
                    channel.color, base.get("color", "FFFFFF"), report
                )
                entry["window"] = _window(
                    base.get("window") or {},
                    channel.display_range or common,
                    image,
                    report,
                )
                if channel.unit is not None:
                    report.approximated["channels"] = (
                        "channel units dropped (omero has no unit)"
                    )
                new.append(entry)
            entries = new
        elif changed["display_range"] is None:
            if entries:
                report.approximated["display_range"] = (
                    "kept: every omero channel needs a window"
                )
            return
        else:
            if not entries:
                entries = [
                    {"color": "FFFFFF"} for _ in range(_channel_count(image))
                ]
            for entry in entries:
                entry["window"] = _window(
                    entry.get("window") or {}, common, image, report
                )
        omero["channels"] = entries
        raw.omero = omero
