"""
The metadata of Zarr images: `ZarrMetadata` for a plain array, and
`OmeZarrMetadata` for an OME-Zarr multiscale pyramid.

**Plain Zarr.** A plain array has no metadata convention, only free-form
attributes. The vocabulary is stored as a BIDS-style sidecar (the keys of
[`to_bids`][brainhops.io.metadata.bids.to_bids]) under
the array attribute `"brainhops"`, and `extra` maps to the other
attributes of the array. Everything but the diffusion fields (which are
not sidecar keys) is supported. The raw record (`raw`) is a `ZarrRaw`:
the array's attributes, and the node they were read from.

**OME-Zarr.** The raw record is an `OmeZarrRaw`: the typed `abczarr`
multiscale (normalised to 0.6), the `omero` block as JSON, the group
attributes that are not OME metadata, and the group they were read
from. What the vocabulary covers:

| Field | Record |
|---|---|
| `name` | multiscale `name` |
| `channels` | `omero.channels`: `label`, `color`, `window.start`/`end` |
| `display_range` | `omero.channels[*].window.start`/`end` |
| `extra` | the other group attributes |
| `data_type` (from the data) | the data type of the arrays |

An omero color (`RRGGBB`) reads as an RGBA string (`RRGGBBFF`). The
display range is read when every channel shares it, and written to every
channel. A window is required for each channel: when nothing gives one,
it is the range of the data type (0 to 1 for floats). OME-Zarr has no
unit for the values, so `data_unit` is unsupported, and the unit of a
channel is dropped (approximated).

In both, `data_type` is derived: it is the data type of the array that
was read, and a writer stores the array as it is, so a `data_type` that
disagrees with it is reported as approximated (by `check_raw`).

Both raw records are rebuilt from the store on each read, so each
remembers the node it was read from (`node`): a parser given metadata
read from its own node keeps it as it is. The node is a handle, not
state: a deep copy or a pickle of the record drops it.

There is one metadata object per pyramid; each level holds a derived copy
(see `OmeZarrImage`).
"""

__all__ = [
    "OmeZarrMetadata",
    "OmeZarrRaw",
    "ZarrMetadata",
    "ZarrRaw",
    "node_attributes",
    "write_attributes",
]

# stdlib
import copy

# dependencies
import numpy as np
import typing_extensions as tx
from abczarr import ZarrArray, ZarrGroup
from abczarr import open as open_node

# internals
from brainhops.datamodel.metadata import (
    Channel,
    ConversionReport,
)
from brainhops.datamodel.metadata._vocabulary import (
    DisplayVocabulary,
    MicroscopyVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    TransformVocabulary,
)
from brainhops.io.base._base import register_format
from brainhops.io.base._metadata_parser import MetadataParser
from brainhops.io.base.parsers import (
    Confidence,
    ParserExistsError,
    ParserNotImplementedError,
    ParserTypeError,
    SnifferContentError,
)
from brainhops.io.metadata import FileBasedMetadata
from brainhops.io.metadata._json import (
    decode_object,
    encode_changes,
    encode_extra,
)

# locals
from ._ome import looks_like_multiscale, read_multiscale

# The array attribute that holds the vocabulary of a plain Zarr image.
BRAINHOPS_KEY = "brainhops"


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


# ----------------------------------------------------------------------
#   PLAIN ZARR
# ----------------------------------------------------------------------


class ZarrRaw:
    """
    The raw record of a plain Zarr array: its attributes, as JSON, and
    the node they were read from (`None` for a record built in memory).

    The node is a handle, not state: a deep copy or a pickle of the
    record drops it, and it takes no part in `==`.
    """

    __slots__ = ("attrs", "node")

    def __init__(
        self,
        attrs: tx.Optional[tx.Mapping[str, tx.Any]] = None,
        node: tx.Any = None,
    ) -> None:
        """
        Parameters
        ----------
        attrs : mapping, optional
            The attributes of the array, as JSON.
        node : object, optional
            The array they were read from.
        """
        self.attrs = dict(attrs or {})
        self.node = node

    def __deepcopy__(self, memo: tx.Dict) -> tx.Self:
        return type(self)(copy.deepcopy(self.attrs, memo))

    def __getstate__(self) -> tx.Dict[str, tx.Any]:
        return self.attrs

    def __setstate__(self, state: tx.Dict[str, tx.Any]) -> None:
        self.attrs, self.node = state, None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZarrRaw):
            return NotImplemented
        return self.attrs == other.attrs

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"ZarrRaw(attrs={sorted(self.attrs)})"


class _ZarrMetadataParser(MetadataParser):
    """
    The metadata parser of a Zarr store: a store is a directory, read
    from its path, never from a stream. A format implements
    `sniff_node(node)` and `from_node(node)` on the opened node, as
    `ZarrImage` does.
    """

    @classmethod
    def sniff_node(cls, node: tx.Any) -> float:
        """
        Score how confident the class is that an opened Zarr node is one
        of its nodes.

        Parameters
        ----------
        node : ZarrArray or ZarrGroup
            The opened node.

        Returns
        -------
        float
            The confidence, in `[0, 1]`. By default, 0.
        """
        return Confidence.NO

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of an opened Zarr node, without its arrays.

        Parameters
        ----------
        node : ZarrArray or ZarrGroup
            The opened node.
        **kwargs
            Ignored.

        Returns
        -------
        Metadata
            The metadata of the node, with its raw record.

        Raises
        ------
        ParserNotImplementedError
            By default: a format implements this method.
        """
        raise ParserNotImplementedError(
            f"{cls.__name__} cannot read the metadata of a Zarr node."
        )

    @classmethod
    def sniff_file(
        cls,
        file: tx.Any,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the class is that a store is one of its
        stores.

        Parameters
        ----------
        file : str or path-like
            The location of the store.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        score = 0.0
        try:
            node = _open(file, "r")
            score = cls.sniff_node(node)
        except Exception:
            pass
        if not score and error:
            raise (SnifferContentError if error is True else error)(
                f"Not a {cls.__name__} store: {file}"
            )
        return score

    sniff_filename = sniff_file

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.Any,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Refuse an open file: a Zarr store is a directory.

        Parameters
        ----------
        file : file object
            An open file.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            0.
        """
        if error:
            raise (ParserTypeError if error is True else error)(
                "A Zarr store is a directory, not a file object."
            )
        return Confidence.NO

    sniff_bytes = sniff_fileobj

    @classmethod
    def from_file(cls, file: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of a store, without its arrays.

        Parameters
        ----------
        file : str or path-like
            The location of the store.
        **kwargs
            Options of `from_node`.

        Returns
        -------
        Metadata
            The metadata of the store, with its raw record.

        Raises
        ------
        ParserTypeError
            If `file` is an open file: a Zarr store is a directory.
        ParserExistsError
            If there is no store at the location.
        """
        if hasattr(file, "read"):
            return cls.from_fileobj(file, **kwargs)
        return cls.from_node(_open(file, "r"), **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs: tx.Any) -> tx.Any:
        """
        Refuse an open file: a Zarr store is a directory.

        Parameters
        ----------
        file : file object
            An open file.
        **kwargs
            Ignored.

        Returns
        -------
        Metadata
            Never: the method always raises.

        Raises
        ------
        ParserTypeError
            Always.
        """
        raise ParserTypeError(
            "A Zarr store is read from a store path, not from a file object."
        )


@register_format
class ZarrMetadata(
    _ZarrMetadataParser,
    FileBasedMetadata[ZarrRaw],
    on={"format": "zarr"},
    # Not the diffusion fields: they are not sidecar keys.
    supports=(
        ProvenanceVocabulary,
        MRIVocabulary,
        DisplayVocabulary,
        MicroscopyVocabulary,
        TransformVocabulary,
        "data_type",
        "extra",
    ),
):
    """
    The metadata of a plain Zarr array: the vocabulary as a sidecar under
    the attribute `"brainhops"`, and `extra` as the other attributes.
    Its raw record (`raw`) is a `ZarrRaw`, the attributes of the array
    that was read, as JSON (and the array they were read from).
    `data_type` is the data type of the array.

    `attributes` is the raw record's attributes under their familiar
    name. `ZarrMetadata.load(store)` reads the attributes of an array
    without its data, and `metadata.save(store)` writes them back into
    the array.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr",)
    HINTS = ("zarr",)

    @property
    def attributes(self) -> tx.Dict[str, tx.Any]:
        """The attributes of the raw record, as JSON."""
        return {} if self.raw is None else self.raw.attrs

    # --- reading and writing the attributes of a store --------------

    @classmethod
    def sniff_node(cls, node: tx.Any) -> float:
        """
        Score how confident the class is that an opened node is a plain
        Zarr array.

        Parameters
        ----------
        node : ZarrArray or ZarrGroup
            The opened node.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        return Confidence.LIKELY if isinstance(node, ZarrArray) else 0.0

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs: tx.Any) -> tx.Self:
        """
        Read the attributes of an opened array, without its data.

        Parameters
        ----------
        node : ZarrArray
            The opened array.
        **kwargs
            Ignored.

        Returns
        -------
        ZarrMetadata
            The metadata of the array, with its `ZarrRaw` as `raw`.
        """
        return cls.from_raw(ZarrRaw(node_attributes(node), node))

    def to_file(self, file: tx.Any, **kwargs: tx.Any) -> None:
        """
        Write the metadata into the attributes of an existing array.

        The fields are encoded with `to_raw`, over a copy of the
        attributes this metadata was read from, and the attributes that
        changed are written into the array.

        Parameters
        ----------
        file : str or path-like
            The location of the array.
        **kwargs
            Options of `to_raw` (`image=`, `on_loss=`).

        Raises
        ------
        ParserExistsError
            If there is no store at the location.
        MetadataLossError
            If something is lost under the `"raise"` policy.
        """
        raw = self.to_raw(**kwargs)
        node = _open(file, "r+")
        write_attributes(node, raw.attrs, self.attributes)

    # --- hooks --------------------------------------------------------

    @classmethod
    def _decode_raw(
        cls, raw: tx.Optional[ZarrRaw], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        dtype = _array_dtype(image) if image is not None else _node_dtype(raw)
        out: tx.Dict[str, tx.Any] = {"data_type": dtype}
        attrs = None if raw is None else raw.attrs
        if not attrs:
            return out
        block = attrs.get(BRAINHOPS_KEY)
        if isinstance(block, tx.Mapping):
            # `data_type` is the array's; other keys are not ours.
            names = cls.supported_fields - {"extra", "data_type"}
            values, _ = decode_object(block, names)
            out.update(values)
        extra = {k: v for k, v in attrs.items() if k != BRAINHOPS_KEY}
        if extra:
            out["extra"] = extra
        return out

    def _encode_raw(
        self,
        raw: ZarrRaw,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> ZarrRaw:
        attrs = raw.attrs
        block = dict(attrs.get(BRAINHOPS_KEY) or {})
        # `data_type` is the array's: nothing to store.
        fields = {
            name: value
            for name, value in changed.items()
            if name not in ("extra", "data_type")
        }
        encode_changes(block, fields, report=report)
        if "extra" in changed:
            encode_extra(
                attrs,
                changed["extra"],
                reserved=(BRAINHOPS_KEY,),
                report=report,
            )
        if block:
            attrs[BRAINHOPS_KEY] = block
        else:
            attrs.pop(BRAINHOPS_KEY, None)
        return raw


# ----------------------------------------------------------------------
#   OME-ZARR
# ----------------------------------------------------------------------


class OmeZarrRaw:
    """
    The raw record of an OME-Zarr pyramid: its typed `abczarr` multiscale
    (normalised to 0.6), its `omero` block as JSON, and the group
    attributes that are not OME metadata.
    """

    __slots__ = ("multiscale", "omero", "attrs", "node")

    def __init__(
        self,
        multiscale: tx.Any = None,
        omero: tx.Optional[tx.Dict[str, tx.Any]] = None,
        attrs: tx.Optional[tx.Dict[str, tx.Any]] = None,
        node: tx.Any = None,
    ) -> None:
        """
        Parameters
        ----------
        multiscale : object, optional
            The typed `abczarr` multiscale, normalised to OME-NGFF 0.6.
        omero : dict, optional
            The `omero` block, as JSON.
        attrs : dict, optional
            The group attributes that are not OME metadata.
        node : object, optional
            The group they were read from.
        """
        self.multiscale = multiscale
        self.omero = omero
        self.attrs = dict(attrs or {})
        self.node = node

    def __deepcopy__(self, memo: tx.Dict) -> tx.Self:
        # The typed multiscale is immutable: it is shared. The node is a
        # handle, not state: it is dropped.
        return type(self)(
            self.multiscale,
            copy.deepcopy(self.omero, memo),
            copy.deepcopy(self.attrs, memo),
        )

    def __getstate__(self) -> tx.Tuple[tx.Any, ...]:
        return (self.multiscale, self.omero, self.attrs)

    def __setstate__(self, state: tx.Tuple[tx.Any, ...]) -> None:
        self.multiscale, self.omero, self.attrs = state
        self.node = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OmeZarrRaw):
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
            f"OmeZarrRaw(multiscale={name!r}, "
            f"omero={self.omero is not None}, attrs={sorted(self.attrs)})"
        )

    @classmethod
    def from_attributes(
        cls,
        multiscale: tx.Any,
        attrs: tx.Mapping[str, tx.Any],
        node: tx.Any = None,
    ) -> tx.Self:
        """
        Build the raw record of an OME-Zarr group.

        Parameters
        ----------
        multiscale : object
            The typed `abczarr` multiscale of the group, normalised to
            OME-NGFF 0.6.
        attrs : mapping
            The attributes of the group, as JSON.
        node : object, optional
            The group the attributes were read from.

        Returns
        -------
        OmeZarrRaw
            The raw record.
        """
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
            node,
        )


@register_format
class OmeZarrMetadata(
    _ZarrMetadataParser,
    FileBasedMetadata[OmeZarrRaw],
    on={"format": "ome-zarr"},
    supports=("name", "channels", "display_range", "extra", "data_type"),
):
    """
    The metadata of an OME-Zarr multiscale pyramid; its raw record
    (`raw`) is the `OmeZarrRaw` of the pyramid that was read: its typed
    multiscale (normalised to OME-NGFF 0.6), its `omero` block and its
    other group attributes. The levels and the coordinate
    transformations are rewritten from the data model on save.
    `data_type` is the data type of the arrays.

    `multiscale` and `omero` are the parts of the raw record under their
    familiar names. `OmeZarrMetadata.load(store)` reads the metadata of a
    pyramid without its arrays.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr", ".ome.zarr")
    HINTS = ("ome-zarr", "ome")

    @property
    def multiscale(self) -> tx.Any:
        """The typed `abczarr` multiscale of the raw record."""
        return None if self.raw is None else self.raw.multiscale

    @property
    def omero(self) -> tx.Optional[tx.Dict[str, tx.Any]]:
        """The `omero` block of the raw record, as JSON."""
        return None if self.raw is None else self.raw.omero

    # --- reading the attributes of a store ---------------------------

    @classmethod
    def sniff_node(cls, node: tx.Any) -> float:
        """
        Score how confident the class is that an opened node is an
        OME-Zarr pyramid.

        Parameters
        ----------
        node : ZarrArray or ZarrGroup
            The opened node.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        if isinstance(node, ZarrGroup) and looks_like_multiscale(node):
            return Confidence.CERTAIN
        return 0.0

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs: tx.Any) -> tx.Self:
        """
        Read the metadata of an opened pyramid, without its arrays.

        Parameters
        ----------
        node : ZarrGroup
            The opened group.
        **kwargs
            Ignored.

        Returns
        -------
        OmeZarrMetadata
            The metadata of the pyramid, with its `OmeZarrRaw` as `raw`.
        """
        multiscale, _ = read_multiscale(node)
        return cls.from_raw(
            OmeZarrRaw.from_attributes(multiscale, node_attributes(node), node)
        )

    # --- hooks --------------------------------------------------------

    @classmethod
    def _decode_raw(
        cls, raw: tx.Optional[OmeZarrRaw], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        dtype = _array_dtype(image) if image is not None else _node_dtype(raw)
        out: tx.Dict[str, tx.Any] = {"data_type": dtype}
        if raw is None:
            return out
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

    def _encode_raw(
        self,
        raw: OmeZarrRaw,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> OmeZarrRaw:
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
            encode_extra(
                raw.attrs, changed["extra"], reserved=OME_KEYS, report=report
            )
        return raw

    def _encode_omero(
        self,
        raw: OmeZarrRaw,
        changed: tx.Dict[str, tx.Any],
        image: tx.Any,
        report: ConversionReport,
    ) -> None:
        """Write the channels (`channels`) and the window of every channel
        (`display_range`) into `omero`, which needs one per channel."""
        omero = dict(raw.omero or {})
        entries = [dict(e) for e in omero.get("channels") or []]
        common = self.display_range or None
        if "channels" in changed:
            channels = changed["channels"]
            if channels is None:
                omero.pop("channels", None)
                raw.omero = omero if omero else None
                return
            entries = [
                _channel_entry(
                    entries[index] if index < len(entries) else {},
                    channel,
                    common,
                    image,
                    report,
                )
                for index, channel in enumerate(channels)
            ]
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


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def node_attributes(node: tx.Any) -> tx.Dict[str, tx.Any]:
    """
    Read the attributes of a Zarr node, as plain JSON.

    Parameters
    ----------
    node : ZarrArray or ZarrGroup
        The node.

    Returns
    -------
    dict
        Its attributes, or an empty dict when they cannot be read.
    """
    try:
        attrs = node.attrs
        return {key: attrs[key] for key in attrs}
    except Exception:
        return {}


def write_attributes(
    node: tx.Any,
    attrs: tx.Mapping[str, tx.Any],
    before: tx.Optional[tx.Mapping[str, tx.Any]] = None,
) -> None:
    """
    Write attributes onto a Zarr node.

    Only the keys whose value changed are written, and the keys of
    `before` (the raw record that was read) that are no longer in `attrs`
    are removed.

    Parameters
    ----------
    node : ZarrArray or ZarrGroup
        The node, open for writing.
    attrs : mapping
        The attributes to write.
    before : mapping, optional
        The attributes the record was read with.
    """
    current = node_attributes(node)
    for key in before or {}:
        if key not in attrs and key in current:
            del node.attrs[key]
    for key, value in attrs.items():
        if current.get(key) != value:
            node.attrs[key] = value


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _open(location: tx.Any, mode: str) -> tx.Any:
    """Open the node of a store, or raise if there is none."""
    node = open_node(location, mode)
    if node is None:
        raise ParserExistsError(f"No Zarr store at {location}")
    return node


def _node_dtype(raw: tx.Any) -> tx.Optional[np.dtype]:
    """The data type of the array a raw record was read from (of the
    first level of a pyramid), without reading its data."""
    node = getattr(raw, "node", None)
    if node is None:
        return None
    multiscale = getattr(raw, "multiscale", None)
    try:
        if multiscale is not None:
            node = node[str(list(multiscale.datasets)[0].path)]
        dtype = getattr(node, "dtype", None)
        return None if dtype is None else np.dtype(dtype)
    except Exception:
        return None


def _array_dtype(image: tx.Any) -> tx.Optional[np.dtype]:
    """The data type of the array of an image (of the first level of a
    pyramid), read from its node when it has one (without reading the
    data), `None` when there is none."""
    if image is None:
        return None
    objects = [image]
    try:
        images = getattr(image, "images", None)
        if images:
            objects.append(images[0])
    except Exception:
        pass
    for obj in objects:
        dtype = getattr(getattr(obj, "node", None), "dtype", None)
        if dtype is not None:
            return np.dtype(dtype)
    for obj in objects:
        if getattr(obj, "node", None) is not None:
            continue
        try:
            dtype = getattr(getattr(obj, "data", None), "dtype", None)
        except Exception:
            dtype = None
        if dtype is not None:
            return np.dtype(dtype)
    return None


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


def _channel_entry(
    base: tx.Mapping[str, tx.Any],
    channel: Channel,
    common: tx.Optional[tx.Tuple[float, float]],
    image: tx.Any,
    report: ConversionReport,
) -> tx.Dict[str, tx.Any]:
    """The omero entry of a channel, over the entry it had (`base`): its
    label, its color, and its window (its own display range, or the
    common one)."""
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
    return entry


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
