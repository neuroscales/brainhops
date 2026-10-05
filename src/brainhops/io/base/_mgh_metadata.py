"""
The metadata of MGH/MGZ files: `MghMetadata`, the metadata of
`MghImage`.

Its raw record (`raw`) is an `MghRaw`: the `nibabel` header, which
holds the footer of MRI acquisition parameters, and the bytes of the
trailing tags. What the vocabulary covers:

| Field | Record | Unit conversion |
|---|---|---|
| `repetition_time` | footer `tr` | ms -> s |
| `echo_time` | footer `te` | ms -> s |
| `inversion_time` | footer `ti` | ms -> s |
| `flip_angle` | footer `flip_angle` | rad -> deg |
| `history` | the `TAG_CMDLINE` tags | one command per tag |
| `data_type` | header `type` | uint8, int16, int32, float32 |

FreeSurfer stores the times in milliseconds and the flip angle in
radians (`mri_info` prints the latter in degrees); a value of zero means
"not recorded", and reads as `None`. The field of view (`fov`) is not in
the vocabulary (it follows from the geometry) and stays in the raw
record. MGH has no free-form store, so `extra` is unsupported. The
writer stores the data as `data_type` when the array's values are of its
kind, or as the nearest type MGH stores (approximated).

**Why not `nibabel`'s footer.** `nibabel` has no separate footer class:
the footer fields (`tr`, `flip_angle`, `te`, `ti`, `fov`) are part of
`MGHHeader` (its `hf_dtype` is the header and the footer), which is the
first half of `MghRaw`. What `nibabel` does not read, nor write, is the
tag stream after the footer (the command lines, `TAG_CMDLINE`), which
`brainhops.io.base._mgh_tags` parses.

**Tags.** The command lines are tags after the footer (see
`brainhops.io.base._mgh_tags`). `history` is decoded only when the whole
tag stream parses; otherwise the tags are kept verbatim and `history` is
unknown (and a new value cannot be written: it is reported as lost).
Writing `history` replaces the command-line tags and keeps every other
tag as it was. The tags sit after the whole volume, so a raw record read
from a file reads them lazily, and `history` is a lazy field
(`lazy=("history",)`, see
[`LazyField`][brainhops._core.fields.LazyField]), decoded on first
access: a load that never touches it never decompresses an MGZ to its
end.
"""

__all__ = ["MghMetadata", "MghRaw"]

# stdlib
import functools
import math
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr
from nibabel.freesurfer import mghformat as _mgh

from brainhops._core.fields import Lazy

# internals
from brainhops._core.numeric import shortest_decimal
from brainhops.datamodel.metadata import ConversionReport, FileBasedMetadata
from brainhops.io.base._base import register_format
from brainhops.io.base._metadata_parser import MetadataParser
from brainhops.io.base._mgh_tags import decode_history, encode_history
from brainhops.io.base.parsers import Confidence, SnifferContentError

# The voxel types MGH stores.
_MGH_DTYPES = tuple(
    np.dtype(t) for t in (np.uint8, np.int16, np.int32, np.float32)
)


# Vocabulary field -> (footer slot, factor from the footer unit to the
# vocabulary unit).
_FOOTER = {
    "repetition_time": ("tr", 1e-3),
    "echo_time": ("te", 1e-3),
    "inversion_time": ("ti", 1e-3),
    "flip_angle": ("flip_angle", None),  # radians -> degrees
}


class MghRaw:
    """
    The raw record of an MGH file: its `nibabel` header (the footer of
    MRI parameters included: `nibabel` keeps it in `MGHHeader`) and the
    bytes of the trailing tags, which `nibabel` does not read.

    The tags follow the whole volume, so reading them decompresses an
    MGZ to its end. A raw record read from a file therefore holds a
    `loader` instead, and reads the tags the first time `tags` is used
    (only `history` needs them).
    """

    __slots__ = ("header", "_tags", "_loader")

    def __init__(
        self,
        header: tx.Optional[_mgh.MGHHeader] = None,
        tags: tx.Optional[bytes] = b"",
        *,
        loader: tx.Optional[tx.Callable[[], bytes]] = None,
    ) -> None:
        """
        Parameters
        ----------
        header : nibabel.freesurfer.mghformat.MGHHeader, optional
            The header, footer included. By default, an empty header.
        tags : bytes or None, optional
            The trailing tags, or `None` to read them with `loader`.
        loader : callable, optional
            A function without arguments that reads the tags, called the
            first time they are used.
        """
        self.header = _mgh.MGHHeader() if header is None else header
        if tags is None and loader is None:
            tags = b""
        self._tags = None if tags is None else bytes(tags)
        self._loader = None if tags is not None else loader

    @property
    def tags(self) -> bytes:
        """The raw trailing tags, read on first use when they are lazy."""
        if self._tags is None:
            loader, self._loader = self._loader, None
            self._tags = bytes(loader() or b"") if loader else b""
        return self._tags

    @tags.setter
    def tags(self, value: tx.Optional[bytes]) -> None:
        self._tags = bytes(value or b"")
        self._loader = None

    @property
    def tags_loaded(self) -> bool:
        """Whether the tags have been read (or were given)."""
        return self._tags is not None

    def __deepcopy__(self, memo: tx.Dict) -> tx.Self:
        return type(self)(self.header.copy(), self._tags, loader=self._loader)

    def __getstate__(self) -> tx.Tuple[tx.Any, ...]:
        return (self.header, self.tags)

    def __setstate__(self, state: tx.Tuple[tx.Any, ...]) -> None:
        self.header, self._tags = state
        self._loader = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MghRaw):
            return NotImplemented
        return self.tags == other.tags and bytes(
            self.header.binaryblock
        ) == bytes(other.header.binaryblock)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        tags = f"{len(self._tags)} bytes" if self._tags is not None else "lazy"
        return f"MghRaw(header=..., tags={tags})"


@register_format
class MghMetadata(
    FileBasedMetadata,
    MetadataParser,
    on={"format": "mgh"},
    supports=(
        "repetition_time",
        "echo_time",
        "inversion_time",
        "flip_angle",
        "history",
        "data_type",
    ),
    lazy=("history",),
):
    """
    The metadata of an MGH/MGZ file; its raw record is an `MghRaw`
    (the `nibabel` header and the trailing tags).

    `header` and `tags` are the parts of the raw record under their
    familiar names. `MghMetadata.load(path)` reads the header and the
    footer of a file without its voxels; the tags, which follow the
    voxels, are read when `history` is first used.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mgh", ".mgz", ".mgh.gz")
    HINTS = ("mgh", "mgz")
    _READ_MODE = "rb"

    raw: tx.Annotated[
        tx.Optional[MghRaw],
        tx.Doc(
            """
            The raw record of the file that was read: its `nibabel`
            header (the footer of MRI parameters included) and its
            trailing tags. Geometry is rewritten from the data model on
            save.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    @property
    def header(self) -> tx.Optional[_mgh.MGHHeader]:
        """The `nibabel` header of the raw record."""
        return None if self.raw is None else self.raw.header

    @property
    def tags(self) -> bytes:
        """The trailing tags of the raw record."""
        return b"" if self.raw is None else self.raw.tags

    # --- reading the record of a file ---------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the class is that an open file holds an MGH
        header, gzipped or not.

        Parameters
        ----------
        file : file object
            A binary stream.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        from brainhops.io.base.mgh import is_mgh_stream

        if is_mgh_stream(file):
            return Confidence.LIKELY
        if error:
            raise (SnifferContentError if error is True else error)(
                "Content is not a valid MGH/MGZ file"
            )
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the class is that bytes hold an MGH header.

        Parameters
        ----------
        content : bytes
            The content of a file, gzipped or not.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        return cls.sniff_fileobj(BytesIO(content), error=error)

    @classmethod
    def _read_raw(cls, file: tx.Any, **kwargs: tx.Any) -> MghRaw:
        from brainhops.io.base.mgh import read_mgh_raw

        return read_mgh_raw(file)

    # --- hooks --------------------------------------------------------

    @classmethod
    def _default_raw(cls) -> MghRaw:
        return MghRaw()

    @classmethod
    def _decode(
        cls, raw: tx.Optional[MghRaw], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        if raw is None:
            return {}
        out: tx.Dict[str, tx.Any] = {}
        for name, (slot, factor) in _FOOTER.items():
            stored = float(raw.header[slot])
            if not stored:
                continue
            # The shortest decimal that is stored as the same number: a
            # flip angle of `9.0`, not `9.000000250447817`.
            if factor is None:
                out[name] = shortest_decimal(
                    math.degrees(stored), math.radians
                )
            else:
                out[name] = shortest_decimal(
                    stored * factor, functools.partial(_divide, factor)
                )
        try:
            out["data_type"] = raw.header.get_data_dtype()
        except Exception:
            pass
        if raw.tags_loaded:
            out["history"] = decode_history(raw.tags)
        else:
            # Decoded on first access: reading the tags reads the file
            # to its end (see `MghRaw`).
            out["history"] = Lazy(functools.partial(_lazy_history, raw))
        return out

    def _encode(
        self,
        raw: MghRaw,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> MghRaw:
        for name, (slot, factor) in _FOOTER.items():
            if name not in changed:
                continue
            value = changed[name]
            if value is None:
                stored = 0.0
            elif factor is None:
                stored = math.radians(value)
            else:
                stored = value / factor
            raw.header[slot] = stored
        if changed.get("data_type") is not None:
            # The writer settles it against the data afterwards.
            dtype = changed["data_type"]
            if dtype in _MGH_DTYPES:
                raw.header.set_data_dtype(dtype)
            else:
                report.approximated["data_type"] = (
                    f"MGH cannot store {dtype.name} (it stores uint8, "
                    f"int16, int32 and float32)"
                )
        if "history" in changed:
            tags = encode_history(raw.tags, changed["history"])
            if tags is None:
                if changed["history"]:
                    report.lost["history"] = tuple(changed["history"])
            else:
                raw.tags = tags
        return raw


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _divide(factor: float, value: float) -> float:
    """A value in the vocabulary unit, in the unit of the footer."""
    return value / factor


def _lazy_history(raw: MghRaw) -> tx.Optional[tx.Tuple[str, ...]]:
    return decode_history(raw.tags)
