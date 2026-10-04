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
tag stream after the footer (the command lines, `TAG_CMDLINE`), which is
why this module parses it.

**Tags.** After the footer, FreeSurfer writes a sequence of tags: a
big-endian `int32` tag id, a length, and the payload. The length is an
`int64`, except for a few legacy ids (`TAG_OLD_MGH_XFORM` has an `int32`
length, and `TAG_OLD_COLORTABLE`, `TAG_OLD_USEREALRAS` and
`TAG_OLD_SURF_GEOM` none). A command line (`TAG_CMDLINE = 3`) is a
NUL-terminated string. `history` is decoded only when the whole tag
stream parses; otherwise the tags are kept verbatim and `history` is
unknown (and a new value cannot be written: it is reported as lost).
Writing `history` replaces the command-line tags and keeps every other
tag as it was. The tags sit after the whole volume, so a raw record read
from a file reads them lazily, and `history` is a lazy field
(`lazy=("history",)`, see
[`LazyField`][brainhops._core.properties.LazyField]), decoded on first
access: a load that never touches it never decompresses an MGZ to its
end.
"""

__all__ = ["MghMetadata", "MghRaw"]

# stdlib
import copy
import functools
import math
import struct

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr
from nibabel.freesurfer import mghformat as _mgh

# internals
from brainhops.datamodel.metadata import (
    ConversionReport,
    FileBasedMetadata,
    Lazy,
)

# FreeSurfer tag ids (`utils/tags.h`).
TAG_OLD_COLORTABLE = 1
TAG_OLD_USEREALRAS = 2
TAG_CMDLINE = 3
TAG_OLD_SURF_GEOM = 20
TAG_OLD_MGH_XFORM = 30

# The voxel types MGH stores.
_MGH_DTYPES = tuple(
    np.dtype(t) for t in (np.uint8, np.int16, np.int32, np.float32)
)

# Legacy tags with no length field.
_NO_LENGTH = (TAG_OLD_COLORTABLE, TAG_OLD_USEREALRAS, TAG_OLD_SURF_GEOM)

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

    def __deepcopy__(self, memo: tx.Dict) -> "MghRaw":
        return MghRaw(self.header.copy(), self._tags, loader=self._loader)

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


# ----------------------------------------------------------------------
#   TAGS
# ----------------------------------------------------------------------


def parse_tags(
    tags: bytes,
) -> tx.Optional[tx.List[tx.Tuple[int, bytes]]]:
    """
    Split trailing tags into `(tag id, chunk)` pairs, where `chunk` is the
    tag's bytes verbatim (id, length and payload). `None` when they do not
    parse as a FreeSurfer tag stream.
    """
    out = []
    pos, end = 0, len(tags)
    while pos < end:
        if end - pos < 4:
            return None
        start = pos
        (tag,) = struct.unpack_from(">i", tags, pos)
        pos += 4
        if tag in _NO_LENGTH:
            # A legacy tag with no length: only safe when it is the last.
            if pos != end:
                return None
            out.append((tag, tags[start:]))
            break
        if tag == TAG_OLD_MGH_XFORM:
            if end - pos < 4:
                return None
            (length,) = struct.unpack_from(">i", tags, pos)
            pos += 4
        else:
            if end - pos < 8:
                return None
            (length,) = struct.unpack_from(">q", tags, pos)
            pos += 8
        if tag <= 0 or length < 0 or pos + length > end:
            return None
        pos += length
        out.append((tag, tags[start:pos]))
    return out


def _cmdline_payload(chunk: bytes) -> str:
    payload = chunk[12:]
    return payload.split(b"\0", 1)[0].decode("utf-8", "replace")


def _cmdline_chunk(command: str) -> bytes:
    payload = command.encode("utf-8") + b"\0"
    return struct.pack(">iq", TAG_CMDLINE, len(payload)) + payload


def decode_history(tags: bytes) -> tx.Optional[tx.Tuple[str, ...]]:
    """The command lines of trailing tags, or `None` (none, or the tags
    do not parse)."""
    parsed = parse_tags(tags)
    if not parsed:
        return None
    history = tuple(
        _cmdline_payload(chunk) for tag, chunk in parsed if tag == TAG_CMDLINE
    )
    return history or None


def encode_history(
    tags: bytes, history: tx.Optional[tx.Sequence[str]]
) -> tx.Optional[bytes]:
    """
    Replace the command-line tags of `tags` with `history`, keeping the
    other tags in place (the new commands go where the first one was, or
    at the end). `None` when the tags do not parse.
    """
    parsed = parse_tags(tags)
    if parsed is None:
        return None
    new = b"".join(_cmdline_chunk(c) for c in history or ())
    out, placed = [], False
    for tag, chunk in parsed:
        if tag == TAG_CMDLINE:
            if not placed:
                out.append(new)
                placed = True
            continue
        out.append(chunk)
    if not placed:
        # FreeSurfer writes the command lines before the legacy tags with
        # no length, which must stay last.
        if out and parsed[-1][0] in _NO_LENGTH:
            out.insert(len(out) - 1, new)
        else:
            out.append(new)
    return b"".join(out)


# ----------------------------------------------------------------------
#   METADATA
# ----------------------------------------------------------------------


def _f32(value: tx.Any) -> float:
    """A single-precision value, as the shortest decimal that reads back
    to it."""
    return float(str(np.float32(value)))


def _degrees(radians: tx.Any) -> float:
    """A single-precision angle in radians, in degrees, as the shortest
    decimal that is stored as the same radians (`9.0`, not
    `9.000000419`)."""
    stored = np.float32(radians)
    exact = math.degrees(float(stored))
    for digits in range(10):
        candidate = round(exact, digits)
        if np.float32(math.radians(candidate)) == stored:
            return candidate
    return exact


def _lazy_history(raw: MghRaw) -> tx.Optional[tx.Tuple[str, ...]]:
    return decode_history(raw.tags)


class MghMetadata(
    FileBasedMetadata,
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
    familiar names.
    """

    format: tx.Annotated[tx.Literal["mgh"], tx.Doc("Always `'mgh'`.")] = "mgh"

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
            value = _f32(raw.header[slot])
            if not value:
                continue
            if factor is None:
                out[name] = _degrees(raw.header[slot])
            else:
                out[name] = round(value * factor, 12)
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

    def _derive_raw(
        self,
        raw: tx.Optional[MghRaw],
        *,
        grid_changed: bool,
        volumes: tx.Optional[tx.Sequence[int]],
    ) -> tx.Optional[MghRaw]:
        return None if raw is None else copy.deepcopy(raw)
