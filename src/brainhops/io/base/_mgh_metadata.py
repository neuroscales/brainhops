"""
The metadata of MGH/MGZ files: `MghMetadata`, the metadata of
`MghImage`.

Its record (`raw`) is an `MghRecord`: the `nibabel` header, which
holds the footer of MRI acquisition parameters, and the raw bytes of the
trailing tags. What the vocabulary covers:

| Field | Record | Unit conversion |
|---|---|---|
| `repetition_time` | footer `tr` | ms -> s |
| `echo_time` | footer `te` | ms -> s |
| `inversion_time` | footer `ti` | ms -> s |
| `flip_angle` | footer `flip_angle` | rad -> deg |
| `history` | the `TAG_CMDLINE` tags | one command per tag |

FreeSurfer stores the times in milliseconds and the flip angle in
radians (`mri_info` prints the latter in degrees); a value of zero means
"not recorded", and reads as `None`. The field of view (`fov`) is not in
the vocabulary (it follows from the geometry) and stays in the record.
MGH has no free-form store, so `extra` is unsupported.

**Tags.** After the footer, FreeSurfer writes a sequence of tags: a
big-endian `int32` tag id, a length, and the payload. The length is an
`int64`, except for a few legacy ids (`TAG_OLD_MGH_XFORM` has an `int32`
length, and `TAG_OLD_COLORTABLE`, `TAG_OLD_USEREALRAS` and
`TAG_OLD_SURF_GEOM` none). A command line (`TAG_CMDLINE = 3`) is a
NUL-terminated string. `history` is decoded only when the whole tag
stream parses; otherwise the tags are kept verbatim and `history` is
unknown (and a new value cannot be written: it is reported as lost).
Writing `history` replaces the command-line tags and keeps every other
tag as it was.
"""

__all__ = ["MghMetadata", "MghRecord"]

# stdlib
import copy
import math
import struct

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr
from nibabel.freesurfer import mghformat as _mgh

# internals
from brainhops.datamodel.metadata import ConversionReport, FormatMetadata

# FreeSurfer tag ids (`utils/tags.h`).
TAG_OLD_COLORTABLE = 1
TAG_OLD_USEREALRAS = 2
TAG_CMDLINE = 3
TAG_OLD_SURF_GEOM = 20
TAG_OLD_MGH_XFORM = 30

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


class MghRecord:
    """
    The record of an MGH file: its `nibabel` header (footer included)
    and the raw bytes of the trailing tags.
    """

    __slots__ = ("header", "tags")

    def __init__(
        self, header: tx.Optional[_mgh.MGHHeader] = None, tags: bytes = b""
    ) -> None:
        self.header = _mgh.MGHHeader() if header is None else header
        self.tags = bytes(tags or b"")

    def __deepcopy__(self, memo: tx.Dict) -> "MghRecord":
        return MghRecord(self.header.copy(), self.tags)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MghRecord):
            return NotImplemented
        return self.tags == other.tags and bytes(
            self.header.binaryblock
        ) == bytes(other.header.binaryblock)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"MghRecord(header=..., tags={len(self.tags)} bytes)"


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


class MghMetadata(
    FormatMetadata,
    on={"format": "mgh"},
    supports=(
        "repetition_time",
        "echo_time",
        "inversion_time",
        "flip_angle",
        "history",
    ),
):
    """
    The metadata of an MGH/MGZ file; its record is an `MghRecord`
    (the `nibabel` header and the trailing tags).

    `header` and `tags` are the parts of the record under their familiar
    names.
    """

    format: tx.Annotated[tx.Literal["mgh"], tx.Doc("Always `'mgh'`.")] = "mgh"

    raw: tx.Annotated[
        tx.Optional[MghRecord],
        tx.Doc(
            """
            The record of the file that was read: its `nibabel` header
            (the footer of MRI parameters included) and its trailing
            tags. Geometry and data type are rewritten from the data
            model on save.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    @property
    def header(self) -> tx.Optional[_mgh.MGHHeader]:
        """The `nibabel` header of the record."""
        return None if self.raw is None else self.raw.header

    @property
    def tags(self) -> bytes:
        """The raw trailing tags of the record."""
        return b"" if self.raw is None else self.raw.tags

    # --- hooks --------------------------------------------------------

    @classmethod
    def _default_raw(cls) -> MghRecord:
        return MghRecord()

    @classmethod
    def _decode(
        cls, raw: tx.Optional[MghRecord], *, image: tx.Any = None
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
        out["history"] = decode_history(raw.tags)
        return out

    def _encode(
        self,
        raw: MghRecord,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> MghRecord:
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
        raw: tx.Optional[MghRecord],
        *,
        grid_changed: bool,
        volumes: tx.Optional[tx.Sequence[int]],
    ) -> tx.Optional[MghRecord]:
        return None if raw is None else copy.deepcopy(raw)
