"""
The shared MRtrix-reading and MRtrix-writing machinery behind every
MRtrix-based image and transformation format.

An MRtrix image is a text header followed by raw voxel data. The header
and the data are either in the same file (`.mif`, or `.mif.gz` when the
whole file is gzip-compressed) or in two files (`.mih` for the header,
and the data file it names, conventionally `.dat`).

The header is a list of `key: value` lines, between a first line that
reads `mrtrix image` and a last line that reads `END`::

    mrtrix image
    dim: 64,64,32,7
    vox: 2,2,2.5,nan
    layout: -0,-1,+2,+3
    datatype: Float32LE
    transform: 0.996, 0.087, 0, -61.2
    transform: -0.087, 0.996, 0, -70.5
    transform: 0, 0, 1, -40
    scaling: 0,1
    dw_scheme: 0,0,1,0
    dw_scheme: 0,1,0,1000
    file: . 552
    END

The conventions below were checked against the MRtrix3 sources
(`core/formats/mrtrix_utils.{h,cpp}`, `core/formats/mrtrix{,_gz}.cpp`,
`core/file/key_value.cpp`, `core/stride.h`, `core/raw.h`,
`core/header.cpp`, `core/transform.h`, `core/datatype.cpp`).

* **Comments.** Everything after a `#` on a line is a comment, and is
  dropped. A value cannot therefore contain a `#`.
* **Keys.** The compulsory keys (`dim`, `vox`, `layout`, `datatype`,
  `transform`, `scaling`) are matched case-insensitively. Any other key
  is kept as it is spelled. A key that appears on several lines (the
  three rows of `transform`, the rows of a `dw_scheme`, the entries of a
  `command_history`) has one value per line; the other keys are
  collected as their lines joined by newlines, which is how MRtrix holds
  them.
* **`dim`, `vox`.** The size and the voxel size of each axis. `vox` may
  hold fewer entries than `dim` (but at least three, or as many as
  `dim` when it has fewer), and may hold `nan` for an axis that has no
  physical size.
* **`layout`.** One signed integer per axis, such as `-0,-1,+2`. The
  absolute value is the *rank* of the axis in the file: the axis of
  rank 0 changes fastest, the one of rank 1 next, and so on. The sign
  says whether the voxels along the axis are stored in increasing
  (`+`) or decreasing (`-`) order of their index. Voxel `[0, 0, ...]`
  is therefore *not* the first value in the file when any axis is
  negative: it is `sum((size[i] - 1) * |stride[i]|)` values in, over
  the negative axes. `layout: +0,+1,+2` is a plain Fortran-ordered
  (x fastest) array.
* **`datatype`.** One of `Bit`, `Int8`, `UInt8`, `[U]Int{16,32,64}`,
  `Float{32,64}` and `CFloat{32,64}`, the multi-byte ones optionally
  suffixed with `LE` or `BE`. Without a suffix the byte order is the
  native one of the machine. `Bit` packs eight voxels per byte, the
  first one in the most significant bit.
* **`transform`.** Three rows of four numbers, the top of a `4x4`
  matrix that maps *voxel coordinates multiplied by the voxel sizes*
  (not plain voxel indices) to scanner coordinates, in millimetres,
  in RAS+ (x to the right, y to the front, z up), like a NIfTI sform.
  The voxel-to-scanner matrix is `transform @ diag(vox[:3], 1)`. MRtrix
  writes the three direction columns at unit length; one that is not is
  normalised and its length moved into the voxel size, which leaves the
  voxel-to-scanner matrix unchanged.
* **Missing transform.** When `transform` is absent (or not finite),
  MRtrix centres the field of view on the origin: the rotation is the
  identity, and the translation is `-0.5 * (size - 1) * vox` along
  each of the first three axes. It is *not* the identity.
* **`scaling`.** `offset,scale`: a stored value `v` means
  `offset + scale * v`.
* **`file`.** `file: <name> [<offset>]`. A name of `.` means the data
  follow the header in the same file, `offset` bytes from its start.
  Any other name is the data file, relative to the header's directory,
  and its offset defaults to zero.

This module holds what an image reader and a transformation reader
(an MRtrix warp, for instance) share: the header, the decoding of the
voxel data into an array whose axes are the header's axes, and the
encoding of such an array back into bytes.
"""

__all__ = [
    "MrtrixHeader",
    "MrtrixParser",
    "MRTRIX_MAGIC",
]

# stdlib
import gzip
import math
import os
import re
from collections import OrderedDict
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._geometry import (
    embed_affine,
    ras_conversion,
    reduce_to_affine,
    split_spatial,
)
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base._utils_files import sibling as _sibling
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    WriterError,
    preserve_position,
)

MRTRIX_MAGIC = "mrtrix image"
"""The line every MRtrix header starts with."""

_MAGIC_BYTES = MRTRIX_MAGIC.encode("ascii")

_END = "END"
"""The line every MRtrix header ends with."""

_RESERVED = ("dim", "vox", "layout", "datatype", "transform", "scaling")
"""The keys MRtrix decodes itself, matched case-insensitively."""

_FILE = "file"
"""The key that says where the voxel data are."""

_MAX_HEADER_LINES = 1_000_000
"""A bound on the header length, so a stray binary file is not read whole
while looking for an `END` that never comes."""


# ----------------------------------------------------------------------
#   DATA TYPES
# ----------------------------------------------------------------------

_DTYPES = {
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "int64": "i8",
    "uint64": "u8",
    "float32": "f4",
    "float64": "f8",
    "cfloat32": "c8",
    "cfloat64": "c16",
}
"""The numpy type of each MRtrix data type, without its byte order."""

_NAMES = {
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
    "cfloat32": "CFloat32",
    "cfloat64": "CFloat64",
}
"""The spelling MRtrix writes each data type with."""

_BIT = "bit"


def mrtrix_dtype(spec: str) -> tx.Optional[np.dtype]:
    """
    The numpy data type of an MRtrix data type, or `None` for `Bit`.

    The byte order is the suffix's (`LE`, `BE`), or the machine's when
    there is none, as MRtrix does. The specifier is matched
    case-insensitively.

    Raises
    ------
    ParserContentError
        If the specifier is not an MRtrix data type.
    """
    key = spec.strip().lower()
    if key == _BIT:
        return None
    order = "="
    if key.endswith(("le", "be")) and key[:-2] in _DTYPES:
        order = "<" if key.endswith("le") else ">"
        key = key[:-2]
    if key not in _DTYPES:
        raise ParserContentError(f"Unknown MRtrix data type: {spec!r}")
    if key in ("int8", "uint8"):
        order = "|"
    return np.dtype(order + _DTYPES[key])


def dtype_to_mrtrix(dtype: tx.Any) -> str:
    """
    The MRtrix data type that stores a numpy data type exactly.

    A multi-byte type is always given its byte order (`LE` or `BE`), as
    MRtrix itself writes it. A boolean is `Bit`. A type MRtrix cannot
    store (`float16`, `float128`, strings, ...) raises `WriterError`.
    """
    if isinstance(dtype, str) and dtype.strip().lower() == _BIT:
        return "Bit"
    if isinstance(dtype, str) and _is_mrtrix_spec(dtype):
        # Already an MRtrix specifier: normalise its spelling.
        np_dtype = mrtrix_dtype(dtype)
        return dtype_to_mrtrix(np_dtype)
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return "Bit"
    for key, code in _DTYPES.items():
        if np.dtype(code) == dtype.newbyteorder("="):
            name = _NAMES[key]
            if dtype.itemsize == 1:
                return name
            order = dtype.byteorder
            if order in ("=", "|"):
                order = "<" if np.little_endian else ">"
            return name + ("LE" if order == "<" else "BE")
    raise WriterError(f"MRtrix cannot store an array of type {dtype}.")


def _is_mrtrix_spec(spec: str) -> bool:
    try:
        mrtrix_dtype(spec)
    except ParserContentError:
        return False
    return True


# ----------------------------------------------------------------------
#   LAYOUT
# ----------------------------------------------------------------------


def parse_layout(spec: str, ndim: int) -> tx.Tuple[int, ...]:
    """
    Parse an MRtrix `layout` into signed, one-based symbolic strides.

    `-0,-1,+2` becomes `(-1, -2, 3)`: the absolute value minus one is
    the rank of the axis in the file (0 changes fastest), and the sign
    is the direction the axis is stored in. The one-based spelling is
    the one MRtrix uses internally, and keeps the sign of rank 0.

    Raises
    ------
    ParserContentError
        If the layout is malformed, has the wrong number of axes, or
        does not give each axis a distinct rank.
    """
    entries = [entry.strip() for entry in spec.split(",")]
    strides = []
    for entry in entries:
        match = re.fullmatch(r"([+-]?)(\d+)", entry)
        if not match:
            raise ParserContentError(f"Malformed MRtrix layout: {spec!r}")
        stride = int(match.group(2)) + 1
        strides.append(-stride if match.group(1) == "-" else stride)
    if len(strides) != ndim:
        raise ParserContentError(
            f"The MRtrix layout {spec!r} has {len(strides)} axes, but the "
            f"image has {ndim}."
        )
    if sorted(abs(s) for s in strides) != list(range(1, ndim + 1)):
        raise ParserContentError(
            f"The MRtrix layout {spec!r} does not give each of the {ndim} "
            f"axes a distinct rank between 0 and {ndim - 1}."
        )
    return tuple(strides)


def format_layout(strides: tx.Sequence[int]) -> str:
    """Spell signed one-based symbolic strides as an MRtrix `layout`."""
    return ",".join(
        ("+" if s > 0 else "-") + str(abs(int(s)) - 1) for s in strides
    )


def default_layout(ndim: int) -> tx.Tuple[int, ...]:
    """The layout of a Fortran-ordered array: `+0,+1,+2,...`."""
    return tuple(range(1, ndim + 1))


def _storage_shape(
    dim: tx.Sequence[int], strides: tx.Sequence[int]
) -> tx.Tuple[tx.Tuple[int, ...], tx.List[int]]:
    """
    The C-ordered shape the values have in the file, and where each
    header axis sits in it.

    The slowest axis (highest rank) comes first in the C-ordered shape.
    `position[i]` is the index, in that shape, of header axis `i`.
    """
    ndim = len(dim)
    ranks = [abs(s) - 1 for s in strides]
    position = [ndim - 1 - rank for rank in ranks]
    shape = [0] * ndim
    for axis, pos in enumerate(position):
        shape[pos] = int(dim[axis])
    return tuple(shape), position


def storage_to_image(
    stored: np.ndarray, dim: tx.Sequence[int], strides: tx.Sequence[int]
) -> np.ndarray:
    """
    View the values as they are stored as an array of the header's axes.

    `stored` holds the values in file order, either flat or already
    shaped. The result is indexed `[i0, i1, ...]` in the order of the
    header's axes, and is a view of `stored` (a transposition, and a
    reversal of the negative axes), so a memory-mapped file stays
    memory-mapped.
    """
    shape, position = _storage_shape(dim, strides)
    stored = stored.reshape(shape)
    image = stored.transpose(position)
    flips = tuple(
        slice(None, None, -1) if s < 0 else slice(None) for s in strides
    )
    return image[flips]


def image_to_storage(image: tx.Any, strides: tx.Sequence[int]) -> np.ndarray:
    """
    Reorder an array of the header's axes into the order of the file.

    The inverse of `storage_to_image`. The result is a C-ordered view
    when possible; `np.ascontiguousarray` of it is what the file holds.
    """
    image = np.asarray(image)
    ndim = image.ndim
    flips = tuple(
        slice(None, None, -1) if s < 0 else slice(None) for s in strides
    )
    image = image[flips]
    _, position = _storage_shape(image.shape, strides)
    order = [0] * ndim
    for axis, pos in enumerate(position):
        order[pos] = axis
    return image.transpose(order)


# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


def _format_float(value: float) -> str:
    """A float spelled the shortest way that reads back exactly."""
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _parse_floats(value: str, key: str) -> tx.List[float]:
    try:
        return [float(v) for v in value.split(",")]
    except ValueError as error:
        raise ParserContentError(
            f"Malformed MRtrix {key!r} entry: {value!r}"
        ) from error


class MrtrixHeader:
    """
    The content of an MRtrix image header.

    The keys MRtrix decodes are held decoded: `dim`, `vox`, `layout`,
    `datatype`, `transform`, `scaling` and `file`. Every other key is
    kept in `keyval`, in the order it was read, with the lines of a
    repeated key joined by newlines, so the header writes back as it was
    read.

    The header is plain data: building or changing one never touches a
    file.
    """

    def __init__(
        self,
        dim: tx.Sequence[int] = (),
        vox: tx.Optional[tx.Sequence[float]] = None,
        layout: tx.Optional[tx.Sequence[int]] = None,
        datatype: str = "Float32LE",
        transform: tx.Optional[tx.Any] = None,
        scaling: tx.Optional[tx.Sequence[float]] = None,
        file: tx.Optional[tx.Tuple[str, int]] = None,
        keyval: tx.Optional[tx.Mapping[str, str]] = None,
    ) -> None:
        self.dim = tuple(int(d) for d in dim)
        """The size of each axis."""
        ndim = len(self.dim)
        if vox is None:
            vox = [1.0] * ndim
        self.vox = tuple(float(v) for v in vox)
        """The voxel size of each axis, as stored (possibly fewer entries
        than axes, possibly `nan`)."""
        if layout is None:
            layout = default_layout(ndim)
        self.layout = tuple(int(s) for s in layout)
        """The signed one-based symbolic strides (see `parse_layout`)."""
        self.datatype = str(datatype)
        """The MRtrix data type, as spelled in the header."""
        if transform is not None:
            transform = np.asarray(transform, dtype=float).reshape(3, 4)
        self.transform = transform
        """The `(3, 4)` stored transform, or `None` when there is none."""
        if scaling is not None:
            scaling = tuple(float(s) for s in scaling)
        self.scaling = scaling
        """`(offset, scale)`, or `None` when there is none."""
        self.file = file
        """`(name, offset)` of the data, or `None` when not known."""
        self.keyval = OrderedDict(keyval or {})
        """Every other key, with its lines joined by newlines."""

    # --- derived ------------------------------------------------------

    @property
    def ndim(self) -> int:
        """The number of axes."""
        return len(self.dim)

    @property
    def dtype(self) -> tx.Optional[np.dtype]:
        """The numpy data type of the stored values, `None` for `Bit`."""
        return mrtrix_dtype(self.datatype)

    @property
    def is_bit(self) -> bool:
        """Whether the values are packed bits."""
        return self.datatype.strip().lower() == _BIT

    @property
    def count(self) -> int:
        """The number of voxels."""
        return int(np.prod(self.dim, dtype=np.int64)) if self.dim else 0

    @property
    def nbytes(self) -> int:
        """The number of bytes the voxel data take in the file."""
        if self.is_bit:
            return (self.count + 7) // 8
        return self.count * self.dtype.itemsize

    @property
    def spacing(self) -> tx.Tuple[float, ...]:
        """
        The voxel size of each axis, as MRtrix uses it.

        Missing entries are `nan`. A spatial voxel size that is not
        finite is replaced by the mean of the finite spatial ones (or 1
        when there is none), which is what MRtrix does.
        """
        vox = list(self.vox[: self.ndim])
        vox += [math.nan] * (self.ndim - len(vox))
        spatial = vox[:3]
        finite = [v for v in spatial if math.isfinite(v)]
        if len(finite) < len(spatial):
            mean = sum(finite) / len(finite) if finite else 1.0
            for i, v in enumerate(spatial):
                if not math.isfinite(v):
                    vox[i] = mean
        return tuple(vox)

    def default_transform(self) -> np.ndarray:
        """
        The `(3, 4)` transform MRtrix assumes when the header has none.

        The rotation is the identity, and the field of view is centred
        on the origin: the translation is `-0.5 * (size - 1) * vox` along
        each of the first three axes (an axis the image does not have
        counts as one voxel).
        """
        dim = list(self.dim[:3]) + [1] * (3 - min(3, self.ndim))
        vox = list(self.spacing[:3]) + [1.0] * (3 - min(3, self.ndim))
        matrix = np.eye(3, 4)
        for i in range(3):
            matrix[i, 3] = -0.5 * (dim[i] - 1) * vox[i]
        return matrix

    def voxel_to_scanner(self) -> np.ndarray:
        """
        The `(4, 4)` matrix from voxel indices to scanner RAS+ mm.

        The stored transform maps coordinates scaled by the voxel sizes,
        so the matrix is `transform @ diag(vox[0], vox[1], vox[2], 1)`.
        An image with fewer than three axes is given unit-size extra
        axes, as MRtrix does. A missing or non-finite transform is
        replaced by `default_transform`.
        """
        transform = self.transform
        if transform is None or not np.all(np.isfinite(transform)):
            transform = self.default_transform()
        vox = list(self.spacing[:3]) + [1.0] * (3 - min(3, self.ndim))
        matrix = np.eye(4)
        matrix[:3, :4] = transform
        return matrix @ np.diag(vox + [1.0])

    # --- reading ------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> "MrtrixHeader":
        """
        Parse header lines, from `mrtrix image` up to `END`.

        Lines after `END` are ignored, so the decoded text of a whole
        file can be handed over.

        Raises
        ------
        ParserContentError
            If the first line is not `mrtrix image`, or a compulsory key
            (`dim`, `vox`, `datatype`) is missing, or a key is malformed.
        """
        lines = iter(lines)
        first = next(lines, None)
        if first is None or not first.startswith(MRTRIX_MAGIC):
            raise ParserContentError(
                f"Not an MRtrix header: the first line is {first!r}, not "
                f"{MRTRIX_MAGIC!r}."
            )
        dim = vox = layout = datatype = scaling = None
        transform: tx.List[tx.List[float]] = []
        keyval: tx.Dict[str, tx.List[str]] = OrderedDict()
        ended = False
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if line == _END:
                ended = True
                break
            if not line:
                continue
            key, colon, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if not colon or not key:
                # MRtrix ignores a malformed line.
                continue
            lkey = key.lower()
            if lkey == "dim":
                try:
                    dim = [int(v) for v in value.split(",")]
                except ValueError as error:
                    raise ParserContentError(
                        f"Malformed MRtrix 'dim' entry: {value!r}"
                    ) from error
            elif lkey == "vox":
                vox = _parse_floats(value, "vox")
            elif lkey == "layout":
                layout = value
            elif lkey == "datatype":
                datatype = value
            elif lkey == "scaling":
                scaling = _parse_floats(value, "scaling")
            elif lkey == "transform":
                transform.append(_parse_floats(value, "transform"))
            elif value:
                keyval.setdefault(key, []).append(value)
        if not ended:
            raise ParserContentError("The MRtrix header has no END line.")

        if not dim:
            raise ParserContentError("The MRtrix header has no 'dim'.")
        if any(d < 1 for d in dim):
            raise ParserContentError(f"Invalid MRtrix dimensions: {dim}")
        if not vox:
            raise ParserContentError("The MRtrix header has no 'vox'.")
        if len(vox) < min(3, len(dim)):
            raise ParserContentError(
                f"Too few entries in the MRtrix 'vox': {vox}"
            )
        if any(v < 0 for v in vox):
            raise ParserContentError(f"Invalid MRtrix voxel sizes: {vox}")
        if not datatype:
            raise ParserContentError("The MRtrix header has no 'datatype'.")
        mrtrix_dtype(datatype)  # validate
        # MRtrix requires a layout. A header without one is read as a
        # Fortran-ordered array, the layout MRtrix gives a new image.
        strides = (
            parse_layout(layout, len(dim))
            if layout
            else default_layout(len(dim))
        )
        if transform:
            if len(transform) < 3 or any(len(r) != 4 for r in transform):
                raise ParserContentError(
                    "Invalid MRtrix 'transform': expected three rows of "
                    "four numbers."
                )
            transform = np.asarray(transform[:3], dtype=float)
        else:
            transform = None
        if scaling is not None and len(scaling) != 2:
            raise ParserContentError(
                f"Invalid MRtrix 'scaling': {scaling} (expected offset,scale)"
            )

        file = None
        if _FILE in keyval:
            entry = keyval.pop(_FILE)[0].split()
            name = entry[0]
            try:
                offset = int(entry[1]) if len(entry) > 1 else 0
            except ValueError as error:
                raise ParserContentError(
                    f"Invalid offset in MRtrix 'file' entry: {entry}"
                ) from error
            file = (name, offset)

        return cls(
            dim=dim,
            vox=vox,
            layout=strides,
            datatype=datatype,
            transform=transform,
            scaling=scaling,
            file=file,
            keyval=OrderedDict(
                (key, "\n".join(values)) for key, values in keyval.items()
            ),
        )

    @classmethod
    def from_text(cls, text: str) -> "MrtrixHeader":
        """Parse a header from its text."""
        return cls.from_lines(text.splitlines())

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO) -> tx.Tuple["MrtrixHeader", int]:
        """
        Read a header from a binary stream, up to its `END` line.

        The stream is left just after the `END` line. Returns the header
        and the number of bytes it took.

        Raises
        ------
        ParserContentError
            If the stream does not hold an MRtrix header.
        """
        first = file.readline()
        if not first.startswith(_MAGIC_BYTES):
            raise ParserContentError(
                f"Not an MRtrix image: the first line is not {MRTRIX_MAGIC!r}."
            )
        raw = [first]
        for _ in range(_MAX_HEADER_LINES):
            line = file.readline()
            if not line:
                break
            raw.append(line)
            if line.split(b"#", 1)[0].strip() == _END.encode():
                break
        nbytes = sum(len(line) for line in raw)
        text = b"".join(raw).decode("utf-8", "replace")
        return cls.from_text(text), nbytes

    # --- writing ------------------------------------------------------

    def to_lines(
        self, file: tx.Optional[tx.Tuple[str, tx.Optional[int]]] = None
    ) -> tx.List[str]:
        """
        The header's lines, from `mrtrix image` up to `END`.

        `file` is the `(name, offset)` of the data, written in the
        `file:` line; an offset of `None` is left out. It defaults to the
        header's own `file`, and no `file:` line is written when there
        is none.
        """
        lines = [MRTRIX_MAGIC]
        lines.append("dim: " + ",".join(str(d) for d in self.dim))
        lines.append("vox: " + ",".join(_format_float(v) for v in self.vox))
        lines.append("layout: " + format_layout(self.layout))
        lines.append("datatype: " + self.datatype)
        if self.transform is not None:
            for row in np.asarray(self.transform).reshape(3, 4):
                lines.append(
                    "transform: " + ", ".join(_format_float(v) for v in row)
                )
        if self.scaling is not None:
            lines.append(
                "scaling: " + ",".join(_format_float(v) for v in self.scaling)
            )
        for key, value in self.keyval.items():
            if key.lower() in _RESERVED or key == _FILE:
                continue
            for line in str(value).split("\n"):
                lines.append(f"{key}: {line}")
        file = self.file if file is None else file
        if file is not None:
            name, offset = file
            entry = name if offset is None else f"{name} {int(offset)}"
            lines.append(f"{_FILE}: {entry}")
        lines.append(_END)
        return lines

    def to_text(self, *args, **kwargs) -> str:
        """The header's text, newline-terminated."""
        return "\n".join(self.to_lines(*args, **kwargs)) + "\n"

    def embedded(self, align: int = 4) -> bytes:
        """
        The header of a single-file image, padded up to its data.

        The `file: . <offset>` line points just past the header, at an
        offset aligned on `align` bytes (MRtrix aligns on four). The
        offset is part of the header it measures, so it is found by
        iteration.
        """
        offset = 0
        while True:
            text = self.to_text(file=(".", offset)).encode("utf-8")
            needed = -(-len(text) // align) * align
            if needed == offset:
                return text + b"\0" * (offset - len(text))
            offset = needed

    def copy(self) -> "MrtrixHeader":
        """A copy that can be changed without changing this one."""
        return MrtrixHeader(
            dim=self.dim,
            vox=self.vox,
            layout=self.layout,
            datatype=self.datatype,
            transform=None
            if self.transform is None
            else self.transform.copy(),
            scaling=self.scaling,
            file=self.file,
            keyval=self.keyval,
        )

    def __repr__(self) -> str:
        return f"MrtrixHeader({self.to_text()!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MrtrixHeader):
            return NotImplemented
        return self.to_text() == other.to_text()

    __hash__ = None


# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def decode_data(header: MrtrixHeader, buffer: tx.Any) -> np.ndarray:
    """
    Decode the stored values into an array of the header's axes.

    `buffer` holds the voxel data, from its first byte: `bytes`, a
    `memoryview`, or a one-dimensional `uint8` array (a memory map). The
    result is a view of it whenever the data type allows one; packed bits
    are unpacked into a new boolean array. Intensity scaling is not
    applied.
    """
    count = header.count
    if header.is_bit:
        raw = np.frombuffer(buffer, dtype=np.uint8, count=header.nbytes)
        stored = np.unpackbits(raw, bitorder="big", count=count)
        stored = stored.astype(bool)
    else:
        if isinstance(buffer, np.ndarray):
            stored = buffer[: header.nbytes].view(header.dtype)
        else:
            stored = np.frombuffer(buffer, dtype=header.dtype, count=count)
    if stored.size < count:
        raise ParserContentError(
            f"The MRtrix data hold {stored.size} values, but the header "
            f"asks for {count}."
        )
    return storage_to_image(stored[:count], header.dim, header.layout)


def encode_data(header: MrtrixHeader, data: tx.Any) -> bytes:
    """
    Encode an array of the header's axes into the bytes of the file.

    The values are converted to the header's data type, rounding when a
    floating-point array is stored as integers, and laid out as the
    header's `layout` says.
    """
    array = np.asarray(data)
    if tuple(array.shape) != tuple(header.dim):
        raise WriterError(
            f"The data have shape {array.shape}, but the header says "
            f"{header.dim}."
        )
    stored = image_to_storage(array, header.layout)
    if header.is_bit:
        flat = np.ascontiguousarray(stored).reshape(-1).astype(bool)
        return np.packbits(flat, bitorder="big").tobytes()
    dtype = header.dtype
    if dtype.kind in "iu" and stored.dtype.kind in "fc":
        stored = np.round(np.real(stored))
    return np.ascontiguousarray(stored, dtype=dtype).tobytes()


def _read_buffer(file: tx.Any, offset: int, nbytes: int, mmap: bool) -> tx.Any:
    """
    The `nbytes` bytes at `offset` in an uncompressed data file.

    A local file is memory-mapped (read-only) unless `mmap` is false, so
    nothing is read until it is indexed. Any other path is read through
    its `open`.
    """
    local = _local_path(file)
    if local is not None and mmap and nbytes > 0:
        return np.memmap(
            local, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,)
        )
    with _open_path(file) as f:
        f.seek(offset)
        return f.read(nbytes)


def _is_gzip(file: tx.Any) -> bool:
    """Whether a path names a gzip-compressed file, from its content."""
    try:
        with _open_path(file) as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


# ----------------------------------------------------------------------
#   GEOMETRY (writing)
# ----------------------------------------------------------------------


def voxel_to_ras(xform: Transformation) -> np.ndarray:
    """
    The `(4, 4)` voxel-to-RAS matrix of a voxel-to-world transformation.

    The transformation is reduced to an affine (a `Scaling`, a `Sequence`
    of affines, ...). A map with fewer than three dimensions is embedded
    in three, with unit extra axes. A map over more axes keeps its three
    spatial ones, when it does not mix them with the others. The world
    space is turned into RAS from the anatomical orientation of its axes;
    a world space with no orientation is taken to be RAS already.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation, or mixes the
        spatial axes with the others.
    """
    affine = reduce_to_affine(xform, "MRtrix", "scanner")
    matrix = affine.homogeneous_matrix
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
    world_ndim = matrix.shape[0] - 1
    # The axes that follow the spatial ones (time, ...) get their voxel
    # sizes elsewhere (see `_extra_vox`), so only the spatial block is kept.
    spatial, _ = split_spatial(matrix, "MRtrix", "scanner")
    embedded = embed_affine(spatial, "MRtrix", "scanner")
    output = getattr(affine, "output", None)
    try:
        if output is not None and output.ndim is None:
            output = output.expand(world_ndim)
    except Exception:
        output = None
    return ras_conversion(output) @ embedded


def split_voxel_to_scanner(
    matrix: np.ndarray,
) -> tx.Tuple[np.ndarray, tx.Tuple[float, float, float]]:
    """
    Split a voxel-to-scanner matrix into an MRtrix transform and voxel
    sizes.

    The voxel sizes are the lengths of the three direction columns, and
    the transform holds the unit-length directions and the translation:
    `matrix == [transform; 0 0 0 1] @ diag(vox, 1)`.

    Raises
    ------
    WriterError
        If a direction column has zero length, so no voxel size can be
        derived.
    """
    matrix = np.asarray(matrix, dtype=float)
    vox = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.all(vox > 0) or not np.all(np.isfinite(matrix)):
        raise WriterError(
            "The voxel-to-scanner matrix is degenerate or not finite, so "
            "it cannot be written as an MRtrix transform and voxel sizes."
        )
    transform = np.array(matrix[:3, :4], copy=True)
    transform[:, :3] /= vox
    return transform, tuple(float(v) for v in vox)


# ----------------------------------------------------------------------
#   PARSER
# ----------------------------------------------------------------------


class MrtrixParser(DataModelBase, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by an MRtrix image file.

    It reads and writes the container -- the header and the raw voxel
    values -- for every MRtrix-based format: an image, and later a warp.
    What the values mean is for the concrete format to say, through
    `_mrtrix_header` and `_mrtrix_data` when writing.

    Reading a `.mif` or a `.mih` from a local path memory-maps the data,
    so nothing but the header is read until the data are indexed. A
    `.mif.gz`, a file object or bytes are read into memory.
    """

    HINTS = ("mrtrix",)

    _header: tx.Annotated[
        tx.Optional[MrtrixHeader],
        tx.Doc(
            """
            The MRtrix header this object was read from, kept so that the
            keys the data model has no slot for (`dw_scheme`,
            `command_history`, ...) are written back.
            """
        ),
    ] = None

    dataobj: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The stored values, as an array of the header's axes, before
            intensity scaling. A view of a memory-mapped file when the file
            allows one.
            """
        ),
    ] = None

    @property
    def header(self) -> tx.Optional[MrtrixHeader]:
        """The MRtrix header this object was read from, if any."""
        return getattr(self, "_header", None)

    @header.setter
    def header(self, value: tx.Optional[MrtrixHeader]) -> None:
        self._header = value

    def _scaled_data(self) -> tx.Optional[tx.Any]:
        """
        The stored values with the header's intensity scaling applied.

        Without scaling (or with the identity one), the stored values are
        returned as they are, so a memory map stays one. With scaling,
        the values are read and scaled, to at least single precision.
        """
        raw = getattr(self, "dataobj", None)
        if raw is None:
            return None
        header = self.header
        scaling = getattr(header, "scaling", None)
        if not scaling or tuple(scaling) == (0.0, 1.0):
            return raw
        offset, scale = scaling
        dtype = np.result_type(np.asarray(raw).dtype, np.float32)
        return offset + scale * np.asarray(raw, dtype=dtype)

    # --- reading ------------------------------------------------------

    @classmethod
    def _from_header(
        cls, header: MrtrixHeader, dataobj: tx.Any, **kwargs
    ) -> tx.Self:
        """Build the object from a decoded header and its stored values."""
        return cls(header=header, dataobj=dataobj, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Build the object from an MRtrix file (path or file object)."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return cls.from_filename(file, **kwargs)
        return super().from_file(file, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> tx.Self:
        """
        Build the object from the path of a `.mif`, `.mih` or `.mif.gz`.

        The data of an uncompressed local file are memory-mapped unless
        `mmap` is false. A `.mih` header is followed to the data file it
        names, relative to the header's directory.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not filename.exists():
            raise ParserExistsError(f"No such file: {filename}")

        if _is_gzip(filename):
            with _open_path(filename) as f:
                return cls.from_fileobj(f, **kwargs)

        with _open_path(filename) as f:
            header, _ = MrtrixHeader.from_fileobj(f)
        name, offset = _data_location(header)
        if name == ".":
            datafile = filename
        else:
            datafile = _sibling(filename, name)
            if not datafile.exists():
                raise ParserExistsError(
                    f"No such file: {datafile} (the data file named by the "
                    f"MRtrix header {filename})"
                )
        buffer = _read_buffer(datafile, offset, header.nbytes, mmap)
        return cls._from_header(header, decode_data(header, buffer), **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """
        Build the object from an open MRtrix file object, gzipped or not.

        The data of a single-file image are read from the stream. A
        header that names a separate data file is resolved against the
        stream's `name`, when it has one.
        """
        kwargs.pop("mmap", None)
        with preserve_position(file):
            stream = open_compressed(file)
            content = stream.read()
        return cls._from_content(
            content, origin=getattr(file, "name", None), **kwargs
        )

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build the object from the bytes of a single-file MRtrix image
        (gzipped or not)."""
        kwargs.pop("mmap", None)
        content = bytes(content)
        if content[:2] == b"\x1f\x8b":
            content = gzip.decompress(content)
        return cls._from_content(content, origin=None, **kwargs)

    @classmethod
    def _from_content(
        cls, content: bytes, origin: tx.Any = None, **kwargs
    ) -> tx.Self:
        header, _ = MrtrixHeader.from_fileobj(BytesIO(content))
        name, offset = _data_location(header)
        if name == ".":
            buffer = memoryview(content)[offset : offset + header.nbytes]
        else:
            if not isinstance(origin, (str, bytes, os.PathLike)):
                raise ParserContentError(
                    f"The MRtrix header names a separate data file "
                    f"({name!r}), which cannot be found from a stream that "
                    f"has no file name. Read it from its path instead."
                )
            datafile = _sibling(path.Path(os.fsdecode(origin)), name)
            if not datafile.exists():
                raise ParserExistsError(f"No such file: {datafile}")
            buffer = _read_buffer(datafile, offset, header.nbytes, False)
        return cls._from_header(header, decode_data(header, buffer), **kwargs)

    # --- sniffing -----------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that a stream holds an MRtrix
        image, gzipped or not."""
        score = Confidence.NO
        base_error = None
        try:
            with preserve_position(file):
                header, _ = MrtrixHeader.from_fileobj(open_compressed(file))
            score = cls._score_header(header)
        except Exception as e:  # noqa: BLE001
            base_error = e
            score = Confidence.NO
        if score:
            return score
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Content is not an MRtrix image") from base_error
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that bytes hold an MRtrix
        image, gzipped or not."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """
        How well a valid MRtrix header matches *this* class.

        Called once the header has been parsed, so the answer is never
        "not MRtrix". A concrete format overrides it to tell its own kind
        of MRtrix file (an image, a warp) from the others.
        """
        return Confidence.MAYBE

    # --- writing ------------------------------------------------------

    def _mrtrix_header(self, **kwargs) -> MrtrixHeader:
        """The header to write. Each concrete format builds its own."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to "
            f"MRtrix."
        )

    def _mrtrix_data(self) -> tx.Any:
        """The array of the header's axes to write."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to "
            f"MRtrix."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """The bytes of a single-file, uncompressed `.mif`."""
        header = self._mrtrix_header(**kwargs)
        data = encode_data(header, self._mrtrix_data_for(header))
        return header.embedded() + data

    def _mrtrix_data_for(self, header: MrtrixHeader) -> tx.Any:
        """The data to write, mapped through the header's scaling."""
        data = self._mrtrix_data()
        if header.scaling and tuple(header.scaling) != (0.0, 1.0):
            offset, scale = header.scaling
            data = (np.asarray(data) - offset) / scale
        return data

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write a single-file, uncompressed `.mif` to a stream."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write to a path, in the variant its extension names.

        * `.mif`: header and data in one file;
        * `.mif.gz`: the same, gzip-compressed;
        * `.mih`: the header, and the data in a `.dat` file next to it
          (named after the header), which the header points to.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        name = str(filename)
        lower = name.lower()
        if lower.endswith(".mih"):
            header = self._mrtrix_header(**kwargs)
            data = encode_data(header, self._mrtrix_data_for(header))
            base = filename.name[: -len(".mih")]
            datafile = _sibling(filename, base + ".dat")
            with datafile.open("wb") as f:
                f.write(data)
            with filename.open("wb") as f:
                text = header.to_text(file=(base + ".dat", 0))
                f.write(text.encode("utf-8"))
            return
        content = self.to_bytes(**kwargs)
        if lower.endswith(".gz"):
            content = gzip.compress(content)
        with filename.open("wb") as f:
            f.write(content)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write to a path (variant chosen by extension) or a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)


def _data_location(header: MrtrixHeader) -> tx.Tuple[str, int]:
    """The `(name, offset)` of the data, checked as MRtrix does."""
    if header.file is None:
        raise ParserContentError(
            "The MRtrix header has no 'file' entry, so its data cannot be "
            "found."
        )
    name, offset = header.file
    if name == "." and offset <= 0:
        raise ParserContentError(
            "Invalid offset for an MRtrix image embedded in its header "
            f"file: {offset}"
        )
    return name, offset
