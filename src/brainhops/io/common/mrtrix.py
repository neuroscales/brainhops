"""
Reading and writing of MRtrix images.

This module holds what the MRtrix image and transformation formats
share: the header, and the decoding and encoding of voxel data. An
MRtrix image is a text header followed by raw voxel values, either in
one file (`.mif`, or `.mif.gz` gzipped) or in two (a `.mih` header and
the data file it names). The header is a list of `key: value` lines
between `mrtrix image` and `END`, where `#` starts a comment:

```text
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
```

`layout` gives the storage rank and direction of each axis (see
[`parse_layout`][]). `transform` maps voxel coordinates multiplied by
the voxel sizes, not voxel indices, to scanner RAS millimetres; without
one, MRtrix centres the field of view on the origin rather than
assuming the identity. `scaling` gives `offset,scale`, by which a
stored value `v` means `offset + scale * v`. `file` names the data file
and an offset, where `.` means the header file itself. The conventions
follow the MRtrix3 sources.
"""

__all__ = [
    "MrtrixHeader",
    "MrtrixParser",
    "MRTRIX_MAGIC",
]

import gzip
import math
import os
import re
from collections import OrderedDict
from io import BytesIO

import numpy as np
import typing_extensions as tx

from brainhops._core import path
from brainhops._core.streams import open_compressed
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.transformations import Transformation
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
from brainhops.io.common._geometry import Arrangement, arrange_voxel_to_ras

MRTRIX_MAGIC = "mrtrix image"
"""The first line of every MRtrix header."""

_MAGIC_BYTES = MRTRIX_MAGIC.encode("ascii")

_END = "END"
"""The last line of every MRtrix header."""

_RESERVED = ("dim", "vox", "layout", "datatype", "transform", "scaling")
"""The keys MRtrix decodes itself, which are case-insensitive."""

_FILE = "file"
"""The key that gives the location of the data."""

_MAX_HEADER_LINES = 1_000_000
"""
A bound on the header length, so that a stray binary file is not read
whole in search of `END`.
"""


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
"""The NumPy type of each MRtrix data type, without byte order."""

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
"""The spelling MRtrix writes for each data type."""

_BIT = "bit"


def mrtrix_dtype(spec: str) -> tx.Optional[np.dtype]:
    """
    Return the NumPy dtype of an MRtrix data type, or `None` for `Bit`.

    The name is case-insensitive. As in MRtrix, the byte order is given by
    an `LE` or `BE` suffix, and is the native one otherwise.

    Raises
    ------
    ParserContentError
        If the name is not an MRtrix data type.
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
    Return the MRtrix data type that stores a dtype exactly.

    Booleans are stored as `Bit`, and multi-byte types always carry an `LE`
    or `BE` suffix, as MRtrix writes them.

    Raises
    ------
    WriterError
        If MRtrix cannot store the dtype, such as float16 or strings.
    """
    if isinstance(dtype, str) and dtype.strip().lower() == _BIT:
        return "Bit"
    if isinstance(dtype, str) and _is_mrtrix_spec(dtype):
        # normalise the spelling of an MRtrix data type
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
    Parse a layout into signed one-based strides.

    For example, `-0,-1,+2` becomes `(-1, -2, 3)`. The absolute value minus
    one is the rank, from 0 for the fastest axis, and the sign is the
    storage direction. Counting from one keeps the sign of rank 0.

    Raises
    ------
    ParserContentError
        If the layout is malformed, has the wrong number of axes, or does
        not give the axes distinct ranks.
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
    """Spell signed one-based strides as a layout."""
    return ",".join(
        ("+" if s > 0 else "-") + str(abs(int(s)) - 1) for s in strides
    )


def default_layout(ndim: int) -> tx.Tuple[int, ...]:
    """Return the Fortran-order layout `+0,+1,+2,...`."""
    return tuple(range(1, ndim + 1))


def _storage_shape(
    dim: tx.Sequence[int], strides: tx.Sequence[int]
) -> tx.Tuple[tx.Tuple[int, ...], tx.List[int]]:
    """
    Return the C-ordered file shape and the position of each header axis
    in it.
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
    View stored values as an array indexed in header axis order.

    The result is a view, so that a memory map stays one.
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
    Reorder an array of header axes into file order.

    This is the inverse of [`storage_to_image`][].
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
    """Spell a float in the shortest way that reads back exactly."""
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

    Keys other than `dim`, `vox`, `layout`, `datatype`, `transform`,
    `scaling` and `file` are kept in `keyval`, in order and with the lines
    of a repeated key joined by newlines, so that the header is written
    back as read. A header is plain data, which never touches a file.
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

    @property
    def ndim(self) -> int:
        """The number of axes."""
        return len(self.dim)

    @property
    def dtype(self) -> tx.Optional[np.dtype]:
        """The NumPy dtype of the stored values, or `None` for `Bit`."""
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

        Missing sizes are `nan`, and a non-finite spatial size is replaced by
        the mean of the finite ones (or 1).
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
        Return the `(3, 4)` transform MRtrix assumes when the header has none.

        The rotation is the identity, and the field of view is centred on the
        origin.
        """
        dim = list(self.dim[:3]) + [1] * (3 - min(3, self.ndim))
        vox = list(self.spacing[:3]) + [1.0] * (3 - min(3, self.ndim))
        matrix = np.eye(3, 4)
        for i in range(3):
            matrix[i, 3] = -0.5 * (dim[i] - 1) * vox[i]
        return matrix

    def voxel_to_scanner(self) -> np.ndarray:
        """
        Return the `(4, 4)` matrix from voxel indices to scanner RAS.

        The matrix is `transform @ diag(vox[0], vox[1], vox[2], 1)`, with
        [`default_transform`][] replacing a missing or non-finite transform.
        """
        transform = self.transform
        if transform is None or not np.all(np.isfinite(transform)):
            transform = self.default_transform()
        vox = list(self.spacing[:3]) + [1.0] * (3 - min(3, self.ndim))
        matrix = np.eye(4)
        matrix[:3, :4] = transform
        return matrix @ np.diag(vox + [1.0])

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> "MrtrixHeader":
        """
        Parse the header lines from `mrtrix image` to `END`.

        Lines after `END` are ignored.

        Raises
        ------
        ParserContentError
            If the first line is wrong, if `END`, `dim`, `vox` or `datatype` is
            missing, or if an entry is malformed.
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
                # MRtrix ignores malformed lines.
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
        mrtrix_dtype(datatype)
        # MRtrix requires a layout; without one, read the Fortran order that
        # MRtrix gives a new image.
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
        """Parse a header from text."""
        return cls.from_lines(text.splitlines())

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO) -> tx.Tuple["MrtrixHeader", int]:
        """
        Read a header from a binary stream, and return it with the number of
        bytes read.

        The stream is left just after the `END` line.

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

    def to_lines(
        self, file: tx.Optional[tx.Tuple[str, tx.Optional[int]]] = None
    ) -> tx.List[str]:
        """
        Return the header lines, from `mrtrix image` to `END`.

        The `file` argument, which defaults to the `file` of the header, is the
        `(name, offset)` of the `file:` line; an offset of `None` is omitted.
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
        """Return the header text, terminated by a newline."""
        return "\n".join(self.to_lines(*args, **kwargs)) + "\n"

    def embedded(self, align: int = 4) -> bytes:
        """
        Return the header of a single-file image, padded up to its data.

        The data offset is aligned on `align` bytes. Since the offset is part
        of the header it measures, it is found by iteration.
        """
        offset = 0
        while True:
            text = self.to_text(file=(".", offset)).encode("utf-8")
            needed = -(-len(text) // align) * align
            if needed == offset:
                return text + b"\0" * (offset - len(text))
            offset = needed

    def copy(self) -> "MrtrixHeader":
        """Return an independent copy."""
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
    Decode stored values into an array of header axes.

    The buffer holds the voxel data from their first byte. The result is a
    view of the buffer, except for `Bit` data, which are unpacked into a
    new boolean array. Intensity scaling is not applied.

    Raises
    ------
    ParserContentError
        If the buffer holds fewer values than the header requires.
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
    Encode an array of header axes into the bytes of the file.

    Floats stored as integers are rounded.

    Raises
    ------
    WriterError
        If the shape of the array disagrees with the header.
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
    Read `nbytes` bytes at `offset` in an uncompressed data file,
    memory-mapping a local file unless `mmap` is false.
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
    """Tell whether a path names a gzip file, judged by its content."""
    try:
        with _open_path(file) as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


# ----------------------------------------------------------------------
#   GEOMETRY (writing)
# ----------------------------------------------------------------------


MRTRIX_POLICY = dict(fill_space=True)
"""
Where MRtrix stores the axes of an array, as a policy of
[`plan_axes`][brainhops.io.common._geometry.plan_axes].

A slice with non-spatial axes gets a `z` axis of size one, so that
those axes are not read as spatial.
"""


def voxel_to_ras(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[tx.Any]] = None
) -> Arrangement:
    """
    Return the voxel-to-RAS geometry of a voxel-to-world transformation,
    with the axes placed where MRtrix stores them.

    The axes are placed by what their spaces declare, where `voxel_axes`
    are the axes of the data (see [`arrange_voxel_to_ras`][]).

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine form, or if it mixes the spatial
        axes with the others.
    """
    return arrange_voxel_to_ras(
        xform, voxel_axes, "MRtrix", "scanner", **MRTRIX_POLICY
    )


def split_voxel_to_scanner(
    matrix: np.ndarray,
) -> tx.Tuple[np.ndarray, tx.Tuple[float, float, float]]:
    """
    Split a voxel-to-scanner matrix into an MRtrix transform of unit
    directions and the voxel sizes.

    Raises
    ------
    WriterError
        If a direction column has zero length, or if the matrix is not
        finite.
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
    The base class of objects encoded as an MRtrix image.

    The class reads and writes the container, a header and its raw values,
    and a concrete format provides `_mrtrix_header` and `_mrtrix_data` on
    write. The data of a local `.mif` or `.mih` file are memory-mapped, so
    only the header is read until they are indexed.
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
        """The header this object was read from, if any."""
        return getattr(self, "_header", None)

    @header.setter
    def header(self, value: tx.Optional[MrtrixHeader]) -> None:
        self._header = value

    def _scaled_data(self) -> tx.Optional[tx.Any]:
        """
        Return the stored values with the intensity scaling of the header.

        Unscaled values are returned unchanged, so that a memory map stays one.
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

    @classmethod
    def _from_header(
        cls, header: MrtrixHeader, dataobj: tx.Any, **kwargs
    ) -> tx.Self:
        """Build an object from a decoded header and its stored values."""
        return cls(header=header, dataobj=dataobj, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Build an object from an MRtrix file, by path or file object."""
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
        Build an object from the path of a `.mif`, `.mih` or `.mif.gz` file.

        Uncompressed local data are memory-mapped unless `mmap` is false.
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
        Build an object from an open MRtrix file, gzipped or not.

        A separate data file is found from the name of the stream.
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
        """Build an object from the bytes of a single-file image."""
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

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an MRtrix image."""
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
        """Return the confidence that bytes hold an MRtrix image."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """
        Score how well a valid header matches this class.

        A concrete format overrides the score to tell its own kind of image,
        such as a warp, from the others.
        """
        return Confidence.MAYBE

    def _mrtrix_header(self, **kwargs) -> MrtrixHeader:
        """Build the header to write; each concrete format defines it."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to "
            f"MRtrix."
        )

    def _mrtrix_data(self) -> tx.Any:
        """Return the array of header axes to write."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to "
            f"MRtrix."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of a single-file, uncompressed `.mif` image."""
        header = self._mrtrix_header(**kwargs)
        data = encode_data(header, self._mrtrix_data_for(header))
        return header.embedded() + data

    def _mrtrix_data_for(self, header: MrtrixHeader) -> tx.Any:
        """Return the data to write, mapped through the header scaling."""
        data = self._mrtrix_data()
        if header.scaling and tuple(header.scaling) != (0.0, 1.0):
            offset, scale = header.scaling
            data = (np.asarray(data) - offset) / scale
        return data

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write a single-file, uncompressed `.mif` image to a stream."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write to a path, in the variant that its extension names.

        A `.mif` file holds the header and the data, gzipped for `.mif.gz`. A
        `.mih` header points to the data, written next to it in a `.dat` file.
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
        """Write to a path or a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)


def _data_location(header: MrtrixHeader) -> tx.Tuple[str, int]:
    """
    Return the `(name, offset)` of the data, checked as MRtrix does.

    Raises
    ------
    ParserContentError
        If the header has no `file` entry, or an embedded image has no
        positive offset.
    """
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
