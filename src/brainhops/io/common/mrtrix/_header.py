"""The MRtrix image header."""

# stdlib
import math
from collections import OrderedDict

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import ParserContentError

# this format
from ._codecs import (
    default_layout,
    format_layout,
    mrtrix_dtype,
    parse_layout,
)
from ._constants import (
    _BIT,
    _END,
    _FILE,
    _MAGIC_BYTES,
    _MAX_HEADER_LINES,
    _RESERVED,
    MRTRIX_MAGIC,
)

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
