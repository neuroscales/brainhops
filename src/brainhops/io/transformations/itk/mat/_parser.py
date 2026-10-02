# stdlib
import re

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Magic

# core
from brainhops._core.streams import preserve_position

# io
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    ParserContentError,
    SnifferContentError,
)

# locals
from .._common import ITKStruct, ITKTransformClass

# constants
_CLASS_RE = re.compile(
    r"^(?P<type>\w+)_"
    r"(?P<precision>float|double)_"
    r"(?P<input_dim>\d+)_"
    r"(?P<output_dim>\d+)$"
)
_FIXED = "fixed"
# A MATLAB v4 variable header is five 32-bit integers:
# type (MOPT), rows, columns, imaginary flag, length of the name.
_HEADER_SIZE = 20
# ITK names a variable after a transform class, which is short. The cap
# only keeps a sniffer from reading a whole foreign file for a name.
_MAX_NAME = 256
# The P digit of MOPT: the precision of the stored values. ITK writes
# `double` transforms as 0 and `float` transforms as 1.
_PRECISIONS = {0: "f8", 1: "f4"}
# The M digit of MOPT: the byte order. ITK writes in the native order of
# the machine that wrote the file, which is almost always little-endian.
_BYTE_ORDERS = {"<": 0, ">": 1}


class _Variable(tx.NamedTuple):
    """One decoded MATLAB v4 variable header."""

    name: str
    dtype: np.dtype
    count: int
    """Number of values: rows times columns."""
    start: int
    """Offset of the first value in the content."""


class MATTransformParser(
    Magic,
    BinaryFileParser,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parses an ITK binary MATLAB (`.mat`) transform file into a chain
    of transform blocks.

    Each block is itself a brainhops transformation, so the parsed blocks
    are stored straight into the `transformations` of the sequence that
    this parser is mixed into.
    """

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the parser is that an open binary file
        object is an ITK MATLAB transform file.

        Only the first variable header is needed, so only enough bytes
        to hold it are read.
        """
        with preserve_position(file):
            head = file.read(_HEADER_SIZE + _MAX_NAME)
        return cls.sniff_content(head, error=error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Text is never an ITK MATLAB transform file."""
        return _reject(error, "Not binary content.")

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the parser is that bytes are an ITK
        MATLAB transform file.

        The file must start with a MATLAB v4 variable header, and that
        variable must be named after an ITK transform class, which is
        how ITK names the parameters of every block it writes.
        """
        variable = _read_header(bytes(content), 0, need_data=False)
        if variable is None:
            return _reject(error, "Not a MATLAB v4 file.")
        if not _CLASS_RE.match(variable.name):
            return _reject(
                error,
                f"MATLAB v4 variable {variable.name!r} is not named after "
                f"an ITK transform class.",
            )
        return Confidence.CERTAIN

    # --- from ---------------------------------------------------------

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build the transform chain from the bytes of an ITK MATLAB
        transform file.

        ITK writes each block as two variables: its parameters, named
        after its transform class, then its fixed parameters, named
        `fixed`.
        """
        variables = list(_read_variables(bytes(content)))
        blocks = []
        index = 0
        while index < len(variables):
            name, parameters = variables[index]
            index += 1
            match = _CLASS_RE.match(name)
            if not match:
                raise ParserContentError(
                    f"Expected a variable named after an ITK transform "
                    f"class, not {name!r}."
                )
            fixed_parameters = np.array([], dtype=parameters.dtype)
            if index < len(variables) and variables[index][0] == _FIXED:
                fixed_parameters = variables[index][1]
                index += 1

            if match.group("type") == "CompositeTransform":
                # skip composite transforms, they just point to the
                # following transforms.
                continue

            blocks.append(
                ITKStruct(
                    type=ITKTransformClass(match.group("type")),
                    precision=match.group("precision"),
                    ndim_input=int(match.group("input_dim")),
                    ndim_output=int(match.group("output_dim")),
                    parameters=parameters,
                    fixed_parameters=fixed_parameters,
                )
            )

        obj = cls()
        obj.transformations = blocks
        return obj


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _reject(error: tx.Union[bool, tx.Type[Exception]], message: str) -> float:
    """Return `Confidence.NO`, or raise if the caller asked for it."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message)
    return Confidence.NO


def _read_header(
    content: bytes, offset: int, need_data: bool = True
) -> tx.Optional[_Variable]:
    """Decode the MATLAB v4 variable header at `offset`.

    Returns `None` if the bytes there are not a header of a real, double
    or single precision, numeric matrix -- the only kind ITK writes. If
    `need_data`, the values the header announces must also be present.
    """
    if len(content) < offset + _HEADER_SIZE:
        return None
    for order, digit in _BYTE_ORDERS.items():
        mopt, rows, cols, imagf, namlen = (
            int(v)
            for v in np.frombuffer(
                content, dtype=order + "i4", count=5, offset=offset
            )
        )
        # MOPT is the decimal number M*1000 + O*100 + P*10 + T: byte
        # order, a reserved zero, precision, and matrix type (0 = full
        # numeric). The byte order a header is read in must agree with
        # the M it declares, which is what tells the two orders apart.
        if not 0 <= mopt < 10000:
            continue
        m, o, p, t = (
            mopt // 1000,
            mopt // 100 % 10,
            mopt // 10 % 10,
            mopt % 10,
        )
        if m != digit or o != 0 or t != 0 or p not in _PRECISIONS:
            continue
        if rows < 0 or cols < 0 or imagf != 0:
            continue
        if not 1 < namlen <= _MAX_NAME:
            continue
        start = offset + _HEADER_SIZE
        raw = content[start : start + namlen]
        # The name is NUL-terminated, and the length counts the NUL.
        if len(raw) < namlen or raw[-1:] != b"\0":
            continue
        try:
            name = raw[:-1].decode("ascii")
        except UnicodeDecodeError:
            continue
        if not name.isidentifier():
            continue
        dtype = np.dtype(order + _PRECISIONS[p])
        count = rows * cols
        start += namlen
        if need_data and len(content) < start + count * dtype.itemsize:
            continue
        return _Variable(name, dtype, count, start)
    return None


def _read_variables(
    content: bytes,
) -> tx.Iterator[tx.Tuple[str, np.ndarray]]:
    """Yield the name and the values of every variable, in file order.

    A mapping would not do: a file that holds several blocks repeats the
    `fixed` name, and may repeat a class name too.
    """
    offset = 0
    while offset < len(content):
        variable = _read_header(content, offset)
        if variable is None:
            raise ParserContentError(
                f"Not a MATLAB v4 variable at byte {offset}."
            )
        values = np.frombuffer(
            content,
            dtype=variable.dtype,
            count=variable.count,
            offset=variable.start,
        )
        # ITK writes every vector as a column, so the column-major order
        # MATLAB stores values in is also the order of the vector. The
        # copy owns its memory and is in native byte order.
        yield variable.name, values.astype(variable.dtype.newbyteorder("="))
        offset = variable.start + variable.count * variable.dtype.itemsize
