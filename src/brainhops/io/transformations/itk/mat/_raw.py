"""The record of an ITK binary MATLAB (`.mat`) file, which holds its blocks."""

__all__ = ["MatRaw"]

import re

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core.streams import preserve_position
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    SnifferContentError,
)

from .._chains import _check_composite, _read_only
from .._common import ItkPrecision, ItkStruct, ItkTransformClass

_CLASS_RE = re.compile(
    r"^(?P<type>\w+)_"
    r"(?P<precision>float|double)_"
    r"(?P<input_dim>\d+)_"
    r"(?P<output_dim>\d+)$"
)
# A MATLAB v4 variable header is five 32-bit integers: the type, the number of
# rows, the number of columns, the complex flag and the length of the name.
_HEADER_SIZE = 20
# ITK names each variable after a transform class, which is short. The cap
# only stops the sniffer from reading a whole foreign file to find a name.
_MAX_NAME = 256
# The P digit of the type gives the precision of the stored values.
_PRECISIONS = {0: "f8", 1: "f4"}
_PRECISION_DIGITS = {ItkPrecision.Double: 0, ItkPrecision.Float: 1}
# The M digit of the type gives the byte order of the header and the values.
_BYTE_ORDERS = {"<": 0, ">": 1}
# MATLAB reserves the O digit as 0, while VNL writes 1 for a matrix stored row
# by row. For the vectors that ITK writes, both values mean the same thing.
_STORAGES = (0, 1)


class _Variable(tx.NamedTuple):
    """One decoded MATLAB v4 variable header."""

    name: str
    dtype: np.dtype
    count: int
    """The number of values."""
    cols: int
    """The number of columns, which is one because ITK only reads column
    vectors."""
    start: int
    """The offset of the first value in the content."""
    order: str
    """The byte order of the header and the values, `"<"` or `">"`."""


class MatRaw(Raw, BinaryFileReader, BinaryFileWriter):
    """Record of an ITK binary MATLAB (`.mat`) file, which holds its blocks.

    A `.mat` file is short and is read in one pass, so the record holds
    every block of the file, in the order of the file, as an
    [`ItkStruct`][brainhops.io.transformations.itk.ItkStruct], with a
    `CompositeTransform` header first when the file holds a composite. It
    also holds the byte order of the file, and the bytes of the file as
    they were read. The metadata of a `.mat` file,
    [`MatMetadata`][brainhops.io.transformations.itk.mat.MatMetadata],
    holds a record, and a
    [`MatTransform`][brainhops.io.transformations.itk.mat.MatTransform]
    decodes its chain from the record of its metadata.

    The bytes that were read are written back as they are while they
    still encode the blocks and the byte order of the record, so that a
    file that is read and written again is kept byte for byte. Encoding
    the blocks again would not always give the same bytes, because ITK
    may name the fixed parameters otherwise, write them in single
    precision, or mark the variables as stored row by row. The blocks of a
    record are also the elements of the chain that a transformation
    decodes, so their parameters are read-only, and a record is never
    changed in place.
    """

    blocks: tx.Tuple[ItkStruct, ...] = ()
    """Every block of the file, in the order of the file.

    A `CompositeTransform` header can only be the first block.
    """

    byteorder: tx.Literal["<", ">"] = "<"
    """The byte order of the file, `"<"` or `">"`.

    It is the byte order of the first variable of the file, and
    little-endian for a file without variables.
    """

    content: NoRepr[tx.Optional[bytes]] = None
    """The bytes of the file as it was read, or `None` for a new record.

    They are written back as they are while they still encode the blocks
    and the byte order of the record.
    """

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the confidence that a binary stream holds an ITK `.mat` file.

        Only the header and the name of the first variable are needed, so
        the sniffer reads no more bytes than those can occupy.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not an ITK
            `.mat` file.
        **kwargs : Any
            Options of the reader, which are not used.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        with preserve_position(file):
            head = file.read(_HEADER_SIZE + _MAX_NAME)
        return cls.sniff_bytes(head, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the confidence that bytes hold an ITK `.mat` file.

        The content must start with a MATLAB v4 header whose variable is
        named after an ITK transform class.

        Parameters
        ----------
        content : bytes
            The content, or its beginning.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not an ITK
            `.mat` file.
        **kwargs : Any
            Options of the reader, which are not used.

        Returns
        -------
        float
            The confidence, between 0 and 1.
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

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Reject text, which is never an ITK `.mat` file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the content.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning 0.
        **kwargs : Any
            Options of the reader, which are not used.

        Returns
        -------
        float
            Always `Confidence.NO`.
        """
        return _reject(error, "Not binary content.")

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "MatRaw":
        """Read the record of an ITK `.mat` file from a binary stream.

        ITK writes each block as two column vectors, which are its
        parameters followed by its fixed parameters. As ITK does, the
        variables are read in pairs, the second variable of each pair is
        taken as the fixed parameters whatever its name, and anything
        other than a column vector is refused. The whole stream is read,
        since the file is read in one pass, and its position is restored.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        **kwargs : Any
            Options of the reader. There are none, so any option is
            refused.

        Returns
        -------
        MatRaw
            The record, which keeps the bytes of the file.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the content is not an ITK `.mat` file.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} takes no options, but was given "
                f"{', '.join(sorted(kwargs))}."
            )
        with preserve_position(file):
            content = bytes(file.read())
        blocks, byteorder = _decode(content)
        return cls(blocks=blocks, byteorder=byteorder, content=content)

    # --- to -----------------------------------------------------------

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the ITK `.mat` file that the record holds.

        The bytes that were read are returned while they still encode the
        blocks and the byte order of the record. Otherwise, each block is
        written as `vnl_matlab_write` writes a column vector: its
        parameters, named after its class, its precision and its
        dimensions, then its fixed parameters, named `fixed` and written
        in double precision.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer. There are none, so any option is
            refused.

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        TypeError
            If an option is given.
        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.to_bytes() takes no options, but "
                f"was given {', '.join(sorted(kwargs))}."
            )
        if self.content is not None and _still_encodes(self):
            return self.content
        return _encode(self.blocks, self.byteorder)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _reject(error: tx.Union[bool, tx.Type[Exception]], message: str) -> float:
    """Return `Confidence.NO`, or raise if the caller asked for an error."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message)
    return Confidence.NO


def _read_header(
    content: bytes, offset: int, need_data: bool = True
) -> tx.Optional[_Variable]:
    """Decode the MATLAB v4 header at `offset`, or return `None`.

    Only headers of real, double or single precision numeric matrices are
    accepted, since ITK writes nothing else. With `need_data`, the values must
    also be present.
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
        # The type is the decimal number M*1000 + O*100 + P*10 + T. A header
        # read in the wrong byte order disagrees with its own M digit, and
        # this disagreement tells the two orders apart.
        if not 0 <= mopt < 10000:
            continue
        m, o, p, t = (
            mopt // 1000,
            mopt // 100 % 10,
            mopt // 10 % 10,
            mopt % 10,
        )
        if m != digit or o not in _STORAGES or t != 0:
            continue
        if p not in _PRECISIONS:
            continue
        if rows < 0 or cols < 0 or imagf != 0:
            continue
        if not 1 < namlen <= _MAX_NAME:
            continue
        start = offset + _HEADER_SIZE
        raw = content[start : start + namlen]
        if len(raw) < namlen:
            continue
        # The name length counts the terminating NUL.
        try:
            name = raw.split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError:
            continue
        if not name.isidentifier():
            continue
        dtype = np.dtype(order + _PRECISIONS[p])
        count = rows * cols
        start += namlen
        if need_data and len(content) < start + count * dtype.itemsize:
            continue
        return _Variable(name, dtype, count, cols, start, order)
    return None


def _read_variables(
    content: bytes,
) -> tx.Iterator[tx.Tuple[_Variable, np.ndarray]]:
    """Yield the header and values of each variable, in file order.

    The variables are not returned as a mapping, because a file with several
    blocks repeats the name `fixed` and may also repeat class names.
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
        # The values are copied so that they own their memory and are in
        # native byte order.
        yield variable, values.astype(variable.dtype.newbyteorder("="))
        offset = variable.start + variable.count * variable.dtype.itemsize


def _write_variable(
    name: str,
    values: tx.Any,
    byteorder: str,
    precision: ItkPrecision,
) -> bytes:
    """Encode a column vector as a MATLAB v4 variable.

    The encoding follows `vnl_matlab_write`: the header, the NUL-terminated
    name and then the values, all in the byte order `byteorder`.
    """
    digit = _PRECISION_DIGITS[precision]
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values.astype(byteorder + _PRECISIONS[digit])
    raw = name.encode("ascii") + b"\0"
    mopt = _BYTE_ORDERS[byteorder] * 1000 + digit * 10
    header = np.array(
        [mopt, values.size, 1, 0, len(raw)], dtype=byteorder + "i4"
    )
    return header.tobytes() + raw + values.tobytes()


def _decode(
    content: bytes,
) -> tx.Tuple[tx.Tuple[ItkStruct, ...], str]:
    """Return the blocks and the byte order that the bytes of a file hold.

    Raises
    ------
    ParserContentError
        If the content is not an ITK `.mat` file.
    """
    variables = list(_read_variables(content))
    if len(variables) % 2:
        raise ParserContentError(
            f"ITK writes variables in pairs (parameters, then fixed "
            f"parameters), but the file holds {len(variables)}."
        )
    for variable, _ in variables:
        if variable.cols != 1:
            raise ParserContentError(
                f"ITK only reads column vectors, but {variable.name!r} "
                f"has {variable.cols} columns."
            )
    blocks = []
    for index in range(0, len(variables), 2):
        (variable, parameters), (_, fixed_parameters) = variables[
            index : index + 2
        ]
        match = _CLASS_RE.match(variable.name)
        if not match:
            raise ParserContentError(
                f"Expected a variable named after an ITK transform "
                f"class, not {variable.name!r}."
            )
        blocks.append(
            ItkStruct(
                type=ItkTransformClass(match.group("type")),
                precision=match.group("precision"),
                ndim_input=int(match.group("input_dim")),
                ndim_output=int(match.group("output_dim")),
                parameters=_read_only(parameters),
                fixed_parameters=_read_only(fixed_parameters),
            )
        )
    _check_composite(blocks)
    byteorder = variables[0][0].order if variables else "<"
    return tuple(blocks), byteorder


def _encode(blocks: tx.Sequence[ItkStruct], byteorder: str) -> bytes:
    """Return the bytes of a file that holds blocks, in a byte order.

    Each block is written as `itk::MatlabTransformIOTemplate::Write`
    writes it, as two column vectors: the parameters, named after the
    class, the precision and the dimensions of the block, and then the
    fixed parameters, named `fixed`. The fixed parameters are written in
    double precision, whereas ITK writes them in the precision of its
    writer.
    """
    content = b""
    for block in blocks:
        precision = ItkPrecision(block.precision)
        name = (
            f"{ItkTransformClass(block.type).value}_{precision.value}_"
            f"{block.ndim_input}_{block.ndim_output}"
        )
        content += _write_variable(
            name, block.parameters, byteorder, precision
        ) + _write_variable(
            "fixed", block.fixed_parameters, byteorder, ItkPrecision.Double
        )
    return content


def _still_encodes(record: MatRaw) -> bool:
    """Tell whether the bytes of a record still encode its other fields."""
    try:
        blocks, byteorder = _decode(record.content)
    except (ParserContentError, ValueError):
        return False
    return byteorder == record.byteorder and _same_blocks(
        blocks, record.blocks
    )


def _same_blocks(
    first: tx.Sequence[ItkStruct], second: tx.Sequence[ItkStruct]
) -> bool:
    """Tell whether two lists of blocks hold the same values."""
    if len(first) != len(second):
        return False
    for one, two in zip(first, second):
        if (
            one.type != two.type
            or one.precision != two.precision
            or one.ndim_input != two.ndim_input
            or one.ndim_output != two.ndim_output
        ):
            return False
        for name in ("parameters", "fixed_parameters"):
            values = np.asarray(getattr(one, name), dtype=np.float64)
            other = np.asarray(getattr(two, name), dtype=np.float64)
            if not np.array_equal(values, other):
                return False
    return True
