# stdlib
import re
import sys

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Magic

# core
from brainhops._core import path
from brainhops._core.streams import preserve_position

# io
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserContentError,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
)

# locals
from .._common import ITKPrecision, ITKStruct, ITKTransformClass
from .._systems import _make_system

# constants
_CLASS_RE = re.compile(
    r"^(?P<type>\w+)_"
    r"(?P<precision>float|double)_"
    r"(?P<input_dim>\d+)_"
    r"(?P<output_dim>\d+)$"
)
# A MATLAB v4 variable header is five 32-bit integers:
# type (MOPT), rows, columns, imaginary flag, length of the name.
_HEADER_SIZE = 20
# ITK names a variable after a transform class, which is short. The cap
# only keeps a sniffer from reading a whole foreign file for a name.
_MAX_NAME = 256
# The P digit of MOPT: the precision of the stored values. ITK writes the
# parameters of a `double` transform as 0 and of a `float` transform as 1;
# fixed parameters are always `double`.
_PRECISIONS = {0: "f8", 1: "f4"}
# The P digit that each ITK precision is written with.
_PRECISION_DIGITS = {ITKPrecision.Double: 0, ITKPrecision.Float: 1}
# The M digit of MOPT: the byte order. `vnl_matlab_write` writes the
# header and the values in the native order of the machine that wrote
# the file, and records which one it was here.
_BYTE_ORDERS = {"<": 0, ">": 1}
# The O digit of MOPT. MATLAB reserves it as 0; VNL sets it to 1 for a
# matrix it writes row by row. ITK writes vectors, where the two agree.
_STORAGES = (0, 1)


class _Variable(tx.NamedTuple):
    """One decoded MATLAB v4 variable header."""

    name: str
    dtype: np.dtype
    count: int
    """Number of values: rows times columns."""
    cols: int
    """Number of columns. ITK only reads column vectors."""
    start: int
    """Offset of the first value in the content."""


class MatTransformParser(
    Magic,
    BinaryFileParserWriter,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parses an ITK binary MATLAB (`.mat`) transform file into a chain
    of transform blocks, and writes one back.

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

        ITK writes each block as two column vectors: its parameters,
        named after its transform class, then its fixed parameters, named
        `fixed`. Like ITK's own reader, this one reads the variables in
        pairs, takes the second of each pair as the fixed parameters
        whatever its name, and refuses anything but column vectors.
        """
        variables = list(_read_variables(bytes(content)))
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

    # --- to -----------------------------------------------------------

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transformation to a file.

        The content is built before the file is opened, so a
        transformation that the format cannot hold is refused without
        creating or truncating the file.
        """
        content = self.to_bytes(**kwargs)
        with path.Path(filename).open(self._WRITE_MODE) as f:
            f.write(content)

    def to_bytes(
        self,
        byteorder: str = "<",
        precision: tx.Optional[tx.Union[ITKPrecision, str]] = None,
        **kwargs,
    ) -> bytes:
        """The content of the ITK MATLAB file that encodes this
        transformation, as `itk::MatlabTransformIO` writes it.

        The transformation must be a single block, which is what ANTs
        writes to a `.mat` file. It is written as two column vectors: its
        parameters, named `{Class}_{Precision}_{D}_{D}`, then its fixed
        parameters, named `fixed`.

        - An ITK block (one that was read from an ITK file, or built as
          an [`ITKStruct`][]) is written as it is: its class, its
          parameters and its fixed parameters -- and so its center.
        - Any other transformation that converts to an
          [`Affine`][brainhops.datamodel.transformations.Affine] is
          written as an `AffineTransform` whose center (`fixed`) is the
          origin. The translation is then the last column of the
          matrix, and ITK reads back exactly that matrix. Its `input`
          and `output` must be ITK's space -- `LPSmm` in 3-D -- or left
          unspecified, in which case they are taken to be ITK's space.

        Parameters
        ----------
        byteorder : {"<", ">", "="}, default="<"
            The byte order of the headers and the values. ITK writes in
            the native order of the machine, which is little-endian on
            every common one; `"="` asks for the native order here.
        precision : {"double", "float"}, optional
            The precision of the parameters, which also goes into the
            class name. By default, that of the block, and `double` for
            a transformation that is not an ITK block. The fixed
            parameters are always `double`, as ITK writes them.

        Raises
        ------
        UnrepresentableTransformationError
            If the transformation is not a single block, or not an ITK
            block or an affine between ITK's spaces.
        """
        if byteorder == "=":
            byteorder = "<" if sys.byteorder == "little" else ">"
        if byteorder not in _BYTE_ORDERS:
            raise WriterError(
                f"byteorder must be '<', '>' or '=', not {byteorder!r}."
            )
        block = _single_block(self)
        if precision is None:
            precision = block.precision
        precision = ITKPrecision(precision)
        name = (
            f"{ITKTransformClass(block.type).value}_{precision.value}_"
            f"{block.ndim_input}_{block.ndim_output}"
        )
        return _write_variable(
            name, block.parameters, byteorder, precision
        ) + _write_variable(
            "fixed", block.fixed_parameters, byteorder, ITKPrecision.Double
        )


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
        # The name is NUL-terminated, and the length counts the NUL.
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
        return _Variable(name, dtype, count, cols, start)
    return None


def _read_variables(
    content: bytes,
) -> tx.Iterator[tx.Tuple[_Variable, np.ndarray]]:
    """Yield the header and the values of every variable, in file order.

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
        # A column vector reads the same column- or row-major. The copy
        # owns its memory and is in native byte order.
        yield variable, values.astype(variable.dtype.newbyteorder("="))
        offset = variable.start + variable.count * variable.dtype.itemsize


def _write_variable(
    name: str,
    values: tx.Any,
    byteorder: str,
    precision: ITKPrecision,
) -> bytes:
    """Encode one column vector as a MATLAB v4 variable, as
    `vnl_matlab_write` does: the header, the `NUL`-terminated name, then
    the values, all in byte order `byteorder`."""
    digit = _PRECISION_DIGITS[precision]
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values.astype(byteorder + _PRECISIONS[digit])
    raw = name.encode("ascii") + b"\0"
    # M*1000 + O*100 + P*10 + T, with O = 0 and T = 0 (full numeric).
    mopt = _BYTE_ORDERS[byteorder] * 1000 + digit * 10
    header = np.array(
        [mopt, values.size, 1, 0, len(raw)], dtype=byteorder + "i4"
    )
    return header.tobytes() + raw + values.tobytes()


def _single_block(chain: tx.Any) -> ITKStruct:
    """The one block that `chain` holds, as an ITK block.

    ANTs writes a single linear transform per `.mat` file, and a chain
    is not written: ITK applies the blocks of a `CompositeTransform` in
    the reverse of their order in the file, so the order a chain should
    be written in is not settled here.
    """
    children = list(chain.transformations or [])
    if len(children) != 1:
        raise UnrepresentableTransformationError(
            f"An ITK MATLAB file is written with a single transform, as "
            f"ANTs writes it, but this chain holds {len(children)}. "
            f"Compose them first, for example with `.compute()`."
        )
    (child,) = children
    if isinstance(child, ITKStruct):
        return child
    return _affine_block(child)


def _affine_block(xform: _xforms.Transformation) -> ITKStruct:
    """Encode an affine between ITK's spaces as an `AffineTransform`
    block centered on the origin."""
    affine = xform
    if not isinstance(affine, _xforms.Affine):
        affine = xform.to(_xforms.Affine, error=None)
    matrix = None if affine is None else affine.matrix
    if matrix is None:
        raise UnrepresentableTransformationError(
            f"An ITK MATLAB file holds an ITK transform or an affine, and "
            f"a {type(xform).__name__} cannot be written as either."
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    ndim = matrix.shape[0]
    if matrix.shape != (ndim, ndim + 1):
        raise UnrepresentableTransformationError(
            f"An ITK affine maps a space to a space of the same "
            f"dimension, of shape (D, D + 1), but the matrix has shape "
            f"{matrix.shape}."
        )
    space = _make_system(ndim)
    for end in ("input", "output"):
        system = getattr(affine, end)
        if system is not None and system != space:
            raise UnrepresentableTransformationError(
                f"An ITK affine maps LPS millimetres to LPS millimetres, "
                f"but the {end} of this one is {type(system).__name__}. "
                f"Convert it to ITK's space first."
            )
    return ITKStruct(
        type=ITKTransformClass.AffineTransform,
        precision=ITKPrecision.Double,
        ndim_input=ndim,
        ndim_output=ndim,
        # ITK stores the matrix row-major, then the translation. With the
        # center at the origin, ITK's translation is the affine's.
        parameters=np.concatenate([matrix[:, :-1].ravel(), matrix[:, -1]]),
        fixed_parameters=np.zeros(ndim),
    )
