import re
import sys

import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Magic

from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserContentError,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
)

from .._common import (
    ItkPrecision,
    ItkStruct,
    ItkTransformClass,
    _application_order,
)
from .._systems import _make_system

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


class MatTransformParser(
    Magic,
    BinaryFileParserWriter,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parser and writer for ITK binary MATLAB (`.mat`) transform files.

    Every block of the file is parsed into a transformation and stored
    directly in `transformations`.
    """

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a binary file holds an ITK MATLAB file.

        Only the header and the name of the first variable are needed, so the
        sniffer reads no more bytes than those can occupy.
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
        """Reject text, which is never an ITK MATLAB transform."""
        return _reject(error, "Not binary content.")

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an ITK MATLAB transform.

        The content must start with a MATLAB v4 header whose variable is named
        after an ITK transform class.
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
    def from_bytes(
        cls,
        content: bytes,
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Parse the content of an ITK MATLAB file.

        ITK writes each block as two column vectors, which are its parameters
        followed by its fixed parameters. As ITK does, this reader reads the
        variables in pairs, takes the second variable of each pair as the
        fixed parameters whatever its name, and refuses anything other than a
        column vector.

        Parameters
        ----------
        content : bytes
            The content of the file.
        position : int, optional
            The index of the top-level transform to read. A file with a
            composite holds the composite as its only top-level transform,
            and a file without one holds one top-level transform per block.
            By default the first one is read, with a warning when there are
            several.

        Returns
        -------
        Self
            The parsed object.
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
        composites = []
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
                # A composite header has no parameters of its own, and its
                # queue consists of the blocks that follow it.
                composites.append(index // 2)
                continue

            blocks.append(
                ItkStruct(
                    type=ItkTransformClass(match.group("type")),
                    precision=match.group("precision"),
                    ndim_input=int(match.group("input_dim")),
                    ndim_output=int(match.group("output_dim")),
                    parameters=parameters,
                    fixed_parameters=fixed_parameters,
                )
            )

        obj = cls()
        obj.transformations = _application_order(blocks, composites, position)
        return obj

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transformation to a file.

        The content is built before the file is opened, so that an
        unrepresentable transformation does not create or truncate the file.
        """
        content = self.to_bytes(**kwargs)
        with path.Path(filename).open(self._WRITE_MODE) as f:
            f.write(content)

    def to_bytes(
        self,
        byteorder: str = "<",
        precision: tx.Optional[tx.Union[ItkPrecision, str]] = None,
        **kwargs,
    ) -> bytes:
        """Return the content of an ITK MATLAB file for this transformation.

        The transformation must hold a single block, because ANTs writes one
        block per file. An [`ItkStruct`][] is written as it is, including its
        center. Any other transformation that can be converted to an
        [`Affine`][brainhops.datamodel.transformations.Affine] is written as an
        `AffineTransform` centered on the origin, which ITK reads back as the
        same matrix. The endpoints of that affine must be the ITK space or left
        unspecified.

        Parameters
        ----------
        byteorder : str, default="<"
            The byte order, `"<"`, `">"` or `"="` for the native order. ITK
            writes the native order, which is little-endian on common machines.
        precision : ItkPrecision or str, optional
            The precision of the parameters, `"double"` or `"float"`, which
            also appears in the class name. By default it is the precision of
            the block, or double for an affine. The fixed parameters are always
            double.

        Returns
        -------
        bytes
            The file content.

        Raises
        ------
        UnrepresentableTransformationError
            If the transformation is not a single ITK block or a single affine
            between ITK spaces.
        WriterError
            If `byteorder` is invalid.
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
        precision = ItkPrecision(precision)
        name = (
            f"{ItkTransformClass(block.type).value}_{precision.value}_"
            f"{block.ndim_input}_{block.ndim_output}"
        )
        return _write_variable(
            name, block.parameters, byteorder, precision
        ) + _write_variable(
            "fixed", block.fixed_parameters, byteorder, ItkPrecision.Double
        )


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
        return _Variable(name, dtype, count, cols, start)
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


def _single_block(chain: tx.Any) -> ItkStruct:
    """Return the single block of `chain`, converted to an ITK block if needed.

    ANTs writes a single linear transform per `.mat` file, so a chain of
    several transformations is refused.
    """
    children = list(chain.transformations or [])
    if len(children) != 1:
        raise UnrepresentableTransformationError(
            f"An ITK MATLAB file is written with a single transform, as "
            f"ANTs writes it, but this chain holds {len(children)}. "
            f"Compose them first, for example with `.compute()`."
        )
    (child,) = children
    if isinstance(child, ItkStruct):
        return child
    return _affine_block(child)


def _affine_block(xform: _xforms.Transformation) -> ItkStruct:
    """Encode an affine between ITK spaces as an origin-centered block."""
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
    return ItkStruct(
        type=ItkTransformClass.AffineTransform,
        precision=ItkPrecision.Double,
        ndim_input=ndim,
        ndim_output=ndim,
        # ITK stores the matrix in row-major order, followed by the
        # translation. With the center at the origin, the ITK translation is
        # the translation of the affine.
        parameters=np.concatenate([matrix[:, :-1].ravel(), matrix[:, -1]]),
        fixed_parameters=np.zeros(ndim),
    )
