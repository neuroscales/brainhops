"""The header of a NetCDF classic file, which is the container of MINC1.

A NetCDF classic file starts with a header that lists its dimensions,
its global attributes and its variables, with the attributes of each
variable and the position of its values in the file. The values follow
the header. The reader below reads the header alone, from a stream that
it reads in order, so that it also reads a gzipped file through a
decompressing stream and a file cut after its header. The layout follows
the NetCDF classic format specification, versions 1 and 2:
<https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html>.
"""

# stdlib
import struct

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import ParserContentError

_ABSENT = 0
_NC_DIMENSION = 0x0A
_NC_VARIABLE = 0x0B
_NC_ATTRIBUTE = 0x0C

_NC_CHAR = 2

_TYPES = {
    1: np.dtype(">i1"),
    2: np.dtype("S1"),
    3: np.dtype(">i2"),
    4: np.dtype(">i4"),
    5: np.dtype(">f4"),
    6: np.dtype(">f8"),
}
"""The type of each NetCDF type code, in the big-endian order of the file."""


class NetcdfVariable(tx.NamedTuple):
    """A variable of a NetCDF classic file, as its header describes it."""

    name: str
    """The name of the variable."""

    dimensions: tx.Tuple[str, ...]
    """The names of its dimensions, slowest first."""

    attributes: tx.Dict[str, tx.Any]
    """Its attributes, as strings or arrays."""

    dtype: np.dtype
    """The type of its values, in the big-endian order of the file."""

    begin: int
    """The position of its values in the file."""


class NetcdfHeader(tx.NamedTuple):
    """The header of a NetCDF classic file."""

    dimensions: tx.Dict[str, int]
    """The length of each dimension, with 0 for the record dimension."""

    records: int
    """The number of records along the record dimension."""

    attributes: tx.Dict[str, tx.Any]
    """The global attributes, as strings or arrays."""

    variables: tx.Dict[str, NetcdfVariable]
    """The variables, by name, in file order."""

    size: int
    """The number of bytes of the header, where the values may begin."""


class _Reader:
    """Read big-endian values from a stream, in order, counting bytes."""

    def __init__(self, file: tx.BinaryIO) -> None:
        self.file = file
        self.count = 0

    def bytes(self, size: int) -> bytes:
        content = self.file.read(size)
        if len(content) != size:
            raise ParserContentError("The NetCDF header ends too early.")
        self.count += size
        return content

    def int(self) -> int:
        return struct.unpack(">i", self.bytes(4))[0]

    def offset(self, version: int) -> int:
        if version == 1:
            return struct.unpack(">i", self.bytes(4))[0]
        return struct.unpack(">q", self.bytes(8))[0]

    def padded(self, size: int) -> bytes:
        content = self.bytes(size)
        self.bytes(-size % 4)
        return content

    def name(self) -> str:
        return self.padded(self.int()).decode("utf-8", "replace")


def read_netcdf_header(file: tx.BinaryIO) -> NetcdfHeader:
    """Read the header of a NetCDF classic file, without its values.

    The stream is read in order from its current position, which must be
    the start of the file.

    Parameters
    ----------
    file : file object
        The stream, open in binary mode and not compressed.

    Returns
    -------
    NetcdfHeader
        The dimensions, the number of records, the global attributes
        and the variables of the file.

    Raises
    ------
    ParserContentError
        If the stream does not hold a NetCDF classic header.
    """
    reader = _Reader(file)
    magic = reader.bytes(4)
    if magic[:3] != b"CDF" or magic[3] not in (1, 2):
        raise ParserContentError("Not a NetCDF classic file.")
    version = magic[3]
    records = reader.int()
    dimensions: tx.Dict[str, int] = {}
    names = []
    for _ in range(_count(reader, _NC_DIMENSION)):
        name = reader.name()
        dimensions[name] = reader.int()
        names.append(name)
    attributes = _attributes(reader)
    variables: tx.Dict[str, NetcdfVariable] = {}
    for _ in range(_count(reader, _NC_VARIABLE)):
        name = reader.name()
        ids = [reader.int() for _ in range(reader.int())]
        try:
            dims = tuple(names[i] for i in ids)
        except IndexError:
            raise ParserContentError(
                f"The NetCDF variable {name!r} names an unknown dimension."
            ) from None
        var_attributes = _attributes(reader)
        dtype = _dtype(reader.int())
        reader.int()  # The size of the values, which the reader recomputes.
        begin = reader.offset(version)
        variables[name] = NetcdfVariable(
            name, dims, var_attributes, dtype, begin
        )
    return NetcdfHeader(
        dimensions, records, attributes, variables, reader.count
    )


def _count(reader: _Reader, tag: int) -> int:
    """Read the tag and the length of a list, which may be absent."""
    found, count = reader.int(), reader.int()
    if found == _ABSENT and count == 0:
        return 0
    if found != tag:
        raise ParserContentError("The NetCDF header is malformed.")
    return count


def _dtype(code: int) -> np.dtype:
    """Return the type of a NetCDF type code."""
    if code not in _TYPES:
        raise ParserContentError(f"Unknown NetCDF type: {code}")
    return _TYPES[code]


def _attributes(reader: _Reader) -> tx.Dict[str, tx.Any]:
    """Read a list of attributes: text as strings, numbers as arrays.

    Trailing NUL characters are removed from text.
    """
    attributes: tx.Dict[str, tx.Any] = {}
    for _ in range(_count(reader, _NC_ATTRIBUTE)):
        name = reader.name()
        code = reader.int()
        dtype = _dtype(code)
        count = reader.int()
        content = reader.padded(count * dtype.itemsize)
        if code == _NC_CHAR:
            # Text is often padded with a NUL, which is not part of it.
            attributes[name] = content.decode("latin-1").rstrip("\0")
        else:
            values = np.frombuffer(content, dtype=dtype, count=count)
            attributes[name] = values.astype(dtype.newbyteorder("="))
    return attributes
