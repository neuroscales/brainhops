"""The record of a NRRD image: its header."""

# stdlib
import math
import re
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, NoRepr

# internals
from brainhops._core.streams import preserve_position
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._internal_api import (
    Raw,
    declined,
    no_options,
    sniffed_raw,
)
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
)

# this format
from ._codecs import (
    escape,
    nrrd_dtype,
    parse_float,
    parse_strings,
    parse_vectors,
    unescape,
)
from ._constants import (
    ALIASES,
    ENCODINGS,
    FIELDS,
    MAX_HEADER_LINES,
    NONE,
    NRRD_MAGIC,
    SPACE_NAMES,
    SPACES,
)

OPTION_OWNER = "NRRD reader"
"""What refuses unknown options, as the error message names it."""

DECLINE_MESSAGE = "Content is not a NRRD image"
"""The message of the error raised when a sniffer declines a stream."""

# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


class NrrdRaw(Raw, BinaryFileReader, BinaryFileWriter, eq=False):
    """The header of a NRRD image, as the file stores it.

    A NRRD image is a text header followed by the samples, either in the
    same `.nrrd` file or in the data files that a `.nhdr` header names.
    The record holds the header: the fields as text, by their canonical
    name, the key/value pairs, and the names of the data files. The
    properties decode the fields that describe the samples and the
    geometry. The record also keeps the bytes that the header was read
    from, so that a header that is read and written again keeps its
    bytes, comments included.

    The record knows where the samples lie and how they are encoded, but
    it never reads them. It is held by
    [`NrrdMetadata`][brainhops.io.common.nrrd.NrrdMetadata], which never
    changes it in place: a writer works on the copy that [`copy`][]
    returns, or builds a new record. Two records are equal when they
    format to the same text, whatever bytes they were read from.
    """

    version: int = 5
    """Format version, from the magic line."""

    fields: tx.Dict[str, str] = Factory(dict)
    """Fields as text, by canonical name, in reading order."""

    keyvalue: tx.Dict[str, str] = Factory(dict)
    """Unescaped `key:=value` pairs."""

    data_files: tx.Tuple[str, ...] = ()
    """Detached data files in order; empty when the data are attached."""

    stored: NoRepr[tx.Optional[bytes]] = None
    """The bytes that the header was read from, or `None`.

    The bytes run from the magic line to the empty line that ends the
    header, or to the end of a detached header that has no empty line.
    They are written back as they are while they still describe the
    record, whichever program wrote them, and the record is formatted as
    the writer spells it otherwise. The bytes are `None` for a record
    that was not read from a file.
    """

    def _field(self, name: str) -> tx.Optional[str]:
        value = self.fields.get(name)
        return None if value is None else value.strip()

    @property
    def dimension(self) -> int:
        """Number of axes."""
        value = self._field("dimension")
        if value is None:
            raise ParserContentError("The NRRD header has no 'dimension'.")
        return int(value)

    @property
    def sizes(self) -> tx.Tuple[int, ...]:
        """Number of samples along each axis, fastest axis first."""
        value = self._field("sizes")
        if value is None:
            raise ParserContentError("The NRRD header has no 'sizes'.")
        sizes = tuple(int(v) for v in value.split())
        if len(sizes) != self.dimension:
            raise ParserContentError(
                f"The NRRD header has {len(sizes)} sizes for "
                f"{self.dimension} dimensions."
            )
        return sizes

    @property
    def encoding(self) -> str:
        """Canonical encoding: `raw`, `ascii`, `hex`, `gzip` or `bzip2`."""
        value = (self._field("encoding") or "").lower()
        if value not in ENCODINGS:
            raise ParserContentError(f"Unsupported NRRD encoding: {value!r}")
        return ENCODINGS[value]

    @property
    def endian(self) -> tx.Optional[str]:
        """Lower-cased byte order, or `None` if absent."""
        value = self._field("endian")
        return None if value is None else value.lower()

    @property
    def dtype(self) -> np.dtype:
        """NumPy data type of the stored values, with the byte order.

        Multi-byte types in a binary encoding require an `endian` field.
        """
        value = self._field("type")
        if value is None:
            raise ParserContentError("The NRRD header has no 'type'.")
        dtype = nrrd_dtype(value)
        binary = self.encoding in ("raw", "gzip", "bzip2", "hex")
        if dtype.itemsize > 1 and binary:
            endian = self.endian
            if endian not in ("little", "big"):
                raise ParserContentError(
                    "The NRRD header needs an 'endian' of 'little' or "
                    f"'big' for its {value!r} values, not {endian!r}."
                )
            dtype = nrrd_dtype(value, endian)
        return dtype

    @property
    def count(self) -> int:
        """Total number of samples."""
        return int(np.prod(self.sizes, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        """Size of the decoded values in bytes."""
        return self.count * self.dtype.itemsize

    @property
    def line_skip(self) -> int:
        """Number of lines to skip before the data."""
        value = int(self._field("line skip") or 0)
        if value < 0:
            raise ParserContentError(f"Invalid NRRD line skip: {value}")
        return value

    @property
    def byte_skip(self) -> int:
        """Number of bytes to skip before the data.

        A value of -1 means that the data are the last bytes of the file.
        """
        value = int(self._field("byte skip") or 0)
        if value < -1:
            raise ParserContentError(f"Invalid NRRD byte skip: {value}")
        return value

    @property
    def space(self) -> tx.Optional[str]:
        """Canonical space name, or `None`; abbreviations are accepted."""
        value = self._field("space")
        if value is None:
            return None
        key = value.lower()
        key = SPACE_NAMES.get(key, key)
        if key not in SPACES:
            raise ParserContentError(f"Unknown NRRD space: {value!r}")
        return key

    @property
    def space_dimension(self) -> tx.Optional[int]:
        """Dimension of the world space, or `None` if unknown."""
        space = self.space
        if space is not None:
            return SPACES[space][1]
        value = self._field("space dimension")
        return None if value is None else int(value)

    @property
    def space_directions(
        self,
    ) -> tx.Optional[tx.List[tx.Optional[np.ndarray]]]:
        """Direction vector of each axis, or `None` if the field is absent.

        The vector of a non-spatial axis is `None`.
        """
        value = self._field("space directions")
        if value is None:
            return None
        vectors = parse_vectors(value)
        if len(vectors) != self.dimension:
            raise ParserContentError(
                f"The NRRD header has {len(vectors)} space directions for "
                f"{self.dimension} dimensions."
            )
        sdim = self.space_dimension
        out = []
        for v in vectors:
            if v is not None:
                v = np.asarray(v, dtype=float)
                if sdim is not None and v.size != sdim:
                    raise ParserContentError(
                        f"A NRRD space direction has {v.size} components "
                        f"in a {sdim}-dimensional space."
                    )
            out.append(v)
        return out

    @property
    def space_origin(self) -> tx.Optional[np.ndarray]:
        """Position of the first sample, or `None`."""
        value = self._field("space origin")
        if value is None:
            return None
        vectors = parse_vectors(value)
        if len(vectors) != 1 or vectors[0] is None:
            return None
        return np.asarray(vectors[0], dtype=float)

    @property
    def measurement_frame(self) -> tx.Optional[np.ndarray]:
        """Measurement frame with one vector per column, or `None`."""
        value = self._field("measurement frame")
        if value is None:
            return None
        vectors = [v for v in parse_vectors(value) if v is not None]
        return np.asarray(vectors, dtype=float).T

    def _per_axis(self, name: str, parse: tx.Callable) -> tx.List[tx.Any]:
        value = self._field(name)
        if value is None:
            return [None] * self.dimension
        values = parse(value)
        if len(values) != self.dimension:
            raise ParserContentError(
                f"The NRRD header has {len(values)} {name} for "
                f"{self.dimension} dimensions."
            )
        return values

    def _floats(self, name: str) -> tx.List[float]:
        return [
            math.nan if v is None else v
            for v in self._per_axis(
                name, lambda s: [parse_float(t) for t in s.split()]
            )
        ]

    def _words(self, name: str) -> tx.List[tx.Optional[str]]:
        return [
            None if v is None or v.lower() in NONE else v.lower()
            for v in self._per_axis(name, str.split)
        ]

    @property
    def kinds(self) -> tx.List[tx.Optional[str]]:
        """Lower-cased kind of each axis, or `None`."""
        return self._words("kinds")

    @property
    def centers(self) -> tx.List[tx.Optional[str]]:
        """Lower-cased centering of each axis, or `None`."""
        return self._words("centers")

    @property
    def spacings(self) -> tx.List[float]:
        """Spacing of each axis, or NaN."""
        return self._floats("spacings")

    @property
    def axis_mins(self) -> tx.List[float]:
        """Minimum position along each axis, or NaN."""
        return self._floats("axis mins")

    @property
    def axis_maxs(self) -> tx.List[float]:
        """Maximum position along each axis, or NaN."""
        return self._floats("axis maxs")

    @property
    def units(self) -> tx.List[tx.Optional[str]]:
        """Unit of each axis, or `None`."""
        return [v or None for v in self._per_axis("units", parse_strings)]

    @property
    def space_units(self) -> tx.List[tx.Optional[str]]:
        """Unit of each world axis, or `None`."""
        value = self._field("space units")
        if value is None:
            return [None] * (self.space_dimension or 0)
        return [v or None for v in parse_strings(value)]

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], version: int = 5
    ) -> "NrrdRaw":
        """Build a record from the lines that follow the magic line.

        The record keeps no bytes, since lines do not say how the header
        was spelled.

        Parameters
        ----------
        lines : iterable of str
            The lines of the header after the magic line, without their
            line breaks.
        version : int, default=5
            The version that the magic line states.

        Returns
        -------
        NrrdRaw
            The record.

        Raises
        ------
        ParserContentError
            If a line cannot be parsed or a required field is missing.
        """
        fields: tx.Dict[str, str] = {}
        keyvalue: tx.Dict[str, str] = {}
        data_files: tx.List[str] = []
        listing = False
        for line in lines:
            line = line.rstrip("\r\n")
            if listing:
                if line.strip():
                    data_files.append(line.strip())
                continue
            if line.startswith("#"):
                continue
            field = re.match(r"([^:]+): ?(.*)$", line)
            name = None
            if field is not None:
                name = " ".join(field.group(1).split()).lower()
                name = ALIASES.get(name, name)
            if (
                field is not None
                and name in FIELDS
                and ":=" not in line[: field.end(1) + 2]
            ):
                value = field.group(2)
                fields[name] = value
                if name == "data file":
                    files, listing = _data_files(value)
                    data_files.extend(files)
            elif ":=" in line:
                key, value = line.split(":=", 1)
                keyvalue[unescape(key)] = unescape(value)
            else:
                raise ParserContentError(f"Not a NRRD header line: {line!r}")
        for name in ("type", "dimension", "sizes", "encoding"):
            if name not in fields:
                raise ParserContentError(
                    f"The NRRD header has no {name!r} field."
                )
        return cls(
            version=version,
            fields=fields,
            keyvalue=keyvalue,
            data_files=tuple(data_files),
        )

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "NrrdRaw":
        """Read the header of a NRRD stream, positioned at the magic line.

        The samples are not read, and the position of the stream is
        restored. The record keeps the bytes of the header, whose length
        is where attached samples begin. The record has no options.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        NrrdRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream does not hold a NRRD header.
        """
        no_options(kwargs, OPTION_OWNER)
        with preserve_position(file):
            return _read_header(file)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a NRRD header.

        The header is parsed, and the position of the stream is restored.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no header.
            With `True`, the error is a `SnifferContentError`, and an
            exception class is raised instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            `Confidence.LIKELY` or `Confidence.NO`.

        Raises
        ------
        SnifferContentError
            If the stream holds no header and `error` is `True`.
        """
        raw, failure = sniffed_raw(NrrdRaw.from_fileobj, file)
        if raw is not None:
            return Confidence.LIKELY
        return declined(error, DECLINE_MESSAGE, failure)

    @takes_error_policy
    def to_text(self) -> str:
        """Return the header text, without the empty line that ends it.

        Returns
        -------
        str
            The text, from the magic line to the last field.
        """
        lines = [f"NRRD000{self.version}"]
        for name in FIELDS:
            if name in self.fields and name != "data file":
                lines.append(f"{name}: {self.fields[name]}")
        for key, value in self.keyvalue.items():
            lines.append(f"{escape(key)}:={escape(value)}")
        if self.data_files:
            if len(self.data_files) == 1:
                lines.append(f"data file: {self.data_files[0]}")
            else:
                lines.append("data file: LIST")
                lines.extend(self.data_files)
        return "\n".join(lines) + "\n"

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the header as the file stores it.

        The bytes that the header was read from are returned while they
        still describe the record. Otherwise, the header is its text, and
        the header of attached samples ends with an empty line, after
        which the samples begin.

        Parameters
        ----------
        **kwargs : Any
            Ignored.

        Returns
        -------
        bytes
            The header.
        """
        if self.stored is not None and _parsed(self.stored) == self:
            return self.stored
        text = self.to_text()
        if not self.data_files:
            text += "\n"
        return text.encode("utf-8")

    def copy(self) -> "NrrdRaw":
        """Return a copy of the record that can be changed freely.

        The fields and the key/value pairs are copied into new
        dictionaries. The bytes that the header was read from are kept,
        since [`to_bytes`][] writes them only while they still describe
        the record.

        Returns
        -------
        NrrdRaw
            The copy.
        """
        return NrrdRaw(
            version=self.version,
            fields=dict(self.fields),
            keyvalue=dict(self.keyvalue),
            data_files=tuple(self.data_files),
            stored=self.stored,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NrrdRaw):
            return NotImplemented
        return self.to_text() == other.to_text()

    __hash__ = None


def _read_header(file: tx.BinaryIO) -> NrrdRaw:
    """Read a header from a stream positioned at its magic line.

    The stream is left just after the empty line that ends the header,
    where attached samples begin. The record keeps the bytes it was read
    from, empty line included.

    Raises
    ------
    ParserContentError
        If the stream does not hold a NRRD header.
    """
    magic = file.readline(64)
    if not (
        magic[:7] == NRRD_MAGIC
        and magic[7:8].isdigit()
        and magic[8:].strip() == b""
    ):
        raise ParserContentError("Not a NRRD file (no NRRD000X magic).")
    version = int(magic[7:8])
    stored = [magic]
    lines = []
    for _ in range(MAX_HEADER_LINES):
        line = file.readline()
        stored.append(line)
        if not line or line in (b"\n", b"\r\n"):
            break
        try:
            lines.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append(line.decode("latin-1"))
    else:
        raise ParserContentError("The NRRD header does not end.")
    record = NrrdRaw.from_lines(lines, version=version)
    return NrrdRaw(
        version=record.version,
        fields=record.fields,
        keyvalue=record.keyvalue,
        data_files=record.data_files,
        stored=b"".join(stored),
    )


def _parsed(stored: bytes) -> NrrdRaw:
    """Parse the bytes of a header."""
    return _read_header(BytesIO(stored))


def _data_files(value: str) -> tx.Tuple[tx.List[str], bool]:
    """Return the names in a `data file` value and whether a `LIST` follows."""
    parts = value.split()
    if parts and parts[0] == "LIST":
        return [], True
    if len(parts) in (4, 5) and "%" in parts[0]:
        try:
            first, last, step = (int(p) for p in parts[1:4])
        except ValueError:
            pass
        else:
            if step == 0:
                raise ParserContentError("A NRRD data file step of 0.")
            stop = last + (1 if step > 0 else -1)
            return [parts[0] % i for i in range(first, stop, step)], False
    return [value.strip()], False
