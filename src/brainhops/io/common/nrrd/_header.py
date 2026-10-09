"""The NRRD header."""

# stdlib
import math
import re

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, Magic

# internals
from brainhops.io.base.parsers import ParserContentError

# this format
from ._codecs import (
    _escape,
    _float,
    _parse_strings,
    _parse_vectors,
    _unescape,
    nrrd_dtype,
)
from ._constants import (
    _ALIASES,
    _ENCODINGS,
    _FIELDS,
    _MAX_HEADER_LINES,
    _NONE,
    _SPACE_NAMES,
    NRRD_MAGIC,
    SPACES,
)

# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


class NrrdHeader(Magic, frozen=True):
    """
    A NRRD header: its fields and key/value pairs, as written.

    The fields are kept as text, under their canonical names (`"data
    file"`, `"space directions"`, ...), so that every one of them is
    written back as it was read. The methods decode the ones that the
    readers need.
    """

    version: int = 5
    """The version of the magic line (`NRRD000X`)."""

    fields: tx.Dict[str, str] = Factory(dict)
    """The fields, by canonical name, as text, in the order they were
    read. The file names of a `data file: LIST` are in `data_files`."""

    keyvalue: tx.Dict[str, str] = Factory(dict)
    """The `key:=value` pairs, unescaped."""

    data_files: tx.Tuple[str, ...] = ()
    """The data files of a detached header, in order (the `LIST` and
    pattern forms expanded); empty for an attached header."""

    # --- decoded fields -----------------------------------------------

    def _field(self, name: str) -> tx.Optional[str]:
        value = self.fields.get(name)
        return None if value is None else value.strip()

    @property
    def dimension(self) -> int:
        """The number of axes."""
        value = self._field("dimension")
        if value is None:
            raise ParserContentError("The NRRD header has no 'dimension'.")
        return int(value)

    @property
    def sizes(self) -> tx.Tuple[int, ...]:
        """The number of samples along each axis, fastest first."""
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
        """The canonical encoding: raw, ascii, hex, gzip or bzip2."""
        value = (self._field("encoding") or "").lower()
        if value not in _ENCODINGS:
            raise ParserContentError(f"Unsupported NRRD encoding: {value!r}")
        return _ENCODINGS[value]

    @property
    def endian(self) -> tx.Optional[str]:
        """`"little"`, `"big"`, or `None` when the header says nothing."""
        value = self._field("endian")
        return None if value is None else value.lower()

    @property
    def dtype(self) -> np.dtype:
        """The numpy type of the stored values, in their byte order."""
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
        """The number of samples."""
        return int(np.prod(self.sizes, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        """The number of bytes of the (decoded) values."""
        return self.count * self.dtype.itemsize

    @property
    def line_skip(self) -> int:
        """The number of lines to skip before the data."""
        value = int(self._field("line skip") or 0)
        if value < 0:
            raise ParserContentError(f"Invalid NRRD line skip: {value}")
        return value

    @property
    def byte_skip(self) -> int:
        """The number of bytes to skip before the data, or -1."""
        value = int(self._field("byte skip") or 0)
        if value < -1:
            raise ParserContentError(f"Invalid NRRD byte skip: {value}")
        return value

    @property
    def space(self) -> tx.Optional[str]:
        """The canonical `space` (`"left-posterior-superior"`, ...), or
        `None` when the header has none."""
        value = self._field("space")
        if value is None:
            return None
        key = value.lower()
        key = _SPACE_NAMES.get(key, key)
        if key not in SPACES:
            raise ParserContentError(f"Unknown NRRD space: {value!r}")
        return key

    @property
    def space_dimension(self) -> tx.Optional[int]:
        """The dimension of the world space, from `space` or `space
        dimension`; `None` when the header has neither."""
        space = self.space
        if space is not None:
            return SPACES[space][1]
        value = self._field("space dimension")
        return None if value is None else int(value)

    @property
    def space_directions(
        self,
    ) -> tx.Optional[tx.List[tx.Optional[np.ndarray]]]:
        """One vector per axis (`None` for a non-spatial axis), or `None`
        when the header has no `space directions`."""
        value = self._field("space directions")
        if value is None:
            return None
        vectors = _parse_vectors(value)
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
        """The position of the centre of the first sample, or `None`."""
        value = self._field("space origin")
        if value is None:
            return None
        vectors = _parse_vectors(value)
        if len(vectors) != 1 or vectors[0] is None:
            return None
        return np.asarray(vectors[0], dtype=float)

    @property
    def measurement_frame(self) -> tx.Optional[np.ndarray]:
        """The measurement frame, one vector per column, or `None`."""
        value = self._field("measurement frame")
        if value is None:
            return None
        vectors = [v for v in _parse_vectors(value) if v is not None]
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
                name, lambda s: [_float(t) for t in s.split()]
            )
        ]

    def _words(self, name: str) -> tx.List[tx.Optional[str]]:
        return [
            None if v is None or v.lower() in _NONE else v.lower()
            for v in self._per_axis(name, str.split)
        ]

    @property
    def kinds(self) -> tx.List[tx.Optional[str]]:
        """The kind of each axis, in lower case (`None` when unknown)."""
        return self._words("kinds")

    @property
    def centers(self) -> tx.List[tx.Optional[str]]:
        """The centering of each axis: `"cell"`, `"node"` or `None`."""
        return self._words("centers")

    @property
    def spacings(self) -> tx.List[float]:
        """The spacing of each axis (`nan` when unknown)."""
        return self._floats("spacings")

    @property
    def axis_mins(self) -> tx.List[float]:
        """The `axis mins` (`nan` when unknown)."""
        return self._floats("axis mins")

    @property
    def axis_maxs(self) -> tx.List[float]:
        """The `axis maxs` (`nan` when unknown)."""
        return self._floats("axis maxs")

    @property
    def units(self) -> tx.List[tx.Optional[str]]:
        """The unit of each axis (`None` when not given)."""
        return [v or None for v in self._per_axis("units", _parse_strings)]

    @property
    def space_units(self) -> tx.List[tx.Optional[str]]:
        """The unit of each world axis (`None` when not given)."""
        value = self._field("space units")
        if value is None:
            return [None] * (self.space_dimension or 0)
        return [v or None for v in _parse_strings(value)]

    # --- parsing ------------------------------------------------------

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], version: int = 5
    ) -> "NrrdHeader":
        """
        Parse the lines of a header, after its magic line and up to the
        empty line that ends it.
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
                name = _ALIASES.get(name, name)
            if (
                field is not None
                and name in _FIELDS
                and ":=" not in line[: field.end(1) + 2]
            ):
                value = field.group(2)
                fields[name] = value
                if name == "data file":
                    files, listing = _data_files(value)
                    data_files.extend(files)
            elif ":=" in line:
                key, value = line.split(":=", 1)
                keyvalue[_unescape(key)] = _unescape(value)
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
    def from_fileobj(cls, file: tx.BinaryIO) -> tx.Tuple["NrrdHeader", int]:
        """
        Read a header from a binary stream, from its magic line.

        Returns the header and the offset, relative to the start of the
        stream, of the first byte after the header (where attached data
        start).
        """
        start = file.tell()
        magic = file.readline(64)
        if not (
            magic[:7] == NRRD_MAGIC
            and magic[7:8].isdigit()
            and magic[8:].strip() == b""
        ):
            raise ParserContentError("Not a NRRD file (no NRRD000X magic).")
        version = int(magic[7:8])
        lines = []
        for _ in range(_MAX_HEADER_LINES):
            line = file.readline()
            if not line or line in (b"\n", b"\r\n"):
                break
            try:
                lines.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                lines.append(line.decode("latin-1"))
        else:
            raise ParserContentError("The NRRD header does not end.")
        header = cls.from_lines(lines, version=version)
        return header, file.tell() - start

    # --- writing ------------------------------------------------------

    def to_text(self) -> str:
        """The text of the header, from its magic line, without the empty
        line that separates it from attached data."""
        lines = [f"NRRD000{self.version}"]
        for name in _FIELDS:
            if name in self.fields and name != "data file":
                lines.append(f"{name}: {self.fields[name]}")
        for key, value in self.keyvalue.items():
            lines.append(f"{_escape(key)}:={_escape(value)}")
        if self.data_files:
            if len(self.data_files) == 1:
                lines.append(f"data file: {self.data_files[0]}")
            else:
                lines.append("data file: LIST")
                lines.extend(self.data_files)
        return "\n".join(lines) + "\n"


def _data_files(value: str) -> tx.Tuple[tx.List[str], bool]:
    """
    The files named by a `data file` value, and whether the names follow
    on the next lines (`LIST`).
    """
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
