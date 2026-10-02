"""
The shared NRRD-reading and NRRD-writing machinery behind every NRRD-based
format.

A NRRD ("Nearly Raw Raster Data") file is a text header followed by the
sample values. The header and the values are either in the same file
(*attached*, `.nrrd`) or the header names the file(s) that hold them
(*detached*, `.nhdr` with its `.raw`, `.raw.gz`, ...).

```
NRRD0004
# Complete NRRD file format specification at:
# http://teem.sourceforge.net/nrrd/format.html
type: short
dimension: 4
space: left-posterior-superior
sizes: 3 64 64 32
space directions: none (2,0,0) (0,2,0) (0,0,2.5)
kinds: vector domain domain domain
endian: little
encoding: gzip
space origin: (-63,-63,-40)
measurement frame: (1,0,0) (0,1,0) (0,0,1)
DWMRI_b-value:=1000

<data>
```

The conventions below follow the format specification
(<https://teem.sourceforge.net/nrrd/format.html>).

* **Magic.** The first line is `NRRD000X`, with `X` the version (1-5).
* **Lines.** `field: value` lines (the field names are matched without
  regard to case, and the spellings without a space -- `datafile`,
  `byteskip`, `axismins`, `centerings`, ... -- are accepted), `key:=value`
  pairs (`\\n` and `\\\\` are escapes in both), and `#` comments. The
  header ends at the first empty line, after which an attached header's
  data start; a detached header may also end at the end of its file.
* **Types.** Every spelling of the specification (`short`, `int16`,
  `int16_t`, `signed short int`, ...) of the signed and unsigned 8 to 64
  bit integers, `float` and `double`. `block` is not supported.
* **Encodings.** `raw`, `ascii` (`txt`, `text`), `hex`, `gzip` (`gz`) and
  `bzip2` (`bz2`). `endian` is required for multi-byte types in a binary
  encoding.
* **Data files.** `data file: <name>`, relative to the header's
  directory unless absolute; `data file: <format> <min> <max> <step>
  [<subdim>]`, a `printf` pattern expanded over the inclusive range; and
  `data file: LIST [<subdim>]`, followed by one file name per line until
  the end of the header. Several files are read in order and concatenated,
  each holding as many samples.
* **Skips.** `line skip` lines are skipped first, in the file as stored.
  `byte skip` bytes are then skipped -- after decompression for `gzip` and
  `bzip2` (as `pynrrd` and teem do) -- and `byte skip: -1` means that the
  data are the last bytes of the (decompressed) file.

Reading an attached `raw` file, or a detached one with a single `raw` data
file, from a local path memory-maps the values, so nothing but the header
is read until they are indexed.

!!! note "Why not `pynrrd`"
    NRRD headers are simple text, and `pynrrd` covers fewer of them than
    this parser does: it reads neither the `LIST` nor the pattern forms
    of `data file`, nor the `hex` encoding, and always reads the values
    into memory. A dedicated parser keeps the dependencies to numpy and
    the standard library (`gzip`, `bz2`).
"""

__all__ = [
    "NRRD_MAGIC",
    "SPACES",
    "NrrdHeader",
    "NrrdParser",
    "dtype_to_nrrd",
    "encode_data",
    "nrrd_dtype",
    "read_data",
]

# stdlib
import bz2
import gzip
import math
import os
import re
import zlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, Magic

# internals
from brainhops._core import path
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base.mrtrix import _local_path, _open_path, _sibling
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    WriterError,
    preserve_position,
)

NRRD_MAGIC = b"NRRD000"
"""The first seven bytes of every NRRD file; a version digit follows."""

_MAX_HEADER_LINES = 1_000_000

# ----------------------------------------------------------------------
#   VOCABULARY
# ----------------------------------------------------------------------

_TYPES = {}
for _names, _code in (
    (("signed char", "int8", "int8_t"), "i1"),
    (("uchar", "unsigned char", "uint8", "uint8_t"), "u1"),
    (
        (
            "short",
            "short int",
            "signed short",
            "signed short int",
            "int16",
            "int16_t",
        ),
        "i2",
    ),
    (
        (
            "ushort",
            "unsigned short",
            "unsigned short int",
            "uint16",
            "uint16_t",
        ),
        "u2",
    ),
    (("int", "signed int", "int32", "int32_t"), "i4"),
    (("uint", "unsigned int", "uint32", "uint32_t"), "u4"),
    (
        (
            "longlong",
            "long long",
            "long long int",
            "signed long long",
            "signed long long int",
            "int64",
            "int64_t",
        ),
        "i8",
    ),
    (
        (
            "ulonglong",
            "unsigned long long",
            "unsigned long long int",
            "uint64",
            "uint64_t",
        ),
        "u8",
    ),
    (("float",), "f4"),
    (("double",), "f8"),
):
    for _name in _names:
        _TYPES[_name] = _code

_TYPE_NAMES = {
    "i1": "int8",
    "u1": "uint8",
    "i2": "int16",
    "u2": "uint16",
    "i4": "int32",
    "u4": "uint32",
    "i8": "int64",
    "u8": "uint64",
    "f4": "float",
    "f8": "double",
}
"""The type name written for each numpy type (the `pynrrd` spelling)."""

_ENCODINGS = {
    "raw": "raw",
    "txt": "ascii",
    "text": "ascii",
    "ascii": "ascii",
    "hex": "hex",
    "gz": "gzip",
    "gzip": "gzip",
    "bz2": "bzip2",
    "bzip2": "bzip2",
}

_ALIASES = {
    "datafile": "data file",
    "lineskip": "line skip",
    "byteskip": "byte skip",
    "centerings": "centers",
    "axismins": "axis mins",
    "axismaxs": "axis maxs",
    "oldmin": "old min",
    "oldmax": "old max",
    "sampleunits": "sample units",
    "blocksize": "block size",
}

_FIELDS = (
    "content",
    "type",
    "block size",
    "dimension",
    "space",
    "space dimension",
    "sizes",
    "space directions",
    "spacings",
    "thicknesses",
    "axis mins",
    "axis maxs",
    "centers",
    "kinds",
    "labels",
    "units",
    "space units",
    "space origin",
    "measurement frame",
    "sample units",
    "number",
    "min",
    "max",
    "old min",
    "old max",
    "endian",
    "encoding",
    "line skip",
    "byte skip",
    "data file",
)
"""Every field of the specification, in the order they are written."""

SPACES = {
    "right-anterior-superior": ("RAS", 3),
    "left-anterior-superior": ("LAS", 3),
    "left-posterior-superior": ("LPS", 3),
    "right-anterior-superior-time": ("RAST", 4),
    "left-anterior-superior-time": ("LAST", 4),
    "left-posterior-superior-time": ("LPST", 4),
    "scanner-xyz": ("scanner-xyz", 3),
    "scanner-xyz-time": ("scanner-xyz-time", 4),
    "3d-right-handed": ("3D-right-handed", 3),
    "3d-left-handed": ("3D-left-handed", 3),
    "3d-right-handed-time": ("3D-right-handed-time", 4),
    "3d-left-handed-time": ("3D-left-handed-time", 4),
}
"""Each `space` of the specification: its abbreviation and dimension."""

_SPACE_NAMES = {abbr.lower(): name for name, (abbr, _) in SPACES.items()}

_NONE = ("none", "???")

# ----------------------------------------------------------------------
#   TYPES
# ----------------------------------------------------------------------


def nrrd_dtype(name: str, endian: tx.Optional[str] = None) -> np.dtype:
    """
    The numpy type of a NRRD `type`, in the byte order of `endian`
    (`"little"` or `"big"`; the native order when it is not given).

    Raises
    ------
    ParserContentError
        If the type is unknown, or is `block`.
    """
    key = " ".join(str(name).split()).lower()
    if key not in _TYPES:
        raise ParserContentError(f"Unsupported NRRD type: {name!r}")
    dtype = np.dtype(_TYPES[key])
    if endian is not None and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    return dtype


def dtype_to_nrrd(dtype: tx.Any) -> str:
    """
    The NRRD `type` of a numpy type (or of a NRRD type name).

    Booleans are stored as `uint8` and half floats as `float`.

    Raises
    ------
    WriterError
        If NRRD has no type for it (complex numbers, strings, ...).
    """
    if isinstance(dtype, str) and " ".join(dtype.split()).lower() in _TYPES:
        return _TYPE_NAMES[_TYPES[" ".join(dtype.split()).lower()]]
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return "uint8"
    if dtype.kind == "f" and dtype.itemsize < 4:
        return "float"
    code = f"{dtype.kind}{dtype.itemsize}"
    if code not in _TYPE_NAMES:
        raise WriterError(f"NRRD has no type for {dtype}.")
    return _TYPE_NAMES[code]


# ----------------------------------------------------------------------
#   VALUES
# ----------------------------------------------------------------------


def _unescape(text: str) -> str:
    return re.sub(
        r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), text
    )


def _escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("\n", "\\n")


def _float(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ParserContentError(f"Not a NRRD number: {token!r}") from None


def _format_float(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _parse_vectors(value: str) -> tx.List[tx.Optional[tx.List[float]]]:
    """`none (1,0,0) (0,1,0)` -> `[None, [1, 0, 0], [0, 1, 0]]`."""
    out: tx.List[tx.Optional[tx.List[float]]] = []
    for token in re.findall(r"\([^)]*\)|[^\s()]+", value):
        if token.lower() in _NONE:
            out.append(None)
        elif token.startswith("("):
            out.append([_float(t) for t in token[1:-1].split(",")])
        else:
            raise ParserContentError(f"Not a NRRD vector: {token!r}")
    return out


def _format_vectors(vectors: tx.Iterable[tx.Any]) -> str:
    return " ".join(
        "none"
        if v is None
        else "(" + ",".join(_format_float(x) for x in v) + ")"
        for v in vectors
    )


def _parse_strings(value: str) -> tx.List[str]:
    """`"mm" "" "s"` -> `["mm", "", "s"]`."""
    return [_unescape(s) for s in re.findall(r'"((?:[^"\\]|\\.)*)"', value)]


def _format_strings(values: tx.Iterable[str]) -> str:
    return " ".join(
        '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'
        for v in values
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


# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def _decode_part(
    file: tx.BinaryIO, start: int, header: NrrdHeader, nbytes: int
) -> bytes:
    """
    The decoded bytes of one data file (or of the data after an attached
    header), from `start`: skips, decompression, text decoding.
    """
    file.seek(start)
    for _ in range(header.line_skip):
        file.readline()
    skip = header.byte_skip
    encoding = header.encoding
    if encoding == "raw":
        if skip == -1:
            file.seek(-nbytes, os.SEEK_END)
        else:
            file.seek(skip, os.SEEK_CUR)
        return file.read(nbytes)
    content = file.read()
    if encoding == "gzip":
        content = _gunzip(content)
    elif encoding == "bzip2":
        content = bz2.decompress(content)
    if skip == -1:
        if encoding not in ("gzip", "bzip2"):
            raise ParserContentError(
                f"A NRRD byte skip of -1 needs a binary encoding, not "
                f"{encoding!r}."
            )
        return content[len(content) - nbytes :]
    content = content[skip:]
    if encoding == "ascii":
        dtype = header.dtype
        tokens = content.decode("ascii").replace(",", " ").split()
        count = nbytes // dtype.itemsize
        values = np.asarray(
            tokens[:count], dtype=float if dtype.kind == "f" else np.int64
        )
        return values.astype(dtype).tobytes()
    if encoding == "hex":
        text = b"".join(content.split()).decode("ascii")
        return bytes.fromhex(text[: 2 * nbytes])
    return content[:nbytes]


def _gunzip(content: bytes) -> bytes:
    """Decompress gzip data, ignoring anything after its last member."""
    out = []
    while content[:2] == b"\x1f\x8b":
        decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out.append(decomp.decompress(content))
        content = decomp.unused_data
    if not out:
        raise ParserContentError("The NRRD data are not gzip-compressed.")
    return b"".join(out)


def read_data(
    header: NrrdHeader,
    file: tx.Any,
    offset: int,
    mmap: bool = True,
) -> np.ndarray:
    """
    The values of a NRRD image, as an array of shape `sizes` in F order.

    Parameters
    ----------
    header : NrrdHeader
        The header.
    file : path | bytes
        The header file (path) or its content (bytes), used when the data
        are attached, and to find detached data files.
    offset : int
        The offset of the first byte after the header.
    mmap : bool
        Memory-map a single uncompressed local data file.
    """
    count, nbytes, dtype = header.count, header.nbytes, header.dtype
    if header.data_files:
        sources = []
        for name in header.data_files:
            if os.path.isabs(name):
                datafile = path.Path(name)
            elif isinstance(file, (bytes, bytearray)) or file is None:
                raise ParserContentError(
                    f"The NRRD header names a separate data file "
                    f"({name!r}), which cannot be found from a stream "
                    f"that has no file name. Read it from its path instead."
                )
            else:
                datafile = _sibling(file, name)
            if not path.exists(datafile):
                raise ParserExistsError(
                    f"No such file: {datafile} (a data file named by the "
                    f"NRRD header {file})"
                )
            sources.append((datafile, 0))
    else:
        sources = [(file, offset)]
    if nbytes % len(sources):
        raise ParserContentError(
            f"The {nbytes} bytes of the NRRD data cannot be split between "
            f"{len(sources)} data files."
        )
    part = nbytes // len(sources)

    if (
        mmap
        and len(sources) == 1
        and header.encoding == "raw"
        and not isinstance(sources[0][0], (bytes, bytearray))
        and _local_path(sources[0][0]) is not None
        and nbytes > 0
    ):
        local = _local_path(sources[0][0])
        with open(local, "rb") as f:
            f.seek(sources[0][1])
            for _ in range(header.line_skip):
                f.readline()
            if header.byte_skip == -1:
                start = os.fstat(f.fileno()).st_size - nbytes
            else:
                start = f.tell() + header.byte_skip
            size = os.fstat(f.fileno()).st_size
        if start < 0 or start + nbytes > size:
            raise ParserContentError(
                f"The NRRD data file {local} is too short for "
                f"{count} values of type {dtype}."
            )
        raw = np.memmap(
            local, dtype=np.uint8, mode="r", offset=start, shape=(nbytes,)
        )
        flat = raw.view(dtype)
    else:
        chunks = []
        for source, start in sources:
            if isinstance(source, (bytes, bytearray)):
                chunks.append(
                    _decode_part(BytesIO(source), start, header, part)
                )
            else:
                with _open_path(source) as f:
                    chunks.append(_decode_part(f, start, header, part))
            if len(chunks[-1]) < part:
                raise ParserContentError(
                    f"The NRRD data hold {len(chunks[-1])} bytes, but "
                    f"{part} are needed."
                )
        flat = np.frombuffer(b"".join(chunks), dtype=dtype, count=count)
    return flat.reshape(header.sizes, order="F")


def encode_data(header: NrrdHeader, data: tx.Any) -> bytes:
    """
    Encode an array of shape `sizes` (indexed in NRRD axis order) into the
    bytes of the data file, in the header's type, byte order and encoding.
    """
    array = np.asarray(data)
    if tuple(array.shape) != tuple(header.sizes):
        raise WriterError(
            f"The data have shape {array.shape}, but the header says "
            f"{header.sizes}."
        )
    dtype = header.dtype
    if dtype.kind in "iu" and array.dtype.kind in "fc":
        array = np.round(np.real(array))
    flat = np.asarray(array, dtype=dtype).ravel(order="F")
    encoding = header.encoding
    if encoding == "ascii":
        width = header.sizes[0] if header.sizes else 1
        if dtype.kind == "f":
            words = [_format_float(v) for v in flat.tolist()]
        else:
            words = [str(v) for v in flat.tolist()]
        rows = [
            " ".join(words[i : i + width]) for i in range(0, len(words), width)
        ]
        return ("\n".join(rows) + "\n").encode("ascii")
    raw = flat.tobytes()
    if encoding == "hex":
        text = raw.hex()
        return (
            "\n".join(text[i : i + 64] for i in range(0, len(text), 64)) + "\n"
        ).encode("ascii")
    if encoding == "gzip":
        return gzip.compress(raw)
    if encoding == "bzip2":
        return bz2.compress(raw)
    return raw


_DATA_EXTENSIONS = {
    "raw": ".raw",
    "gzip": ".raw.gz",
    "bzip2": ".raw.bz2",
    "ascii": ".txt",
    "hex": ".hex",
}


# ----------------------------------------------------------------------
#   PARSER
# ----------------------------------------------------------------------


class NrrdParser(DataModelBase, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by a NRRD file.

    It reads and writes the container -- the header and the sample
    values -- for every NRRD-based format. What the values mean is for
    the concrete format to say, through `_nrrd_header` and `_nrrd_data`
    when writing.

    Reading a `raw` file from a local path memory-maps the values, so
    nothing but the header is read until they are indexed.
    """

    HINTS = ("nrrd",)

    _header: tx.Annotated[
        tx.Optional[NrrdHeader],
        tx.Doc(
            """
            The NRRD header this object was read from, kept so that the
            fields and key/value pairs the data model has no slot for
            (`measurement frame`, `DWMRI_gradient_0000`, ...) are written
            back.
            """
        ),
    ] = None

    dataobj: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The stored values, as an array of the file's axes (fastest
            first, F order). A view of a memory-mapped file when the file
            allows one.
            """
        ),
    ] = None

    @property
    def header(self) -> tx.Optional[NrrdHeader]:
        """The NRRD header this object was read from, if any."""
        return getattr(self, "_header", None)

    @header.setter
    def header(self, value: tx.Optional[NrrdHeader]) -> None:
        self._header = value

    # --- reading ------------------------------------------------------

    @classmethod
    def _from_header(
        cls, header: NrrdHeader, dataobj: tx.Any, **kwargs
    ) -> tx.Self:
        """Build the object from a decoded header and its stored values."""
        return cls(header=header, dataobj=dataobj, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Build the object from a NRRD file (path or file object)."""
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
        Build the object from the path of a `.nrrd` or `.nhdr` file.

        The values of a `raw` local data file are memory-mapped unless
        `mmap` is false. Detached data files are found relative to the
        header's directory.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        with _open_path(filename) as f:
            header, offset = NrrdHeader.from_fileobj(f)
        data = read_data(header, filename, offset, mmap=mmap)
        return cls._from_header(header, data, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """
        Build the object from an open NRRD file object.

        Detached data files are resolved against the stream's `name`,
        when it has one.
        """
        kwargs.pop("mmap", None)
        name = getattr(file, "name", None)
        with preserve_position(file):
            content = file.read()
        header, offset = NrrdHeader.from_fileobj(BytesIO(content))
        if header.data_files and isinstance(name, (str, os.PathLike)):
            source: tx.Any = path.Path(os.fsdecode(name))
        else:
            source = content
        data = read_data(header, source, offset, mmap=False)
        return cls._from_header(header, data, **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build the object from the bytes of an attached NRRD file."""
        kwargs.pop("mmap", None)
        content = bytes(content)
        header, offset = NrrdHeader.from_fileobj(BytesIO(content))
        data = read_data(header, content, offset, mmap=False)
        return cls._from_header(header, data, **kwargs)

    # --- sniffing -----------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that a stream holds a NRRD
        file of its own kind."""
        base_error = None
        score = Confidence.NO
        try:
            with preserve_position(file):
                header, _ = NrrdHeader.from_fileobj(file)
                score = cls._score_header(header)
        except Exception as e:  # noqa: BLE001
            base_error = e
            score = Confidence.NO
        if score:
            return score
        if error:
            if error is True:
                error = SnifferContentError
            raise error(f"Content is not a {cls.__name__}") from base_error
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that bytes hold a NRRD file of
        its own kind."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: NrrdHeader) -> float:
        """
        How well a valid NRRD header matches *this* class.

        Called once the header has been parsed, so the answer is never
        "not NRRD". A concrete format overrides it.
        """
        return Confidence.MAYBE

    # --- writing ------------------------------------------------------

    def _nrrd_header(self, **kwargs) -> NrrdHeader:
        """The header to write. Each concrete format builds its own."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to NRRD."
        )

    def _nrrd_data(self, header: NrrdHeader) -> tx.Any:
        """The array, in the header's axis order, to write."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to NRRD."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """The bytes of an attached NRRD file."""
        kwargs.pop("data_file", None)
        header = self._nrrd_header(**kwargs)
        data = encode_data(header, self._nrrd_data(header))
        return header.to_text().encode("utf-8") + b"\n" + data

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write an attached NRRD file to a stream."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(
        self,
        filename: path.FilenameLike,
        data_file: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Write to a path: an attached file, or, for a `.nhdr` name (or when
        `data_file` is given), a detached header and its data file.

        The data file of a detached header is named after it, with an
        extension that says its encoding (`.raw`, `.raw.gz`, `.raw.bz2`,
        `.txt`, `.hex`), unless `data_file` names it (relative to the
        header's directory).
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        name = filename.name
        if not name.lower().endswith(".nhdr") and data_file is None:
            with filename.open("wb") as f:
                f.write(self.to_bytes(**kwargs))
            return
        header = self._nrrd_header(**kwargs)
        data = encode_data(header, self._nrrd_data(header))
        if data_file is None:
            base = (
                name[: -len(".nhdr")]
                if name.lower().endswith(".nhdr")
                else name
            )
            data_file = base + _DATA_EXTENSIONS[header.encoding]
        with _sibling(filename, data_file).open("wb") as f:
            f.write(data)
        header = NrrdHeader(
            version=header.version,
            fields=header.fields,
            keyvalue=header.keyvalue,
            data_files=(data_file,),
        )
        with filename.open("wb") as f:
            f.write(header.to_text().encode("utf-8"))

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write to a path (variant chosen by extension) or a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)
