"""The AFNI `.HEAD` header: its attributes, read and written."""

# stdlib
import re
from collections import OrderedDict

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

# internals
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base.parsers import ParserContentError

# this format
from ._constants import (
    _BRICK_DTYPES,
    _BYTEORDERS,
    _FLOAT_ATTRIBUTES,
    _TAXIS_UNITS,
    AFNI_VIEWS,
)
from ._geometry import (
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
)

_ATTRIBUTE = re.compile(
    r"\s*type\s*=\s*(\S+)\s+name\s*=\s*(\S+)\s+count\s*=\s*([+-]?\d+)"
)
_TOKEN = re.compile(r"\s*(\S+)")
_KINDS = ("integer-attribute", "float-attribute", "string-attribute")

_Value = tx.Union[str, tx.Tuple[int, ...], tx.Tuple[float, ...]]


# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


def _decode_string(chars: str) -> str:
    """Decode a string attribute (`~` to NUL, final NUL dropped)."""
    value = chars.replace("~", "\0")
    return value[:-1] if value.endswith("\0") else value


def _encode_string(value: str) -> str:
    """Encode a string attribute (`~` to `*`, NUL to `~`)."""
    return (value.replace("~", "*") + "\0").replace("\0", "~")


def _parse_attributes(text: str) -> "OrderedDict[str, _Value]":
    """Parse the attributes of a `.HEAD` file."""
    if "<AFNI_" in text[:1024]:
        raise ParserContentError(
            "This AFNI header is in the NIML (XML) variant, which is not "
            "supported. Rewrite it with AFNI_WRITE_NIML unset (for "
            "instance with `3dcopy`)."
        )
    attributes = OrderedDict()
    pos = 0
    end = len(text.rstrip())
    while pos < end:
        match = _ATTRIBUTE.match(text, pos)
        if match is None:
            if attributes:
                # As AFNI does, keep what was read before unreadable content.
                break
            snippet = text[pos : pos + 40].strip()
            raise ParserContentError(
                f"Not an AFNI attribute, at character {pos}: {snippet!r}"
            )
        kind, name, count = match.group(1), match.group(2), int(match[3])
        pos = match.end()
        if kind not in _KINDS:
            raise ParserContentError(
                f"Unknown kind of AFNI attribute {name}: {kind}"
            )
        if count <= 0:
            continue  # AFNI skips empty attributes
        if kind == "string-attribute":
            quote = pos
            while quote < len(text) and text[quote].isspace():
                quote += 1
            if quote >= len(text) or text[quote] != "'":
                raise ParserContentError(
                    f"The AFNI string attribute {name} does not start "
                    f"with a quote."
                )
            chars = text[quote + 1 : quote + 1 + count]
            if len(chars) != count:
                raise ParserContentError(
                    f"The AFNI string attribute {name} is truncated: "
                    f"{len(chars)} characters instead of {count}."
                )
            attributes[name] = _decode_string(chars)
            pos = quote + 1 + count
            continue
        convert = int if kind == "integer-attribute" else float
        values = []
        for _ in range(count):
            token = _TOKEN.match(text, pos)
            if token is None:
                raise ParserContentError(
                    f"The AFNI attribute {name} holds fewer than {count} "
                    f"values."
                )
            try:
                values.append(convert(token.group(1)))
            except ValueError:
                raise ParserContentError(
                    f"The AFNI attribute {name} holds a value that is not "
                    f"{'an integer' if convert is int else 'a number'}: "
                    f"{token.group(1)!r}"
                ) from None
            pos = token.end()
        attributes[name] = tuple(values)
    return attributes


def _format_float(value: float) -> str:
    """Format a float with enough digits to round-trip a float32."""
    return f"{float(value):.9g}"


def _format_attribute(name: str, value: _Value) -> str:
    """Format one attribute as AFNI writes it (`thd_writeatr.c`)."""
    if isinstance(value, str):
        chars = _encode_string(value)
        return (
            f"\ntype = string-attribute\nname = {name}\n"
            f"count = {len(chars)}\n'{chars}\n"
        )
    values = list(value)
    if not values:
        return ""
    is_float = name in _FLOAT_ATTRIBUTES or any(
        isinstance(v, (float, np.floating)) for v in values
    )
    if is_float:
        tokens = [_format_float(v) for v in values]
        kind = "float"
    else:
        tokens = [str(int(v)) for v in values]
        kind = "integer"
    lines = [
        " " + " ".join(tokens[i : i + 5]) for i in range(0, len(tokens), 5)
    ]
    return (
        f"\ntype = {kind}-attribute\nname = {name}\ncount = {len(values)}\n"
        + "\n".join(lines)
        + "\n"
    )


class AfniHeader(Magic, frozen=True, eq=False):
    """
    The attributes of an AFNI `.HEAD` file.

    String attributes are strings, with substrings separated by NUL, and
    numeric attributes are tuples. The properties decode the attributes
    that describe the data and the geometry. All others, such as
    `HISTORY_NOTE` or `BRICK_LABS`, are written back unchanged.
    """

    attributes: tx.Dict[str, _Value]
    """All attributes by name, in file order."""

    def __getitem__(self, name: str) -> _Value:
        return self.attributes[name]

    def __contains__(self, name: object) -> bool:
        return name in self.attributes

    def get(self, name: str, default: tx.Any = None) -> tx.Any:
        """Return the value of an attribute, or `default` if it is absent."""
        return self.attributes.get(name, default)

    def _ints(self, name: str, n: int) -> tx.Tuple[int, ...]:
        value = self.attributes.get(name)
        if isinstance(value, str) or value is None or len(value) < n:
            raise ParserContentError(
                f"The AFNI header has no valid {name} attribute (it needs "
                f"at least {n} values)."
            )
        return tuple(int(v) for v in value[:n])

    def _floats(self, name: str, n: int) -> tx.Tuple[float, ...]:
        value = self.attributes.get(name)
        if isinstance(value, str) or value is None or len(value) < n:
            raise ParserContentError(
                f"The AFNI header has no valid {name} attribute (it needs "
                f"at least {n} values)."
            )
        return tuple(float(v) for v in value[:n])

    @property
    def shape(self) -> tx.Tuple[int, int, int]:
        """The number of voxels along each axis, `(nx, ny, nz)`."""
        return self._ints("DATASET_DIMENSIONS", 3)

    @property
    def nvals(self) -> int:
        """The number of sub-bricks."""
        return self._ints("DATASET_RANK", 2)[1]

    @property
    def brick_types(self) -> tx.Tuple[int, ...]:
        """
        The `BRICK_TYPES` code of each sub-brick (int16 when absent; a short
        list repeats its last code).
        """
        codes = self.attributes.get("BRICK_TYPES")
        if isinstance(codes, str) or not codes:
            codes = (1,)
        codes = tuple(int(c) for c in codes[: self.nvals])
        return codes + codes[-1:] * (self.nvals - len(codes))

    @property
    def byteorder(self) -> str:
        """
        The byte order of the BRIK: `"<"`, `">"`, or `"="` (native) when the
        header does not give one.
        """
        value = self.attributes.get("BYTEORDER_STRING")
        if isinstance(value, str):
            return _BYTEORDERS.get(value.split("\0")[0].strip(), "=")
        return "="

    @property
    def dtypes(self) -> tx.Tuple[np.dtype, ...]:
        """The stored dtype of each sub-brick, with its byte order."""
        dtypes = []
        for code in self.brick_types:
            if code not in _BRICK_DTYPES:
                raise ParserContentError(
                    f"AFNI sub-bricks of type code {code} are not "
                    f"supported (supported: byte, short, int, float, "
                    f"double, complex)."
                )
            dtypes.append(_BRICK_DTYPES[code].newbyteorder(self.byteorder))
        return tuple(dtypes)

    @property
    def nbytes(self) -> int:
        """The size of the uncompressed BRIK, in bytes."""
        nvox = int(np.prod(self.shape))
        return sum(nvox * dtype.itemsize for dtype in self.dtypes)

    @property
    def float_facs(self) -> tx.Tuple[float, ...]:
        """The scale factor of each sub-brick, 0 for an unscaled sub-brick."""
        facs = self.attributes.get("BRICK_FLOAT_FACS")
        facs = () if isinstance(facs, str) or facs is None else facs
        facs = tuple(float(f) for f in facs[: self.nvals])
        return facs + (0.0,) * (self.nvals - len(facs))

    @property
    def labels(self) -> tx.Optional[tx.List[str]]:
        """The sub-brick labels (`BRICK_LABS`), or `None`."""
        value = self.attributes.get("BRICK_LABS")
        if not isinstance(value, str):
            return None
        return value.split("\0")

    @property
    def view(self) -> str:
        """The view given by `SCENE_DATA[0]`, `"orig"` by default."""
        scene = self.attributes.get("SCENE_DATA")
        if isinstance(scene, str) or not scene:
            return "orig"
        code = int(scene[0])
        return AFNI_VIEWS[code] if 0 <= code < len(AFNI_VIEWS) else "orig"

    @property
    def real_matrix(self) -> tx.Optional[np.ndarray]:
        """The `IJK_TO_DICOM_REAL` matrix, or `None` if absent or singular."""
        value = self.attributes.get("IJK_TO_DICOM_REAL")
        if isinstance(value, str) or value is None or len(value) < 12:
            return None
        matrix = np.eye(4)
        matrix[:3] = np.asarray(value[:12], dtype=np.float64).reshape(3, 4)
        if not np.all(np.isfinite(matrix)) or not np.linalg.det(matrix):
            return None
        return matrix

    @property
    def orient(self) -> tx.Tuple[int, int, int]:
        """The `ORIENT_SPECIFIC` code of each axis."""
        if "ORIENT_SPECIFIC" not in self and self.real_matrix is not None:
            return afni_geometry_from_matrix(self.real_matrix)[0]
        return self._ints("ORIENT_SPECIFIC", 3)

    @property
    def origin(self) -> tx.Tuple[float, float, float]:
        """The DICOM coordinate of the first voxel centre along each axis."""
        if "ORIGIN" not in self and self.real_matrix is not None:
            return afni_geometry_from_matrix(self.real_matrix)[1]
        return self._floats("ORIGIN", 3)

    @property
    def delta(self) -> tx.Tuple[float, float, float]:
        """The signed voxel size along each axis."""
        if "DELTA" not in self and self.real_matrix is not None:
            return afni_geometry_from_matrix(self.real_matrix)[2]
        return self._floats("DELTA", 3)

    @property
    def cardinal_matrix(self) -> np.ndarray:
        """The cardinal voxel-to-DICOM matrix, the grid AFNI computes on."""
        try:
            return afni_cardinal_matrix(self.orient, self.origin, self.delta)
        except ValueError as e:
            raise ParserContentError(str(e)) from None

    @property
    def voxel_to_dicom(self) -> np.ndarray:
        """
        The true voxel-to-DICOM matrix: the real matrix if any, else the
        cardinal one.
        """
        real = self.real_matrix
        return self.cardinal_matrix if real is None else real

    @property
    def is_oblique(self) -> bool:
        """Whether the true matrix differs from the cardinal matrix."""
        real = self.real_matrix
        if real is None:
            return False
        return not np.allclose(real, self.cardinal_matrix, atol=1e-4)

    @property
    def taxis(self) -> tx.Optional[tx.Tuple[float, tx.Optional[str]]]:
        """
        The repetition time and its unit, or `None` without a time axis.

        The unit is `"millisecond"`, `"second"`, `"hertz"`, or `None`.
        """
        nums = self.attributes.get("TAXIS_NUMS")
        floats = self.attributes.get("TAXIS_FLOATS")
        if isinstance(nums, str) or isinstance(floats, str):
            return None
        if not nums or not floats or len(floats) < 2:
            return None
        unit = _TAXIS_UNITS.get(int(nums[2])) if len(nums) > 2 else None
        return float(floats[1]), unit

    def validate(self) -> "AfniHeader":
        """
        Check that the header describes a readable, non-empty dataset.

        Raises
        ------
        ParserContentError
            If a mandatory attribute is missing or invalid.
        """
        rank = self._ints("DATASET_RANK", 2)
        shape = self.shape
        if rank[1] < 1 or any(n < 1 for n in shape):
            raise ParserContentError(
                f"The AFNI header describes an empty dataset: "
                f"{shape} voxels and {rank[1]} sub-bricks."
            )
        # Both raise on an unsupported type or an invalid orientation.
        _ = self.dtypes, self.cardinal_matrix
        return self

    @classmethod
    def from_text(cls, text: str) -> "AfniHeader":
        """Parse the text of a `.HEAD` file."""
        return cls(attributes=_parse_attributes(text))

    @classmethod
    def from_bytes(cls, content: bytes) -> "AfniHeader":
        """Parse the content of a `.HEAD` file."""
        return cls.from_text(bytes(content).decode("latin-1"))

    def to_text(self) -> str:
        """Format the `.HEAD` file as AFNI writes it."""
        return "".join(
            _format_attribute(name, value)
            for name, value in self.attributes.items()
        )

    def replace(self, **attributes: tx.Optional[_Value]) -> "AfniHeader":
        """Return a copy with attributes set, or removed when given `None`."""
        merged = OrderedDict(self.attributes)
        for name, value in attributes.items():
            if value is None:
                merged.pop(name, None)
            else:
                merged[name] = value
        return type(self)(attributes=merged)


def _looks_like_head(head: str) -> bool:
    """Tell whether a text starts like a `.HEAD` file."""
    return re.match(r"\s*type\s*=\s*\S+-attribute\s", head) is not None


def _read_head(file: tx.Any) -> AfniHeader:
    with _open_path(file) as f:
        return AfniHeader.from_bytes(f.read())


def _read_header_stream(file: tx.IO) -> AfniHeader:
    """
    Read a header from a stream, giving up early on content that does not
    start like one.
    """
    start = file.read(256)
    if isinstance(start, bytes):
        start = start.decode("latin-1")
    if not _looks_like_head(start):
        raise ParserContentError("Not an AFNI header")
    rest = file.read()
    if isinstance(rest, bytes):
        rest = rest.decode("latin-1")
    return AfniHeader.from_text(start + rest)
