"""The record of an AFNI dataset: the attributes of its `.HEAD` file."""

# stdlib
import os
import re
from collections import OrderedDict
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic, NoRepr

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
)

# this format
from ._constants import (
    _BRICK_DTYPES,
    _BYTEORDERS,
    _FLOAT_ATTRIBUTES,
    _TAXIS_UNITS,
    AFNI_VIEWS,
)
from ._files import afni_dataset_files
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
#   ATTRIBUTES
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


def _format_attributes(attributes: tx.Mapping[str, _Value]) -> str:
    """Format the attributes of a `.HEAD` file as the writer spells them."""
    return "".join(
        _format_attribute(name, value) for name, value in attributes.items()
    )


# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


class AfniRaw(
    Magic, BinaryFileReader, BinaryFileWriter, frozen=True, eq=False
):
    """The attributes of an AFNI `.HEAD` file, as the file stores them.

    An AFNI dataset is a text header, the `.HEAD` file, and a file of
    voxel values, the `.BRIK` file. The record holds the whole header:
    every attribute by name, and the text that it was read from. String
    attributes are strings, with substrings separated by NUL, and numeric
    attributes are tuples. The properties decode the attributes that
    describe the voxels and the geometry, and the other attributes, such
    as `HISTORY_NOTE` or `BRICK_LABS`, are written back unchanged.

    The record knows where the voxels lie and how they are stored, but
    it never reads them. It is held by
    [`AfniMetadata`][brainhops.io.common.afni.AfniMetadata], which never
    changes it in place: a writer works on the copy that [`copy`][]
    returns, or builds a new record with [`replace`][].

    A record compares by identity. Two records hold the same header when
    their `attributes` are equal.
    """

    attributes: tx.Dict[str, _Value]
    """All attributes by name, in file order."""

    text: NoRepr[tx.Optional[str]] = None
    """The text of the `.HEAD` file that the attributes were read from.

    The text is written back as it is while it still describes the
    attributes, so that a header that is read and written again keeps
    its bytes, whichever program wrote it. Otherwise, and for a record
    built from attributes alone, the attributes are formatted as AFNI
    writes them. The text is `None` for a record built from attributes.
    """

    def copy(self) -> "AfniRaw":
        """Return a copy of the record that can be changed freely.

        The attributes are copied into a new dictionary, and the text is
        kept, since [`to_text`][] writes it only while it still describes
        the attributes.

        Returns
        -------
        AfniRaw
            The copy.
        """
        return AfniRaw(attributes=OrderedDict(self.attributes), text=self.text)

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

    def validate(self) -> "AfniRaw":
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

    def replace(self, **attributes: tx.Optional[_Value]) -> "AfniRaw":
        """Return a record with attributes set, or removed when given `None`.

        The new record has no text, so it is formatted from its
        attributes when it is written.
        """
        merged = OrderedDict(self.attributes)
        for name, value in attributes.items():
            if value is None:
                merged.pop(name, None)
            else:
                merged[name] = value
        return type(self)(attributes=merged)

    # --- reading ------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "AfniRaw":
        """Parse the text of a `.HEAD` file, which the record keeps.

        The keyword arguments are ignored.

        Raises
        ------
        ParserContentError
            If the text is not an AFNI header.
        """
        return cls(attributes=_parse_attributes(text), text=text)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "AfniRaw":
        """Parse the content of a `.HEAD` file.

        The content is decoded as Latin-1, as AFNI reads it, and the
        keyword arguments are ignored.
        """
        return cls.from_text(bytes(content).decode("latin-1"))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "AfniRaw":
        """Read the record from an open `.HEAD` or `.BRIK` file.

        A `.HEAD` stream is read whole. A stream of any other content is
        taken to be the BRIK of a dataset, whose header is found from the
        name of the stream. The position of the stream is restored, and
        the keyword arguments are ignored.

        Raises
        ------
        ParserContentError
            If the stream is not a header and has no file name.
        """
        with preserve_position(file):
            content = file.read()
        if isinstance(content, str):
            content = content.encode("latin-1")
        if _looks_like_head(bytes(content[:256]).decode("latin-1")):
            return cls.from_bytes(content)
        name = getattr(file, "name", None)
        if isinstance(name, (str, bytes, os.PathLike)):
            return cls.from_filename(os.fsdecode(name))
        raise ParserContentError(
            "This stream is not an AFNI header, and has no file name to "
            "find one from."
        )

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> "AfniRaw":
        """Read the record from the path of a dataset.

        The path may name the `.HEAD` file, the `.BRIK` file or the bare
        dataset, and only the `.HEAD` file is read. The keyword arguments
        are ignored.

        Raises
        ------
        ParserExistsError
            If the `.HEAD` file does not exist.
        """
        head, _, _ = afni_dataset_files(filename)
        if not path.exists(head):
            raise ParserExistsError(f"No such file: {head}")
        with _open_path(head) as f:
            return cls.from_bytes(f.read())

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a valid AFNI header.

        Only the start of a stream that does not look like a header is
        read, and the position of the stream is restored.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no valid
            header. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            `Confidence.LIKELY` or `Confidence.NO`.

        Raises
        ------
        SnifferContentError
            If the stream holds no valid header and `error` is `True`.
        """
        raw, failure = _sniffed_raw(file)
        if raw is not None:
            return Confidence.LIKELY
        return _declined(error, failure)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a path names an AFNI dataset.

        The `.HEAD` file of the dataset is tested, whichever file of the
        dataset the path names.
        """
        raw, failure = _sniffed_raw_at(filename)
        if raw is not None:
            return Confidence.LIKELY
        return _declined(error, failure)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a valid AFNI header."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    # --- writing ------------------------------------------------------

    def to_text(self, **kwargs) -> str:
        """Return the text of the `.HEAD` file.

        The text that the record was read from is returned while it still
        describes the attributes. Otherwise, the attributes are formatted
        as AFNI writes them. The keyword arguments are ignored.
        """
        formatted = _format_attributes(self.attributes)
        if self.text is not None:
            stored = _format_attributes(_parse_attributes(self.text))
            if stored == formatted:
                return self.text
        return formatted

    def to_bytes(self, **kwargs) -> bytes:
        """Return the content of the `.HEAD` file, encoded as Latin-1."""
        return self.to_text().encode("latin-1")


def _looks_like_head(head: str) -> bool:
    """Tell whether a text starts like a `.HEAD` file."""
    return re.match(r"\s*type\s*=\s*\S+-attribute\s", head) is not None


def _read_header_stream(file: tx.IO) -> AfniRaw:
    """
    Read a record from a stream, giving up early on content that does not
    start like a header.
    """
    start = file.read(256)
    if isinstance(start, bytes):
        start = start.decode("latin-1")
    if not _looks_like_head(start):
        raise ParserContentError("Not an AFNI header")
    rest = file.read()
    if isinstance(rest, bytes):
        rest = rest.decode("latin-1")
    return AfniRaw.from_text(start + rest)


def _sniffed_raw(
    file: tx.IO,
) -> tx.Tuple[tx.Optional[AfniRaw], tx.Optional[Exception]]:
    """Read and validate the record of a stream for sniffing.

    The position of the stream is restored. The result is the record, or
    `None` with the error that refused it.
    """
    try:
        with preserve_position(file):
            return _read_header_stream(file).validate(), None
    except Exception as e:  # noqa: BLE001
        return None, e


def _sniffed_raw_at(
    filename: path.FilenameLike,
) -> tx.Tuple[tx.Optional[AfniRaw], tx.Optional[Exception]]:
    """Read and validate the record of the dataset that a path names."""
    head, _, _ = afni_dataset_files(filename)
    if not path.exists(head):
        return None, ParserExistsError(f"No such file: {head}")
    with _open_path(head) as f:
        return _sniffed_raw(f)


def _declined(
    error: tx.Union[bool, tx.Type[Exception]],
    failure: tx.Optional[Exception],
) -> float:
    """Decline a sniffed input, raising the requested error if any."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error("Content is not an AFNI dataset") from failure
    return Confidence.NO
