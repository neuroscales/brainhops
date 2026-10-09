"""
Reading and writing of AFNI datasets.

This module holds what all AFNI formats, images and transformations
alike, share: the header, the geometry, and the decoding and encoding of
voxel values. All of these formats derive from [`AfniFormat`][]. The
conventions follow the AFNI sources (`README.attributes`, `3ddata.h`).

A dataset is a pair of files. `prefix+view.HEAD` holds text attributes
and `prefix+view.BRIK` holds the voxel values, possibly compressed. The
view is `orig` (scanner space), `acpc`, or `tlrc` (Talairach or any
other template). The NIML (XML) variant of the header is not
supported.

The BRIK stores `DATASET_RANK[1]` sub-bricks (time points, statistics
or warp components) one after the other, with `x` varying fastest,
which gives a Fortran-ordered `(nx, ny, nz, nvals)` array. Each
sub-brick has its own type (`BRICK_TYPES`) and, optionally, a scale
factor (`BRICK_FLOAT_FACS`).

The world space is DICOM space, that is LPS in millimetres. The
`ORIENT_SPECIFIC`, `ORIGIN` and `DELTA` attributes define a signed
permutation from voxels to DICOM coordinates, the cardinal matrix, which
is recomputed on read rather than taken from `IJK_TO_DICOM`. An oblique
dataset also stores its true matrix in `IJK_TO_DICOM_REAL`. AFNI
programs compute on the cardinal grid, while `3dAFNItoNIFTI` stores the
true matrix as the NIfTI sform.
"""

__all__ = [
    "AFNI_ORIENTATIONS",
    "AFNI_VIEWS",
    "DICOM_TO_RAS",
    "AfniFormat",
    "AfniHeader",
    "AfniParser",
    "afni_cardinal_matrix",
    "afni_dataset_files",
    "afni_geometry_from_matrix",
    "afni_view",
    "afni_voxel_to_dicom",
    "afni_world",
    "brick_code",
    "brick_dtype",
    "decode_bricks",
    "encode_bricks",
    "scale_bricks",
]

import bz2
import gzip
import itertools
import os
import re
from collections import OrderedDict
from io import BytesIO

import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import open_compressed
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import LPSmm
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    WriterError,
    preserve_position,
)
from brainhops.io.common._geometry import (
    embed_affine,
    ras_conversion,
    reduce_to_affine,
)

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

AFNI_VIEWS: tx.Tuple[str, ...] = ("orig", "acpc", "tlrc")
"""The AFNI views, indexed by their `SCENE_DATA[0]` code."""

AFNI_ORIENTATIONS: tx.Tuple[str, ...] = (
    "right-to-left",
    "left-to-right",
    "posterior-to-anterior",
    "anterior-to-posterior",
    "inferior-to-superior",
    "superior-to-inferior",
)
"""The anatomical orientation name of each `ORIENT_SPECIFIC` code."""

# DICOM axis of a code is code // 2; R2L, A2P and I2S point toward +LPS.
_ORIENT_SIGN = (1, -1, -1, 1, 1, -1)

# (DICOM axis, points toward positive LPS) -> orientation code.
_ORIENT_CODE = {
    (0, True): 0,
    (0, False): 1,
    (1, True): 3,
    (1, False): 2,
    (2, True): 4,
    (2, False): 5,
}

# AFNI stores these attributes as floats, so integer values are still
# written as a float attribute.
_FLOAT_ATTRIBUTES = frozenset(
    {
        "ORIGIN",
        "DELTA",
        "IJK_TO_DICOM",
        "IJK_TO_DICOM_REAL",
        "BRICK_FLOAT_FACS",
        "BRICK_STATS",
        "BRICK_STATAUX",
        "STAT_AUX",
        "TAXIS_FLOATS",
        "TAXIS_OFFSETS",
        "MARKS_XYZ",
        "TAGSET_FLOATS",
        "TAGALIGN_MATVEC",
        "WARP_DATA",
        "VOLREG_MATVEC_000000",
        "VOLREG_CENTER_OLD",
        "VOLREG_CENTER_BASE",
    }
)

DICOM_TO_RAS: np.ndarray = np.diag([-1.0, -1.0, 1.0, 1.0])
"""The `(4, 4)` matrix from DICOM (LPS) to RAS, its own inverse."""

# BRICK_TYPES code -> dtype without byte order (MRI_TYPE in mrilib.h).
# RGB (6) and RGBA (7) are not read.
_BRICK_DTYPES = {
    0: np.dtype(np.uint8),
    1: np.dtype(np.int16),
    2: np.dtype(np.int32),
    3: np.dtype(np.float32),
    4: np.dtype(np.float64),
    5: np.dtype(np.complex64),
}
_BRICK_CODES = {dtype: code for code, dtype in _BRICK_DTYPES.items()}
_BRICK_NAMES = {
    "byte": 0,
    "short": 1,
    "int": 2,
    "float": 3,
    "double": 4,
    "complex": 5,
}

_BYTEORDERS = {"LSB_FIRST": "<", "MSB_FIRST": ">"}

# Suffixes AFNI tries after .BRIK, uncompressed first; only gzip and
# bzip2 can be decompressed.
_BRIK_SUFFIXES = ("", ".gz", ".bz2", ".Z", ".briz")
_READABLE_SUFFIXES = ("", ".gz", ".bz2")

_TAXIS_UNITS = {77001: "millisecond", 77002: "second", 77003: "hertz"}
"""The time units of the `TAXIS_NUMS[2]` codes."""

# Like AFNI (thd_niftiread.c), template world spaces map to the tlrc view.
_VIEW_NAMES = {
    "orig": "orig",
    "acpc": "acpc",
    "tlrc": "tlrc",
    "talairach": "tlrc",
    "mni": "tlrc",
    "template": "tlrc",
}

_ATTRIBUTE = re.compile(
    r"\s*type\s*=\s*(\S+)\s+name\s*=\s*(\S+)\s+count\s*=\s*([+-]?\d+)"
)
_TOKEN = re.compile(r"\s*(\S+)")
_KINDS = ("integer-attribute", "float-attribute", "string-attribute")

_Value = tx.Union[str, tx.Tuple[int, ...], tx.Tuple[float, ...]]


# ----------------------------------------------------------------------
#   FORMAT FAMILY
# ----------------------------------------------------------------------


class AfniFormat:
    """
    The base format of the AFNI family, images and transformations.

    Subclass hints such as `"brik"` can also be reached as `"afni.brik"`.
    """

    HINTS = ("afni",)


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def afni_cardinal_matrix(
    orient: tx.Sequence[int],
    origin: tx.Sequence[float],
    delta: tx.Sequence[float],
) -> np.ndarray:
    """
    Build the cardinal voxel-to-DICOM matrix of an AFNI grid.

    As in `THD_daxes_to_mat44`, voxel axis `i` runs on DICOM axis
    `orient[i] // 2`; in that row, column `i` holds `delta[i]` and the last
    column holds `origin[i]`.

    Raises
    ------
    ValueError
        If an orientation code is invalid, or if the codes do not name three
        different DICOM axes.
    """
    orient = [int(o) for o in orient[:3]]
    if len(orient) != 3 or any(not 0 <= o <= 5 for o in orient):
        raise ValueError(f"Invalid AFNI orientation codes: {orient}")
    rows = [o // 2 for o in orient]
    if len(set(rows)) != 3:
        raise ValueError(
            f"The AFNI orientation codes {orient} do not name three "
            f"different axes."
        )
    matrix = np.eye(4)
    matrix[:3, :3] = 0.0
    for column, row in enumerate(rows):
        matrix[row, column] = float(delta[column])
        matrix[row, 3] = float(origin[column])
    return matrix


def afni_geometry_from_matrix(
    matrix: np.ndarray,
) -> tx.Tuple[
    tx.Tuple[int, int, int],
    tx.Tuple[float, float, float],
    tx.Tuple[float, float, float],
]:
    """
    Decompose a voxel-to-DICOM matrix into `(orient, origin, delta)`.

    As in `THD_daxes_from_mat44`, each voxel axis takes its closest
    orientation, under the constraint that the three axes run on different
    DICOM axes. The voxel size is the length of the column and the origin is
    the translation projected on the unit column, both negated for the R, A
    and I orientations. A cardinal matrix round-trips exactly.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    norms = np.linalg.norm(linear, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    unit = linear / safe

    # The closest orientation is the permutation with the largest entries,
    # as in nifti_mat44_to_orientation.
    best = max(
        itertools.permutations(range(3)),
        key=lambda rows: sum(abs(unit[r, c]) for c, r in enumerate(rows)),
    )
    orient = []
    origin = []
    delta = []
    for column, row in enumerate(best):
        code = _ORIENT_CODE[row, bool(unit[row, column] >= 0)]
        sign = _ORIENT_SIGN[code]
        orient.append(code)
        delta.append(float(sign * norms[column]))
        origin.append(float(sign * (unit[:, column] @ translation)))
    return tuple(orient), tuple(origin), tuple(delta)


def afni_voxel_to_dicom(xform: Transformation) -> np.ndarray:
    """
    Compute the voxel-to-DICOM matrix of a voxel-to-world transformation.

    The world coordinates are converted to RAS according to the orientation
    of the world axes (axes without one are taken to be RAS already), and
    then to DICOM.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation.
    WriterError
        If the transformation has more than three spatial dimensions.
    """
    affine = reduce_to_affine(xform, "AFNI", "DICOM")
    matrix = affine.homogeneous_matrix
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
    matrix = _spatial_block(matrix)
    embedded = embed_affine(matrix, "AFNI", "DICOM")
    output = getattr(affine, "output", None)
    try:
        if output is not None and output.ndim is None:
            output = output.expand(matrix.shape[0] - 1)
    except Exception:
        output = None
    return DICOM_TO_RAS @ ras_conversion(output) @ embedded


def _spatial_block(matrix: np.ndarray) -> np.ndarray:
    """
    Keep the three leading axes of a matrix when they do not mix with the
    others.

    A `Scaling` of a time series that also scales time is an example. Any
    other matrix is returned unchanged.
    """
    out_dim, in_dim = matrix.shape[0] - 1, matrix.shape[1] - 1
    if out_dim <= 3 and in_dim <= 3:
        return matrix
    if out_dim < 3 or in_dim < 3:
        return matrix
    if np.any(matrix[:3, 3:in_dim]) or np.any(matrix[3:out_dim, :3]):
        return matrix
    block = np.eye(4)
    block[:3, :3] = matrix[:3, :3]
    block[:3, 3] = matrix[:3, in_dim]
    return block


def afni_world(name: str) -> LPSmm:
    """The DICOM (LPS mm) world space of an AFNI view."""
    return LPSmm(name=name)


def afni_view(name: tx.Any) -> tx.Optional[str]:
    """
    Return the AFNI view that a name stands for, or `None`.

    The name may be a view (`"tlrc"`), a world-space name (`"mni"` stands
    for `"tlrc"`, `"orig-cardinal"` for `"orig"`), or a dataset file name
    (`"anat+tlrc.HEAD"`).
    """
    if not isinstance(name, (str, os.PathLike)):
        return None
    text = os.fspath(name)
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    base = text.replace("\\", "/").rsplit("/", 1)[-1]
    match = re.search(r"\+(orig|acpc|tlrc)(\.|$)", base)
    if match:
        return match.group(1)
    key = base.lower()
    if key.endswith(_CARDINAL):
        key = key[: -len(_CARDINAL)]
    return _VIEW_NAMES.get(key)


_CARDINAL = "-cardinal"
"""The suffix of the cardinal world-space name of a view."""


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


# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def brick_dtype(dtype: tx.Any) -> np.dtype:
    """
    Return the AFNI dtype that stores a dtype, or the one an AFNI type
    name (`"short"`, `"float"`, ...) names.

    Supported dtypes are kept. Otherwise booleans become bytes, int8
    shorts, uint16 ints, uint32 and wider integers doubles, float16 floats,
    and complex128 single-precision complex.

    Raises
    ------
    WriterError
        If AFNI cannot store the dtype.
    """
    if isinstance(dtype, str) and dtype.lower() in _BRICK_NAMES:
        return _BRICK_DTYPES[_BRICK_NAMES[dtype.lower()]]
    dtype = np.dtype(dtype).newbyteorder("=")
    if dtype in _BRICK_CODES:
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype == np.int8:
        return np.dtype(np.int16)
    if dtype.kind in "iu" and dtype.itemsize <= 4:
        return np.dtype(np.int32) if dtype != np.uint32 else np.dtype(float)
    if dtype.kind in "iu":
        return np.dtype(np.float64)
    if dtype.kind == "f":
        return np.dtype(np.float32 if dtype.itemsize < 8 else np.float64)
    if dtype.kind == "c":
        return np.dtype(np.complex64)
    raise WriterError(f"AFNI cannot store data of type {dtype}.")


def brick_code(dtype: tx.Any) -> int:
    """Return the `BRICK_TYPES` code of an AFNI dtype."""
    return _BRICK_CODES[np.dtype(dtype).newbyteorder("=")]


def decode_bricks(header: AfniHeader, buffer: tx.Any) -> np.ndarray:
    """
    Decode an uncompressed BRIK into an `(nx, ny, nz, nvals)` array.

    The buffer may be bytes, a memoryview or a one-dimensional uint8 array
    such as a memory map. When all sub-bricks share a type, the result is a
    view of the buffer; otherwise the sub-bricks are copied into their
    common type. Scale factors are not applied.

    Raises
    ------
    ParserContentError
        If the buffer is shorter than the header requires.
    """
    shape = header.shape
    nvox = int(np.prod(shape))
    dtypes = header.dtypes
    size = len(buffer) if not isinstance(buffer, np.ndarray) else buffer.size
    if size < header.nbytes:
        raise ParserContentError(
            f"The AFNI BRIK holds {size} bytes, but the header asks for "
            f"{header.nbytes}."
        )

    def _brick(offset: int, dtype: np.dtype) -> np.ndarray:
        if isinstance(buffer, np.ndarray):
            chunk = buffer[offset : offset + nvox * dtype.itemsize]
            return chunk.view(dtype)
        return np.frombuffer(buffer, dtype=dtype, count=nvox, offset=offset)

    if len(set(dtypes)) == 1:
        if isinstance(buffer, np.ndarray):
            flat = buffer[: header.nbytes].view(dtypes[0])
        else:
            count = nvox * header.nvals
            flat = np.frombuffer(buffer, dtype=dtypes[0], count=count)
        return flat.reshape(shape + (header.nvals,), order="F")

    common = np.result_type(*dtypes)
    out = np.empty(shape + (header.nvals,), dtype=common, order="F")
    offset = 0
    for i, dtype in enumerate(dtypes):
        out[..., i] = _brick(offset, dtype).reshape(shape, order="F")
        offset += nvox * dtype.itemsize
    return out


def scale_bricks(header: AfniHeader, stored: tx.Any) -> tx.Any:
    """
    Apply the `BRICK_FLOAT_FACS` scale factors to stored values.

    When no sub-brick is scaled, the values are returned unchanged, so that
    a memory map stays one. Otherwise they are scaled in at least single
    precision.
    """
    facs = np.asarray(header.float_facs, dtype=np.float64)
    if not np.any(facs) or np.all((facs == 0) | (facs == 1)):
        return stored
    facs = np.where(facs == 0, 1.0, facs)
    stored = np.asarray(stored)
    dtype = np.result_type(stored.dtype, np.float32)
    if stored.ndim == len(header.shape):
        return stored.astype(dtype) * dtype.type(facs[0])
    return stored.astype(dtype) * facs.astype(dtype)


def encode_bricks(header: AfniHeader, data: tx.Any) -> tx.Iterator[bytes]:
    """
    Encode an `(nx, ny, nz, nvals)` array into BRIK content, sub-brick by
    sub-brick.

    A three-dimensional array is a single sub-brick. Each sub-brick is
    divided by its nonzero scale factor, rounded if it is stored as
    integers, and converted to the type and byte order of the header.

    Raises
    ------
    WriterError
        If the shape disagrees with the header, or if integer values
        overflow their stored type.
    """
    array = np.asarray(data)
    shape = header.shape
    if array.ndim == 3:
        array = array[..., None]
    if tuple(array.shape) != tuple(shape) + (header.nvals,):
        raise WriterError(
            f"The data have shape {np.shape(data)}, but the AFNI header "
            f"says {shape} with {header.nvals} sub-brick(s)."
        )
    for i, (dtype, fac) in enumerate(zip(header.dtypes, header.float_facs)):
        brick = array[..., i]
        if fac:
            brick = brick / fac
        if dtype.kind in "iu":
            if brick.dtype.kind in "fc":
                brick = np.round(np.real(brick))
            if brick.size:
                info = np.iinfo(dtype)
                low, high = np.nanmin(brick), np.nanmax(brick)
                if low < info.min or high > info.max:
                    raise WriterError(
                        f"Sub-brick {i} holds values in [{low}, {high}], "
                        f"which AFNI's {dtype.name} cannot store."
                    )
        elif dtype.kind == "f" and brick.dtype.kind == "c":
            brick = np.real(brick)
        yield np.asarray(brick, dtype=dtype).tobytes(order="F")


# ----------------------------------------------------------------------
#   FILES
# ----------------------------------------------------------------------


def afni_dataset_files(
    filename: path.FilenameLike,
) -> tx.Tuple[tx.Any, tx.Any, str]:
    """
    Find the `.HEAD` and `.BRIK` files of the dataset that a path names.

    The path may name the `.HEAD` file, the `.BRIK` file with any
    compression, or the bare dataset (`anat+orig`). The BRIK is the first
    existing file among the compression suffixes AFNI tries, or else the
    uncompressed name.

    Returns
    -------
    head : Path
        The `.HEAD` file.
    brik : Path
        The `.BRIK` file, which may not exist.
    stem : str
        The dataset name, without directory or extension.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    name = filename.name
    stem, head_ext, brik_ext = name, ".HEAD", ".BRIK"
    match = re.search(r"\.(HEAD|head|BRIK|brik)(\.[A-Za-z0-9]+)?$", name)
    if match and match.group(2) in (None,) + _BRIK_SUFFIXES[1:]:
        stem = name[: match.start()]
        if match.group(1).islower():
            head_ext, brik_ext = ".head", ".brik"
    parent = filename.parent
    head = parent / (stem + head_ext)
    brik = None
    for suffix in _BRIK_SUFFIXES:
        candidate = parent / (stem + brik_ext + suffix)
        if path.exists(candidate):
            brik = candidate
            break
    if brik is None:
        brik = parent / (stem + brik_ext)
    return head, brik, stem


def _read_head(file: tx.Any) -> AfniHeader:
    with _open_path(file) as f:
        return AfniHeader.from_bytes(f.read())


def _brik_suffix(brik: tx.Any) -> str:
    name = str(brik.name if hasattr(brik, "name") else brik)
    for suffix in _BRIK_SUFFIXES[1:]:
        if name.endswith(suffix):
            return suffix
    return ""


def _read_brik(header: AfniHeader, brik: tx.Any, mmap: bool) -> np.ndarray:
    """
    Read a BRIK, memory-mapping it when it is local, uncompressed, of a
    single type, and `mmap` is true.
    """
    if not path.exists(brik):
        raise ParserExistsError(
            f"No such file: {brik} (the BRIK of the AFNI dataset). A "
            f"dataset without a BRIK (a 'warp-on-demand' dataset) cannot "
            f"be read."
        )
    suffix = _brik_suffix(brik)
    if suffix not in _READABLE_SUFFIXES:
        raise ParserContentError(
            f"The AFNI BRIK {brik} is compressed with {suffix!r}, which "
            f"cannot be read here. Decompress it, or recompress it with "
            f"gzip."
        )
    nbytes = header.nbytes
    local = _local_path(brik)
    with _open_path(brik) as f:
        stream = open_compressed(f)
        compressed = stream is not f
        if (
            local is not None
            and mmap
            and not compressed
            and nbytes > 0
            and len(set(header.dtypes)) == 1
        ):
            size = os.path.getsize(local)
            if size < nbytes:
                raise ParserContentError(
                    f"The AFNI BRIK {brik} holds {size} bytes, but the "
                    f"header asks for {nbytes}."
                )
            buffer = np.memmap(
                local, dtype=np.uint8, mode="r", offset=0, shape=(nbytes,)
            )
        else:
            buffer = stream.read(nbytes)
    return decode_bricks(header, buffer)


def _write_brik(header: AfniHeader, data: tx.Any, brik: tx.Any) -> None:
    """Write a BRIK, compressed according to its suffix."""
    suffix = _brik_suffix(brik)
    with brik.open("wb") as f:
        if suffix == ".gz":
            out = gzip.GzipFile(fileobj=f, mode="wb")
        elif suffix == ".bz2":
            out = bz2.BZ2File(f, mode="wb")
        elif suffix:
            raise WriterError(
                f"Cannot write an AFNI BRIK compressed with {suffix!r}: "
                f"use '.BRIK', '.BRIK.gz' or '.BRIK.bz2'."
            )
        else:
            out = f
        try:
            for chunk in encode_bricks(header, data):
                out.write(chunk)
        finally:
            if out is not f:
                out.close()


# ----------------------------------------------------------------------
#   PARSER
# ----------------------------------------------------------------------


class AfniParser(DataModelBase, AfniFormat, BinaryFileParserWriter):
    """
    The base class of objects stored as an AFNI dataset.

    The class reads and writes the container, a header and its sub-bricks.
    A concrete format gives it a meaning by defining `_from_header` for
    reading, and `_afni_header` and `_afni_data` for writing. A local
    uncompressed BRIK is memory-mapped, so only the header is read until
    the data are indexed.
    """

    HINTS = ("brik",)

    _header: tx.Annotated[
        tx.Optional[AfniHeader],
        tx.Doc(
            """
            The AFNI header this object was read from, kept so that the
            attributes the data model has no slot for (`HISTORY_NOTE`,
            `BRICK_LABS`, `BRICK_STATAUX`, ...) are written back.
            """
        ),
    ] = None

    dataobj: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The stored values, before scaling by `BRICK_FLOAT_FACS`: a
            `(x, y, z, sub-brick)` array, a view of a memory-mapped BRIK
            when the file allows one.
            """
        ),
    ] = None

    header = smartproperty("header")

    def _scaled_data(self) -> tx.Optional[tx.Any]:
        """The stored values, scaled by `BRICK_FLOAT_FACS`."""
        raw = getattr(self, "dataobj", None)
        if raw is None or self.header is None:
            return raw
        return scale_bricks(self.header, raw)

    @classmethod
    def _from_header(
        cls, header: AfniHeader, bricks: np.ndarray, **kwargs
    ) -> tx.Self:
        """
        Build an object from a header and its `(x, y, z, sub-brick)` stored
        values.
        """
        return cls(header=header, dataobj=bricks, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Build an object from an AFNI dataset, by path or file object."""
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
        Build an object from the path of a `.HEAD`, a `.BRIK` or a bare
        dataset.
        """
        head, brik, _ = afni_dataset_files(filename)
        if not path.exists(head):
            raise ParserExistsError(f"No such file: {head}")
        header = _read_head(head).validate()
        bricks = _read_brik(header, brik, mmap)
        return cls._from_header(header, bricks, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """
        Build an object from an open `.HEAD` or `.BRIK` file.

        The other file is found from the name of the stream, which must have
        one.
        """
        kwargs.pop("mmap", None)
        name = getattr(file, "name", None)
        with preserve_position(file):
            content = file.read()
        if isinstance(content, str):
            content = content.encode("latin-1")
        if _looks_like_head(bytes(content[:256]).decode("latin-1")):
            header = AfniHeader.from_bytes(content).validate()
            if not isinstance(name, (str, bytes, os.PathLike)):
                raise ParserContentError(
                    "An AFNI header was given as a stream with no file "
                    "name, so its BRIK cannot be found. Read it from its "
                    "path instead."
                )
            _, brik, _ = afni_dataset_files(os.fsdecode(name))
            bricks = _read_brik(header, brik, mmap=False)
            return cls._from_header(header, bricks, **kwargs)
        if isinstance(name, (str, bytes, os.PathLike)):
            return cls.from_filename(os.fsdecode(name), mmap=False, **kwargs)
        raise ParserContentError(
            "This stream is not an AFNI header, and has no file name to "
            "find one from."
        )

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Refuse to read bytes, which cannot hold a dataset of two files."""
        raise ParserContentError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
            "alone cannot hold. Read it from its path instead."
        )

    @classmethod
    def _sniff_header(
        cls,
        read: tx.Callable[[], AfniHeader],
        error: tx.Union[bool, tx.Type[Exception]],
    ) -> float:
        base_error = None
        try:
            score = cls._score_header(read().validate())
        except Exception as e:  # noqa: BLE001
            base_error = e
            score = Confidence.NO
        if score:
            return score
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Content is not an AFNI dataset") from base_error
        return Confidence.NO

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Return the confidence that a path names an AFNI dataset, from its
        `.HEAD` file.
        """
        head, _, _ = afni_dataset_files(filename)

        def _read() -> AfniHeader:
            if not path.exists(head):
                raise ParserExistsError(f"No such file: {head}")
            with _open_path(head) as f:
                return _read_header_stream(f)

        return cls._sniff_header(_read, error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an AFNI header."""

        def _read() -> AfniHeader:
            with preserve_position(file):
                return _read_header_stream(file)

        return cls._sniff_header(_read, error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an AFNI header."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: AfniHeader) -> float:
        """
        Score how well a valid header matches this class.

        A concrete format overrides the score to recognise its own kind of
        dataset, such as an image or a warp.
        """
        return Confidence.MAYBE

    def _afni_header(self, **kwargs) -> AfniHeader:
        """Build the header to write; each concrete format defines it."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to AFNI."
        )

    def _afni_data(self) -> tx.Any:
        """Return the `(x, y, z[, sub-brick])` array to write."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to AFNI."
        )

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write the `.HEAD` and `.BRIK` files of a dataset.

        The path may name the `.HEAD` file, the `.BRIK` file (a `.gz` or `.bz2`
        suffix compresses it) or the bare dataset. As AFNI does, a sibling BRIK
        with another suffix is removed, so that it cannot shadow the new one.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        head, _, stem = afni_dataset_files(filename)
        brik_ext = ".brik" if head.name.endswith(".head") else ".BRIK"
        suffix = ""
        if ".BRIK" in filename.name.upper():
            suffix = _brik_suffix(filename)
        brik = filename.parent / (stem + brik_ext + suffix)
        header = self._afni_header(filename=filename, **kwargs)
        _write_brik(header, self._afni_data(), brik)
        with head.open("wb") as f:
            f.write(header.to_text().encode("latin-1"))
        for other in _BRIK_SUFFIXES:
            stale = filename.parent / (stem + brik_ext + other)
            if other != suffix and path.exists(stale):
                local = _local_path(stale)
                if local is not None:
                    os.remove(local)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write to a path; AFNI datasets cannot be written to a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return self.to_fileobj(file, **kwargs)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Refuse a stream, which cannot hold a dataset of two files."""
        raise WriterError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which a "
            "single stream cannot hold. Write it to a path instead."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """Refuse to write bytes, which cannot hold a dataset of two files."""
        raise WriterError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
            "alone cannot hold. Write it to a path instead."
        )


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
