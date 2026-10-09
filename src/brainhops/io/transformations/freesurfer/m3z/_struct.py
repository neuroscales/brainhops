"""Raw content of a FreeSurfer morph (`.m3z`), as it is stored.

The structs mirror the file one to one, so that reading and writing a
file reproduces it. The layout follows `__m3zRead` and `__m3zWrite` in
FreeSurfer's `utils/gcamorph.cpp`; the package documentation gives the
full specification.
"""

__all__ = [
    "GCAM_RAS",
    "GCAM_VOX",
    "M3zGeometry",
    "M3zStruct",
    "M3zXform",
    "read_m3z",
    "write_m3z",
]

import gzip
import struct as _struct

import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Magic, field

from brainhops.io.base.parsers import ParserContentError

# io
from brainhops.io.common.freesurfer._geometry import (
    FS_DEFAULT_XRAS,
    FS_DEFAULT_YRAS,
    FS_DEFAULT_ZRAS,
    fs_geometry_from_vox2ras,
    fs_vox2ras,
)

_3Ints = tx.Tuple[int, int, int]
_3Floats = tx.Tuple[float, float, float]

GCAM_VERSION = 1.0
"""The only morph version that FreeSurfer reads and writes."""

GCAM_RAS = 1
"""Node positions are scanner RAS coordinates of the source image."""

GCAM_VOX = 2
"""Node positions are voxel coordinates of the source image (default)."""

TAG_GCAMORPH_GEOM = 10
TAG_GCAMORPH_TYPE = 11
TAG_GCAMORPH_LABELS = 12
TAG_MGH_XFORM = 31

_TAGS = (TAG_GCAMORPH_GEOM, TAG_GCAMORPH_TYPE, TAG_GCAMORPH_LABELS)
"""The tags that this module decodes, besides `TAG_MGH_XFORM`."""

MATRIX_STRLEN = 4 * 4 * 100
"""Length of the text buffer into which FreeSurfer writes a matrix."""

_FNAME_LEN = 512
"""Length of the file-name buffer of a volume geometry."""

# A morph is always big-endian: `__m3zWrite` writes through `znzwriteInt`
# and related functions (`fio.cpp`), which swap bytes on little-endian
# hosts, and `__m3zRead` swaps back.

# version (float), width, height, depth, spacing (int), exp_k (float)
_HEADER = _struct.Struct(">f4if")

# origx, origy, origz, x, y, z (float), then xn, yn, zn (int)
_NODE = np.dtype(
    [("orig", ">f4", (3,)), ("pos", ">f4", (3,)), ("index", ">i4", (3,))]
)

# valid, width, height, depth (int), then 15 floats: xsize, ysize, zsize,
# the cosines x_r to z_s, and c_r, c_a, c_s
_GEOM = _struct.Struct(">4i15f")
_GEOM_SIZE = _GEOM.size + _FNAME_LEN

_GZIP_MAGIC = b"\x1f\x8b"


# ----------------------------------------------------------------------
#   STRUCTS
# ----------------------------------------------------------------------


class M3zGeometry(Magic, frozen=True, repr=HIDE_IF_NONE):
    """Geometry of a volume, as a morph stores it (`VOL_GEOM`).

    The defaults, used when a file records no geometry, are those of
    FreeSurfer's `initVolGeom`: an invalid volume of 256³ voxels of 1 mm,
    coronal LIA, centred on the origin.
    """

    valid: int = 0
    """1 if the geometry is valid, 0 otherwise."""

    shape: _3Ints = (256, 256, 256)
    """The shape `(width, height, depth)`."""

    voxel_size: _3Floats = (1.0, 1.0, 1.0)
    """The voxel size, in millimetres."""

    xras: _3Floats = FS_DEFAULT_XRAS
    """Direction cosines of the first voxel axis, in RAS."""

    yras: _3Floats = FS_DEFAULT_YRAS
    """Direction cosines of the second voxel axis, in RAS."""

    zras: _3Floats = FS_DEFAULT_ZRAS
    """Direction cosines of the third voxel axis, in RAS."""

    cras: _3Floats = (0.0, 0.0, 0.0)
    """RAS coordinates of the centre of the volume (voxel `shape / 2`)."""

    fname: bytes = field(
        default=b"unknown".ljust(_FNAME_LEN, b"\0"), repr=False
    )
    """The 512-byte, NUL-padded file-name buffer, as stored."""

    @property
    def filename(self) -> str:
        """The file name of the volume, decoded from its buffer."""
        return self.fname.split(b"\0", 1)[0].decode("utf-8", "replace")

    @property
    def vox2ras(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-scanner-RAS matrix."""
        return fs_vox2ras(
            self.shape,
            self.voxel_size,
            self.xras,
            self.yras,
            self.zras,
            self.cras,
        )

    @classmethod
    def from_vox2ras(
        cls,
        vox2ras: np.ndarray,
        shape: tx.Sequence[int],
        valid: int = 1,
        filename: tx.Union[str, bytes] = "",
    ) -> tx.Self:
        """Return the geometry of a volume of `shape` placed by `vox2ras`.

        See `fs_geometry_from_vox2ras`.
        """
        shape = tuple(int(s) for s in shape)
        size, xras, yras, zras, cras = fs_geometry_from_vox2ras(vox2ras, shape)
        if isinstance(filename, str):
            filename = filename.encode("utf-8")
        return cls(
            valid=int(valid),
            shape=shape,
            voxel_size=size,
            xras=xras,
            yras=yras,
            zras=zras,
            cras=cras,
            fname=filename[:_FNAME_LEN].ljust(_FNAME_LEN, b"\0"),
        )


class M3zXform(Magic, frozen=True, repr=HIDE_IF_NONE):
    """Matrix of a `TAG_MGH_XFORM` tag.

    The matrix is the linear transform that initialised the morph, written
    as text into a fixed-size buffer behind its own tag and length: tag 0
    and keyword `Matrix` in recent FreeSurfer, `TAG_AUTO_ALIGN` (33) and
    `AutoAlign` in FreeSurfer 6 and 7.1. The buffer is kept verbatim.
    """

    tag: int = 0
    """The inner tag."""

    length: int = MATRIX_STRLEN
    """The length that the inner tag records."""

    buffer: bytes = field(default=b"\0" * MATRIX_STRLEN, repr=False)
    """The text buffer, as stored."""

    @property
    def matrix(self) -> np.ndarray:
        """The `(4, 4)` matrix that the buffer encodes."""
        text = self.buffer.split(b"\0", 1)[0].decode("ascii", "replace")
        values = text.split()[1:17]
        if len(values) != 16:
            raise ParserContentError(
                "The matrix of a TAG_MGH_XFORM tag does not hold 16 values."
            )
        return np.asarray([float(v) for v in values]).reshape(4, 4)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> tx.Self:
        """Encode a matrix as FreeSurfer's `znzWriteMatrix` does."""
        matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        text = "Matrix " + " ".join(f"{v:10f}" for v in matrix.ravel())
        buffer = text.encode("ascii")[:MATRIX_STRLEN]
        return cls(buffer=buffer.ljust(MATRIX_STRLEN, b"\0"))


class M3zStruct(Magic, frozen=True, eq=False, repr=HIDE_IF_NONE):
    """Content of a morph file, as `M3zMorph.struct` holds it.

    Node arrays are indexed `[x, y, z]` with x slowest, the Fortran order of
    the node grid. Structs compare by identity, because the arrays are large.

    Attributes
    ----------
    version : float
        Always `1.0`.
    spacing : int
        Node `n` is atlas voxel `n * spacing`.
    exp_k : float
        Exponent of the area-preserving penalty.
    original : (W, H, D, 3) float32 array
        Positions before registration, in the units of `positions`.
    positions : (W, H, D, 3) float32 array
        Node positions, in source voxels (`GCAM_VOX`) or scanner RAS
        (`GCAM_RAS`).
    index : (W, H, D, 3) int32 array
        The GCA node to which each node maps.
    image : M3zGeometry or None
        Source geometry from `TAG_GCAMORPH_GEOM`, or `None`.
    atlas : M3zGeometry or None
        Target geometry from `TAG_GCAMORPH_GEOM`, or `None`.
    type : int or None
        `GCAM_VOX` or `GCAM_RAS` from `TAG_GCAMORPH_TYPE`. `None` means voxels.
    labels : (W, H, D) int32 array or None
        Node labels, from `TAG_GCAMORPH_LABELS`.
    xform : M3zXform or None
        The linear transform of `TAG_MGH_XFORM`.
    tags : tuple of int
        The tags, in file order.
    trailing : bytes
        The bytes after an unknown tag, kept verbatim.
    """

    version: float = GCAM_VERSION
    spacing: int = 1
    exp_k: float = 20.0
    original: tx.Optional[np.ndarray] = field(default=None, repr=False)
    positions: tx.Optional[np.ndarray] = field(default=None, repr=False)
    index: tx.Optional[np.ndarray] = field(default=None, repr=False)
    image: tx.Optional[M3zGeometry] = None
    atlas: tx.Optional[M3zGeometry] = None
    type: tx.Optional[int] = None
    labels: tx.Optional[np.ndarray] = field(default=None, repr=False)
    xform: tx.Optional[M3zXform] = None
    tags: tx.Tuple[int, ...] = ()
    trailing: bytes = field(default=b"", repr=False)

    @property
    def shape(self) -> _3Ints:
        """The shape `(W, H, D)` of the node grid."""
        if self.positions is None:
            return (0, 0, 0)
        return tuple(int(s) for s in self.positions.shape[:3])

    @property
    def coordinates(self) -> int:
        """`GCAM_VOX` or `GCAM_RAS`: the units of `positions`."""
        return GCAM_VOX if self.type is None else int(self.type)

    @property
    def image_geometry(self) -> M3zGeometry:
        """The source geometry, or the FreeSurfer default if there is none."""
        return self.image if self.image is not None else M3zGeometry()

    @property
    def atlas_geometry(self) -> M3zGeometry:
        """The atlas geometry, or the FreeSurfer default if there is none."""
        return self.atlas if self.atlas is not None else M3zGeometry()

    @property
    def invalid(self) -> np.ndarray:
        """The `(W, H, D)` mask of the nodes that FreeSurfer marks invalid.

        These nodes (`GCAM_POSITION_INVALID`) have all-zero positions and
        originals, and FreeSurfer does not sample through them.
        """
        zero = (self.original == 0).all(-1) & (self.positions == 0).all(-1)
        return np.asarray(zero)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def is_gzip(content: bytes) -> bool:
    """Return whether bytes are gzip-compressed."""
    return bytes(content[:2]) == _GZIP_MAGIC


def read_header(content: bytes) -> tx.Optional[tx.Tuple]:
    """Decode a morph header.

    The result is `(version, width, height, depth, spacing, exp_k)`, or `None`
    if the bytes do not start with a morph header.
    """
    if len(content) < _HEADER.size:
        return None
    version, width, height, depth, spacing, exp_k = _HEADER.unpack_from(
        content
    )
    if version != GCAM_VERSION:
        return None
    if min(width, height, depth) <= 0 or spacing <= 0:
        return None
    return version, width, height, depth, spacing, exp_k


def _read_geometry(content: bytes, offset: int) -> M3zGeometry:
    values = _GEOM.unpack_from(content, offset)
    fname = bytes(content[offset + _GEOM.size : offset + _GEOM_SIZE])
    return M3zGeometry(
        valid=values[0],
        shape=tuple(values[1:4]),
        voxel_size=tuple(values[4:7]),
        xras=tuple(values[7:10]),
        yras=tuple(values[10:13]),
        zras=tuple(values[13:16]),
        cras=tuple(values[16:19]),
        fname=fname,
    )


def _need(content: bytes, offset: int, size: int, what: str) -> None:
    if len(content) < offset + size:
        raise ParserContentError(f"The morph file is truncated in {what}.")


def read_m3z(content: bytes) -> M3zStruct:
    """Decode the bytes of a morph, gzipped (`.m3z`) or not (`.m3d`).

    Raises
    ------
    ParserContentError
        If the bytes are not a morph.
    """
    content = bytes(content)
    if is_gzip(content):
        try:
            content = gzip.decompress(content)
        except (OSError, EOFError) as e:
            raise ParserContentError(f"Corrupt gzip stream: {e}") from e
    header = read_header(content)
    if header is None:
        raise ParserContentError("Not a FreeSurfer morph (m3z) file.")
    version, width, height, depth, spacing, exp_k = header
    shape = (width, height, depth)
    count = width * height * depth
    offset = _HEADER.size
    _need(content, offset, count * _NODE.itemsize, "its nodes")
    nodes = np.frombuffer(content, dtype=_NODE, count=count, offset=offset)
    nodes = nodes.reshape(shape)
    offset += count * _NODE.itemsize

    fields: tx.Dict[str, tx.Any] = {}
    tags: tx.List[int] = []
    trailing = b""
    # Tags follow until the end of the file or a zero tag. Morph tags have no
    # length, so an unknown tag cannot be skipped, and whatever follows it is
    # kept verbatim.
    while len(content) >= offset + 4:
        (tag,) = _struct.unpack_from(">i", content, offset)
        if tag == 0:
            trailing = content[offset:]
            break
        if tag == TAG_GCAMORPH_GEOM:
            _need(content, offset + 4, 2 * _GEOM_SIZE, "its geometry")
            fields["image"] = _read_geometry(content, offset + 4)
            fields["atlas"] = _read_geometry(content, offset + 4 + _GEOM_SIZE)
            size = 2 * _GEOM_SIZE
        elif tag == TAG_GCAMORPH_TYPE:
            _need(content, offset + 4, 4, "its type")
            (fields["type"],) = _struct.unpack_from(">i", content, offset + 4)
            size = 4
        elif tag == TAG_GCAMORPH_LABELS:
            _need(content, offset + 4, 4 * count, "its labels")
            labels = np.frombuffer(
                content, dtype=">i4", count=count, offset=offset + 4
            )
            fields["labels"] = labels.reshape(shape).astype(np.int32)
            size = 4 * count
        elif tag == TAG_MGH_XFORM:
            _need(content, offset + 4, 12 + MATRIX_STRLEN, "its matrix")
            inner, length = _struct.unpack_from(">iq", content, offset + 4)
            start = offset + 16
            fields["xform"] = M3zXform(
                tag=inner,
                length=length,
                buffer=content[start : start + MATRIX_STRLEN],
            )
            size = 12 + MATRIX_STRLEN
        else:
            trailing = content[offset:]
            break
        tags.append(tag)
        offset += 4 + size
    else:
        trailing = content[offset:]

    return M3zStruct(
        version=version,
        spacing=spacing,
        exp_k=exp_k,
        original=nodes["orig"].astype(np.float32),
        positions=nodes["pos"].astype(np.float32),
        index=nodes["index"].astype(np.int32),
        tags=tuple(tags),
        trailing=bytes(trailing),
        **fields,
    )


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _write_geometry(geom: M3zGeometry) -> bytes:
    values = (
        int(geom.valid),
        *(int(v) for v in geom.shape),
        *geom.voxel_size,
        *geom.xras,
        *geom.yras,
        *geom.zras,
        *geom.cras,
    )
    fname = bytes(geom.fname)[:_FNAME_LEN].ljust(_FNAME_LEN, b"\0")
    return _GEOM.pack(*values) + fname


def write_m3z(struct: M3zStruct, compress: bool = True) -> bytes:
    """Encode a morph as FreeSurfer's `__m3zWrite` does.

    Tags are written in the order of `struct.tags`, followed by any tag whose
    content the struct holds but does not list, in FreeSurfer's order.
    """
    positions = np.asarray(struct.positions)
    shape = tuple(positions.shape[:3])
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError(
            f"Node positions must have shape (W, H, D, 3), not "
            f"{positions.shape}."
        )
    nodes = np.empty(shape, dtype=_NODE)
    nodes["pos"] = positions
    original = positions if struct.original is None else struct.original
    nodes["orig"] = np.asarray(original)
    if struct.index is None:
        nodes["index"] = 0
    else:
        nodes["index"] = np.asarray(struct.index)
    chunks = [
        _HEADER.pack(
            struct.version, *shape, int(struct.spacing), struct.exp_k
        ),
        nodes.tobytes(),
    ]

    present = {
        TAG_GCAMORPH_GEOM: struct.image is not None
        or struct.atlas is not None,
        TAG_GCAMORPH_TYPE: struct.type is not None,
        TAG_GCAMORPH_LABELS: struct.labels is not None,
        TAG_MGH_XFORM: struct.xform is not None,
    }
    order = [t for t in struct.tags if present.get(t)]
    order += [t for t, p in present.items() if p and t not in order]
    for tag in order:
        chunks.append(_struct.pack(">i", tag))
        if tag == TAG_GCAMORPH_GEOM:
            chunks.append(_write_geometry(struct.image_geometry))
            chunks.append(_write_geometry(struct.atlas_geometry))
        elif tag == TAG_GCAMORPH_TYPE:
            chunks.append(_struct.pack(">i", int(struct.type)))
        elif tag == TAG_GCAMORPH_LABELS:
            labels = np.asarray(struct.labels).reshape(shape)
            chunks.append(labels.astype(">i4").tobytes())
        elif tag == TAG_MGH_XFORM:
            xform = struct.xform
            buffer = bytes(xform.buffer)[:MATRIX_STRLEN]
            chunks.append(_struct.pack(">iq", xform.tag, xform.length))
            chunks.append(buffer.ljust(MATRIX_STRLEN, b"\0"))
    chunks.append(bytes(struct.trailing))
    content = b"".join(chunks)
    if compress:
        content = gzip.compress(content, mtime=0)
    return content
