__all__ = ["M3zFormat", "M3zParser", "M3zMorph"]

# stdlib
import zlib

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Magic, field, replace

# core
from brainhops._core import path
from brainhops._core.streams import preserve_position

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.transformations.base import WritableFileBasedTransformation
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    homogeneous_matrix,
)

# locals
from .._formats import FreesurferTransformationFormat
from ._struct import (
    GCAM_RAS,
    GCAM_VOX,
    M3zGeometry,
    M3zStruct,
    is_gzip,
    read_header,
    read_m3z,
    write_m3z,
)

# Enough compressed bytes to hold the 24-byte header of the
# decompressed stream.
_SNIFF_SIZE = 1024


class M3zFormat(FreesurferTransformationFormat):
    """A non-linear transformation stored in a FreeSurfer morph file."""

    HINTS = ("m3z",)


class M3zParser(
    Magic,
    BinaryFileParserWriter,
    repr=HIDE_IF_NONE,
):
    """Reads and writes the raw content of a FreeSurfer morph file."""

    struct: tx.Optional[M3zStruct] = field(default=None, repr=False)
    """The raw content of the file: every node and every tag."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open binary file: only its first bytes are read."""
        with preserve_position(file):
            head = file.read(_SNIFF_SIZE)
        return cls.sniff_bytes(head, error=error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Text is never a morph."""
        return _reject(error, "Not binary content.")

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes, gzipped or not: a morph starts with the version
        `1.0` (a big-endian float), then a positive shape and spacing."""
        if not isinstance(content, (bytes, bytearray, memoryview)):
            return _reject(error, "Not binary content.")
        head = bytes(content)
        if is_gzip(head):
            head = _gunzip_head(head)
        if read_header(head) is None:
            return _reject(error, "Not a FreeSurfer morph (m3z) file.")
        return Confidence.CERTAIN

    # --- from ---------------------------------------------------------

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build the object from the bytes of a `.m3z` (gzipped) or
        `.m3d` (plain) file."""
        return cls(struct=read_m3z(content), **kwargs)

    @classmethod
    def from_struct(cls, struct: M3zStruct, **kwargs) -> tx.Self:
        """Build the object from the raw content of a morph."""
        return cls(struct=struct, **kwargs)

    # --- to -----------------------------------------------------------

    def to_struct(self, **kwargs) -> M3zStruct:
        """The raw content that encodes this object."""
        if self.struct is None:
            raise WriterError("This morph has no content to write.")
        return self.struct

    def to_bytes(self, compress: bool = True, **kwargs) -> bytes:
        """
        The content of the morph file, gzipped (`.m3z`) unless
        `compress=False` (`.m3d`).

        Other keyword arguments go to `to_struct`.
        """
        return write_m3z(self.to_struct(**kwargs), compress=compress)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write the morph to a file, gzipped unless its name ends in `.m3d`
        (FreeSurfer gzips a morph whose name contains `.m3z`).

        The content is built before the file is opened, so a
        transformation that the format cannot hold is refused without
        creating or truncating the file.
        """
        filename = path.Path(filename)
        kwargs.setdefault("compress", not str(filename).endswith(".m3d"))
        content = self.to_bytes(**kwargs)
        with filename.open(self._WRITE_MODE) as f:
            f.write(content)


@register_format
class M3zMorph(
    M3zFormat,
    M3zParser,
    _xforms.ImmutableSequence,
    WritableFileBasedTransformation,
):
    """
    A non-linear transformation stored in a FreeSurfer morph (`.m3z`).

    It maps atlas (target) scanner RAS to source scanner RAS, as
    `mri_vol2vol --m3z` applies it: the source image is resampled onto
    the atlas grid, pulled through the field. It is the
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of

    1. [`RASToVoxel`][brainhops.io.transformations.base.affines.RASToVoxel]:
       atlas RAS to the voxels of the node grid;
    2. a
       [`CoordinatesField`][brainhops.datamodel.transformations.CoordinatesField]:
       the position of each node in source voxels;
    3. [`VoxelToRAS`][brainhops.io.transformations.base.affines.VoxelToRAS]:
       source voxels to source RAS.

    A morph whose positions are in RAS (`GCAM_RAS`) is the sequence of
    the first step and a
    [`RASCoordinatesField`][brainhops.io.transformations.base.fields.RASCoordinatesField].

    The raw content -- original positions, GCA node indices, labels,
    the linear transform, the file names of both volumes -- stays in
    [`struct`][.struct].

    !!! note "What is written"
        A morph read from a file, whose chain has not been assigned, is
        written back as it was read. One whose `transformations` were
        assigned -- including one built from scratch -- is written from
        them (see [`to_struct`][.to_struct]).
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".m3z", ".m3d")

    # --- properties ---------------------------------------------------

    @property
    def spacing(self) -> int:
        """The distance between nodes, in atlas voxels."""
        return 1 if self.struct is None else int(self.struct.spacing)

    @property
    def image_vox2ras(self) -> np.ndarray:
        """The voxel-to-RAS matrix of the source image."""
        return self.struct.image_geometry.vox2ras

    @property
    def atlas_vox2ras(self) -> np.ndarray:
        """The voxel-to-RAS matrix of the atlas (target) volume."""
        return self.struct.atlas_geometry.vox2ras

    @property
    def node_vox2ras(self) -> np.ndarray:
        """The voxel-to-RAS matrix of the node grid.

        Node `n` is atlas voxel `n * spacing` (`GCAMsampleMorph`).
        """
        scale = np.diag([float(self.spacing)] * 3 + [1.0])
        return self.atlas_vox2ras @ scale

    @property
    def affine(self) -> tx.Optional[np.ndarray]:
        """The `(4, 4)` linear transform the morph records, if any."""
        xform = None if self.struct is None else self.struct.xform
        return None if xform is None else xform.matrix

    # --- chain --------------------------------------------------------

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations, in the order they are applied.

        It is derived from `struct` unless it has been assigned.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is not None:
            return explicit
        if self.struct is None:
            return ()
        cached = self.__dict__.get("_m3z_chain")
        if cached is not None and cached[0] is self.struct:
            return cached[1]
        chain = self._build_chain()
        self.__dict__["_m3z_chain"] = (self.struct, chain)
        return chain

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.Sequence[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else tuple(value)

    def _build_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
        struct = self.struct
        ras2node = RASToVoxel(matrix=np.linalg.inv(self.node_vox2ras)[:3])
        options = dict(
            field=struct.positions,
            order=InterpolationOrder.linear,
            bound=BoundaryCondition.nearest,
        )
        if struct.coordinates == GCAM_RAS:
            return (ras2node, RASCoordinatesField(**options))
        voxel = _systems.VoxelCoordinateSystem()
        return (
            ras2node,
            _xforms.CoordinatesField(input=voxel, output=voxel, **options),
            VoxelToRAS(matrix=self.image_vox2ras[:3]),
        )

    # --- to -----------------------------------------------------------

    def to_struct(
        self,
        spacing: tx.Optional[int] = None,
        image_shape: tx.Optional[tx.Sequence[int]] = None,
        atlas_shape: tx.Optional[tx.Sequence[int]] = None,
        **kwargs,
    ) -> M3zStruct:
        """
        The raw content that encodes this morph.

        A morph whose chain has not been assigned is its `struct`. One
        whose chain was assigned must hold, as the reader builds it,
        either three transformations -- atlas RAS to node voxels (an
        affine), a field of source voxel coordinates, source voxels to
        RAS (an affine) -- or two -- the affine and a field of source
        RAS coordinates. The geometries of both volumes are rebuilt from
        the affines; what the chain does not say is taken from `struct`
        when it has one, and from the arguments otherwise.

        Parameters
        ----------
        spacing : int, optional
            The distance between nodes in atlas voxels. By default, that
            of `struct`, or 1.
        image_shape : (int, int, int), optional
            The shape of the source image, which its RAS centre depends
            on. By default, that of `struct`. Required when there is no
            `struct` and the field is in voxels.
        atlas_shape : (int, int, int), optional
            The shape of the atlas. By default, that of `struct`, or the
            shape of the node grid times the spacing.

        Raises
        ------
        UnrepresentableTransformationError
            If the chain does not have one of these two shapes.
        WriterError
            If the shape of the source image is needed and unknown.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is None:
            return super().to_struct()

        old = self.struct
        chain = tuple(explicit)
        field = chain[1] if len(chain) in (2, 3) else None
        if not isinstance(field, _xforms.CoordinatesField) or isinstance(
            field, _xforms.CartesianField
        ):
            raise UnrepresentableTransformationError(
                "A morph is written from a chain of three transformations "
                "(atlas RAS to node voxels, a field of source voxel "
                "coordinates, source voxels to RAS) or two (atlas RAS to "
                "node voxels, a field of source RAS coordinates)."
            )
        if field.coeff:
            raise UnrepresentableTransformationError(
                "A morph stores sampled positions, and this field holds "
                "spline coefficients. Convert it to values first."
            )
        positions = np.asarray(field.field, dtype=np.float32)
        if positions.ndim != 4 or positions.shape[-1] != 3:
            raise UnrepresentableTransformationError(
                f"A morph holds one 3-vector per node of a 3-D grid, not "
                f"an array of shape {positions.shape}."
            )
        shape = tuple(positions.shape[:3])
        same_grid = old is not None and old.shape == shape

        what = "A FreeSurfer morph"
        ras2node = homogeneous_matrix(chain[0], what, 3)
        if spacing is None:
            spacing = 1 if old is None else int(old.spacing)
        scale = np.diag([1.0 / spacing] * 3 + [1.0])
        atlas_vox2ras = np.linalg.inv(ras2node) @ scale
        if atlas_shape is None:
            if old is not None and old.atlas is not None:
                atlas_shape = old.atlas.shape
            else:
                atlas_shape = tuple(s * spacing for s in shape)
        atlas = M3zGeometry.from_vox2ras(
            atlas_vox2ras,
            atlas_shape,
            filename=b"" if old is None else old.atlas_geometry.fname,
        )

        if len(chain) == 3:
            coordinates = GCAM_VOX
            if image_shape is None and old is not None and old.image:
                image_shape = old.image.shape
            if image_shape is None:
                raise WriterError(
                    "The geometry of the source image includes its shape, "
                    "which the chain does not say: pass image_shape=..."
                )
            image = M3zGeometry.from_vox2ras(
                homogeneous_matrix(chain[2], what, 3),
                image_shape,
                filename=b"" if old is None else old.image_geometry.fname,
            )
        else:
            coordinates = GCAM_RAS
            image = None if old is None else old.image
            if image is None:
                image = M3zGeometry()

        if old is None:
            old = M3zStruct()
        return replace(
            old,
            positions=positions,
            original=old.original if same_grid else positions,
            index=old.index if same_grid else np.zeros_like(positions, "i4"),
            labels=old.labels if same_grid else np.zeros(shape, "i4"),
            spacing=int(spacing),
            image=image,
            atlas=atlas,
            type=coordinates,
            xform=old.xform,
        )


def _gunzip_head(content: bytes) -> bytes:
    """Decompress as much of a gzip stream as the bytes allow."""
    try:
        return zlib.decompressobj(zlib.MAX_WBITS | 16).decompress(content)
    except zlib.error:
        return b""


def _reject(error: tx.Union[bool, tx.Type[Exception]], message: str) -> float:
    """Return `Confidence.NO`, or raise if the caller asked for it."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message)
    return Confidence.NO
