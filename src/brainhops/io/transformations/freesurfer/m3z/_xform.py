__all__ = ["M3zFormat", "M3zParser", "M3zMorph"]

import zlib

import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Magic, field, replace

from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
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

# Enough compressed bytes to hold the 24-byte header once decompressed.
_SNIFF_SIZE = 1024


class M3zFormat(FreesurferTransformationFormat):
    """Marker of non-linear transformations stored in a FreeSurfer morph."""

    HINTS = ("m3z",)


class M3zParser(
    Magic,
    BinaryFileParserWriter,
    repr=HIDE_IF_NONE,
    eq=False,
):
    """Reader and writer of the raw content of a morph file.

    The parser compares by identity, since its struct holds arrays.
    """

    struct: tx.Optional[M3zStruct] = field(default=None, repr=False)
    """The raw content of the file, every node and every tag.

    See [`M3zStruct`][].
    """

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open binary file from its first bytes only.

        A morph weighs tens of megabytes, but its header only 24 bytes.
        """
        with preserve_position(file):
            head = file.read(_SNIFF_SIZE)
        return cls.sniff_content(head, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes, gzipped or not.

        A morph starts with the version `1.0`, followed by a positive shape and
        spacing (see `read_header`).
        """
        head = bytes(content)
        if is_gzip(head):
            head = _gunzip_head(head)
        if read_header(head) is not None:
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Not a FreeSurfer morph (m3z) file.")
        return Confidence.NO

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build a parser from the bytes of a `.m3z` or `.m3d` file."""
        return cls(struct=read_m3z(content), **kwargs)

    @classmethod
    def from_struct(cls, struct: M3zStruct, **kwargs) -> tx.Self:
        """Build a parser from the raw content of a morph."""
        return cls(struct=struct, **kwargs)

    def to_struct(self, **kwargs) -> M3zStruct:
        """Return the raw content that encodes this object."""
        if self.struct is None:
            raise WriterError("This morph has no content to write.")
        return self.struct

    def to_bytes(self, compress: bool = True, **kwargs) -> bytes:
        """Return the content of a morph file.

        The content is gzipped (`.m3z`) unless `compress=False` (`.m3d`). Other
        keyword arguments go to `to_struct`.
        """
        return write_m3z(self.to_struct(**kwargs), compress=compress)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the morph to a file.

        The file is gzipped unless its name ends in `.m3d`, as in FreeSurfer.
        The content is built first, so an unrepresentable transformation leaves
        the file untouched.
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
    """Non-linear transformation stored in a FreeSurfer morph.

    The morph maps atlas (target) scanner RAS to source scanner RAS, as
    `mri_vol2vol --m3z` applies it to pull the source onto the atlas grid. It
    is the
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of

    1. a [`RASToVoxel`][], from atlas RAS to the voxels of the node grid; 2. a
    [`CoordinatesField`][brainhops.datamodel.transformations.CoordinatesField]
       holding the position of each node in source voxels;
    3. a [`VoxelToRAS`][], from source voxels to source RAS.

    With RAS positions (`GCAM_RAS`), the first step is followed by a
    [`RASCoordinatesField`][] instead.

    The raw content of the file (spacing, geometries, original positions,
    GCA node indices, labels and linear transform) stays in [`struct`][], an
    [`M3zStruct`][], for example `morph.struct.atlas_geometry.vox2ras`.

    !!! note "What is written"
        A morph whose chain was not assigned is written back as it was read.
        One whose `transformations` were assigned, including one built from
        scratch, is written from them (see [`to_struct`][]).
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".m3z", ".m3d")

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain, in the order of application.

        The chain is derived from `struct` unless it was assigned.
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
        # Node n is atlas voxel n * spacing (GCAMsampleMorph).
        scale = np.diag([float(struct.spacing)] * 3 + [1.0])
        node2ras = struct.atlas_geometry.vox2ras @ scale
        ras2node = RASToVoxel(matrix=np.linalg.inv(node2ras)[:3])
        options = dict(
            field=struct.positions,
            degree=InterpolationOrder.linear,
            bound=BoundaryCondition.nearest,
        )
        if struct.coordinates == GCAM_RAS:
            return (ras2node, RASCoordinatesField(**options))
        voxel = _systems.VoxelCoordinateSystem()
        return (
            ras2node,
            _xforms.CoordinatesField(input=voxel, output=voxel, **options),
            VoxelToRAS(matrix=struct.image_geometry.vox2ras[:3]),
        )

    def to_struct(
        self,
        spacing: tx.Optional[int] = None,
        image_shape: tx.Optional[tx.Sequence[int]] = None,
        atlas_shape: tx.Optional[tx.Sequence[int]] = None,
        **kwargs,
    ) -> M3zStruct:
        """Return the raw content that encodes this morph.

        Without an assigned chain, the result is `struct`. An assigned chain
        must have the shape that the reader builds: three transformations
        (atlas RAS to node voxels, a field of source voxel coordinates, source
        voxels to RAS), or two (the first affine and a field of source RAS
        coordinates). Both geometries are rebuilt from the affines, and
        whatever the chain does not describe comes from `struct` if there is
        one, or from the arguments.

        Parameters
        ----------
        spacing : int, optional
            Atlas voxels between nodes. Defaults to that of `struct`, or 1.
        image_shape : (int, int, int), optional
            Shape of the source image, on which its RAS centre depends.
            Defaults to that of `struct`; required without a struct for a voxel
            field.
        atlas_shape : (int, int, int), optional
            Shape of the atlas. Defaults to that of `struct`, or to the node
            grid shape times `spacing`.

        Raises
        ------
        UnrepresentableTransformationError
            If the chain has neither of the two shapes.
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
        # A morph stores sampled positions, not spline coefficients.
        positions = np.asarray(field.to(store="values").data, dtype=np.float32)
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
    """Decompress the start of a gzip stream, up to `_SNIFF_SIZE` bytes."""
    try:
        stream = zlib.decompressobj(zlib.MAX_WBITS | 16)
        return stream.decompress(content, _SNIFF_SIZE)
    except zlib.error:
        return b""
