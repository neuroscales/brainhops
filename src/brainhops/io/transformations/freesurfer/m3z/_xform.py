"""The chain of transformations stored in a FreeSurfer morph."""

__all__ = ["M3zFormat", "M3zMorph"]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    homogeneous_matrix,
)

# this format
from .._formats import FreesurferTransformationFormat
from ._metadata import M3zMetadata
from ._raw import GCAM_RAS, GCAM_VOX, M3zGeometry, M3zRaw

# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# A morph is parsed in one pass, so the positions of its nodes live in
# the record of its metadata. The field of the decoded chain holds a
# read-only view of the array of the record, so the chain is a view of
# the record and decoding it copies nothing. A chain that is assigned
# is held by the data model, and it is encoded into a new record only
# when the morph is written, because encoding needs the options of the
# writer.


def _record(xform: "M3zMorph") -> tx.Optional[M3zRaw]:
    """Return the record of the metadata of a morph, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _positions_to_model(positions: np.ndarray) -> np.ndarray:
    """Return the data of the field of a chain from the positions of a record.

    The field holds the positions in the layout of the record, indexed
    `[x, y, z]`, so the result is a view of the array of the record,
    without a copy. The view is read-only even when the array of the
    record is not, as after the record was unpickled, because an edit in
    place would change the record.
    """
    view = positions.view()
    view.flags.writeable = False
    return view


def _positions_to_disk(data: ArrayProtocol) -> np.ndarray:
    """Return the positions that a record stores for the data of a field.

    The record stores single-precision floats. The array is a new one,
    and it is read-only, as the arrays of a record that was read are.
    """
    positions = np.array(data, dtype=np.float32)
    positions.flags.writeable = False
    return positions


def _decode(record: M3zRaw) -> tx.Tuple[_xforms.Transformation, ...]:
    """Return the chain of transformations that a record describes.

    The chain is described in [`M3zMorph`][]. Its field holds a
    read-only view of the positions of the record, without a copy.
    """
    # Node n is atlas voxel n * spacing (GCAMsampleMorph).
    scale = np.diag([float(record.spacing)] * 3 + [1.0])
    node2ras = record.atlas_geometry.vox2ras @ scale
    ras2node = RASToVoxel(matrix=np.linalg.inv(node2ras)[:3])
    options = dict(
        field=_positions_to_model(record.positions),
        degree=InterpolationOrder.linear,
        bound=BoundaryCondition.nearest,
    )
    if record.coordinates == GCAM_RAS:
        return (ras2node, RASCoordinatesField(**options))
    voxel = _systems.VoxelCoordinateSystem()
    return (
        ras2node,
        _xforms.CoordinatesField(input=voxel, output=voxel, **options),
        VoxelToRAS(matrix=record.image_geometry.vox2ras[:3]),
    )


def _set_transformations(
    self: "M3zMorph",
    value: tx.Optional[tx.Sequence[_xforms.Transformation]],
) -> None:
    """Store an assigned chain as a tuple and forget the decoded chain."""
    self._transformations = None if value is None else tuple(value)
    self.__dict__.pop("_cache_transformations", None)


# ----------------------------------------------------------------------
#   FORMAT
# ----------------------------------------------------------------------


class M3zFormat(FreesurferTransformationFormat):
    """Marker of non-linear transformations stored in a FreeSurfer morph."""

    HINTS = ("m3z",)


@register_format
class M3zMorph(
    M3zFormat,
    BinaryFileReader,
    BinaryFileWriter,
    _xforms.ImmutableSequence,
    TransformationFormat,
):
    """Non-linear transformation stored in a FreeSurfer morph.

    The morph maps scanner RAS coordinates of the atlas, which is the
    target, to scanner RAS coordinates of the source image. This is the
    direction in which `mri_vol2vol --m3z` applies the morph to pull the
    source onto the atlas grid. The morph is the
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of

    1. a [`RASToVoxel`][], from atlas RAS to the voxels of the node grid;
    2. a
       [`CoordinatesField`][brainhops.datamodel.transformations.CoordinatesField]
       that holds the position of each node in source voxels;
    3. a [`VoxelToRAS`][], from source voxels to source RAS.

    When the positions are RAS coordinates (`GCAM_RAS`), the first step is
    followed by a [`RASCoordinatesField`][] instead, and there is no third
    step.

    A morph file is read in one pass, so the whole file is held as an
    [`M3zRaw`][] record by the [`M3zMetadata`][] of the morph. The
    record includes the spacing, the geometries, the positions, the
    original positions, the GCA node indices, the labels and the linear
    transform. The chain is decoded from the record when it is first
    used, and it is then cached. Its field holds a read-only view of the
    positions of the record. For example, the voxel-to-RAS matrix of
    the atlas is `morph.metadata.raw.atlas_geometry.vox2ras`. A morph
    built from a chain has no metadata.

    !!! note "What is written"
        A morph whose chain was not assigned is written back as the
        record that it read. A morph whose `transformations` were
        assigned, including a morph built from a chain, is written from
        the assigned chain, as described in [`to_raw`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".m3z", ".m3d")

    _metadata: KwOnly[NoRepr[tx.Optional[M3zMetadata]]] = None

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds the file as a record.

    A morph built from a chain has no metadata. Assigning other metadata
    drops the chain decoded from the old record, and a chain that was
    assigned is kept.
    """

    @smartproperty(cache=True, fset=_set_transformations)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain, in the order of application.

        The chain is decoded from the record when it is first used, and
        it is then cached. A morph without a record and without an
        assigned chain has an empty chain. Assigning a chain replaces
        the decoded one, and the assigned chain is what the writer
        encodes.
        """
        record = _record(self)
        if record is None:
            return ()
        return _decode(record)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open binary file from its first bytes only.

        The file is scored by [`M3zMetadata.sniff_fileobj`][].
        """
        return M3zMetadata.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes, gzipped or not.

        The bytes are scored by [`M3zMetadata.sniff_bytes`][].
        """
        return M3zMetadata.sniff_bytes(content, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read a morph from the rest of an open binary file.

        The record is read by [`M3zMetadata.from_fileobj`][], and the
        chain is decoded from it when it is first used. Keyword
        arguments go to the constructor.
        """
        return cls(metadata=M3zMetadata.from_fileobj(file), **kwargs)

    @classmethod
    def from_raw(cls, raw: M3zRaw, **kwargs) -> tx.Self:
        """Build a morph from an already read [`M3zRaw`][].

        The record is held by new metadata, without a copy. Keyword
        arguments go to the constructor.
        """
        return cls(metadata=M3zMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a morph from a record, a file or a data model.

        An [`M3zRaw`][] is read with [`from_raw`][], and any other value
        as the bases read it.
        """
        if isinstance(other, M3zRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def to_raw(
        self,
        spacing: tx.Optional[int] = None,
        image_shape: tx.Optional[tx.Sequence[int]] = None,
        atlas_shape: tx.Optional[tx.Sequence[int]] = None,
    ) -> M3zRaw:
        """Return the [`M3zRaw`][] that encodes this morph.

        When no chain has been assigned, the record that the metadata
        holds is returned as it is, without a copy, and the arguments
        are not used. An assigned chain is encoded into a new record,
        and it must have one of the two shapes that the reader builds.
        The first shape has three transformations: atlas RAS to node
        voxels, a field of source voxel coordinates, and source voxels
        to RAS. The second shape has two: the same first affine,
        followed by a field of source RAS coordinates.

        The atlas geometry is rebuilt from the first affine, and with
        three transformations, the source geometry is rebuilt from the
        last one. Whatever the chain does not describe is taken from the
        arguments, then from the current record if there is one, and
        otherwise from defaults. The original positions, the GCA node
        indices and the labels of the current record are kept when the
        node grid keeps its shape, and they are shared with the current
        record.

        Parameters
        ----------
        spacing : int, optional
            Number of atlas voxels between nodes. The default is the
            spacing of the current record, or 1 when there is no record.
        image_shape : (int, int, int), optional
            Shape of the source image, on which the RAS centre of the
            image depends. The default is the shape that the current
            record records. The argument is required for a field of
            voxel coordinates when no source geometry is recorded.
        atlas_shape : (int, int, int), optional
            Shape of the atlas. The default is the shape that the
            current record records, or else the shape of the node grid
            multiplied by `spacing`.

        Returns
        -------
        M3zRaw
            The record that encodes the morph.

        Raises
        ------
        UnrepresentableTransformationError
            If the chain has neither of the two shapes, or if its field is
            not a 3-D grid of 3-vectors.
        WriterError
            If the morph has neither a record nor a chain, or if the
            shape of the source image is needed but unknown.
        """
        record = _record(self)
        if getattr(self, "_transformations", None) is None:
            if record is None:
                raise WriterError("This morph has no content to write.")
            return record
        return _encode(
            self.transformations, record, spacing, image_shape, atlas_shape
        )

    def to_bytes(self, compress: bool = True, **kwargs) -> bytes:
        """Return the content of the morph file that encodes this morph.

        The content is gzipped (`.m3z`) unless `compress=False`
        (`.m3d`). Other keyword arguments go to [`to_raw`][].
        """
        return self.to_raw(**kwargs).to_bytes(compress=compress)

    def to_filename(
        self,
        filename: path.FilenameLike,
        compress: tx.Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Write the morph to a file.

        By default, the file is gzipped unless its name ends in `.m3d`,
        as in FreeSurfer. Other keyword arguments go to [`to_raw`][].
        The record is built first, so a morph that cannot be encoded
        leaves the file untouched.
        """
        self.to_raw(**kwargs).to_filename(filename, compress=compress)


# ----------------------------------------------------------------------
#   ENCODING A CHAIN AS A RECORD
# ----------------------------------------------------------------------


def _encode(
    chain: tx.Sequence[_xforms.Transformation],
    old: tx.Optional[M3zRaw],
    spacing: tx.Optional[int],
    image_shape: tx.Optional[tx.Sequence[int]],
    atlas_shape: tx.Optional[tx.Sequence[int]],
) -> M3zRaw:
    """Encode a chain of transformations as a new record.

    The rules are described in [`M3zMorph.to_raw`][]. The current
    record, `old`, is never changed: the new record replaces its fields,
    and it shares the arrays that it keeps.
    """
    chain = tuple(chain)
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
    positions = _positions_to_disk(field.to(store="values").data)
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

    if same_grid:
        original, index, labels = old.original, old.index, old.labels
    else:
        original = positions
        index = _zeros(positions.shape, np.int32)
        labels = _zeros(shape, np.int32)
    if old is None:
        old = M3zRaw()
    return replace(
        old,
        positions=positions,
        original=original,
        index=index,
        labels=labels,
        spacing=int(spacing),
        image=image,
        atlas=atlas,
        type=coordinates,
    )


def _zeros(shape: tx.Tuple[int, ...], dtype: type) -> np.ndarray:
    """Return a read-only array of zeros for a new record."""
    zeros = np.zeros(shape, dtype)
    zeros.flags.writeable = False
    return zeros
