"""The OME-Zarr multiscale field reader.

An OME-Zarr coordinate or displacement field is read as a sequence of a
world-to-voxel affine, a multiscale field, and a voxel-to-world affine.
The reader keeps the arrays and the metadata exactly as they were read,
so a field that is read and written again re-emits its OME metadata
unchanged.
"""

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# internals
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.transformations import (
    MultiscaleCoordinatesField,
    MultiscaleDisplacementField,
    check_ome_axes,
    check_ome_displacement_placement,
    normalize_ome_coordinates,
    normalize_ome_displacement,
    place_ome_field,
)


class OmeZarrField(_xforms.Sequence):
    """A coordinate or displacement field read from OME-Zarr.

    The field is placed in world space with the sandwich construction, so
    it composes with the rest of the data model exactly as
    [`SPMCoordinatesField`][brainhops.io.transformations.spm.SPMCoordinatesField]
    does. The active resolution level is the finest one by default, and a
    reslice onto a coarser grid activates the matching level.

    The reader holds the array of every level, the placement, the axes,
    and the raw OME metadata, all exactly as they were read. A field that
    is read and not modified therefore re-emits its OME metadata
    unchanged.

    A displacement field placed by a non-linear transformation, and a
    field whose axes mix the displacement and coordinate types, are both
    refused with an
    [`OmePlacementError`][brainhops.datamodel.transformations.OmePlacementError].
    """

    raw_levels: tx.Annotated[
        tx.Optional[tx.List[ArrayProtocol]],
        tx.Doc(
            """
            The array of each resolution level, ordered from finest to
            coarsest, exactly as stored in the file. The values are in
            world units.
            """
        ),
    ] = None

    placement: tx.Annotated[
        tx.Optional[_xforms.Transformation],
        tx.Doc(
            """
            The voxel-to-world transformation of the finest level, called
            `xform_0`. It is the coordinate transformation of the finest
            dataset in the OME metadata.
            """
        ),
    ] = None

    level_transforms: tx.Annotated[
        tx.Optional[tx.List[_xforms.Transformation]],
        tx.Doc(
            """
            The transformation that maps each level's grid to the finest
            level's grid, ordered from finest to coarsest. The first entry
            is the identity.
            """
        ),
    ] = None

    axes: tx.Annotated[
        tx.Optional[tx.List[tx.Any]],
        tx.Doc("The axes declared by the OME metadata."),
    ] = None

    ome_metadata: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The OME metadata exactly as read. The reader never rewrites
            this object, so it can be re-emitted unchanged.
            """
        ),
    ] = None

    @property
    def kind(self) -> str:
        """Whether the field is a coordinate field or a displacement field.

        The axes are classified by their type. A field whose axes mix the
        displacement and coordinate types is refused.
        """
        return check_ome_axes(self.axes)

    @property
    def field(self) -> _xforms.Transformation:
        """The placed multiscale field, in voxel-to-voxel form.

        The values read from the file are in world units. Each level is
        converted to voxel-to-voxel form against its own placement, so
        the field works in voxel coordinates inside the sandwich.
        """
        kind = self.kind
        placement = self.placement
        transforms = self.level_transforms or [
            _xforms.Identity() for _ in (self.raw_levels or [])
        ]
        if kind == "displacement":
            check_ome_displacement_placement(placement)
            normalize = normalize_ome_displacement
            field_cls = MultiscaleDisplacementField
        else:
            normalize = normalize_ome_coordinates
            field_cls = MultiscaleCoordinatesField
        levels = []
        for index, raw in enumerate(self.raw_levels or []):
            level_placement = _level_placement(placement, transforms, index)
            levels.append(normalize(raw, level_placement))
        return field_cls(
            levels=levels,
            level_transforms=transforms,
            input=placement.output,
            output=placement.output,
        )

    @property
    def transformations(
        self,
    ) -> tx.List[_xforms.Transformation]:
        """The world-to-voxel affine, the field, and the voxel-to-world affine.

        This is the sandwich `Sequence([~xform_0, field, xform_0])`.
        """
        cached = getattr(self, "_transformations", None)
        if cached is not None:
            return cached
        return place_ome_field(self.field, self.placement).transformations

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.List[_xforms.Transformation]]
    ) -> None:
        # `None` is the inherited default and means "derive the sandwich
        # from the levels and the placement". Only a real sequence is
        # frozen onto the instance.
        self._transformations = None if value is None else list(value)

    def to_ome_metadata(self) -> tx.Any:
        """Return the OME metadata to write for this field.

        An untouched field returns the metadata it was read with, the
        identical object, so a read followed by a write re-emits the OME
        metadata unchanged.
        """
        return self.ome_metadata


def _level_placement(
    placement: _xforms.Transformation,
    transforms: tx.List[_xforms.Transformation],
    index: int,
) -> _xforms.Transformation:
    # The voxel-to-world transformation of one level, obtained by
    # composing the finest level's placement with the transformation that
    # maps that level's grid to the finest grid.
    if index < len(transforms) and transforms[index] is not None:
        cross = transforms[index]
        if not _xforms.is_identity(cross):
            return (placement @ cross).compute()
    return placement
