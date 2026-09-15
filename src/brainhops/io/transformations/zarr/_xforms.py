"""The OME-Zarr multiscale field reader.

An OME-Zarr coordinate or displacement field is read as a
[`MultiscaleField`][brainhops.datamodel.transformations.MultiscaleField].
Each resolution level is built as a sequence that samples the field on
that level's grid. The reader keeps the arrays and the metadata exactly
as they were read, so a field that is read and written again re-emits its
OME metadata unchanged.
"""

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# core
from brainhops._core.affines import inv as _affine_inv

# backends
from brainhops.backends import get_array_backend

# internals
from brainhops.datamodel.axes import Axis, vector_axis
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    MultiscaleField,
    Sequence,
    Transformation,
    _affine_matrix,
    is_identity,
)
from brainhops.io.transformations.zarr import _node


class OmeFieldError(ValueError):
    """Raised when an OME-Zarr field cannot be read.

    A displacement field placed by a non-affine transformation is refused
    with this error, because its world-unit displacements cannot be
    rescaled to voxel units without a linear part. A field whose axes
    cannot be read as one vector field is refused separately, with an
    [`AxisError`][brainhops.datamodel.axes.AxisError].
    """


class OmeZarrField(MultiscaleField):
    """A coordinate or displacement field read from OME-Zarr.

    The field is a
    [`MultiscaleField`][brainhops.datamodel.transformations.MultiscaleField],
    so it composes with the rest of the data model exactly as any
    multiscale field does, and a reslice onto a coarser grid selects the
    matching resolution level. The finest level is used unless a level is
    selected.

    The reader holds the array of every level, the finest level's
    voxel-to-world transformation, the axes, and the raw OME metadata, all
    exactly as they were read. A field that is read and not modified
    therefore re-emits its OME metadata unchanged through
    [`to_ome`][brainhops.io.transformations.zarr.OmeZarrField.to_ome].

    A displacement field placed by a non-affine transformation is refused
    with an
    [`OmeFieldError`][brainhops.io.transformations.zarr.OmeFieldError]. A
    field whose axes mix the displacement and coordinate types is refused
    with an [`AxisError`][brainhops.datamodel.axes.AxisError].
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

    voxel2world: tx.Annotated[
        tx.Optional[Transformation],
        tx.Doc(
            """
            The voxel-to-world transformation of the finest level. It is
            the coordinate transformation of the finest dataset in the OME
            metadata.
            """
        ),
    ] = None

    level_transforms: tx.Annotated[
        tx.Optional[tx.List[Transformation]],
        tx.Doc(
            """
            The transformation that maps each level's grid to the finest
            level's grid, ordered from finest to coarsest. The first entry
            is the identity.
            """
        ),
    ] = None

    axes: tx.Annotated[
        tx.Optional[tx.List[Axis]],
        tx.Doc("The axes declared by the OME metadata."),
    ] = None

    ome: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The OME metadata exactly as read. The reader never rewrites
            this object, so it can be re-emitted unchanged.
            """
        ),
    ] = None

    # `scales` is built on demand from the arrays and the placement rather
    # than stored, so it is not a constructor-taken field here. Declaring
    # it a `ClassVar` overrides the inherited init-field from
    # `MultiscaleField` and keeps it out of `__init__`, `fields()` and
    # `replace()`, while the property builds and caches each scale once.
    scales: tx.ClassVar[tx.Optional[tx.List[Sequence]]]

    @property
    def scales(self) -> tx.Optional[tx.List[Sequence]]:
        """The resolution scales, each built as a sequence."""
        cached = getattr(self, "_scales", None)
        if cached is not None:
            return cached
        count = len(self.raw_levels or [])
        if self._kind == "displacement":
            self._check_affine_voxel2world()
        built = [self._level(index) for index in range(count)]
        self._scales = built
        return built

    def to_ome(self) -> tx.Any:
        """Return the OME metadata to write for this field.

        An untouched field returns the metadata it was read with, the
        identical object, so a read followed by a write re-emits the OME
        metadata unchanged.
        """
        return self.ome

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> "OmeZarrField":
        """Read a coordinate or displacement field from an OME-Zarr node.

        `node` is an opened abczarr node whose own ``ome`` metadata
        describes the field. The field's typed axes name the axis that holds
        the vector components, the resolution levels are read from the
        datasets the metadata names, and the placement of each level is
        mapped from the level's coordinate transformation. The arrays and
        the metadata are kept exactly as read, so a field that is read and
        written again re-emits its OME metadata unchanged.

        A node that carries no OME field metadata, or metadata that names no
        datasets, is refused with an
        [`OmeFieldError`][brainhops.io.transformations.zarr.OmeFieldError].
        """
        ome = getattr(node, "ome", None)
        if ome is None:
            raise OmeFieldError(
                "This node carries no OME metadata, so no field can be read "
                "from it."
            )
        try:
            normalized = ome.to_version("0.6rc0")
        except Exception as error:
            raise OmeFieldError(
                "This node's OME metadata could not be read as a field. "
                + str(error)
            ) from error
        multiscales = getattr(normalized, "multiscales", None) or []
        datasets = list(multiscales[0].datasets) if multiscales else []
        if not datasets:
            raise OmeFieldError(
                "This node's OME metadata names no field datasets, so it has "
                "no resolution levels to read."
            )
        backend = get_array_backend()
        raw_levels = [
            backend.asarray(_node.follow_path(node, str(ds.path))[...])
            for ds in datasets
        ]
        ndim = int(raw_levels[0].ndim)
        axes = _node.typed_axes(node, ndim)
        vector_count = sum(
            1
            for axis in (axes or [])
            if axis.type in ("displacement", "coordinate")
        )
        spatial_ndim = ndim - vector_count
        perm = list(range(spatial_ndim))
        placements = [
            _dataset_placement(ds, perm, spatial_ndim) for ds in datasets
        ]
        voxel2world = placements[0]
        world2voxel = voxel2world.inverse()
        level_transforms = [
            (world2voxel @ placement).compute() for placement in placements
        ]
        return cls(
            raw_levels=raw_levels,
            voxel2world=voxel2world,
            level_transforms=level_transforms,
            axes=axes,
            ome=ome,
            **kwargs,
        )

    # --- level construction ---

    @property
    def _kind(self) -> str:
        # Whether the field is a coordinate field or a displacement field,
        # read from the axes. A field with no axes is read as a coordinate
        # field with its vector components on the last array axis.
        if not self.axes:
            return "coordinate"
        return vector_axis(self.axes)[0]

    @property
    def _vector_axis(self) -> tx.Optional[int]:
        # The index of the array axis that holds the vector components, or
        # `None` when no axes are given.
        if not self.axes:
            return None
        return vector_axis(self.axes)[1]

    def _level_voxel2world(self, index: int, ndim: int) -> Transformation:
        # The voxel-to-world transformation of one level, always a defined
        # affine. The finest level's placement is composed with the
        # transformation that maps this level's grid to the finest grid.
        base = _as_affine(
            self.voxel2world if self.voxel2world is not None else Identity(),
            ndim,
        )
        transforms = self.level_transforms or []
        if (
            index < len(transforms)
            and transforms[index] is not None
            and not is_identity(transforms[index])
        ):
            return (base @ transforms[index]).compute()
        return base

    def _level(self, index: int) -> Sequence:
        # One resolution level, built as a sequence sampled on that level's
        # grid. A coordinate level is the world-to-voxel affine followed by
        # the coordinate array, stored without a copy. A displacement level
        # adds the voxel-to-world affine, because a brainhops displacement
        # works from voxel coordinates to voxel coordinates.
        raw = _vector_axis_last(self.raw_levels[index], self._vector_axis)
        ndim = int(raw.shape[-1])
        voxel2world = self._level_voxel2world(index, ndim)
        world2voxel = voxel2world.inverse()
        if self._kind == "displacement":
            field = DisplacementField(
                field=_to_voxel_displacement(raw, voxel2world)
            )
            elements = [world2voxel, field, voxel2world]
        else:
            elements = [world2voxel, CoordinatesField(field=raw)]
        return Sequence(
            transformations=elements,
            input=self.input,
            output=self.output,
        )

    def _check_affine_voxel2world(self) -> None:
        # A displacement field is placed by mapping its world-unit vectors
        # through the linear part of the placement, so the placement must
        # reduce to an affine. An identity placement and a
        # `CartesianField` placement are pure regrids and are accepted.
        placement = self.voxel2world
        if placement is None or is_identity(placement):
            return
        if isinstance(placement, CartesianField):
            return
        if _affine_matrix(placement) is None:
            raise OmeFieldError(
                "This OME displacement field is placed by a transformation "
                "that cannot be reduced to an affine, so the displacements "
                "cannot be rescaled to voxel units. Only an affine "
                "placement is supported for a displacement field."
            )


def _vector_axis_last(
    raw: ArrayProtocol, vector_axis: tx.Optional[int]
) -> ArrayProtocol:
    # Move the vector-component axis to the last position, so the array
    # has the `(*grid_shape, ndim)` shape the field expects. A `None`
    # index, or one already last, leaves the array unchanged.
    if vector_axis is None:
        return raw
    ndim = len(raw.shape)
    if vector_axis in (-1, ndim - 1):
        return raw
    ab = get_array_backend(raw)
    return ab.moveaxis(raw, vector_axis, -1)


def _to_voxel_displacement(
    raw: ArrayProtocol, voxel2world: Transformation
) -> ArrayProtocol:
    # Convert a world-unit displacement field to voxel-to-voxel form. A
    # brainhops displacement works in voxel units, so each world-unit
    # vector is mapped through the inverse of the linear part of the
    # level's placement. With `L` the linear part, the voxel-space field
    # is `raw @ inverse(L).T`.
    matrix = _affine_matrix(voxel2world)
    inverse = _affine_inv(matrix)
    ab = get_array_backend(matrix)
    raw = ab.asarray(raw)
    return raw @ inverse[:, :-1].T


def _dataset_placement(
    dataset: tx.Any, perm: tx.Sequence[int], ndim: int
) -> Transformation:
    # The voxel-to-world placement of one field level, mapped from the
    # dataset's first coordinate transformation. A dataset that names no
    # transformation is placed by the identity.
    from brainhops.io.transformations.zarr import _map

    transforms = list(
        getattr(dataset, "coordinateTransformations", None) or []
    )
    if not transforms:
        return Identity()
    return _map.from_ome(transforms[0], perm, ndim)


def _as_affine(xform: Transformation, ndim: int) -> Transformation:
    # A defined affine for a level's placement. A transformation that
    # already reduces to an affine with a matrix is returned unchanged, so
    # an incoming affine keeps its identity. Otherwise an identity affine
    # of the given number of dimensions stands in, which keeps the leading
    # world-to-voxel element a defined affine and avoids composing with a
    # bare identity.
    if _affine_matrix(xform) is not None:
        return xform
    return Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim + 1)]
            for i in range(ndim)
        ],
        input=xform.input,
        output=xform.output,
    )
