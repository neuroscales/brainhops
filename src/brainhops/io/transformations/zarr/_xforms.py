"""The OME-Zarr multiscale field format.

An OME-Zarr coordinate or displacement field is a Zarr store, so it is a
file format. It is read into a
[`MultiscaleField`][brainhops.datamodel.transformations.MultiscaleField]
and written back out, and it registers so that
[`load`][brainhops.io.transformations.load] discovers it like any other
transformation format. Each resolution level is built as a sequence that
samples the field on that level's grid. The reader keeps the arrays and the
metadata exactly as they were read, so a field that is read and written
again re-emits its OME metadata unchanged.
"""

# dependencies
import abczarr
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
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.base.zarr import ZarrParser
from brainhops.io.transformations.base import WritableFileBasedTransformation
from brainhops.io.transformations.zarr import _map, _node


class OmeFieldError(ValueError):
    """Raised when an OME-Zarr field cannot be read.

    A displacement field placed by a non-affine transformation is refused
    with this error, because its world-unit displacements cannot be
    rescaled to voxel units without a linear part. A field whose axes
    cannot be read as one vector field is refused separately, with an
    [`AxisError`][brainhops.datamodel.axes.AxisError].
    """


@register_format
class OmeZarrField(
    ZarrParser, WritableFileBasedTransformation, MultiscaleField
):
    """A coordinate or displacement field stored as OME-Zarr.

    An OME-Zarr field is a Zarr store, so this is a file format. It is read
    from a store with
    [`from_store`][brainhops.io.base.zarr.ZarrParser.from_store] and written
    with [`to_store`][brainhops.io.base.zarr.ZarrParser.to_store], and it is
    discoverable through
    [`load`][brainhops.io.transformations.load] like any other
    transformation format. An already-opened Zarr node is read with
    [`from_node`][brainhops.io.base.zarr.ZarrParser.from_node] and written
    with [`to_node`][brainhops.io.base.zarr.ZarrParser.to_node].

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

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr", ".ome.zarr")

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
    def _score_store(cls, node: tx.Any) -> float:
        # An OME-Zarr field is a group whose own OME metadata names a
        # displacement or coordinate component axis. A plain image pyramid
        # names no such axis, so it is not read as a field.
        if not isinstance(node, abczarr.ZarrGroup):
            return Confidence.NO
        for system in _node.coordinate_systems(node):
            for axis in getattr(system, "axes", None) or []:
                kind = getattr(axis, "type", None)
                if kind in ("displacement", "coordinate"):
                    return Confidence.CERTAIN
        return Confidence.NO

    @classmethod
    def _read_node(cls, node: tx.Any, **kwargs) -> "OmeZarrField":
        # Build the field from an opened node whose own OME metadata
        # describes it. The typed axes name the axis that holds the vector
        # components, the resolution levels are read from the datasets the
        # metadata names, and the placement of each level is mapped from the
        # level's coordinate transformation. The arrays and the metadata are
        # kept exactly as read, so a field that is read and written again
        # re-emits its OME metadata unchanged.
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

    def _write_node(self, node: tx.Any, **kwargs) -> None:
        # Write the field's arrays and its OME metadata into an opened group.
        # Each level array is written to the dataset path the metadata names,
        # and the metadata is re-emitted unchanged, so a read followed by a
        # write round-trips the store.
        if not isinstance(node, abczarr.ZarrGroup):
            raise WriterError(
                "An OME-Zarr field is written into a group, not a plain array."
            )
        ome = self.to_ome()
        if ome is None:
            raise WriterError(
                "This OME-Zarr field has no OME metadata, so there is nothing "
                "to write."
            )
        raw_levels = self.raw_levels or []
        if not raw_levels:
            raise WriterError(
                "This OME-Zarr field has no resolution levels to write."
            )
        try:
            normalized = ome.to_version("0.6rc0")
        except Exception:
            normalized = ome
        multiscales = getattr(normalized, "multiscales", None) or []
        datasets = list(multiscales[0].datasets) if multiscales else []
        backend = get_array_backend()
        for index, array in enumerate(raw_levels):
            dataset_path = (
                str(datasets[index].path)
                if index < len(datasets)
                else str(index)
            )
            node.create_array(
                dataset_path, data=backend.asarray(array), **kwargs
            )
        node.ome = ome

    def _create_store(self, location: str, **kwargs) -> None:
        # Create a store at `location` and write the field into it.
        group = abczarr.open_group(location, mode="w")
        self._write_node(group, **kwargs)

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
