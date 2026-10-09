"""The OME-Zarr format of multiscale transformation fields.

A field stored in a Zarr store is read into a [`MultiscaleField`][], in which
each resolution level is a sequence that samples the field on the grid of that
level. The format is registered, so that
[`load`][brainhops.io.transformations.load] discovers it. Arrays and metadata
are kept exactly as read, so that writing an unchanged field re-emits its OME
metadata unchanged.
"""

import abczarr
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

from brainhops._core.affines import inv as _affine_inv
from brainhops.backends import get_array_backend
from brainhops.datamodel._sugar import vector_axis
from brainhops.datamodel._transformations.multiscale import _as_affine
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    MultiscaleField,
    Sequence,
    Transformation,
    is_identity,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserTypeError,
    WriterError,
)
from brainhops.io.common.zarr import StoreLike, ZarrParserWriter
from brainhops.io.common.zarr._parsers import _as_node
from brainhops.io.transformations.base import FileBasedTransformation
from brainhops.io.transformations.zarr import _map, _node


class OmeFieldError(ValueError):
    """An OME-Zarr field that cannot be read.

    The error is raised, in particular, for a displacement field placed by a
    transformation that is not affine, since displacements in world units
    cannot be rescaled to voxels without a linear part. A field whose axes
    cannot be read as one vector field raises an
    [`AxisError`][brainhops.errors.AxisError] instead.
    """


@register_format
class OmeZarrField(ZarrParserWriter, FileBasedTransformation, MultiscaleField):
    """A coordinate or displacement field stored as OME-Zarr.

    The Zarr store is the file format. It is read with `from_store` and written
    with `to_store`, or `from_node` and `to_node` for an opened node (see
    [`ZarrParser`][brainhops.io.common.zarr.ZarrParser]), and
    [`load`][brainhops.io.transformations.load] discovers it.

    The field is a [`MultiscaleField`][], so it composes like any multiscale
    field: reslicing onto a coarser grid picks the matching level, and the
    finest level is used unless a level is selected. The reader holds the array
    of every level, the voxel-to-world transformation of the finest level, the
    axes and the raw OME metadata, all as read, so that an untouched field
    re-emits its metadata unchanged through [`OmeZarrField.to_ome`][].
    """

    HINTS = ("ome", "ome-zarr")

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

    # scales is built on demand from the arrays and the placement, not stored.
    # Declaring it a ClassVar overrides the init field inherited from
    # MultiscaleField, which keeps it out of __init__, fields() and replace().
    scales: tx.ClassVar[tx.Optional[tx.List[Sequence]]]

    @property
    def scales(self) -> tx.Optional[tx.List[Sequence]]:
        """The resolution scales, each a sequence.

        The scales are built on first access and cached. For a displacement
        field, the placement is first checked to be affine.
        """
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
        """Return the OME metadata to write.

        An untouched field returns the object it was read with, so that reading
        and writing round-trips the metadata.
        """
        return self.ome

    @classmethod
    def _score_store(cls, node: tx.Any) -> float:
        # A field is a group whose OME metadata names a displacement or
        # coordinate axis. A plain image pyramid has no such axis.
        if not isinstance(node, abczarr.ZarrGroup):
            return Confidence.NO
        for system in _node.coordinate_systems(node):
            for axis in getattr(system, "axes", None) or []:
                kind = getattr(axis, "type", None)
                if kind in ("displacement", "coordinate"):
                    return Confidence.CERTAIN
        return Confidence.NO

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        """Read a field from an opened Zarr group.

        The node's own OME metadata describes the field. The arrays of all
        levels are read eagerly, and the metadata is kept as read.
        """
        node = _as_node(node)
        if node is None:
            raise ParserTypeError(
                "from_node expects an opened Zarr array or group; pass a "
                "store path to from_store instead."
            )
        ome = getattr(node, "ome", None)
        if ome is None:
            raise OmeFieldError(
                "This node carries no OME metadata, so no field can be read "
                "from it."
            )
        try:
            normalized = ome.to_version("0.6")
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
            node=node,
            **kwargs,
        )

    def to_node(self, node: tx.Any, **kwargs) -> tx.Any:
        """Write the field into an opened Zarr group, and return the group.

        Raises
        ------
        WriterError
            If the node is a plain array, or if the field has no OME metadata
            or no resolution levels.
        """
        # Each level is written at the dataset path named by the metadata, and
        # the metadata is re-emitted unchanged.
        node = _as_node(node)
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
            normalized = ome.to_version("0.6")
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
        return node

    def to_store(self, location: StoreLike, **kwargs) -> None:
        """Write the field to a store location or an opened store.

        An opened group is written into as it stands. A location names a store
        that does not exist yet, so the group is created first.
        """
        node = _as_node(location)
        if node is None:
            node = abczarr.open_group(location, mode="w")
        self.to_node(node, **kwargs)

    # --- level construction ---

    @property
    def _kind(self) -> str:
        # Without axes, the field holds coordinates, with its vector components
        # on the last array axis.
        if not self.axes:
            return "coordinate"
        return vector_axis(self.axes)[0]

    @property
    def _vector_axis(self) -> tx.Optional[int]:
        if not self.axes:
            return None
        return vector_axis(self.axes)[1]

    def _level_voxel2world(self, index: int, ndim: int) -> Transformation:
        # Always a defined affine: the finest placement composed with the
        # transformation from this level's grid to the finest grid.
        base = _affine_or_identity(
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
        # A coordinate level is the world-to-voxel affine followed by the
        # coordinate array. A displacement level also maps back to the world,
        # because brainhops displacements work from voxel to voxel.
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
        # Displacement vectors in world units are mapped through the linear
        # part of the placement, so the placement must reduce to an affine.
        # Identity and CartesianField placements are pure regrids, and are
        # accepted.
        placement = self.voxel2world
        if placement is None or is_identity(placement):
            return
        if isinstance(placement, CartesianField):
            return
        if _as_affine(placement) is None:
            raise OmeFieldError(
                "This OME displacement field is placed by a transformation "
                "that cannot be reduced to an affine, so the displacements "
                "cannot be rescaled to voxel units. Only an affine "
                "placement is supported for a displacement field."
            )


def _vector_axis_last(
    raw: ArrayProtocol, vector_axis: tx.Optional[int]
) -> ArrayProtocol:
    # Move the vector axis last, so that the array is (*grid_shape, ndim).
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
    # World-unit displacements become voxel displacements through the inverse
    # L^-1 of the linear part of the placement: raw @ L^-T.
    matrix = _as_affine(voxel2world).matrix
    inverse = _affine_inv(matrix)
    ab = get_array_backend(matrix)
    raw = ab.asarray(raw)
    return raw @ inverse[:, :-1].T


def _dataset_placement(
    dataset: tx.Any, perm: tx.Sequence[int], ndim: int
) -> Transformation:
    # A level's placement is the dataset's first coordinate transformation, or
    # the identity.
    transforms = list(
        getattr(dataset, "coordinateTransformations", None) or []
    )
    if not transforms:
        return Identity()
    return _map.from_ome(transforms[0], perm, ndim)


def _affine_or_identity(xform: Transformation, ndim: int) -> Transformation:
    # An affine placement is returned unchanged. Otherwise, an identity affine
    # stands in, so that the leading world-to-voxel element stays a defined
    # affine rather than a bare identity.
    if _as_affine(xform) is not None:
        return xform
    return Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim + 1)]
            for i in range(ndim)
        ],
        input=xform.input,
        output=xform.output,
    )
