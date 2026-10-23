"""The OME-Zarr format of multiscale transformation fields.

A field stored in a Zarr store is read into a [`MultiscaleField`][], in
which each resolution level is a sequence that samples the field on the
grid of that level. The format is registered, so that
[`load`][brainhops.io.transformations.load] discovers it. Arrays and
metadata are kept exactly as read, so that writing an unchanged field
re-emits its OME metadata unchanged.
"""

import abczarr
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import Forget, KwOnly

from brainhops._core.affines import inv as _affine_inv
from brainhops.backends import get_array_backend, to_backend, to_numpy
from brainhops.datamodel._sugar import vector_axis
from brainhops.datamodel._transformations.multiscale import _as_affine
from brainhops.datamodel.axes import AxisList
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinateField,
    DisplacementField,
    Identity,
    MultiscaleField,
    Scaling,
    Sequence,
    Transformation,
    Translation,
    is_identity,
)
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import is_local, replace_directory
from brainhops.io.base.parsers import (
    Confidence,
    ParserTypeError,
    WriterError,
)
from brainhops.io.common.zarr import StoreLike
from brainhops.io.common.zarr._data import (
    _ZarrProxy,
    reads_inside,
    store_chunks,
    store_location,
    write_into,
)
from brainhops.io.common.zarr._parsers import (
    _as_node,
    _ZarrNodeReaderWriter,
    absolute_location,
)
from brainhops.io.transformations.base import TransformationFormat

from . import _map, _node
from ._errors import OmeFieldError

# typing
if tx.TYPE_CHECKING:
    # This is imported for mkdocstrings to resolve cross-references
    from brainhops.io.common.zarr import ZarrReader  # noqa: F401
    from brainhops.io.transformations import load  # noqa: F401


@register_format
class OmeZarrField(
    _ZarrNodeReaderWriter, TransformationFormat, MultiscaleField
):
    """A coordinate or displacement field stored as OME-Zarr.

    The Zarr store is the file format. It is read with [`from_store`][]
    and written with [`to_store`][], or [`from_node`][] and [`to_node`][]
    for an opened node (see [`ZarrReader`][]), and [`load`][] discovers
    it.

    The field is a [`MultiscaleField`][], so it composes like any
    multiscale field: reslicing onto a coarser grid picks the matching
    level, and the finest level is used unless a level is selected. The
    reader holds the array of every level, the voxel-to-world
    transformation of the finest level, the axes and the raw OME metadata,
    all as read, so that an untouched field re-emits its metadata
    unchanged through [`to_ome`][].
    """

    HINTS = "ome", "ome-zarr"
    EXTENSIONS = ".zarr", ".ome.zarr"

    raw_levels: tx.Optional[tx.List[ArrayProtocol]] = None
    """
    The array of each resolution level, ordered from finest to coarsest,
    exactly as stored in the file. The values are in world units. A
    field read from a store holds a lazy array of each level, which
    reads only the chunks that a selection needs (see
    [`ArrayProxy`][brainhops._core.proxy.ArrayProxy]), and which becomes
    a Dask array with the chunks of the store under the backend setting
    `"dask"`.
    """

    voxel2world: tx.Optional[Transformation] = None
    """
    The voxel-to-world transformation of the finest level. It is the
    coordinate transformation of the finest dataset in the OME metadata.
    """

    level_transforms: tx.Optional[tx.List[Transformation]] = None
    """
    The transformation that maps each level's grid to the finest level's
    grid, ordered from finest to coarsest. The first entry is the identity.
    """

    axes: tx.Optional[AxisList] = None
    """The axes declared by the OME metadata."""

    ome: tx.Optional[tx.Any] = None
    """
    The OME metadata exactly as read. The reader never rewrites this
    object, so it can be re-emitted unchanged.
    """

    # The degree is an option of the reading, so it is keyword-only and does
    # not move the positional fields of the reader.
    degree: KwOnly[tx.Optional[int]] = None
    """The spline degree with which the vectors of every level are read.

    A transformation that names the field may ask for an interpolation,
    such as `nearest`, which is degree 0. `None` keeps the default, which
    is linear.
    """

    # scales is built on demand from the arrays and the placement, not stored.
    # Forgetting the field inherited from MultiscaleField keeps it out of
    # __init__, fields() and replace().
    scales: Forget

    @property
    def scales(self) -> tx.Optional[tx.List[Sequence]]:
        """The resolution scales, each a sequence.

        The scales are built on first access and cached. For a displacement
        field, the placement is first checked to be affine. The lazy array
        of a level is converted with the backend setting in effect at that
        access, as [`to_backend`][brainhops.backends.to_backend] says, so
        that under `with backend("dask"):` each level is a Dask array with
        the chunks of the store, and nothing is read.
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

        The node's own OME metadata describes the field. Each level is
        held as a lazy array of its Zarr array, so that nothing but the
        metadata is read here, and the metadata is kept as read.
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
        raw_levels = [
            _level_proxy(_node.follow_path(node, str(ds.path)))
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
        # The placement of a level is mapped over every stored axis, and its
        # part on the spatial axes is kept. OME-NGFF 0.6 stores the vector
        # axis before the spatial axes, so they are found by their type.
        if axes:
            spatial = [
                index
                for index, axis in enumerate(axes)
                if axis.type not in ("displacement", "coordinate")
            ]
        else:
            spatial = list(range(spatial_ndim))
        placements = [_dataset_placement(ds, spatial, ndim) for ds in datasets]
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

        Each level is created with its shape and its type, and then
        written into, so that a lazy array or a Dask array is written
        block by block and never read whole into memory. A level that is
        a lazy array of a store or a Dask array keeps its chunks, unless
        `chunks` is given among the options.

        Parameters
        ----------
        node : ZarrGroup
            The group that receives the levels and the metadata.
        **kwargs : Any
            Options of the creation of each level array, such as
            `chunks`.

        Returns
        -------
        ZarrGroup
            The group that was written into.

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
        for index, array in enumerate(raw_levels):
            dataset_path = (
                str(datasets[index].path)
                if index < len(datasets)
                else str(index)
            )
            # An array on a GPU is copied to memory, one block at a time
            # for a Dask array, since Zarr writes arrays from memory.
            options = dict(kwargs)
            chunks = store_chunks(array)
            if chunks is not None:
                options.setdefault("chunks", chunks)
            target = node.create_array(
                dataset_path,
                shape=tuple(int(n) for n in array.shape),
                dtype=array.dtype,
                **options,
            )
            write_into(target, array)
        node.ome = ome
        return node

    def to_store(self, location: StoreLike, **kwargs) -> None:
        """Write the field to a store location or an opened store.

        An opened group is written into as it stands. A location names a store
        that does not exist yet, so the group is created first.

        The levels of a field read from a store are lazy arrays of that
        store. When the location is the local store that they read, the
        new store is filled beside it and then moved into place, so that
        the levels are read before the old store is removed.

        Parameters
        ----------
        location : str or PathLike or ZarrGroup
            The location of the store, or an opened group.
        **kwargs : Any
            Options of [`to_node`][].

        Raises
        ------
        WriterError
            If the location is the remote store that a level reads, which
            cannot be replaced without the level reading a store half
            written.
        """
        node = _as_node(location)
        if node is not None:
            self.to_node(node, **kwargs)
            return
        location = absolute_location(location)
        if not any(
            reads_inside(raw, location) for raw in self.raw_levels or []
        ):
            self.to_node(abczarr.open_group(location, mode="w"), **kwargs)
            return
        if not is_local(location):
            raise WriterError(
                "This field reads the remote store that it would be written "
                f"to, {location}, and a remote store cannot be replaced in "
                "one step, so the field would read a store half written. "
                "Write the field to another location."
            )
        replace_directory(
            location,
            lambda name: self.to_node(
                abczarr.open_group(name, mode="w"), **kwargs
            ),
        )

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
        # because brainhops displacements work from voxel to voxel. The lazy
        # array of a store is converted as the backend setting says.
        raw = _vector_axis_last(
            to_backend(self.raw_levels[index]), self._vector_axis
        )
        ndim = int(raw.shape[-1])
        voxel2world = self._level_voxel2world(index, ndim)
        world2voxel = voxel2world.inverse()
        options = {} if self.degree is None else {"degree": self.degree}
        if self._kind == "displacement":
            field = DisplacementField(
                field=_to_voxel_displacement(raw, voxel2world), **options
            )
            elements = [world2voxel, field, voxel2world]
        else:
            elements = [world2voxel, CoordinateField(field=raw, **options)]
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


def _level_proxy(array: tx.Any) -> tx.Any:
    # A level of a store is held as a lazy array of its Zarr array, which
    # reads only the chunks that a selection needs. A node that is not a
    # Zarr array, such as an array given in memory, is kept as it is.
    if isinstance(array, abczarr.ZarrArray):
        return _ZarrProxy(array, store_location(array))
    return array


def _vector_axis_last(
    raw: ArrayProtocol, vector_axis: tx.Optional[int]
) -> ArrayProtocol:
    # Move the vector axis last, so that the array is (*grid_shape, ndim).
    if vector_axis is None:
        return raw
    ndim = len(raw.shape)
    if vector_axis in (-1, ndim - 1):
        return raw
    xp = get_array_backend(raw)
    return xp.moveaxis(raw, vector_axis, -1)


def _to_voxel_displacement(
    raw: ArrayProtocol, voxel2world: Transformation
) -> ArrayProtocol:
    # World-unit displacements become voxel displacements through the inverse
    # L^-1 of the linear part of the placement: raw @ L^-T. The product is
    # computed in the backend of the displacements, so that a Dask array is
    # converted block by block rather than read into memory.
    matrix = _as_affine(voxel2world).matrix
    inverse = to_numpy(_affine_inv(matrix))
    ab = get_array_backend(raw)
    return raw @ ab.asarray(inverse[:, :-1].T)


def _dataset_placement(
    dataset: tx.Any, spatial: tx.Sequence[int], ndim: int
) -> Transformation:
    # A level's placement is the dataset's first coordinate transformation, or
    # the identity. It is mapped over all `ndim` stored axes, and then
    # restricted to the `spatial` ones, which keep their stored order.
    transforms = list(
        getattr(dataset, "coordinateTransformations", None) or []
    )
    if not transforms:
        return Identity()
    stored = list(range(ndim))
    if list(spatial) == stored:
        return _map.from_ome(transforms[0], stored, stored)
    try:
        placement = _map.from_ome(transforms[0], stored, stored)
    except _map.OmeMappingError:
        # Fields written before OME-NGFF 0.6 give their placement over the
        # spatial axes alone, which is read as it is.
        spatial_only = list(range(len(spatial)))
        return _map.from_ome(transforms[0], spatial_only, spatial_only)
    return _spatial_part(placement, list(spatial), ndim)


def _spatial_part(
    placement: Transformation, spatial: tx.List[int], ndim: int
) -> Transformation:
    # Restrict a placement to the spatial axes. A scale, a translation and a
    # sequence of them keep their type, and any other affine is restricted
    # through its matrix.
    if is_identity(placement):
        return Identity()
    if isinstance(placement, Scaling):
        return Scaling(scale=[placement.scale[i] for i in spatial])
    if isinstance(placement, Translation):
        return Translation(
            translation=[placement.translation[i] for i in spatial]
        )
    if isinstance(placement, Sequence):
        return Sequence(
            [
                _spatial_part(part, spatial, ndim)
                for part in placement.transformations
            ]
        )
    affine = _as_affine(placement)
    if affine is None:
        raise OmeFieldError(
            "This OME field is placed by a transformation that is not "
            "affine, so its spatial part cannot be told apart from its "
            "vector axis."
        )
    matrix = np.asarray(affine.matrix, dtype=float)
    return Affine(matrix=matrix[np.ix_(spatial, spatial + [ndim])])


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
