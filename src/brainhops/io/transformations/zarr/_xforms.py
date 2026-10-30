"""The OME-Zarr format of multiscale transformation fields.

A field stored in a Zarr store is read into a [`MultiscaleField`][], in
which each resolution level is a sequence that samples the field on the
grid of that level. The format is registered, so that
[`load`][brainhops.io.transformations.load] discovers it. Arrays and
metadata are kept exactly as read, so that writing an unchanged field
re-emits its OME metadata unchanged. A field that has no metadata, such
as a field built from arrays, is written with metadata built from its
levels and their placements by [`field_ome`][].

The dataset of a level of OME-NGFF 0.6 only scales and translates its
grid. A rotation, a shear or a flip that every level shares is stored as
a transformation of the whole multiscale, which [`common_placement`][]
reads and [`field_ome`][] writes, as the levels of an image do.
"""

# dependencies
import abczarr
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import Forget, KwOnly, replace

# core
from brainhops._core.affines import axis_scales
from brainhops._core.affines import inv as _affine_inv
from brainhops._core.backends import (
    array_backend,
    is_dask,
    to_backend,
    to_numpy,
)
from brainhops._core.path import Path, replace_directory
from brainhops._core.pyramids import downsample, level_geometry

# datamodel
from brainhops.datamodel._internal_api import (
    PyramidLayout,
    adopt_scales,
    build_scales,
    field_scale,
    keeps_scale,
    level_matches,
    next_scale,
    pop_pyramid_options,
    pyramid_layout,
    vector_axis,
)
from brainhops.datamodel.axes import AxisList, CoordinateAxis, SpaceAxis
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
from brainhops.errors import (
    ParserTypeError,
    UnrepresentableTransformationError,
    WriterError,
)

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.common.zarr import StoreLike
from brainhops.io.common.zarr._internal_api import (
    ZarrNodeReaderWriter,
    ZarrProxy,
    absolute_location,
    as_node,
    check_ome,
    read_back,
    reads_inside,
    store_chunks,
    store_location,
    write_into,
    written_plan,
)
from brainhops.io.transformations.base import TransformationFormat

# this format
from . import _map, _node
from ._errors import OmeFieldError

# typing
if tx.TYPE_CHECKING:
    # This is imported for mkdocstrings to resolve cross-references
    from brainhops.io.common.zarr import ZarrReader  # noqa: F401
    from brainhops.io.transformations import load  # noqa: F401


@register_format
class OmeZarrField(
    ZarrNodeReaderWriter, TransformationFormat, MultiscaleField
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

    # A field converted from a single scale is written with its coarser
    # scales, as `with_scales()` builds them.
    BUILDS_SCALES = True

    # A single scale saved with the options of a pyramid is written with
    # coarser scales built from the store, as `to_node()` builds them.
    WRITES_SCALES = True
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
    coordinate transformation of the finest dataset in the OME metadata,
    followed by the transformation that the multiscale gives every
    level, when it has one.
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
    object, so it can be re-emitted unchanged. `None`, the default,
    means that the writer builds the metadata from the levels.
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

    @property
    def nscales(self) -> int:
        # The number of levels is that of the arrays, which builds no
        # scale.
        return len(self.raw_levels or [])

    def to_ome(self) -> tx.Any:
        """Return the OME metadata to write.

        An untouched field returns the object it was read with, so that
        reading and writing round-trips the metadata. A field without
        metadata, such as a field built from arrays, gets metadata built
        from its levels, their placements and its axes. The dataset of
        each level then holds a positive scale and a translation, and a
        rotation, a shear or a flip that every level shares becomes a
        transformation of the whole multiscale, as for the levels of an
        image (see the section "Limitations" of
        [`brainhops.io.images.zarr`][]).

        Raises
        ------
        WriterError
            If the field has no metadata and no axes, or if the
            placement of a level is not an affine.
        """
        if self.ome is not None:
            return self.ome
        if not self.raw_levels:
            return None
        if not self.axes:
            raise WriterError(
                "This OME-Zarr field has neither OME metadata nor axes, so "
                "the axis that holds its vectors is not known."
            )
        return field_ome(self.axes, self._placements())

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
        node = as_node(node)
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
        placements = [dataset_placement(ds, spatial, ndim) for ds in datasets]
        grid2finest = placements[0].inverse()
        level_transforms = [
            (grid2finest @ placement).compute() for placement in placements
        ]
        # A placement that every level shares, such as a rotation, is
        # stored in the transformations of the multiscale, after the
        # placement of each level.
        shared = common_placement(multiscales[0], spatial, ndim)
        voxel2world = placements[0]
        if shared is not None:
            voxel2world = Sequence([voxel2world, shared])
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
            `chunks`, and the options of a pyramid (`nscales=`,
            `factor=`, `mode=`, `align=`, `min_size=` and `strict=`).
            Given to a field that holds a single level, as `io.save`
            gives them, the options of a pyramid build its coarser
            levels as `with_scales` builds them, except that each level
            is computed from the levels already written, read back from
            the store.

        Returns
        -------
        ZarrGroup
            The group that was written into.

        Raises
        ------
        WriterError
            If the node is a plain array, if the field has no resolution
            levels, if its OME metadata cannot be built, as [`to_ome`][]
            explains, if its OME metadata breaks a rule of OME-NGFF, or
            if the options of a pyramid are given to a field of several
            levels or are wrong.
        """
        # Each level is written at the dataset path named by the
        # metadata, which is re-emitted unchanged when it was read, and
        # built from the levels otherwise.
        pyramid = pop_pyramid_options(kwargs)
        node = as_node(node)
        if not isinstance(node, abczarr.ZarrGroup):
            raise WriterError(
                "An OME-Zarr field is written into a group, not a plain array."
            )
        field = self
        plan = written_plan(self, pyramid)
        if plan is not None:
            # The metadata is built again from the levels that are built.
            field = replace(self, ome=None)
        ome = field.to_ome()
        if ome is None:
            raise WriterError(
                "This OME-Zarr field has no OME metadata, so there is nothing "
                "to write."
            )
        raw_levels = list(self.raw_levels or [])
        if not raw_levels:
            raise WriterError(
                "This OME-Zarr field has no resolution levels to write."
            )
        check_ome(ome)
        try:
            normalized = ome.to_version("0.6")
        except Exception:
            normalized = ome
        multiscales = getattr(normalized, "multiscales", None) or []
        datasets = list(multiscales[0].datasets) if multiscales else []
        # Every level is stored with one chunk shape, and one shard
        # shape when the levels are sharded, capped at the shape of the
        # level, as the levels of an image are: `chunks`, or else the
        # chunks of the finest level when it is a lazy array or a Dask
        # array, or else the chunks that abczarr chooses for the finest
        # level.
        common = kwargs.get("chunks") or store_chunks(raw_levels[0])
        shards = kwargs.get("shards")
        count = len(raw_levels) if plan is None else len(plan.factors) + 1
        targets = []  # type: tx.List[tx.Any]
        for index in range(count):
            if index == len(raw_levels):
                # The next level of the plan is built from the levels
                # just written, read back from the store.
                written = [read_back(target) for target in targets]
                field = next_scale(replace(field, raw_levels=written), plan)
                raw_levels.append(field.raw_levels[-1])
            array = raw_levels[index]
            dataset_path = (
                str(datasets[index].path)
                if index < len(datasets)
                else str(index)
            )
            shape = tuple(int(n) for n in array.shape)
            # An array on a GPU is copied to memory, one block at a time
            # for a Dask array, since Zarr writes arrays from memory.
            options = dict(kwargs)
            if common is not None:
                chunks = tuple(
                    min(int(size), n) for size, n in zip(common, shape)
                )
                options["chunks"] = chunks
                if shards is not None:
                    options["shards"] = tuple(
                        min(int(shard), -(-n // size) * size)
                        for shard, n, size in zip(shards, shape, chunks)
                    )
                if is_dask(array):
                    array = array.rechunk(chunks)
            target = node.create_array(
                dataset_path, shape=shape, dtype=array.dtype, **options
            )
            if common is None:
                common = tuple(int(size) for size in target.chunks)
            write_into(target, array)
            targets.append(target)
        if plan is not None:
            ome = field.to_ome()
            check_ome(ome)
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
        node = as_node(location)
        if node is not None:
            self.to_node(node, **kwargs)
            return
        location = absolute_location(location)
        if not any(
            reads_inside(raw, location) for raw in self.raw_levels or []
        ):
            self.to_node(abczarr.open_group(location, mode="w"), **kwargs)
            return
        if not Path(location).is_local():
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

    # --- scales built from the finest level ---
    # The method `with_scales` of
    # `MultiscaleField` call these hooks. A field of this format stores
    # its levels as arrays of vectors in world units, so the vectors of
    # a coarser level are not rescaled, and the metadata is built again
    # from the levels when the field is written.

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
        base = affine_or_identity(
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

    def _placements(self) -> tx.List[Transformation]:
        # The placement of each level over its spatial axes, in the
        # stored order: the placement of the finest level after the
        # transformation from the grid of the level to the finest grid.
        base = self.voxel2world
        if base is None:
            base = Identity()
        transforms = self.level_transforms or []
        placements = []  # type: tx.List[Transformation]
        for index in range(len(self.raw_levels or [])):
            level = transforms[index] if index < len(transforms) else None
            if level is None or is_identity(level):
                placements.append(base)
            else:
                placements.append(base @ level)
        return placements

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
        affine = placement.to_affine(on_error=None)
        if affine is None:
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
        return ZarrProxy(array, store_location(array))
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
    xp = array_backend(raw)
    return xp.moveaxis(raw, vector_axis, -1)


def _to_voxel_displacement(
    raw: ArrayProtocol, voxel2world: Transformation
) -> ArrayProtocol:
    # World-unit displacements become voxel displacements through the inverse
    # L^-1 of the linear part of the placement: raw @ L^-T. The product is
    # computed in the backend of the displacements, so that a Dask array is
    # converted block by block rather than read into memory.
    # `_check_affine_voxel2world` checks the placement of the finest
    # level only, and the placement of a coarser level also goes through
    # its level transform, which may not reduce to an affine.
    affine = voxel2world.to_affine(on_error=None)
    if affine is None:
        raise OmeFieldError(
            "A level of this OME displacement field is placed by a "
            "transformation that cannot be reduced to an affine, so its "
            "displacements cannot be rescaled to voxel units. Only an "
            "affine placement is supported for a displacement field."
        )
    inverse = to_numpy(_affine_inv(affine.matrix))
    ab = array_backend(raw)
    return raw @ ab.asarray(inverse[:, :-1].T)


def dataset_placement(
    dataset: tx.Any, spatial: tx.Sequence[int], ndim: int
) -> Transformation:
    """Return the placement of a level of an OME-Zarr field.

    The placement is the first coordinate transformation of the dataset
    of the level, or the identity when it has none. It is mapped over
    all the `ndim` stored axes, and then restricted to the `spatial`
    ones, which keep their stored order.

    Parameters
    ----------
    dataset : Any
        The dataset of the level, from the multiscale metadata.
    spatial : Sequence[int]
        The indices of the spatial axes among the stored axes.
    ndim : int
        The number of stored axes.

    Returns
    -------
    Transformation
        The placement of the level over its spatial axes.
    """
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


def common_placement(
    multiscale: tx.Any, spatial: tx.Sequence[int], ndim: int
) -> tx.Optional[Transformation]:
    """Return the placement that every level of an OME-Zarr field shares.

    The datasets of a multiscale of OME-NGFF 0.6 only scale and
    translate their levels, and a rotation or a shear that every level
    shares is a transformation of the whole multiscale, from the system
    onto which the levels map, called intrinsic, to another system. The
    last such transformation is the preferred one, as for an image. A
    transformation that ends in the intrinsic system is read as its
    inverse, and one that refers to a system of another group is
    skipped.

    Parameters
    ----------
    multiscale : Any
        The multiscale metadata, normalized to version 0.6.
    spatial : sequence of int
        The indices of the spatial axes among the stored axes.
    ndim : int
        The number of stored axes.

    Returns
    -------
    Transformation or None
        The shared placement over the spatial axes, in their stored
        order, or `None` when the multiscale has none.

    Raises
    ------
    OmeFieldError
        If the shared placement cannot be read.
    """
    common = list(getattr(multiscale, "coordinateTransformations", None) or [])
    if not common:
        return None
    intrinsic = _intrinsic_name(multiscale)
    chosen = None  # type: tx.Optional[tx.Tuple[tx.Any, bool]]
    for transform in common:
        source = getattr(transform, "input", None)
        target = getattr(transform, "output", None)
        if _in_other_group(source) or _in_other_group(target):
            continue
        source = getattr(source, "name", None)
        target = getattr(target, "name", None)
        if source in (None, intrinsic):
            chosen = (transform, False)
        elif target == intrinsic:
            chosen = (transform, True)
    if chosen is None:
        return None
    transform, inverted = chosen
    stored = list(range(ndim))
    try:
        mapped = _map.from_ome(transform, stored, stored)
    except _map.OmeMappingError as error:
        raise OmeFieldError(
            "The placement that the levels of this OME field share cannot "
            f"be read. {error}"
        ) from error
    placement = _spatial_part(mapped, list(spatial), ndim)
    return placement.inverse() if inverted else placement


def _intrinsic_name(multiscale: tx.Any) -> tx.Optional[str]:
    # The intrinsic system is the output of the transformations of the
    # datasets.
    for dataset in getattr(multiscale, "datasets", None) or []:
        for transform in dataset.coordinateTransformations or []:
            name = getattr(getattr(transform, "output", None), "name", None)
            if isinstance(name, str):
                return name
    return None


def _in_other_group(system: tx.Any) -> bool:
    # A system named with a path lies in another group of the hierarchy,
    # unless the path leads back to the group of the multiscale.
    path = getattr(system, "path", None)
    return isinstance(path, str) and path.strip("/") not in ("", ".")


FIELD_SYSTEM = "field"
"""The name of the system in which a written field places its vectors."""

INTRINSIC_SYSTEM = "intrinsic"
"""The name of the system onto which the levels of a written field map.

It is written only when the levels share a placement that their
datasets cannot hold, such as a rotation. Otherwise, the levels map
directly onto [`FIELD_SYSTEM`][].
"""


def field_ome(
    axes: tx.Sequence[tx.Any], placements: tx.Sequence[Transformation]
) -> tx.Any:
    """Build the OME-NGFF 0.6 metadata of the levels of a field.

    Each level is stored in the array named after its index, with the
    axes in the order of `axes`, which holds one axis whose type is
    `displacement` or `coordinate`. The dataset of a level holds a
    positive scale and a translation of its spatial axes. When the
    levels are also rotated, sheared or flipped, their placements are
    split into that part and a placement that every level shares, which
    is written as a transformation of the multiscale, as for the levels
    of an image (see the section "Limitations" of
    [`brainhops.io.images.zarr`][]). The vectors are given in the system
    that the levels are placed in, whatever the split, so their values
    do not change.

    Parameters
    ----------
    axes : sequence of Axis
        The axes of the stored arrays, in the order in which they are
        stored. An axis without a name is named after its position.
    placements : sequence of Transformation
        The placement of each level, finest first, from its voxels to
        the system of the field, over the spatial axes in their stored
        order.

    Returns
    -------
    OME
        The typed metadata of the field.

    Raises
    ------
    WriterError
        If a placement is not an affine, or if the levels are rotated,
        sheared or flipped in different ways.
    """
    from abczarr.ome import v0_6

    ndim = len(axes)
    spatial = [
        index
        for index, axis in enumerate(axes)
        if getattr(axis, "type", None) not in ("displacement", "coordinate")
    ]
    full = []  # type: tx.List[Transformation]
    for index, placement in enumerate(placements):
        matrix = _map.affine_matrix(placement, len(spatial))
        if matrix is None:
            raise WriterError(
                f"Level {index} of this OME-Zarr field is placed by a "
                "transformation that is not an affine of its spatial axes, "
                "which OME-Zarr cannot store."
            )
        # The axis of the vectors is left unchanged by the placement.
        expanded = np.eye(ndim, ndim + 1)
        expanded[np.ix_(spatial, spatial)] = matrix[:, :-1]
        expanded[spatial, -1] = matrix[:, -1]
        full.append(Affine(matrix=expanded))
    stored = list(range(ndim))
    entries = [_map.to_dataset_ome(level, stored, ndim) for level in full]
    shared = None  # type: tx.Optional[tx.Dict[str, tx.Any]]
    if any(entry is None for entry in entries):
        split = _map.split_levels(full, ndim)
        if split is None:
            raise UnrepresentableTransformationError(
                "The levels of this OME-Zarr field are not rotated, sheared "
                "or flipped in the same way, so their placements cannot be "
                "split into the positive scale and the translation that the "
                "dataset of each level holds in OME-NGFF 0.6, and a "
                "placement that every level shares."
            )
        placement, parts = split
        entries = [_map.to_dataset_ome(part, stored, ndim) for part in parts]
        shared = _map.to_ome(placement, stored, ndim)
    json_axes = [_axis_json(index, axis) for index, axis in enumerate(axes)]
    output = INTRINSIC_SYSTEM if shared is not None else FIELD_SYSTEM
    systems = [{"name": FIELD_SYSTEM, "axes": json_axes}]
    if shared is not None:
        systems.insert(0, {"name": INTRINSIC_SYSTEM, "axes": json_axes})
    block = {
        "coordinateSystems": systems,
        "datasets": [
            {
                "path": str(index),
                "coordinateTransformations": [
                    dict(
                        entry,
                        input={"path": str(index)},
                        output={"name": output},
                    )
                ],
            }
            for index, entry in enumerate(entries)
        ],
    }  # type: tx.Dict[str, tx.Any]
    if shared is not None:
        block["coordinateTransformations"] = [
            dict(
                shared,
                input={"name": INTRINSIC_SYSTEM},
                output={"name": FIELD_SYSTEM},
            )
        ]
    return v0_6.OME.from_json({"version": "0.6", "multiscales": [block]})


def _axis_json(index: int, axis: tx.Any) -> tx.Dict[str, tx.Any]:
    # The OME description of an axis. OME-NGFF requires a name, so an
    # axis without one is named after its position.
    name = getattr(axis, "name", None)
    entry = {
        "name": name if isinstance(name, str) and name else f"axis{index}"
    }  # type: tx.Dict[str, tx.Any]
    kind = getattr(axis, "type", None)
    if isinstance(kind, str):
        entry["type"] = kind
    # A brainhops unit is an object whose text is the name of the unit.
    unit = getattr(axis, "unit", None)
    if unit is not None:
        entry["unit"] = str(getattr(unit, "value", unit))
    discrete = getattr(axis, "discrete", None)
    if isinstance(discrete, bool):
        entry["discrete"] = discrete
    return entry


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
    affine = placement.to_affine(on_error=None)
    if affine is None:
        raise OmeFieldError(
            "This OME field is placed by a transformation that is not "
            "affine, so its spatial part cannot be told apart from its "
            "vector axis."
        )
    matrix = np.asarray(affine.matrix, dtype=float)
    return Affine(matrix=matrix[np.ix_(spatial, spatial + [ndim])])


def affine_or_identity(xform: Transformation, ndim: int) -> Transformation:
    """Return a placement, or an identity affine in its place.

    An affine placement is returned unchanged. Otherwise, an identity
    affine between the same systems stands in, so that the leading
    world-to-voxel element of a chain stays a defined affine rather than
    a bare identity.

    Parameters
    ----------
    xform : Transformation
        The placement.
    ndim : int
        The number of dimensions of the identity affine.

    Returns
    -------
    Transformation
        The placement, or an identity affine.
    """
    affine = xform.to_affine(on_error=None)
    if affine is not None:
        return xform
    return Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim + 1)]
            for i in range(ndim)
        ],
        input=xform.input,
        output=xform.output,
    )


@pyramid_layout.register(OmeZarrField)
def _(obj: "OmeZarrField") -> PyramidLayout:
    raw_levels = list(obj.raw_levels or [])
    if not raw_levels:
        raise ValueError(
            "This OME-Zarr field has no level, so no coarser level can be "
            "built from it."
        )
    shape = tuple(int(n) for n in raw_levels[0].shape)
    ndim = len(shape)
    vector = obj._vector_axis
    vector = ndim - 1 if vector is None else vector % ndim
    spatial = tuple(axis for axis in range(ndim) if axis != vector)
    if obj.axes and len(obj.axes) == ndim:
        axes = AxisList(list(obj.axes))
    else:
        axes = AxisList([SpaceAxis() for _ in spatial] + [CoordinateAxis()])
    sizes = [1.0] * ndim
    matrix = _map.affine_matrix(obj._placements()[0], len(spatial))
    if matrix is not None:
        for axis, size in zip(spatial, axis_scales(matrix)):
            sizes[axis] = float(size)
    return PyramidLayout(
        axes=axes,
        shape=shape,
        sizes=tuple(sizes),
        spatial=spatial,
        fixed=(vector,),
        field=True,
    )


@build_scales.register(OmeZarrField)
def _(
    obj: "OmeZarrField",
    factors: tx.Sequence[tx.Tuple[int, ...]],
    mode: str,
    align: str,
    kept: int,
) -> "OmeZarrField":
    layout = pyramid_layout(obj)
    raw_levels = list(obj.raw_levels[: kept + 1])
    transforms = list(obj.level_transforms or [])[: kept + 1]
    transforms += [Identity()] * (kept + 1 - len(transforms))
    finest_shape = [raw_levels[0].shape[axis] for axis in layout.spatial]
    for level in factors[kept:]:
        _, scales, shifts = level_geometry(
            align, finest_shape, [level[axis] for axis in layout.spatial]
        )
        transforms.append(
            Affine(
                matrix=np.concatenate(
                    [np.diag(scales), shifts[:, None]], axis=1
                )
            )
        )
    previous = None
    if kept:
        previous = (factors[kept - 1], to_backend(raw_levels[kept]))
    # The vectors are stored in world units, which do not depend on
    # the grid, so the modes only average them.
    coarser = downsample(
        to_backend(raw_levels[0]),
        factors[kept:],
        mode,
        edge="linear",
        align=align,
        previous=previous,
    )
    return replace(
        obj,
        raw_levels=raw_levels + coarser,
        voxel2world=obj._placements()[0],
        level_transforms=transforms,
        ome=None,
    )


@keeps_scale.register(OmeZarrField)
def _(
    obj: "OmeZarrField", index: int, factors: tx.Tuple[int, ...], align: str
) -> bool:
    layout = pyramid_layout(obj)
    raw_levels = list(obj.raw_levels or [])
    transforms = list(obj.level_transforms or [])
    count = len(layout.spatial)
    level = transforms[index] if index < len(transforms) else None
    matrix = _map.affine_matrix(level or Identity(), count)
    if matrix is None or len(raw_levels[index].shape) != len(layout.shape):
        return False
    shapes = [
        [int(raw.shape[axis]) for axis in layout.spatial]
        for raw in (raw_levels[0], raw_levels[index])
    ]
    return level_matches(
        np.eye(count, count + 1),
        matrix,
        shapes[0],
        shapes[1],
        [factors[axis] for axis in layout.spatial],
        align,
    )


@adopt_scales.register(OmeZarrField)
def _(obj: "OmeZarrField", scales: tx.Sequence[Sequence]) -> "OmeZarrField":
    parts = [field_scale(scale) for scale in scales]
    if parts[0].kind != obj._kind:
        raise ValueError(
            f"This OME-Zarr field holds {obj._kind}s, but the scales "
            f"that it is given hold {parts[0].kind}s."
        )
    ndim = int(parts[0].field.data.ndim)
    vector = obj._vector_axis
    vector = ndim - 1 if vector is None else vector % ndim
    grids = [np.asarray(_affine_inv(part.world2voxel)) for part in parts]
    raw_levels = []
    for part, grid2world in zip(parts, grids):
        values = to_backend(part.field.field)
        if part.kind == "displacement":
            # The level stores its displacements in world units.
            linear = array_backend(values).asarray(grid2world[:, :-1].T)
            values = values @ linear
        raw_levels.append(array_backend(values).moveaxis(values, -1, vector))
    finest2world = grids[0]
    world2finest = np.asarray(_affine_inv(finest2world))
    transforms = [Identity()]  # type: tx.List[Transformation]
    for grid2world in grids[1:]:
        linear = world2finest[:, :-1] @ grid2world[:, :-1]
        shift = world2finest[:, :-1] @ grid2world[:, -1]
        shift = shift + world2finest[:, -1]
        transforms.append(
            Affine(matrix=np.concatenate([linear, shift[:, None]], 1))
        )
    return replace(
        obj,
        raw_levels=raw_levels,
        voxel2world=Affine(matrix=finest2world),
        level_transforms=transforms,
        ome=None,
    )
