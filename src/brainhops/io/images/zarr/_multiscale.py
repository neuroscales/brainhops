# dependencies
import typing_extensions as tx
from abczarr import ZarrGroup, ZarrNode, open_group
from abczarr.ome.v0_6.images import Multiscale
from bagof.magic import Inherit, replace

# core
from brainhops._core.backends import array_backend, is_dask
from brainhops._core.path import Path, replace_directory
from brainhops._core.properties import smartproperty

# datamodel
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import (
    ArrayCoordinateSystem,
    AxisList,
    CoordinateSystem,
)
from brainhops.datamodel.transformations import (
    Sequence,
    Transformation,
)

# io
from brainhops.errors import UnrepresentableTransformationError, WriterError
from brainhops.io.base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.common.zarr import StoreLike, ZarrMetadata, ZarrRaw
from brainhops.io.common.zarr._internal_api import (
    ZarrNodeReaderWriter,
    ZarrProxy,
    absolute_location,
    as_node,
    opened_node,
    reads_inside,
    store_chunks,
    store_location,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.zarr import _axisorder
from brainhops.io.transformations.zarr import _map

# this format
from ._image import ZarrImage
from ._ome import (
    OmeImageError,
    check_parameter_paths,
    common_transformations,
    group_path,
    intrinsic_name,
    level_transformation,
    looks_like_multiscale,
    multiscale_axes,
    read_multiscale,
    resolve_world_names,
    resolve_write_version,
    split_system_name,
    system_axes,
    system_name,
    write_multiscale,
)

if tx.TYPE_CHECKING:
    from types import EllipsisType
else:
    # `types.EllipsisType` only exists from Python 3.10 onwards.
    EllipsisType = type(Ellipsis)


class OmeZarrLevel(ZarrImage):
    """One resolution level of an OME-Zarr pyramid.

    The level holds the lazy array of its dataset in `rawdata`, which presents
    the stored axes in the brainhops order, and the record of the array of
    its dataset. Its transformations are those of its dataset, assigned
    when the pyramid is read. Opening a pyramid reads no array data, and a
    copy made with `replace` keeps the lazy array. Step 4b-ii of the
    fourth pass makes the level hold the record of its pyramid instead.
    """


# `kw_only=True` is repeated here because bagof takes each class option
# from the first base in the method resolution order, whether that base
# sets the option or keeps the default. That base is `ZarrNodeReaderWriter`,
# which keeps the default, so the fields declared by this class would
# otherwise be positional.
@register_format
class OmeZarrImage(
    ZarrNodeReaderWriter,
    ImageFormat,
    MultiScaleImage,
):
    """Multiscale image backed by an OME-Zarr pyramid.

    Each level is a single-scale image placed by the coordinate
    transformation of its dataset. The metadata is read with abczarr and
    normalized to OME-NGFF 0.6. Levels run from finest to coarsest, their
    axes are permuted into the brainhops order, and their arrays are read
    lazily. A plain Zarr array is read by [`ZarrImage`][] instead.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr", ".ome.zarr")

    # An image converted or saved from a single scale is written with
    # its coarser scales, as `with_scales()` builds them.
    BUILDS_SCALES = True

    # ---- attributes --------------------------------------------------

    _axes: tx.Optional[AxisList[tx.Union[Axis, EllipsisType]]] = None
    """
    The axes to store a pyramid under, in the brainhops order, when it
    is built from scratch, as an `AxisList`. As in a coordinate system,
    a single `...` may stand for the axes about which nothing is known;
    it is closed to the number of axes of the data when the pyramid is
    written. `None` is not `[...]`: it leaves the axes unset, so a
    pyramid read from a store takes its axes from `ome`.
    """

    _ome: tx.Optional[Multiscale] = None
    """The OME multiscale metadata, normalized to 0.6."""

    _group: str = ""
    """
    The path of the group of the pyramid from the root of its Zarr
    hierarchy, such as `labels/cells`.

    The coordinate systems of a pyramid in another group than the root
    are named after that path, as in `labels/cells//physical`, so that
    the systems of two groups stay distinct. A Zarr group does not know
    its path, so the reader that opened the root of the hierarchy gives
    it.

    The empty string, the default, is the root group, whose systems keep
    their OME names.
    """

    _images: tx.Optional[tx.List[SingleScaleImage]] = Inherit(
        "images", default=None
    )
    """
    The levels given to the pyramid, or `None` to read them.

    The pyramid takes over the `images` field of [`MultiScaleImage`][]
    to store it under this private name. The constructor still takes
    `images` in the same place, and the property of the same name reads
    it. `None`, the default, means that the levels are read from `node`,
    while a list, even an empty one, is used as it is.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """
    The placements given to the pyramid, or `None` to read them.

    The pyramid takes over the `transformations` field of
    [`MultiScaleImage`][] in the same way as `images`. `None`, the
    default, means that the placements are read from `node`, while a
    list, even an empty one, is used as it is.
    """

    # ---- properties --------------------------------------------------

    @property
    def node(self) -> ZarrGroup:
        return getattr(self, "_node", None)

    @node.setter
    def node(self, value: ZarrGroup) -> None:
        self._node = value
        # Discard the values cached from the previous node; `smartproperty`
        # caches them as `_cache_<name>`.
        for name in (
            "_cache_ome",
            "_cache__layout",
            "_cache_images",
            "_cache_transformations",
        ):
            if hasattr(self, name):
                delattr(self, name)

    @property
    def group(self) -> str:
        """Path of the group of the pyramid from the root of its hierarchy."""
        return getattr(self, "_group", "")

    @property
    def axes(self) -> tx.Optional[AxisList[tx.Union[Axis, EllipsisType]]]:
        """Axes under which a pyramid built from scratch is stored."""
        return getattr(self, "_axes", None)

    @smartproperty
    def ome(self) -> tx.Optional[Multiscale]:
        # The `_ome` helpers expect the multiscale, not the enclosing OME
        # block.
        return self._layout["multiscale"]

    @property
    def _ome_version(self) -> tx.Optional[str]:
        """OME version of the source group, reused by the writer."""
        return self._layout["version"]

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """Levels of the pyramid, read from `node` on first access.

        The levels read from `node` are cached, and assigning another node
        drops them. Assigning a list replaces the levels that were read,
        and an empty list means that the pyramid has no level. Assigning
        `None` reads the levels from `node` again.
        """
        return self._layout["images"]

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Placements of the pyramid, read from `node` on first access.

        There is one placement per named world space, and the preferred
        one is last. The list is empty when the levels sit directly in
        world space. The placements read from `node` are cached, so a
        placement appended to the list in place is kept, and assigning
        another node drops them. Assigning a list replaces the placements
        that were read, and an empty list means that the pyramid has no
        placement. Assigning `None` reads the placements from `node` again.
        """
        return list(self._layout["commons"])

    # ---- load --------------------------------------------------------

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        """Read a pyramid from an opened Zarr group.

        The metadata is parsed immediately, so unreadable metadata is refused
        when the group is opened rather than on first access.
        """
        image = cls(node=opened_node(node), **kwargs)
        _ = image._layout
        return image

    # ---- workers  ----------------------------------------------------

    @smartproperty(cache=True, fset=False)
    def _layout(self) -> tx.Dict[str, tx.Any]:
        # A single parse yields both levels and placements, so that they agree
        # on their coordinate systems.
        if self.node is None:
            return {
                "images": [],
                "commons": [],
                "multiscale": None,
                "version": None,
            }
        return self._read_layout()

    def _read_layout(self) -> tx.Dict[str, tx.Any]:
        node = self.node
        if not isinstance(node, ZarrGroup):
            raise OmeImageError(
                "This Zarr store is an array, not a group, so it cannot be "
                "read as a multiscale image."
            )

        # `read_multiscale` reports specific schema faults; a separate scoring
        # check would parse twice and give a vaguer error.
        multiscale, source_version = read_multiscale(node)
        if multiscale is None:
            raise OmeImageError(
                "This Zarr group carries no OME multiscale metadata, so it "
                "cannot be read as a multiscale image."
            )

        check_parameter_paths(multiscale, source_version)
        datasets = list(multiscale.datasets)
        if not datasets:
            raise OmeImageError(
                "This OME multiscale names no datasets, so it has no "
                "resolution levels to read."
            )

        store_axes = multiscale_axes(multiscale)
        base_array = node[str(datasets[0].path)]
        ndim = base_array.ndim
        if len(store_axes) != ndim:
            raise OmeImageError(
                "This OME multiscale names "
                f"{len(store_axes)} axes, but its arrays have {ndim} "
                "dimensions."
            )

        try:
            group = group_path(self.group)
        except ValueError as error:
            raise OmeImageError(str(error)) from error
        perm = _axisorder.to_canonical(store_axes)
        canonical_axes = _axisorder.permute(store_axes, perm)
        # Each named system sorts its own axes into the brainhops order, and
        # the mapping of a transformation reorders its parameters by the
        # orders of its two systems. Without common transformations, the
        # intrinsic space is the world space.
        declared = system_axes(multiscale)
        orders = {
            name: _axisorder.to_canonical(axes)
            for name, axes in declared.items()
        }

        def system(name: tx.Optional[str]) -> CoordinateSystem:
            # A name that no system declares takes the axes of the array. A
            # system is named after its group, unless it is in the root.
            axes = declared.get(name) if isinstance(name, str) else None
            label = system_name(name, group) if isinstance(name, str) else None
            if axes is None:
                return CoordinateSystem(name=label, axes=canonical_axes)
            return CoordinateSystem(
                name=label, axes=_axisorder.permute(axes, orders[name])
            )

        # Stored arrays are C-ordered. A level whose permutation reverses the
        # stored axes is F-ordered, as in nibabel; any other permutation leaves
        # the order unspecified.
        reverses = list(perm) == list(range(ndim))[::-1]
        voxel_system = ArrayCoordinateSystem(
            name="voxel",
            axes=[replace(axis, unit="index") for axis in canonical_axes],
            order="F" if reverses else None,
        )
        intrinsic_system = system(intrinsic_name(multiscale))
        systems = {name: system(name) for name in declared}

        images: tx.List[OmeZarrLevel] = []
        for dataset in datasets:
            array = node[str(dataset.path)]
            geometry = level_transformation(
                dataset,
                perm,
                node=node,
                store_axes=store_axes,
                input=voxel_system,
                output=intrinsic_system,
                orders=orders,
            )
            level = _level(array, perm, geometry)
            images.append(level)

        commons = common_transformations(
            multiscale,
            orders,
            perm,
            node=node,
            store_axes=store_axes,
            input=intrinsic_system,
            systems=systems,
        )
        return {
            "images": images,
            "commons": commons,
            "multiscale": multiscale,
            "version": source_version,
        }

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        if not isinstance(node, ZarrGroup):
            return Confidence.NO
        if not looks_like_multiscale(node):
            return Confidence.NO
        if _holds_a_field(node):
            # A displacement or a coordinate axis makes the group a field,
            # which the OME-Zarr field reads.
            return Confidence.NO
        return Confidence.CERTAIN

    def to_node(
        self,
        node: tx.Any,
        chunks: tx.Any = None,
        version: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Write the pyramid into an opened Zarr group.

        Each level becomes an array named after its index, and the
        placements of the pyramid become common transformations of the
        multiscale. When a level does more than scale and translate each
        axis, the geometries of the levels are split into the scale and
        the translation of each level and a placement that they share,
        as the section "Limitations" of [`brainhops.io.images.zarr`][]
        explains.

        Parameters
        ----------
        node : ZarrGroup
            Group that receives the levels and the metadata.
        chunks : sequence of int, optional
            Chunk shape of every level, in the brainhops axis order,
            capped at the shape of each level. By default, the chunks of
            the finest level when it is a lazy array of a chunked store
            or a Dask array, and otherwise the chunks that abczarr
            chooses for the finest level. A Dask array is rechunked to
            the chunk shape of its level, so that each of its blocks is
            written into chunks of its own. A shard shape given as
            `shards` is capped in the same way.
        version : str, optional
            OME-NGFF version. By default, the source version or 0.6 is used.
        **kwargs : Any
            Passed to the creation of each level array.

        Raises
        ------
        WriterError
            If `node` is an array, if a level is missing or empty, or if
            the axes or transformations cannot be written.
        UnrepresentableTransformationError
            If the levels are rotated, sheared or flipped in different
            ways, so that their geometries cannot be split into the
            positive scale and the translation that an OME-NGFF 0.6
            dataset holds and a placement that every level shares. This
            error is a kind of `WriterError`.
        """
        node = as_node(node)
        if not isinstance(node, ZarrGroup):
            raise WriterError(
                "A multiscale image is written into a group, not a plain "
                "array."
            )
        images = list(self.images or [])
        if not images:
            raise WriterError(
                "This multiscale image has no levels, so there is nothing "
                "to write."
            )
        for index, image in enumerate(images):
            if image.data is None:
                raise WriterError(
                    f"Level {index} of this multiscale image has no data."
                )
        ndim = len(images[0].data.shape)
        axes = self._write_axes(ndim)
        storage_perm = _axisorder.to_storage(axes)
        storage_axes = _axisorder.permute(axes, storage_perm)
        # One chunk shape for all levels, reordered to the stored axis order.
        stored_chunks = (
            None if chunks is None else tuple(chunks[p] for p in storage_perm)
        )

        geometries = [image.transformation for image in images]
        datasets = [
            _dataset_entry(index, geometry, storage_perm, ndim)
            for index, geometry in enumerate(geometries)
        ]
        placements = list(self.transformations or [])
        moved_placement = False
        if any(entry is None for entry in datasets):
            # A dataset only holds a positive scale and a translation
            # (items 131 and 135). The rotation, the shear or the flip
            # that every level shares therefore goes before the
            # placements of the pyramid, and each level keeps a scale
            # and a translation.
            shared, geometries = _split(geometries, ndim)
            datasets = [
                _dataset_entry(index, geometry, storage_perm, ndim)
                for index, geometry in enumerate(geometries)
            ]
            if placements:
                placements = [
                    Sequence(
                        [shared, placement],
                        input=getattr(placement, "input", None),
                        output=getattr(placement, "output", None),
                    )
                    for placement in placements
                ]
            else:
                output = getattr(images[0].transformation, "output", None)
                placements = [replace(shared, output=output)]
            moved_placement = True

        # Every level is stored with one chunk shape, and one shard
        # shape when the levels are sharded, capped at the shape of the
        # level, so that several chunks of a finer level make one chunk
        # of a coarser level. The shape is `chunks`, or else the chunks
        # of the finest level when it is a lazy array of a chunked store
        # or a Dask array, or else the chunks that abczarr chooses for
        # the finest level.
        common = stored_chunks
        if common is None:
            native = store_chunks(getattr(images[0], "data", None))
            if native is not None and len(native) == len(storage_perm):
                common = tuple(native[p] for p in storage_perm)
        shards = kwargs.get("shards")
        levels = []  # type: tx.List[tx.Tuple[str, tx.Dict[str, tx.Any]]]
        rich = False
        for index, image in enumerate(images):
            data = image.data
            entry = datasets[index]
            rich = rich or _map.needs_rich_version(entry)
            xp = array_backend(data)
            stored = xp.transpose(data, storage_perm)
            options = dict(kwargs)
            if common is not None:
                level_chunks = tuple(
                    min(int(size), int(n))
                    for size, n in zip(common, stored.shape)
                )
                options["chunks"] = level_chunks
                if shards is not None:
                    options["shards"] = tuple(
                        min(int(shard), -(-int(n) // size) * size)
                        for shard, n, size in zip(
                            shards, stored.shape, level_chunks
                        )
                    )
                if is_dask(stored):
                    # Each block of a Dask array is then written into
                    # chunks of its own.
                    stored = stored.rechunk(level_chunks)
            array = node.create_array(str(index), data=stored, **options)
            if common is None:
                common = tuple(int(size) for size in array.chunks)
            levels.append((str(index), entry))

        # Placements are written once as common transformations instead of
        # being folded into the levels, so a round trip keeps them apart.
        entries = []  # type: tx.List[tx.Dict[str, tx.Any]]
        for placement in placements:
            try:
                entry = _map.to_ome(placement, storage_perm, ndim)
            except _map.OmeMappingError as error:
                raise WriterError(
                    "The intrinsic-to-world placement of this pyramid cannot "
                    f"be written as OME-Zarr metadata. {error}"
                ) from error
            rich = rich or _map.needs_rich_version(entry)
            entries.append(entry)
        rich = rich or moved_placement
        # Only 0.6 and later name coordinate systems, which several world
        # spaces require.
        rich = rich or len(entries) > 1
        # A system of the group of the pyramid is written under its OME name.
        # A system of another group cannot be the world space of this one.
        try:
            own = group_path(self.group)
        except ValueError as error:
            raise WriterError(str(error)) from error
        names = []  # type: tx.List[tx.Optional[str]]
        for placement in placements:
            name = getattr(getattr(placement, "output", None), "name", None)
            if isinstance(name, str):
                group, name = split_system_name(name)
                if group != own:
                    raise WriterError(
                        "This pyramid is placed in the coordinate system "
                        f"{name!r} of the group {group!r}, which is not its "
                        "own group, so it cannot be written as one of its "
                        "world spaces."
                    )
            names.append(name)
        worlds = resolve_world_names(names)
        commons = list(zip(worlds, entries))

        resolved = resolve_write_version(version, self._ome_version, rich)
        write_multiscale(node, storage_axes, levels, commons, None, resolved)

    def to_store(
        self,
        location: StoreLike,
        chunks: tx.Any = None,
        version: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Write the pyramid to a store location or an opened group.

        A location is opened with mode `"w"`, which replaces any existing store
        there. The other arguments are those of [`to_node`][].

        The levels of a pyramid read from a store are lazy arrays of that
        store. When the location is the local store that they read, the
        new store is filled beside it and then moved into place, so that
        the levels are read before the old store is removed.

        Raises
        ------
        WriterError
            If the location is the remote store that a level reads, which
            cannot be replaced without the level reading a store half
            written.
        """
        node = as_node(location)
        if node is not None:
            self.to_node(node, chunks=chunks, version=version, **kwargs)
            return
        location = absolute_location(location)
        levels = [getattr(level, "data", None) for level in self.images or ()]
        if not any(reads_inside(raw, location) for raw in levels):
            node = open_group(location, mode="w")
            self.to_node(node, chunks=chunks, version=version, **kwargs)
            return
        if not Path(location).is_local():
            raise WriterError(
                "This image reads the remote store that it would be written "
                f"to, {location}, and a remote store cannot be replaced in "
                "one step, so the image would read a store half written. "
                "Write the image to another location."
            )
        replace_directory(
            location,
            lambda name: self.to_node(
                open_group(name, mode="w"),
                chunks=chunks,
                version=version,
                **kwargs,
            ),
        )

    def _pyramid_axes(self, ndim: int) -> AxisList:
        # The scales that `with_scales` builds downsample the axes that
        # the writer stores, which a pyramid built from scratch takes
        # from `axes`.
        return AxisList(self._write_axes(ndim))

    def _write_axes(self, ndim: int) -> tx.List[Axis]:
        # An axis whose unit is unknown takes the unit of the axis of the
        # same name in the world of the geometry, when that world knows
        # it. Otherwise the unit is left out rather than guessed (#626).
        axes = self._stored_axes(ndim)
        images = list(self.images or [])
        if not images:
            return axes
        world = getattr(images[0].transformation, "output", None)
        units = {
            getattr(axis, "name", None): getattr(axis, "unit", None)
            for axis in (getattr(world, "axes", None) or ())
        }
        return [
            replace(axis, unit=units[axis.name])
            if axis.unit is None and units.get(axis.name) is not None
            else axis
            for axis in axes
        ]

    def _stored_axes(self, ndim: int) -> tx.List[Axis]:
        axes = self.axes
        if axes and (axes.is_variadic or axes.ndim == ndim):
            # OME-Zarr cannot store `...`, so an open axis list is closed to
            # the dimensionality of the data.
            try:
                return axes.expand(ndim)
            except ValueError as error:
                raise WriterError(
                    f"Cannot store the axes {axes} under data of {ndim} "
                    f"dimensions: {error}"
                ) from error
        if self.ome is not None:
            store_axes = multiscale_axes(self.ome)
            if len(store_axes) == ndim:
                perm = _axisorder.to_canonical(store_axes)
                return _axisorder.permute(store_axes, perm)
        return default_canonical_axes(ndim)


def default_canonical_axes(ndim: int) -> tx.List[Axis]:
    """Return the axes that a pyramid without axes is written under.

    The axes are `x`, `y` and `z` in space, then `t` in time and `c` for
    the channels, in the brainhops order, and the axes beyond these five
    are channels named `dim<i>`.
    """
    spatial = [
        SpaceAxis(name=name) for name in ("x", "y", "z")[: min(3, ndim)]
    ]
    trailing = [TimeAxis(name="t"), ChannelAxis(name="c")]
    extra = ndim - len(spatial)
    non_spatial = [
        trailing[i]
        if i < len(trailing)
        else Axis(name="dim" + str(i), type="channel")
        for i in range(extra)
    ]
    return spatial + non_spatial


def _dataset_entry(
    index: int,
    geometry: Transformation,
    storage_perm: tx.Sequence[int],
    ndim: int,
) -> tx.Optional[tx.Dict[str, tx.Any]]:
    # Return the OME transformation of the dataset of a level, or `None`
    # when its geometry does more than scale and translate each axis.
    try:
        return _map.to_dataset_ome(geometry, storage_perm, ndim)
    except _map.OmeMappingError as error:
        raise WriterError(
            f"Level {index} of this image cannot be written as OME-Zarr "
            f"metadata. {error}"
        ) from error


def _split(
    geometries: tx.Sequence[Transformation], ndim: int
) -> tx.Tuple[Transformation, tx.List[Transformation]]:
    # Split the geometries of the levels into a placement that they
    # share and a scale and a translation of each level (#447).
    split = _map.split_levels(geometries, ndim)
    if split is None:
        raise UnrepresentableTransformationError(
            "The levels of this image are not rotated, sheared or flipped "
            "in the same way, so their geometries cannot be split into the "
            "positive scale and the translation that the dataset of each "
            "level holds in OME-NGFF 0.6, and a placement that every level "
            "shares. Give the levels geometries that differ only by a "
            "positive scale and a translation along each axis."
        )
    shared, parts = split
    return shared, list(parts)


def _level(
    array: tx.Any, perm: tx.Sequence[int], geometry: Transformation
) -> OmeZarrLevel:
    """Return the level of a dataset, which reads its array lazily."""
    return OmeZarrLevel(
        transformations=[geometry],
        rawdata=ZarrProxy(array, store_location(array), perm),
        metadata=ZarrMetadata.from_raw(ZarrRaw.from_node(array)),
    )


def _holds_a_field(node: ZarrNode) -> bool:
    """Whether the OME metadata of a group names a field axis.

    A group whose coordinate systems have a displacement or a coordinate
    axis stores a transformation field rather than an image.
    """
    # The transformation formats import the image formats, so their helper
    # is imported when it is used.
    from brainhops.io.transformations.zarr import _node

    for system in _node.coordinate_systems(node):
        for axis in getattr(system, "axes", None) or []:
            if getattr(axis, "type", None) in ("displacement", "coordinate"):
                return True
    return False
