import typing_extensions as tx
from abczarr import ZarrGroup, ZarrNode, open_group
from abczarr.ome.v0_6.images import Multiscale
from bagof.magic import Inherit, replace

from brainhops._core.properties import smartproperty
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import AxisList, CoordinateSystem
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.common.zarr import StoreLike, ZarrMetadata, ZarrRaw
from brainhops.io.common.zarr._data import _ZarrProxy
from brainhops.io.common.zarr._parsers import (
    _as_node,
    _ZarrNodeReaderWriter,
    opened_node,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.zarr import _axisorder
from brainhops.io.transformations.zarr import _map

from ._image import ZarrImage
from ._ome import (
    OmeImageError,
    common_transformations,
    intrinsic_name,
    level_transformation,
    looks_like_multiscale,
    multiscale_axes,
    read_multiscale,
    resolve_world_names,
    resolve_write_version,
    system_axes,
    write_multiscale,
)

_Ellipsis = type(Ellipsis)
# `types.EllipsisType` only exists from Python 3.10 onwards.


class OmeZarrLevel(ZarrImage):
    """One resolution level of an OME-Zarr pyramid.

    The level holds the lazy array of its dataset in `raw`, which presents
    the stored axes in the brainhops order, and the record of the array of
    its dataset. Its transformations are those of its dataset, assigned
    when the pyramid is read. Opening a pyramid reads no array data, and a
    copy made with `replace` keeps the lazy array. Step 4b-ii of the
    fourth pass makes the level hold the record of its pyramid instead.
    """


# `kw_only=True` is repeated here because bagof takes each class option
# from the first base in the method resolution order, whether that base
# sets the option or keeps the default. That base is `_ZarrNodeReaderWriter`,
# which keeps the default, so the fields declared by this class would
# otherwise be positional.
@register_format
class OmeZarrImage(
    _ZarrNodeReaderWriter, ImageFormat, MultiScaleImage, kw_only=True
):
    """Multiscale image backed by an OME-Zarr pyramid.

    Each level is a single-scale image placed by the coordinate
    transformation of its dataset. The metadata is read with abczarr and
    normalized to OME-NGFF 0.6. Levels run from finest to coarsest, their
    axes are permuted into the brainhops order, and their arrays are read
    lazily. A plain Zarr array is read by [`ZarrImage`][] instead.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr", ".ome.zarr")

    # ---- attributes --------------------------------------------------

    _axes: tx.Annotated[
        tx.Optional[AxisList[tx.Union[Axis, _Ellipsis]]],
        tx.Doc(
            "The axes to store a pyramid under, in the brainhops order, "
            "when it is built from scratch, as an `AxisList`. As in a "
            "coordinate system, a single `...` may stand for the axes "
            "about which nothing is known; it is closed to the number of "
            "axes of the data when the pyramid is written. `None` is not "
            "`[...]`: it leaves the axes unset, so a pyramid read from a "
            "store takes its axes from `ome`."
        ),
    ] = None

    _ome: tx.Annotated[
        tx.Optional[Multiscale],
        tx.Doc("The OME multiscale metadata, normalized to 0.6."),
    ] = None

    _images: tx.Optional[tx.List[SingleScaleImage]] = Inherit(
        "images", default=None
    )
    """The levels given to the pyramid, or `None` to read them.

    The pyramid takes over the `images` field of [`MultiScaleImage`][]
    to store it under this private name. The constructor still takes
    `images` in the same place, and the property of the same name reads
    it. `None`, the default, means that the levels are read from `node`,
    while a list, even an empty one, is used as it is.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The placements given to the pyramid, or `None` to read them.

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
    def axes(self) -> tx.Optional[AxisList[tx.Union[Axis, _Ellipsis]]]:
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

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy an instance of this class or of another multiscale image.

        The pyramid stores its levels in the private field `_images`, where
        `None` means that they are read from `node`. Within this class, the
        stored value is copied, so a copy of a pyramid whose levels were
        not assigned reads them from the same node. From another class,
        the levels that `other` reports are copied, read if need be,
        because the private field of `other` means nothing to this class.
        The placements in `_transformations` are copied in the same way by
        the `from_instance` of [`ImageFormat`][].

        Parameters
        ----------
        other : object
            The instance to copy.
        *args : Any
            Constructor arguments, which take precedence over the fields
            of `other`.
        **kwargs : Any
            Constructor arguments, which take precedence over the fields
            of `other`.

        Returns
        -------
        Self
            The copy.
        """
        if not isinstance(other, cls):
            images = getattr(other, "images", None)
            if images is not None:
                kwargs.setdefault("images", images)
        return super().from_instance(other, *args, **kwargs)

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

        perm = _axisorder.to_canonical(store_axes)
        canonical_axes = _axisorder.permute(store_axes, perm)
        # Without common transformations, the intrinsic space is the world
        # space.
        declared = system_axes(multiscale)

        def system(name: tx.Optional[str]) -> CoordinateSystem:
            # Fall back to the canonical axes when the axis count of a named
            # system does not match the array.
            axes = declared.get(name) if isinstance(name, str) else None
            if axes is None or len(axes) != ndim:
                axes = canonical_axes
            else:
                axes = _axisorder.permute(axes, perm)
            return CoordinateSystem(name=name, axes=axes)

        # Stored arrays are C-ordered. A level whose permutation reverses the
        # stored axes is F-ordered, as in nibabel; any other permutation leaves
        # the order unspecified.
        reverses = list(perm) == list(range(ndim))[::-1]
        voxel_system = CoordinateSystem(
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
                ndim,
                node=node,
                store_axes=store_axes,
                input=voxel_system,
                output=intrinsic_system,
            )
            level = _level(array, perm, geometry)
            images.append(level)

        commons = common_transformations(
            multiscale,
            perm,
            ndim,
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
        multiscale.

        Parameters
        ----------
        node : ZarrGroup
            Group that receives the levels and the metadata.
        chunks : sequence of int, optional
            Chunk shape of every level, in the brainhops axis order.
        version : str, optional
            OME-NGFF version. By default, the source version or 0.6 is used.
        **kwargs : Any
            Passed to the creation of each level array.

        Raises
        ------
        WriterError
            If `node` is an array, if a level is missing or empty, or if the
            axes or transformations cannot be written.
        """
        node = _as_node(node)
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
        ndim = len(images[0].data.shape)
        axes = self._write_axes(ndim)
        storage_perm = _axisorder.to_storage(axes)
        storage_axes = _axisorder.permute(axes, storage_perm)
        # One chunk shape for all levels, reordered to the stored axis order.
        stored_chunks = (
            None if chunks is None else tuple(chunks[p] for p in storage_perm)
        )

        levels = []  # type: tx.List[tx.Tuple[str, tx.Dict[str, tx.Any]]]
        rich = False
        for index, image in enumerate(images):
            data = image.data
            if data is None:
                raise WriterError(
                    f"Level {index} of this multiscale image has no data."
                )
            try:
                entry = _map.to_ome(
                    image.transformation,
                    storage_perm,
                    len(data.shape),
                )
            except _map.OmeMappingError as error:
                raise WriterError(
                    f"Level {index} of this image cannot be written as "
                    f"OME-Zarr metadata. {error}"
                ) from error
            rich = rich or _map.needs_rich_version(entry)
            backend = get_array_backend(data)
            stored = backend.transpose(data, storage_perm)
            options = dict(kwargs)
            if stored_chunks is not None:
                options["chunks"] = stored_chunks
            node.create_array(str(index), data=stored, **options)
            levels.append((str(index), entry))

        # Placements are written once as common transformations instead of
        # being folded into the levels, so a round trip keeps them apart.
        placements = list(self.transformations or [])
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
        # Only 0.6 and later name coordinate systems, which several world
        # spaces require.
        rich = rich or len(entries) > 1
        worlds = resolve_world_names(
            [
                getattr(getattr(placement, "output", None), "name", None)
                for placement in placements
            ]
        )
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
        """
        node = _as_node(location)
        if node is None:
            node = open_group(location, mode="w")
        self.to_node(node, chunks=chunks, version=version, **kwargs)

    def _write_axes(self, ndim: int) -> tx.List[Axis]:
        axes = self.axes
        if axes and (axes.is_open or axes.ndim == ndim):
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
        return _default_canonical_axes(ndim)


def _default_canonical_axes(ndim: int) -> tx.List[Axis]:
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


def _level(
    array: tx.Any, perm: tx.Sequence[int], geometry: Transformation
) -> OmeZarrLevel:
    """Return the level of a dataset, which reads its array lazily."""
    return OmeZarrLevel(
        transformations=[geometry],
        raw=_ZarrProxy(array, array.store_path, perm),
        metadata=ZarrMetadata.from_raw(ZarrRaw.from_node(array)),
    )
