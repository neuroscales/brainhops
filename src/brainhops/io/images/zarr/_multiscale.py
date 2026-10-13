import typing_extensions as tx
from abczarr import ZarrGroup, ZarrNode, open_group
from abczarr.ome.v0_6.images import Multiscale
from bagof.magic import Inherit, NoRepr, replace

from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import AxisList, CoordinateSystem
from brainhops.datamodel.transformations import (
    Identity,
    Sequence,
    Transformation,
)
from brainhops.io.base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.common.zarr import StoreLike, ZarrReaderWriter
from brainhops.io.common.zarr._parsers import _as_node
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.zarr import _axisorder
from brainhops.io.transformations.zarr import _map

from ._image import ZarrImage
from ._ome import (
    OmeImageError,
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

_Ellipsis = type(Ellipsis)
# `types.EllipsisType` only exists from Python 3.10 onwards.


class OmeZarrLevel(ZarrImage):
    """One resolution level of an OME-Zarr pyramid.

    The array is transposed into the brainhops axis order only when `data`
    is read, so opening a pyramid reads no array data.
    """

    _perm: NoRepr[tx.Optional[tx.Tuple[int, ...]]] = None
    """The permutation from the stored axes to the brainhops order.

    The level keeps it as a field, so that a copy made with `replace`
    keeps the node and still transposes the array that it reads.
    """

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        raw = super().data
        backend = get_array_backend(raw)
        if self._perm is not None:
            raw = backend.transpose(raw, self._perm)
        return raw


# `kw_only=True` is repeated here because bagof takes each class option
# from the first base in the method resolution order, whether that base
# sets the option or keeps the default. That base is `ZarrReaderWriter`,
# which keeps the default, so the fields declared by this class would
# otherwise be positional.
@register_format
class OmeZarrImage(
    ZarrReaderWriter, ImageFormat, MultiScaleImage, kw_only=True
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

    _group: tx.Annotated[
        str,
        tx.Doc(
            "The path of the group of the pyramid from the root of its Zarr "
            "hierarchy, such as `labels/cells`. The coordinate systems of a "
            "pyramid in another group than the root are named after that "
            "path, as in `labels/cells//physical`, so that the systems of "
            "two groups stay distinct. A Zarr group does not know its path, "
            "so the reader that opened the root of the hierarchy gives it. "
            "The empty string, the default, is the root group, whose "
            "systems keep their OME names."
        ),
    ] = ""

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
    def group(self) -> str:
        """Path of the group of the pyramid from the root of its hierarchy."""
        return getattr(self, "_group", "")

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
        image = super().from_node(node, **kwargs)
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
                node=node,
                store_axes=store_axes,
                input=voxel_system,
                output=intrinsic_system,
                orders=orders,
            )
            level = OmeZarrLevel(
                transformations=[geometry], node=array, perm=perm
            )
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
            axes or transformations cannot be written. This includes a
            level that does not only scale and translate each axis in a
            pyramid of several levels, or in a pyramid that has its own
            placements, since an OME-NGFF 0.6 dataset only holds a scale and
            a translation. The geometry of a single level becomes the
            placement of a pyramid that has none.
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
        placements = list(self.transformations or [])
        moved_placement = False
        # A dataset only holds a positive scale, and a flip is never written
        # as a negative one (item 135). The flip that every level shares
        # therefore goes before the placements of the pyramid, as a linear
        # transformation, and each level keeps the rest of its geometry.
        geometries = [image.transformation for image in images]
        signs = _shared_flip(geometries, ndim)
        if signs is not None:
            geometries = [
                _map.unflipped(geometry, signs, ndim)
                for geometry in geometries
            ]
            linear = _map.flip(signs)
            if placements:
                placements = [
                    Sequence(
                        [linear, placement],
                        input=getattr(placement, "input", None),
                        output=getattr(placement, "output", None),
                    )
                    for placement in placements
                ]
            else:
                output = getattr(images[0].transformation, "output", None)
                placements = [replace(linear, output=output)]
            moved_placement = True
        for index, image in enumerate(images):
            data = image.data
            if data is None:
                raise WriterError(
                    f"Level {index} of this multiscale image has no data."
                )
            try:
                entry = _map.to_dataset_ome(
                    geometries[index],
                    storage_perm,
                    len(data.shape),
                )
            except _map.OmeMappingError as error:
                raise WriterError(
                    f"Level {index} of this image cannot be written as "
                    f"OME-Zarr metadata. {error}"
                ) from error
            if entry is None:
                # A dataset only scales and translates. The geometry of a
                # single level becomes the placement of the pyramid, which is
                # exact. Splitting the geometry of several levels into a
                # diagonal part per level and a shared placement is not done
                # yet (#447).
                if len(images) != 1 or placements:
                    raise WriterError(
                        f"Level {index} of this image is placed by a "
                        "transformation that does not only scale and "
                        "translate each axis, which an OME-NGFF 0.6 dataset "
                        "cannot hold. Give the levels a scale and a "
                        "translation, and the multiscale image the rest of "
                        "the placement. Splitting it when writing is tracked "
                        "in neuroscales/brainhops#447."
                    )
                placements.append(geometries[index])
                entry = _map.to_ome(Identity(), storage_perm, len(data.shape))
                moved_placement = True
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


def _shared_flip(
    geometries: tx.Sequence[Transformation], ndim: int
) -> tx.Optional[tx.Any]:
    # Return the signs of the flip that every level shares, or `None` when
    # no level flips an axis or when a level does not only scale and
    # translate, which the writer handles level by level. Levels that flip
    # different axes cannot share one placement (#447).
    signs = [_map.diagonal_signs(geometry, ndim) for geometry in geometries]
    if any(sign is None for sign in signs):
        return None
    if not any(bool((sign < 0).any()) for sign in signs):
        return None
    if any(list(sign) != list(signs[0]) for sign in signs):
        raise WriterError(
            "The levels of this image flip different axes, so the flip "
            "cannot be moved to a placement that they share, and an "
            "OME-NGFF dataset only holds a positive scale. Splitting the "
            "geometry of the levels when writing is tracked in "
            "neuroscales/brainhops#447."
        )
    return signs[0]
