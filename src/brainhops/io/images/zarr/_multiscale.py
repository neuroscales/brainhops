# dependencies
import typing_extensions as tx
from abczarr import ZarrGroup, ZarrNode, open_group
from abczarr.ome.v0_6.images import Multiscale

# internals
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol

# backends
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    SpatialAxis,
    TimeAxis,
)
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.base.zarr import (
    StoreLike,
    ZarrParserWriter,
    _as_node,
)
from brainhops.io.images.base import WritableFileBasedImage
from brainhops.io.images.zarr import _axisorder
from brainhops.io.images.zarr._ome import (
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
from brainhops.io.transformations.zarr import _map

from ._image import ZarrImage


class OmeZarrLevel(ZarrImage):
    """One resolution level, read from its array node on first access.

    The level holds the array handle and permutes it into the brainhops
    order only when its data is read, so opening a pyramid does not read
    any level.
    """

    @smartproperty
    def data(self) -> tx.Optional[ArrayProtocol]:
        node = self.node
        if node is None:
            return None
        backend = get_array_backend()
        raw = backend.asarray(node[...])
        perm = getattr(self, "_perm", None)
        if perm is not None:
            raw = backend.transpose(raw, perm)
        return raw


@register_format
class OmeZarrImage(
    ZarrParserWriter, WritableFileBasedImage, MultiScaleImage
):
    """A multiscale image that is encoded by an OME-Zarr pyramid.

    Each resolution level of the pyramid is read as a single-scale image
    whose voxel-to-world geometry comes from the level's coordinate
    transformation. The metadata is read through abczarr and normalized so
    that each level carries one transformation. The levels are ordered from
    finest to coarsest, and the axes are permuted from the OME storage order
    into the brainhops order at the boundary. A plain Zarr array, which
    carries no multiscale geometry, is read by
    [ZarrImage][brainhops.io.images.zarr.ZarrImage] instead.

    Every level holds its array handle and reads it on first access, so
    opening a large pyramid does not read the arrays.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr", ".ome.zarr")

    # ---- attributes --------------------------------------------------

    _axes: tx.Annotated[
        tx.Optional[tx.List[Axis]],
        tx.Doc(
            "The axes to store a pyramid under, in the brainhops order, "
            "when it is built from scratch. A pyramid read from a store "
            "leaves this unset and takes its axes from `ome`."
        ),
    ] = None

    _ome: tx.Annotated[
        tx.Optional[Multiscale],
        tx.Doc("The OME multiscale metadata, normalized to 0.6."),
    ] = None

    # ---- properties --------------------------------------------------

    @property
    def node(self) -> ZarrGroup:
        return getattr(self, "_node", None)

    @node.setter
    def node(self, value: ZarrGroup) -> None:
        self._node = value
        # Everything derived from the node is dropped, so a new node is read
        # afresh. `smartproperty` caches under `_cache_<name>`.
        for name in (
            "_cache_ome",
            "_cache_layout",
            "_cache_images",
            "_cache_transformations",
        ):
            if hasattr(self, name):
                delattr(self, name)

    @property
    def axes(self) -> tx.Optional[tx.List[Axis]]:
        """The axes a from-scratch pyramid is stored under, or `None`.

        bagof stores the `axes` argument under `_axes` but generates no
        reader for it, so the reader is spelled out here.
        """
        return getattr(self, "_axes", None)

    @smartproperty
    def ome(self) -> tx.Optional[Multiscale]:
        # The multiscale itself, not the OME block that holds it: this is
        # what `multiscale_axes` and the rest of `_ome` take.
        return self._layout()["multiscale"]

    @property
    def _ome_version(self) -> tx.Optional[str]:
        """The version of the OME metadata, as written in the node.

        This is the version the group was read from, which the writer falls
        back to so that a pyramid is written back in the version it came in.
        """
        return self._layout()["version"]

    @smartproperty(empty_as_unset=True)
    def images(self) -> tx.List[SingleScaleImage]:
        return self._layout()["images"]

    @smartproperty(empty_as_unset=True)
    def transformations(self) -> tx.List[Transformation]:
        # The intrinsic-to-world placements the whole pyramid shares, one per
        # world space the metadata names, the preferred one last. A
        # multiscale that declares no common transformations places its
        # levels directly in world space, so the list is empty and
        # `transformation` is the identity.
        return list(self._layout()["commons"])

    # ---- load --------------------------------------------------------

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        """
        Read the pyramid from an opened Zarr group.

        The multiscale metadata is parsed here, so a group whose metadata
        cannot be read as a pyramid is refused at open rather than on first
        access. The levels themselves stay unread: each holds its array
        handle and reads it when its data is asked for.
        """
        image = super().from_node(node, **kwargs)
        image._layout()
        return image

    # ---- workers  ----------------------------------------------------

    def _layout(self) -> tx.Dict[str, tx.Any]:
        # The levels and the shared placement, read from the node together.
        # Both come from one parse of the metadata, so `images` and
        # `transformations` agree on the coordinate systems they name and the
        # metadata is read once however they are reached.
        #
        # A pyramid built from scratch is backed by no node, so there is
        # nothing to derive and the empty layout stands. Only a pyramid that
        # is backed by a node, but whose metadata cannot be read as one, is
        # refused -- by `_read_layout`.
        cached = getattr(self, "_cache_layout", None)
        if cached is None:
            if self.node is None:
                return {
                    "images": [],
                    "commons": [],
                    "multiscale": None,
                    "version": None,
                }
            cached = self._read_layout()
            self._cache_layout = cached
        return cached

    def _read_layout(self) -> tx.Dict[str, tx.Any]:
        node = self.node
        if not isinstance(node, ZarrGroup):
            raise OmeImageError(
                "This Zarr store is an array, not a group, so it cannot be "
                "read as a multiscale image."
            )

        # `read_multiscale` parses the metadata and reports the specific
        # fault when it contradicts the schema, so the scoring check is not
        # repeated here: it would parse a second time only to report the
        # vaguer "carries no multiscale metadata" for a group whose real
        # problem is known.
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
        # The levels end in the intrinsic space that every level shares, and
        # the pyramid's own transformations carry that space to each world
        # space the metadata names. When the multiscale declares no common
        # transformations the intrinsic space *is* the world space, so a
        # level lands directly in world space.
        declared = system_axes(multiscale)

        def system(name: tx.Optional[str]) -> CoordinateSystem:
            # A named system keeps its own axes, permuted into the brainhops
            # order like the levels are. A system that names a different
            # number of axes than the arrays have cannot be laid out that
            # way, so the pyramid's own axes stand in.
            axes = declared.get(name) if isinstance(name, str) else None
            if axes is None or len(axes) != ndim:
                axes = canonical_axes
            else:
                axes = _axisorder.permute(axes, perm)
            return CoordinateSystem(name=name, axes=axes)

        voxel_system = CoordinateSystem(name="voxel", axes=canonical_axes)
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
            level = OmeZarrLevel(transformations=[geometry], node=array)
            level._perm = perm
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
        # An OME-Zarr image is a group that carries multiscale metadata. A
        # plain array is left to the single-scale reader. The raw attributes
        # are inspected, so a malformed pyramid is still recognized here and
        # reported by the reader rather than passed over.
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
        """Write the pyramid into an opened Zarr group, and return it."""
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
        # One chunking is used for every level, reordered from the brainhops
        # axis order into the stored order. This matches abczarr, whose
        # pyramid construction chunks every level the same way.
        stored_chunks = (
            None if chunks is None else tuple(chunks[p] for p in storage_perm)
        )

        backend = get_array_backend()
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
            stored = backend.transpose(data, storage_perm)
            options = dict(kwargs)
            if stored_chunks is not None:
                options["chunks"] = stored_chunks
            node.create_array(str(index), data=stored, **options)
            levels.append((str(index), entry))

        # The pyramid's own intrinsic-to-world placements are written once, as
        # the multiscale's common transformations, rather than folded into
        # every level. A read of what was written therefore returns the same
        # split of level and pyramid that was written. A pyramid placed in
        # several world spaces writes one transformation per space, each
        # leaving the intrinsic space.
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
        # Only 0.6 and later name their coordinate systems, so only they
        # can carry a pyramid placed in more than one world space.
        rich = rich or len(entries) > 1
        worlds = resolve_world_names([
            getattr(getattr(placement, "output", None), "name", None)
            for placement in placements
        ])
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
        """Write the pyramid to a store location, or into an opened store.

        An opened group is written into as it stands. A location names a
        store that does not exist yet, so the group is created there first.
        """
        node = _as_node(location)
        if node is None:
            node = open_group(location, mode="w")
        self.to_node(node, chunks=chunks, version=version, **kwargs)

    def _write_axes(self, ndim: int) -> tx.List[Axis]:
        # The axes to store the pyramid under, in the brainhops order. An
        # explicit `axes` (a from-scratch pyramid) is used first. A pyramid
        # read from a store has none, so its axes are derived from `ome`. A
        # pyramid with neither falls back to a default axis list.
        if self.axes and len(self.axes) == ndim:
            return list(self.axes)
        if self.ome is not None:
            store_axes = multiscale_axes(self.ome)
            if len(store_axes) == ndim:
                perm = _axisorder.to_canonical(store_axes)
                return _axisorder.permute(store_axes, perm)
        return _default_canonical_axes(ndim)


def _default_canonical_axes(ndim: int) -> tx.List[Axis]:
    # The axes to assume when none are recorded, in the brainhops order:
    # the spatial axes x, y, z first, then time, then channel.
    spatial = [
        SpatialAxis(name=name) for name in ("x", "y", "z")[: min(3, ndim)]
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
