# dependencies
import abczarr
import typing_extensions as tx
from abczarr.abc.sync import ZarrNode
from abczarr.ome.v0_6rc0.images import Multiscale

# internals
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
from brainhops.datamodel.transformations import (
    Affine,
    Transformation,
    _affine_matrix,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.images.base import WritableFileBasedImage
from brainhops.io.images.zarr import _axisorder
from brainhops.io.images.zarr._ome import (
    OmeImageError,
    affine_from_matrix,
    level_matrix,
    looks_like_multiscale,
    multiscale_axes,
    permute_affine,
    read_multiscale,
    resolve_write_version,
    scale_translation_from_affine,
    write_multiscale,
)
from brainhops.io.images.zarr._store import ZarrParser


class _ZarrLevel(SingleScaleImage):
    """One resolution level, read from its array node on first access.

    The level holds the array handle and permutes it into the brainhops
    order only when its data is read, so opening a pyramid does not read
    any level.
    """

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        cached = getattr(self, "_data", None)
        if cached is not None:
            return cached
        node = getattr(self, "_node", None)
        if node is None:
            return None
        backend = get_array_backend()
        raw = backend.asarray(node[...])
        perm = getattr(self, "_perm", None)
        if perm is not None:
            raw = backend.transpose(raw, perm)
        self._data = raw
        return self._data

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._data = value


@register_format
class OmeZarrImage(ZarrParser, WritableFileBasedImage, MultiScaleImage):
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

    axes: tx.Annotated[
        tx.Optional[tx.List[Axis]],
        tx.Doc(
            "The axes to store a pyramid under, in the brainhops order, "
            "when it is built from scratch. A pyramid read from a store "
            "leaves this unset and takes its axes from `ome`."
        ),
    ] = None

    ome: tx.Annotated[
        tx.Optional[Multiscale],
        tx.Doc("The OME multiscale metadata, normalized to 0.6rc0."),
    ] = None

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        # An OME-Zarr image is a group that carries multiscale metadata. A
        # plain array is left to the single-scale reader. The raw attributes
        # are inspected, so a malformed pyramid is still recognized here and
        # reported by the reader rather than passed over.
        if not isinstance(node, abczarr.ZarrGroup):
            return Confidence.NO
        if not looks_like_multiscale(node):
            return Confidence.NO
        return Confidence.CERTAIN

    @classmethod
    def _read_node(cls, node: ZarrNode, **kwargs) -> tx.Self:
        if not isinstance(
            node, abczarr.ZarrGroup
        ) or not looks_like_multiscale(node):
            raise OmeImageError(
                "This Zarr group carries no OME multiscale metadata, so it "
                "cannot be read as a multiscale image."
            )
        result = read_multiscale(node)
        if result is None:
            raise OmeImageError(
                "This Zarr group carries no OME multiscale metadata, so it "
                "cannot be read as a multiscale image."
            )
        multiscale, source_version = result
        datasets = list(multiscale.datasets)
        if not datasets:
            raise OmeImageError(
                "This OME multiscale names no datasets, so it has no "
                "resolution levels to read."
            )

        store_axes = multiscale_axes(multiscale)
        first = node[str(datasets[0].path)]
        ndim = first.ndim
        if len(store_axes) != ndim:
            raise OmeImageError(
                "This OME multiscale names "
                f"{len(store_axes)} axes, but its arrays have {ndim} "
                "dimensions."
            )
        perm = _axisorder.to_canonical(store_axes)
        canonical_axes = _axisorder.permute(store_axes, perm)
        input_system = CoordinateSystem(name="voxel", axes=canonical_axes)
        output_system = CoordinateSystem(name="world", axes=canonical_axes)

        images = []  # type: tx.List[SingleScaleImage]
        for dataset in datasets:
            array = node[str(dataset.path)]
            matrix = level_matrix(multiscale, dataset, ndim)
            affine = affine_from_matrix(
                permute_affine(matrix, perm),
                input=input_system,
                output=output_system,
            )
            level = _ZarrLevel(transformations=[affine])
            level._node = array
            level._perm = perm
            images.append(level)

        # `axes` is left unset: a read pyramid takes its axes from `ome`,
        # rather than storing a second copy that could drift from it.
        image = cls(images=images, ome=multiscale)
        image._source_version = source_version
        return image

    def _write_node(
        self,
        node: ZarrNode,
        chunks: tx.Any = None,
        version: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        if not isinstance(node, abczarr.ZarrGroup):
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
        sperm = _axisorder.to_storage(axes)
        storage_axes = _axisorder.permute(axes, sperm)
        per_level_chunks = _resolve_chunks(chunks, len(images))

        backend = get_array_backend()
        levels = []  # type: tx.List[tx.Tuple[str, tx.List[float], tx.List[float]]]
        for index, image in enumerate(images):
            data = image.data
            if data is None:
                raise WriterError(
                    f"Level {index} of this multiscale image has no data."
                )
            matrix = _affine_matrix(self._level_transform(image))
            if matrix is None:
                raise WriterError(
                    f"Level {index} of this image is placed by a "
                    "transformation that is not an affine, so it cannot be "
                    "written as OME-Zarr metadata."
                )
            scale, translation = scale_translation_from_affine(
                Affine(matrix=matrix), len(data.shape)
            )
            stored = backend.transpose(data, sperm)
            options = dict(kwargs)
            level_chunks = per_level_chunks[index]
            if level_chunks is not None:
                options["chunks"] = tuple(level_chunks[p] for p in sperm)
            node.create_array(str(index), data=stored, **options)
            levels.append(
                (
                    str(index),
                    _axisorder.permute(list(scale), sperm),
                    _axisorder.permute(list(translation), sperm),
                )
            )

        resolved = resolve_write_version(
            version, getattr(self, "_source_version", None)
        )
        write_multiscale(node, storage_axes, levels, None, resolved)

    def _create_store(
        self,
        location: str,
        chunks: tx.Any = None,
        version: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        group = abczarr.open_group(location, mode="w")
        self._write_node(group, chunks=chunks, version=version, **kwargs)

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

    def _level_transform(self, image: SingleScaleImage) -> Transformation:
        # The voxel-to-world transformation of one level. The pyramid's own
        # preferred transformation, when it has one, is composed onto the
        # level's transformation, so a whole-pyramid placement is written
        # into every level.
        if self.transformations:
            return (self.transformation @ image.transformation).compute()
        return image.transformation


def _resolve_chunks(
    chunks: tx.Any, nlevels: int
) -> tx.List[tx.Optional[tx.Sequence[int]]]:
    # Normalize the `chunks` argument to one entry per level. `None` lets
    # each level choose its own chunking. A single shape is used for every
    # level. A per-level sequence gives one shape (or `None`) per level, so
    # a coarse level need not carry the finest level's chunk shape.
    if chunks is None:
        return [None] * nlevels
    if (
        isinstance(chunks, (list, tuple))
        and len(chunks) == nlevels
        and all(
            item is None or isinstance(item, (list, tuple)) for item in chunks
        )
    ):
        return list(chunks)
    return [chunks] * nlevels


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
