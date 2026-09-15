# dependencies
import abczarr
import typing_extensions as tx

# backends
from brainhops.backends import get_array_backend

# internals
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    SpatialAxis,
    TimeAxis,
)
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, _affine_matrix
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.images.base import WritableFileBasedImage
from brainhops.io.images.zarr import _axisorder
from brainhops.io.images.zarr._ome import (
    OmeImageError,
    affine_from_scale_translation,
    build_multiscale,
    level_scale_translation,
    multiscale_axes,
    read_multiscale,
    scale_translation_from_affine,
)
from brainhops.io.images.zarr._store import ZarrParser


@register_format
class OmeZarrImage(ZarrParser, WritableFileBasedImage, MultiScaleImage):
    """A multiscale image that is encoded by an OME-Zarr pyramid.

    Each resolution level of the pyramid is read as a single-scale image
    whose voxel-to-world geometry comes from the level's OME coordinate
    transformations. The levels are ordered from finest to coarsest, and
    the axes are permuted from the OME storage order into the brainhops
    order at the boundary. A plain Zarr array, which carries no multiscale
    geometry, is read by
    [ZarrImage][brainhops.io.images.zarr.ZarrImage] instead.

    Only the OME-NGFF 0.4 scale and translation placements are handled. A
    pyramid placed by any other coordinate transformation is refused.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr", ".ome.zarr")

    axes: tx.Annotated[
        tx.Optional[tx.List[Axis]],
        tx.Doc("The axes of the pyramid, in the brainhops order."),
    ] = None

    ome: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc("The OME multiscale metadata exactly as read."),
    ] = None

    @classmethod
    def _score_store(cls, node: tx.Any) -> float:
        # An OME-Zarr image is a group that carries a multiscale block. A
        # plain array is left to the single-scale reader.
        if not isinstance(node, abczarr.ZarrGroup):
            return Confidence.NO
        if read_multiscale(node) is None:
            return Confidence.NO
        return Confidence.CERTAIN

    @classmethod
    def _from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        multiscale = read_multiscale(node)
        if multiscale is None:
            raise OmeImageError(
                "This Zarr group carries no OME multiscale metadata, so it "
                "cannot be read as a multiscale image."
            )
        datasets = list(multiscale.get("datasets") or [])
        if not datasets:
            raise OmeImageError(
                "This OME multiscale names no datasets, so it has no "
                "resolution levels to read."
            )
        backend = get_array_backend()

        # The axes are shared by every level, so the permutation into the
        # brainhops order is computed once from the first level's rank.
        first = backend.asarray(node[str(datasets[0]["path"])][...])
        ndim = len(first.shape)
        axes = multiscale_axes(multiscale)
        if len(axes) != ndim:
            axes = _default_storage_axes(ndim)
        perm = _axisorder.to_canonical(axes)
        canonical_axes = _axisorder.permute(axes, perm)
        input_system = CoordinateSystem(name="voxel", axes=canonical_axes)
        output_system = CoordinateSystem(name="world", axes=canonical_axes)

        images = []
        for index, dataset in enumerate(datasets):
            raw = (
                first
                if index == 0
                else backend.asarray(node[str(dataset["path"])][...])
            )
            scale, translation = level_scale_translation(
                multiscale, dataset, len(raw.shape)
            )
            data = backend.transpose(raw, perm)
            affine = affine_from_scale_translation(
                _axisorder.permute(list(scale), perm),
                _axisorder.permute(list(translation), perm),
                input=input_system,
                output=output_system,
            )
            images.append(
                SingleScaleImage(data=data, transformations=[affine])
            )

        return cls(images=images, axes=canonical_axes, ome=multiscale)

    def _to_store(
        self, location: str, chunks: tx.Any = None, **kwargs
    ) -> None:
        images = list(self.images or [])
        if not images:
            raise WriterError(
                "This multiscale image has no levels, so there is nothing "
                "to write."
            )
        ndim = len(images[0].data.shape)
        axes = (
            self.axes
            if self.axes and len(self.axes) == ndim
            else (_default_canonical_axes(ndim))
        )
        sperm = _axisorder.to_storage(axes)
        storage_axes = _axisorder.permute(axes, sperm)

        backend = get_array_backend()
        group = abczarr.open_group(location, mode="w")
        levels = []
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
                    "written as OME-NGFF metadata."
                )
            scale, translation = scale_translation_from_affine(
                Affine(matrix=matrix), len(data.shape)
            )
            stored = backend.transpose(data, sperm)
            options = dict(kwargs)
            if chunks is not None:
                options["chunks"] = chunks
            group.create_array(str(index), data=stored, **options)
            levels.append(
                (
                    str(index),
                    _axisorder.permute(list(scale), sperm),
                    _axisorder.permute(list(translation), sperm),
                )
            )
        group.update_attributes(build_multiscale(storage_axes, levels))

    def _level_transform(self, image: SingleScaleImage) -> tx.Any:
        # The voxel-to-world transformation of one level. The pyramid's own
        # preferred transformation, when it has one, is composed onto the
        # level's transformation, so a whole-pyramid placement is written
        # into every level.
        if self.transformations:
            return (self.transformation @ image.transformation).compute()
        return image.transformation


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
        else Axis(name=f"dim{i}", type="channel")
        for i in range(extra)
    ]
    return spatial + non_spatial


def _default_storage_axes(ndim: int) -> tx.List[Axis]:
    # The axes to assume when none are recorded, in the OME storage order:
    # time, then channel, then the spatial axes z, y, x.
    spatial_names = ("z", "y", "x")
    nspace = min(3, ndim)
    spatial = [SpatialAxis(name=name) for name in spatial_names[3 - nspace :]]
    leading = [TimeAxis(name="t"), ChannelAxis(name="c")]
    extra = ndim - nspace
    non_spatial = [
        leading[i]
        if i < len(leading)
        else Axis(name=f"dim{i}", type="channel")
        for i in range(extra)
    ]
    return non_spatial + spatial
