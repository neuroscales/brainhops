"""Read and build the OME-Zarr multiscale metadata for an image pyramid.

An OME-Zarr image group carries a ``multiscales`` block in its
attributes. The block names the axes and lists the datasets of the
pyramid, each with the coordinate transformations that place its array in
world space. The functions here read that block into brainhops axes and
placements, and build it back from them.

Only the ``scale`` and ``translation`` coordinate transformations of
OME-NGFF 0.4 are handled. A pyramid placed by any other transformation is
refused, and a placement that is written out must reduce to a per-axis
scale and translation.
"""

# stdlib
from collections.abc import Mapping

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import Affine
from brainhops.io.base.parsers import ParserContentError, WriterError
from brainhops.io.transformations.zarr._axes import _to_axis

#: The OME-NGFF version the writer emits.
OME_VERSION = "0.4"


class OmeImageError(ParserContentError):
    """Raised when an OME-Zarr image group cannot be read.

    A group with no ``multiscales`` block, or one placed by a coordinate
    transformation other than a scale or a translation, is refused with
    this error.
    """


def read_multiscale(node: tx.Any) -> tx.Optional[tx.Dict[str, tx.Any]]:
    """Return the first multiscale block of a group, or `None`.

    The block is read from the group's attributes. From OME-NGFF 0.5 on it
    is nested under an ``"ome"`` attribute, and before 0.5 it sits directly
    in the attributes. Both layouts are read.
    """
    attrs = dict(node.attrs)
    inner = attrs.get("ome")
    if isinstance(inner, Mapping):
        attrs = dict(inner)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, (list, tuple)) or not multiscales:
        return None
    first = multiscales[0]
    if not isinstance(first, Mapping):
        return None
    return dict(first)


def multiscale_axes(multiscale: tx.Mapping[str, tx.Any]) -> tx.List[Axis]:
    """Return the axes of a multiscale block as brainhops axes.

    A block that names no axes, as in OME-NGFF 0.2 and 0.3, yields an empty
    list. The caller then falls back to a default axis list built from the
    array's number of dimensions.
    """
    axes = multiscale.get("axes")
    if not isinstance(axes, (list, tuple)):
        return []
    return [_to_axis(axis) for axis in axes]


def _compose(
    transforms: tx.Sequence[tx.Mapping[str, tx.Any]],
    ndim: int,
) -> tx.Tuple[np.ndarray, np.ndarray]:
    # The scale and translation that a list of coordinate transformations
    # composes to, applied in order. Only scale and translation transforms
    # are accepted. A transform of any other type is refused.
    scale = np.ones(ndim)
    translation = np.zeros(ndim)
    for transform in transforms:
        kind = (
            transform.get("type") if isinstance(transform, Mapping) else None
        )
        if kind == "scale":
            factor = np.asarray(transform["scale"], dtype=float)
            scale = scale * factor
            translation = translation * factor
        elif kind == "translation":
            offset = np.asarray(transform["translation"], dtype=float)
            translation = translation + offset
        elif kind in ("identity", None):
            continue
        else:
            raise OmeImageError(
                "This OME-Zarr image is placed by a "
                f"{kind!r} coordinate transformation, which is not "
                "supported. Only scale and translation placements can be "
                "read as an image geometry."
            )
    return scale, translation


def level_scale_translation(
    multiscale: tx.Mapping[str, tx.Any],
    dataset: tx.Mapping[str, tx.Any],
    ndim: int,
) -> tx.Tuple[np.ndarray, np.ndarray]:
    """Return the scale and translation that place one dataset in world space.

    The dataset's own coordinate transformations are applied first, then
    the multiscale-level coordinate transformations that apply to every
    dataset. The result is a per-axis scale and translation, in the stored
    axis order.
    """
    dataset_transforms = dataset.get("coordinateTransformations") or []
    scale, translation = _compose(dataset_transforms, ndim)
    common = multiscale.get("coordinateTransformations") or []
    common_scale, common_translation = _compose(common, ndim)
    return (
        common_scale * scale,
        common_scale * translation + common_translation,
    )


def affine_from_scale_translation(
    scale: tx.Sequence[float],
    translation: tx.Sequence[float],
    input: tx.Any = None,
    output: tx.Any = None,
) -> Affine:
    """Build the voxel-to-world affine of a per-axis scale and translation."""
    scale = np.asarray(scale, dtype=float)
    translation = np.asarray(translation, dtype=float)
    ndim = scale.shape[0]
    matrix = np.zeros((ndim, ndim + 1))
    matrix[np.arange(ndim), np.arange(ndim)] = scale
    matrix[:, -1] = translation
    return Affine(matrix=matrix, input=input, output=output)


def scale_translation_from_affine(
    affine: Affine, ndim: int
) -> tx.Tuple[np.ndarray, np.ndarray]:
    """Return the per-axis scale and translation of a diagonal affine.

    The affine must reduce to a per-axis scale and translation, so its
    matrix must be diagonal apart from the translation column. A placement
    with any off-diagonal term, such as a rotation or a shear, cannot be
    written as OME-NGFF 0.4 metadata and is refused.
    """
    if affine is None or getattr(affine, "matrix", None) is None:
        return np.ones(ndim), np.zeros(ndim)
    matrix = np.asarray(affine.matrix, dtype=float)
    linear = matrix[:, :-1]
    off_diagonal = linear - np.diag(np.diag(linear))
    if linear.shape[0] != linear.shape[1] or np.any(
        np.abs(off_diagonal) > 1e-8
    ):
        raise WriterError(
            "This image is placed by a transformation that is not a "
            "per-axis scale and translation, so it cannot be written as "
            "OME-NGFF 0.4 metadata. Only an axis-aligned geometry, whose "
            "matrix is diagonal apart from the translation, is supported."
        )
    return np.diag(linear), matrix[:, -1]


def axis_to_json(axis: tx.Any) -> tx.Dict[str, tx.Any]:
    """Return the OME-Zarr JSON description of one axis."""
    entry = {}  # type: tx.Dict[str, tx.Any]
    name = getattr(axis, "name", None)
    type_ = getattr(axis, "type", None)
    unit = getattr(axis, "unit", None)
    entry["name"] = name if isinstance(name, str) else ""
    if isinstance(type_, str):
        entry["type"] = type_
    if unit is not None:
        value = getattr(unit, "value", unit)
        if isinstance(value, str):
            entry["unit"] = value
    return entry


def build_multiscale(
    axes: tx.Sequence[tx.Any],
    levels: tx.Sequence[tx.Tuple[str, tx.Sequence[float], tx.Sequence[float]]],
    name: tx.Optional[str] = None,
) -> tx.Dict[str, tx.Any]:
    """Build the OME-Zarr metadata for an image pyramid.

    `axes` are the axes in the stored order. `levels` gives, for each
    resolution level, its array path and the per-axis scale and translation
    that place it in world space. The result is the group attributes to
    write, a mapping with a single ``multiscales`` block. The block records
    its OME-NGFF version, in the OME-NGFF 0.4 layout that keeps the version
    inside the block rather than at the top level.
    """
    datasets = []
    for path, scale, translation in levels:
        transforms = [{"type": "scale", "scale": [float(s) for s in scale]}]  # type: tx.List[tx.Dict[str, tx.Any]]
        if np.any(np.asarray(translation, dtype=float) != 0.0):
            transforms.append(
                {
                    "type": "translation",
                    "translation": [float(t) for t in translation],
                }
            )
        datasets.append(
            {"path": path, "coordinateTransformations": transforms}
        )
    block = {
        "version": OME_VERSION,
        "axes": [axis_to_json(axis) for axis in axes],
        "datasets": datasets,
    }  # type: tx.Dict[str, tx.Any]
    if name is not None:
        block["name"] = name
    return {"multiscales": [block]}
