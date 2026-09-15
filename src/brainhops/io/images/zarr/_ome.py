"""Read and build OME-Zarr multiscale metadata through abczarr.

The OME-Zarr metadata of a group is read with abczarr, which parses it
into a typed object and validates it. The typed object is normalized to
OME-NGFF 0.6rc0, in which each resolution level carries a single
coordinate transformation. The reader can therefore assume one
transformation specification per level, whatever version the group was
written in.

Malformed or missing metadata is rejected by abczarr while it parses, so a
level with no transformation, or one placed by a transformation that is
misspelled or of the wrong shape, raises rather than being read as an
identity placement.
"""

# stdlib
from collections.abc import Mapping

# dependencies
import numpy as np
import typing_extensions as tx
from abczarr.ome import v0_6rc0 as _v6

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import Affine
from brainhops.io.base.parsers import ParserContentError, WriterError
from brainhops.io.transformations.zarr._axes import _to_axis

#: The OME-NGFF version the reader normalizes every group to.
NORMALIZED_VERSION = "0.6rc0"

#: The OME-NGFF version the writer emits when nothing else selects one. A
#: per-axis scale and translation is expressible in every version, and this
#: is the leanest that carries it.
DEFAULT_WRITE_VERSION = "0.4"


class OmeImageError(ParserContentError):
    """Raised when an OME-Zarr image group cannot be read.

    A group with no multiscale metadata, or one whose metadata abczarr
    cannot parse into a valid pyramid, is refused with this error.
    """


def looks_like_multiscale(node: tx.Any) -> bool:
    """Whether a group's attributes carry image multiscale metadata.

    This inspects the raw attributes only, so a group that names a
    multiscale is recognized even when the metadata is malformed. The
    reader then reports the specific fault instead of the group being
    passed over.
    """
    attrs = dict(node.attrs)
    if "multiscales" in attrs:
        return True
    inner = attrs.get("ome")
    return isinstance(inner, Mapping) and "multiscales" in inner


def read_multiscale(
    node: tx.Any,
) -> tx.Optional[tx.Tuple[tx.Any, tx.Optional[str]]]:
    """Return a group's first multiscale, normalized to 0.6rc0.

    The result is ``(multiscale, source_version)``, where `multiscale` is
    the abczarr 0.6rc0 multiscale object and `source_version` is the
    OME-NGFF version the group was written in. The result is `None` when
    the group carries no OME metadata at all.
    """
    try:
        ome = node.ome
    except Exception as error:
        raise OmeImageError(
            "This OME-Zarr group carries metadata that could not be read as "
            "a valid image pyramid. " + str(error)
        ) from error
    if ome is None:
        return None
    source_version = getattr(ome, "version", None)
    try:
        normalized = ome.to_version(NORMALIZED_VERSION)
    except Exception as error:
        raise OmeImageError(
            "This OME-Zarr image could not be converted to a form with one "
            "transformation per level. " + str(error)
        ) from error
    multiscales = getattr(normalized, "multiscales", None)
    if not multiscales:
        return None
    return multiscales[0], source_version


def _output_system(multiscale: tx.Any) -> tx.Any:
    # The coordinate system a dataset maps its array onto, read from the
    # first dataset's transformation. The first coordinate system is used
    # when no transformation names one.
    systems = list(multiscale.coordinateSystems)
    for dataset in multiscale.datasets:
        for transform in dataset.coordinateTransformations:
            output = getattr(transform, "output", None)
            name = getattr(output, "name", None)
            if isinstance(name, str):
                for system in systems:
                    if system.name == name:
                        return system
    return systems[0]


def multiscale_axes(multiscale: tx.Any) -> tx.List[Axis]:
    """Return the axes of a multiscale as brainhops axes, in stored order."""
    system = _output_system(multiscale)
    return [_to_axis(axis.to_json()) for axis in system.axes]


def _homogeneous(matrix: np.ndarray, ndim: int) -> np.ndarray:
    # Pad an ``(n, n)`` or ``(n, n + 1)`` matrix to a square homogeneous
    # ``(ndim + 1, ndim + 1)`` matrix, so a chain of transforms composes by
    # matrix multiplication.
    out = np.eye(ndim + 1)
    rows = matrix.shape[0]
    out[:rows, : matrix.shape[1]] = matrix
    return out


def _transform_matrix(transform: tx.Any, ndim: int) -> np.ndarray:
    # The homogeneous matrix of one 0.6rc0 coordinate transformation, in the
    # stored axis order. Only the transformations that describe a static,
    # axis-placed geometry are accepted; a transformation defined by a
    # sampled field, or one that reorders or drops axes, is refused.
    kind = getattr(transform, "type", None)
    if kind == "identity":
        return np.eye(ndim + 1)
    if kind == "scale" and isinstance(getattr(transform, "scale", None), list):
        matrix = np.eye(ndim + 1)
        matrix[np.arange(ndim), np.arange(ndim)] = transform.scale
        return matrix
    if kind == "translation" and isinstance(
        getattr(transform, "translation", None), list
    ):
        matrix = np.eye(ndim + 1)
        matrix[:ndim, ndim] = transform.translation
        return matrix
    if kind in ("affine", "rotation"):
        inline = getattr(transform, kind, None)
        if isinstance(inline, list):
            return _homogeneous(np.asarray(inline, dtype=float), ndim)
    if kind == "sequence":
        matrix = np.eye(ndim + 1)
        for inner in transform.transformations:
            matrix = _transform_matrix(inner, ndim) @ matrix
        return matrix
    raise OmeImageError(
        "This OME-Zarr image is placed by a "
        f"{kind!r} coordinate transformation, which is not a static axis "
        "placement and cannot be read as an image geometry."
    )


def level_matrix(multiscale: tx.Any, dataset: tx.Any, ndim: int) -> np.ndarray:
    """Return the voxel-to-world matrix of one level, in stored axis order.

    The level's own transformation is composed with the multiscale
    transformations that apply to every level. The result is an ``(ndim,
    ndim + 1)`` matrix.
    """
    matrix = np.eye(ndim + 1)
    transforms = list(dataset.coordinateTransformations)
    if transforms:
        matrix = _transform_matrix(transforms[0], ndim) @ matrix
    common = getattr(multiscale, "coordinateTransformations", None)
    if isinstance(common, list):
        for transform in common:
            matrix = _transform_matrix(transform, ndim) @ matrix
    return matrix[:ndim]


def permute_affine(matrix: np.ndarray, perm: tx.Sequence[int]) -> np.ndarray:
    """Reorder the rows and linear columns of an affine matrix by `perm`.

    The matrix is ``(n, n + 1)``: a linear block and a translation column.
    Both the output axes (rows) and the input axes (linear columns) are
    reordered by `perm`. The translation column keeps its place.
    """
    matrix = np.asarray(matrix, dtype=float)
    perm = list(perm)
    linear = matrix[:, :-1][np.ix_(perm, perm)]
    translation = matrix[:, -1][perm]
    return np.concatenate([linear, translation[:, None]], axis=1)


def affine_from_matrix(
    matrix: np.ndarray,
    input: tx.Any = None,
    output: tx.Any = None,
) -> Affine:
    """Build a voxel-to-world affine from an ``(n, n + 1)`` matrix."""
    return Affine(
        matrix=np.asarray(matrix, dtype=float), input=input, output=output
    )


def scale_translation_from_affine(
    affine: Affine, ndim: int
) -> tx.Tuple[np.ndarray, np.ndarray]:
    """Return the per-axis scale and translation of a diagonal affine.

    The affine must reduce to a per-axis scale and translation, so its
    matrix must be diagonal apart from the translation column. A placement
    with any off-diagonal term, such as a rotation or a shear, cannot be
    written as scale-and-translation OME metadata and is refused.
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
            "OME-Zarr multiscale metadata. Only an axis-aligned geometry, "
            "whose matrix is diagonal apart from the translation, is "
            "supported."
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


def resolve_write_version(
    version: tx.Optional[str], source_version: tx.Optional[str]
) -> str:
    """Choose the OME-NGFF version to write.

    An explicit `version` is used as given. Otherwise the version of the
    OME-Zarr the image was read from is used, so a pyramid is written back
    in the version it came from. When neither is available, the leanest
    version that carries a per-axis scale and translation is used.
    """
    if version is not None:
        return version
    if source_version is not None:
        return source_version
    return DEFAULT_WRITE_VERSION


def _level_transforms(
    scale: tx.Sequence[float],
    translation: tx.Sequence[float],
    path: str,
) -> tx.List[tx.Dict[str, tx.Any]]:
    refs = {"input": {"path": path}, "output": {"name": "physical"}}
    transforms = [{"type": "scale", "scale": [float(s) for s in scale]}]  # type: tx.List[tx.Dict[str, tx.Any]]
    if np.any(np.asarray(translation, dtype=float) != 0.0):
        transforms.append(
            {
                "type": "translation",
                "translation": [float(t) for t in translation],
            }
        )
    if len(transforms) == 1:
        return [dict(transforms[0], **refs)]
    return [dict({"type": "sequence", "transformations": transforms}, **refs)]


def build_ome(
    axes: tx.Sequence[tx.Any],
    levels: tx.Sequence[tx.Tuple[str, tx.Sequence[float], tx.Sequence[float]]],
    name: tx.Optional[str],
    version: str,
) -> tx.Any:
    """Build the typed OME metadata for an image pyramid.

    `axes` are the axes in the stored order. `levels` gives, for each
    resolution level, its array path and the per-axis scale and translation
    that place it in world space. The metadata is built in 0.6rc0 and then
    converted to `version`, so any supported version can be written from one
    code path. The returned object is assigned to a group's ``ome``.
    """
    block = {
        "coordinateSystems": [
            {
                "name": "physical",
                "axes": [axis_to_json(axis) for axis in axes],
            }
        ],
        "datasets": [
            {
                "path": path,
                "coordinateTransformations": _level_transforms(
                    scale, translation, path
                ),
            }
            for path, scale, translation in levels
        ],
    }  # type: tx.Dict[str, tx.Any]
    if name is not None:
        block["name"] = name
    ome = _v6.OME.from_json(
        {"version": NORMALIZED_VERSION, "multiscales": [block]}
    )
    if version != NORMALIZED_VERSION:
        ome = ome.to_version(version)
    return ome


def write_multiscale(
    node: tx.Any,
    axes: tx.Sequence[tx.Any],
    levels: tx.Sequence[tx.Tuple[str, tx.Sequence[float], tx.Sequence[float]]],
    name: tx.Optional[str],
    version: str,
) -> None:
    """Write an image pyramid's OME metadata onto a group through abczarr."""
    node.ome = build_ome(axes, levels, name, version)
