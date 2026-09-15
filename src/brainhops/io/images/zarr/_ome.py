"""Read and build OME-Zarr multiscale metadata through abczarr.

The OME-Zarr metadata of a group is read with abczarr, which parses it
into a typed object and validates it. The typed object is normalized to
OME-NGFF 0.6rc0, in which each resolution level carries a single
coordinate transformation. The reader can therefore assume one
transformation specification per level, whatever version the group was
written in.

Each OME coordinate transformation is mapped to the brainhops
transformation of the same kind, rather than being collapsed into an
affine. A scale becomes a `Scaling`, a translation a `Translation`, a
rotation a `Rotation`, and a sequence a `Sequence` of the mapped children,
so the geometry a level carries is the one the metadata describes.

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
from abczarr.abc.sync import ZarrGroup
from abczarr.ome import v0_6rc0 as _v6
from abczarr.ome.v0_6rc0.images import Dataset, Multiscale
from abczarr.ome.v0_6rc0.ome import OME
from abczarr.ome.v0_6rc0.transformations import CoordinateTransformation
from bagof.magic import replace

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    Permutation,
    Rotation,
    Scaling,
    Sequence,
    Transformation,
    Translation,
)
from brainhops.io.base.parsers import ParserContentError, WriterError
from brainhops.io.transformations.zarr._axes import _to_axis

#: A per-axis scale or translation, and the placement of one level: its
#: array path with the per-axis scale and translation that place it.
_Vector = tx.Sequence[float]
_Level = tx.Tuple[str, _Vector, _Vector]

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


def looks_like_multiscale(node: ZarrGroup) -> bool:
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
    node: ZarrGroup,
) -> tx.Optional[tx.Tuple[Multiscale, tx.Optional[str]]]:
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


def _output_system(multiscale: Multiscale) -> tx.Any:
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


def multiscale_axes(multiscale: Multiscale) -> tx.List[Axis]:
    """Return the axes of a multiscale as brainhops axes, in stored order."""
    system = _output_system(multiscale)
    return [_to_axis(axis.to_json()) for axis in system.axes]


def _permute_vector(
    values: tx.Sequence[float], perm: tx.Sequence[int]
) -> tx.List[float]:
    # Reorder a per-axis vector, such as a scale or a translation, from the
    # stored axis order into the brainhops order.
    return [float(values[p]) for p in perm]


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


def _permute_linear(matrix: np.ndarray, perm: tx.Sequence[int]) -> np.ndarray:
    # Reorder the rows and columns of a square linear matrix by `perm`.
    matrix = np.asarray(matrix, dtype=float)
    perm = list(perm)
    return matrix[np.ix_(perm, perm)]


def _invert_perm(perm: tx.Sequence[int]) -> tx.List[int]:
    # The inverse permutation: `inverse[perm[i]] == i`. `perm[i]` is the
    # stored index at brainhops position `i`, so `inverse` maps a stored
    # index back to its brainhops position.
    inverse = [0] * len(perm)
    for position, stored in enumerate(perm):
        inverse[stored] = position
    return inverse


def _map_transform(
    transform: CoordinateTransformation, perm: tx.Sequence[int], ndim: int
) -> Transformation:
    # Map one 0.6rc0 coordinate transformation to the brainhops
    # transformation of the same kind, with its parameters reordered from
    # the stored axis order into the brainhops order.
    kind = getattr(transform, "type", None)
    if kind == "identity":
        return Identity()
    if kind == "scale" and isinstance(getattr(transform, "scale", None), list):
        return Scaling(scale=_permute_vector(transform.scale, perm))
    if kind == "translation" and isinstance(
        getattr(transform, "translation", None), list
    ):
        return Translation(
            translation=_permute_vector(transform.translation, perm)
        )
    if kind == "affine" and isinstance(
        getattr(transform, "affine", None), list
    ):
        matrix = np.asarray(transform.affine, dtype=float)
        if matrix.shape == (ndim + 1, ndim + 1):
            matrix = matrix[:ndim]
        return Affine(matrix=permute_affine(matrix, perm))
    if kind == "rotation" and isinstance(
        getattr(transform, "rotation", None), list
    ):
        return Rotation(matrix=_permute_linear(transform.rotation, perm))
    if kind == "mapAxis" and isinstance(
        getattr(transform, "mapAxis", None), list
    ):
        mapping = list(transform.mapAxis)
        if sorted(mapping) == list(range(ndim)):
            # Rewrite the axis map from the stored order into the brainhops
            # order on both its input and output sides. The output axis at
            # brainhops position `o` is stored axis `perm[o]`, and the input
            # axis it names maps back through the inverse permutation.
            inverse = _invert_perm(perm)
            permutation = [inverse[mapping[p]] for p in perm]
            return Permutation(permutation=permutation)
    if kind == "sequence":
        return Sequence(
            [
                _map_transform(inner, perm, ndim)
                for inner in transform.transformations
            ]
        )
    if kind in ("displacements", "coordinates"):
        raise OmeImageError(
            "This OME-Zarr image is placed by a "
            f"{kind!r} field transformation, which brainhops does not yet "
            "read from a group. The coordinate transformation around such a "
            "field must be affine, so that the field can be inverted and its "
            "vectors rotated."
        )
    raise OmeImageError(
        "This OME-Zarr image is placed by a "
        f"{kind!r} coordinate transformation, which brainhops cannot yet "
        "read as an image geometry."
    )


def level_transformation(
    multiscale: Multiscale,
    dataset: Dataset,
    perm: tx.Sequence[int],
    ndim: int,
    input: tx.Optional[CoordinateSystem] = None,
    output: tx.Optional[CoordinateSystem] = None,
) -> Transformation:
    """Return the voxel-to-world transformation of one level.

    Each OME coordinate transformation is mapped to the brainhops
    transformation of the same kind. The level's own transformation runs
    first, then the multiscale transformations that apply to every level. A
    level with more than one transformation becomes a `Sequence` in that
    application order.
    """
    mapped = []  # type: tx.List[Transformation]
    transforms = list(dataset.coordinateTransformations)
    if transforms:
        mapped.append(_map_transform(transforms[0], perm, ndim))
    common = getattr(multiscale, "coordinateTransformations", None)
    if isinstance(common, list):
        mapped.extend(_map_transform(one, perm, ndim) for one in common)

    if not mapped:
        return Identity(input=input, output=output)
    if len(mapped) == 1:
        return replace(mapped[0], input=input, output=output)
    return Sequence(mapped, input=input, output=output)


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


def axis_to_json(axis: Axis) -> tx.Dict[str, tx.Any]:
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
    scale: _Vector, translation: _Vector, path: str
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
    axes: tx.Sequence[Axis],
    levels: tx.Sequence[_Level],
    name: tx.Optional[str],
    version: str,
) -> OME:
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
    node: ZarrGroup,
    axes: tx.Sequence[Axis],
    levels: tx.Sequence[_Level],
    name: tx.Optional[str],
    version: str,
) -> None:
    """Write an image pyramid's OME metadata onto a group through abczarr."""
    node.ome = build_ome(axes, levels, name, version)
