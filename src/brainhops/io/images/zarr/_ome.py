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
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Identity,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    Transformation,
    Translation,
)
from brainhops.io.base.parsers import ParserContentError, WriterError
from brainhops.io.images.zarr import _axisorder
from brainhops.io.transformations.zarr._axes import _to_axis

#: The brainhops spline order each OME interpolation name maps to.
_INTERPOLATION_ORDER = {"nearest": 0, "linear": 1, "bspline-cubic": 3}

#: The brainhops transformations that reduce to an affine. A field must be
#: surrounded only by these, so it can be inverted and its vectors rotated.
_AFFINE_ISH = (Affine, Rotation, Scaling, Translation, Identity)

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


def _map_axis_transform(
    mapping: tx.Sequence[int], perm: tx.Sequence[int], ndim: int
) -> Transformation:
    # A bijective axis map is a permutation; one that names a subset of the
    # input axes is a projection that drops the rest.
    mapping = list(mapping)
    inverse = _invert_perm(perm)
    if sorted(mapping) == list(range(ndim)):
        # Rewrite the axis map from the stored order into the brainhops order
        # on both its input and output sides. The output axis at brainhops
        # position `o` is stored axis `perm[o]`, and the input axis it names
        # maps back through the inverse permutation.
        return Permutation(permutation=[inverse[mapping[p]] for p in perm])
    if mapping == sorted(mapping) and set(mapping) <= set(range(ndim)):
        # A strictly increasing subset drops the input axes it omits, keeping
        # the rest in order. The dropped axes are reported in the brainhops
        # order, and no axis is created.
        dropped_stored = [i for i in range(ndim) if i not in mapping]
        dropped = sorted(inverse[i] for i in dropped_stored)
        return Projection(dropped=dropped, created=[])
    raise OmeImageError(
        "This OME-Zarr image is placed by a mapAxis transformation that both "
        "drops and reorders axes, which brainhops does not read as a single "
        "transformation."
    )


def _follow_path(node: ZarrGroup, path: str) -> tx.Any:
    # Open the node the field transformation points at, following a path that
    # may descend through subgroups (``"coordinateTransformations/dfield"``).
    current = node  # type: tx.Any
    for segment in path.strip("/").split("/"):
        current = current[segment]
    return current


def _read_field(
    transform: CoordinateTransformation,
    kind: str,
    node: ZarrGroup,
    store_axes: tx.Sequence[Axis],
    ndim: int,
) -> Transformation:
    # Build a brainhops displacement or coordinate field from the array the
    # transformation points at. The field array's own axis names, read from
    # the node's ``dimension_names``, say which axis holds the vector
    # components and how the spatial axes are ordered.
    path = getattr(transform, "path", None)
    if not isinstance(path, str):
        raise OmeImageError(
            f"This OME-Zarr {kind} field names no array, so its field cannot "
            "be read."
        )
    field_node = _follow_path(node, path)

    names = getattr(field_node.metadata, "dimension_names", None)
    if not names or None in names or len(names) != field_node.ndim:
        raise OmeImageError(
            f"The {kind} field at {path!r} does not name its axes (it has no "
            "dimension_names), so brainhops cannot tell which axis holds the "
            "vector components."
        )

    # The field array shares the image's spatial axes by name; the remaining
    # axis is the component axis. Reading the names from the node avoids any
    # assumption about which axis leads.
    image_names = [axis.name for axis in store_axes]
    spatial_dims = [i for i, name in enumerate(names) if name in image_names]
    component_dims = [i for i in range(len(names)) if i not in spatial_dims]
    if len(component_dims) != 1 or len(spatial_dims) != len(store_axes):
        raise OmeImageError(
            f"The {kind} field at {path!r} has axes {tuple(names)}, which do "
            f"not match the image axes {tuple(image_names)} together with a "
            "single component axis."
        )

    # Lay the field out as (*spatial, component): the spatial axes in the
    # brainhops order, then the component axis. Only axes are moved; the
    # component values are not reordered, since rotating the vectors is the
    # job of the affine that surrounds the field.
    canonical_axes = _axisorder.permute(
        store_axes, _axisorder.to_canonical(store_axes)
    )
    by_name = {names[i]: i for i in spatial_dims}
    target = [by_name[axis.name] for axis in canonical_axes] + component_dims

    backend = get_array_backend()
    data = backend.transpose(backend.asarray(field_node[...]), target)
    order = _INTERPOLATION_ORDER.get(
        getattr(transform, "interpolation", None), 1
    )
    field_cls = (
        DisplacementField if kind == "displacements" else CoordinatesField
    )
    return field_cls(field=data, order=order)


def _map_transform(
    transform: CoordinateTransformation,
    perm: tx.Sequence[int],
    ndim: int,
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
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
        return _map_axis_transform(transform.mapAxis, perm, ndim)
    if kind in ("displacements", "coordinates"):
        if node is None or store_axes is None:
            raise OmeImageError(
                "A field transformation can only be read from a group."
            )
        return _read_field(transform, kind, node, store_axes, ndim)
    if kind == "sequence":
        children = [
            _map_transform(inner, perm, ndim, node, store_axes)
            for inner in transform.transformations
        ]
        _gate_field_surround(children)
        return Sequence(children)
    raise OmeImageError(
        "This OME-Zarr image is placed by a "
        f"{kind!r} coordinate transformation, which brainhops cannot yet "
        "read as an image geometry."
    )


def _is_field(transformation: Transformation) -> bool:
    return isinstance(transformation, (DisplacementField, CoordinatesField))


def _is_affine_ish(transformation: Transformation) -> bool:
    if isinstance(transformation, Sequence):
        return all(
            _is_affine_ish(one)
            for one in (transformation.transformations or [])
        )
    return isinstance(transformation, _AFFINE_ISH)


def _gate_field_surround(mapped: tx.Sequence[Transformation]) -> None:
    # A field must be surrounded only by affine transformations, so that the
    # field can be inverted and its vectors rotated. More than one field, or
    # a field beside a non-affine transformation, is refused.
    fields = [one for one in mapped if _is_field(one)]
    if not fields:
        return
    if len(fields) > 1:
        raise OmeImageError(
            "This OME-Zarr image composes more than one field, which "
            "brainhops does not read. A field must be surrounded only by "
            "affine transformations."
        )
    for one in mapped:
        if not _is_field(one) and not _is_affine_ish(one):
            raise OmeImageError(
                "This OME-Zarr image surrounds a field with a "
                f"{type(one).__name__} transformation. A field must be "
                "surrounded only by affine transformations, so that it can "
                "be inverted and its vectors rotated."
            )


def level_transformation(
    multiscale: Multiscale,
    dataset: Dataset,
    perm: tx.Sequence[int],
    ndim: int,
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
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
        mapped.append(
            _map_transform(transforms[0], perm, ndim, node, store_axes)
        )
    common = getattr(multiscale, "coordinateTransformations", None)
    if isinstance(common, list):
        mapped.extend(
            _map_transform(one, perm, ndim, node, store_axes) for one in common
        )
    _gate_field_surround(mapped)

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
