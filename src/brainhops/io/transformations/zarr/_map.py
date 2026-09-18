"""Map coordinate transformations between OME-Zarr and brainhops.

The same knowledge places an image in both directions. Reading turns an
OME-Zarr coordinate transformation into the brainhops transformation of the
same kind. Writing turns a brainhops transformation back into an OME-Zarr
coordinate transformation. Both directions live here, so the pairing of an
OME kind with a brainhops kind is written once.

Each direction is dispatch-driven rather than a chain of type tests. A
converter is registered for one type, and the mapping selects the converter
whose registered type is closest to the value's type in the class
hierarchy. The closeness is measured by [`distance`][], the same measure the
brainhops transformation converters use, so a new kind is added by
registering a converter rather than by extending a conditional.

A scale maps to a [`Scaling`][brainhops.datamodel.transformations.Scaling],
a translation to a
[`Translation`][brainhops.datamodel.transformations.Translation], a
rotation to a [`Rotation`][brainhops.datamodel.transformations.Rotation], an
affine to an [`Affine`][brainhops.datamodel.transformations.Affine], and a
sequence to a [`Sequence`][brainhops.datamodel.transformations.Sequence] of
the mapped children. The reverse mapping is the mirror of this. A brainhops
transformation that reduces to a per-axis scale and translation is written
in that leanest form, and any other affine is written as a full affine, so
a rotation or a shear is kept rather than refused.
"""

# dependencies
import numpy as np
import typing_extensions as tx
from abczarr.ome.v0_6 import transformations as _ot

# internals
from brainhops.datamodel._transformations.registries import distance
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    Linear,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    Transformation,
    Translation,
)


class OmeMappingError(ValueError):
    """Raised when a coordinate transformation cannot be mapped.

    An OME-Zarr coordinate transformation of a kind brainhops does not
    read, or a brainhops transformation that OME-Zarr cannot express, is
    refused with this error.
    """


# ----------------------------------------------------------------------
#   axis reordering
# ----------------------------------------------------------------------
#
# A transformation's parameters are stored in the OME axis order and read
# into the brainhops axis order, and back again when writing. The reorder
# is a permutation of the axes, applied to the rows and linear columns of a
# matrix, or to the entries of a per-axis vector.


def permute_vector(
    values: tx.Sequence[float], perm: tx.Sequence[int]
) -> tx.List[float]:
    """Return a per-axis vector reordered by `perm`."""
    return [float(values[p]) for p in perm]


def permute_affine(matrix: tx.Any, perm: tx.Sequence[int]) -> np.ndarray:
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


def permute_linear(matrix: tx.Any, perm: tx.Sequence[int]) -> np.ndarray:
    """Reorder the rows and columns of a square linear matrix by `perm`."""
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
    raise OmeMappingError(
        "This OME-Zarr image is placed by a mapAxis transformation that both "
        "drops and reorders axes, which brainhops does not read as a single "
        "transformation."
    )


# ----------------------------------------------------------------------
#   read: OME-Zarr coordinate transformation -> brainhops transformation
# ----------------------------------------------------------------------
_FROM_OME = {}  # type: tx.Dict[type, tx.Callable]


def _from_ome(ome_type: type) -> tx.Callable:
    # Register a reader for one OME-Zarr coordinate transformation type.
    def register(func: tx.Callable) -> tx.Callable:
        _FROM_OME[ome_type] = func
        return func

    return register


def from_ome(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable] = None,
) -> Transformation:
    """Map an OME-Zarr coordinate transformation to a brainhops one.

    `perm` reorders each transformation's parameters from the stored OME
    axis order into the brainhops order. `read_field` reads a displacement
    or coordinate field from the node it names, and is required only when
    the transformation is a field. A transformation of a kind brainhops
    does not read is refused with an
    [`OmeMappingError`][brainhops.io.transformations.zarr._map.OmeMappingError].
    """
    kind = type(transform)
    best_distance, best = float("inf"), None
    for registered, func in _FROM_OME.items():
        dist = distance(kind, registered)
        if dist < best_distance:
            best_distance, best = dist, func
    if best is None or best_distance == float("inf"):
        name = getattr(transform, "type", kind.__name__)
        raise OmeMappingError(
            "This OME-Zarr image is placed by a "
            f"{name!r} coordinate transformation, which brainhops cannot yet "
            "read as an image geometry."
        )
    return best(transform, perm, ndim, read_field)


@_from_ome(_ot.Identity)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    return Identity()


@_from_ome(_ot.Scale)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    return Scaling(scale=permute_vector(transform.scale, perm))


@_from_ome(_ot.Translation)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    return Translation(translation=permute_vector(transform.translation, perm))


@_from_ome(_ot.Affine)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    matrix = np.asarray(transform.affine, dtype=float)
    if matrix.shape == (ndim + 1, ndim + 1):
        matrix = matrix[:ndim]
    return Affine(matrix=permute_affine(matrix, perm))


@_from_ome(_ot.Rotation)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    return Rotation(matrix=permute_linear(transform.rotation, perm))


@_from_ome(_ot.MapAxis)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    return _map_axis_transform(transform.mapAxis, perm, ndim)


@_from_ome(_ot.Displacements)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    return _read_field(transform, "displacements", read_field)


@_from_ome(_ot.Coordinates)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    return _read_field(transform, "coordinates", read_field)


@_from_ome(_ot.Sequence)
def _(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable],
) -> Transformation:
    children = [
        from_ome(inner, perm, ndim, read_field)
        for inner in transform.transformations
    ]
    return Sequence(children)


def _read_field(
    transform: tx.Any, kind: str, read_field: tx.Optional[tx.Callable]
) -> Transformation:
    if read_field is None:
        raise OmeMappingError(
            "A field transformation can only be read from a group."
        )
    return read_field(transform, kind)


# ----------------------------------------------------------------------
#   write: brainhops transformation -> OME-Zarr coordinate transformation
# ----------------------------------------------------------------------
#
# The result of the write mapping is the JSON of one OME coordinate
# transformation, ready to be placed in a dataset's
# `coordinateTransformations`. The parameters are reordered from the
# brainhops axis order into the stored OME order by `storage_perm`.

#: The OME transformation types that only a richer OME-NGFF version carries.
#: A per-axis scale and a translation are expressible in every version; a
#: rotation, an affine, or an axis map is not.
_RICH_TYPES = frozenset(
    {"rotation", "affine", "mapAxis", "displacements", "coordinates"}
)

_TO_OME = {}  # type: tx.Dict[type, tx.Callable]


def _to_ome_for(brainhops_type: type) -> tx.Callable:
    # Register a writer for one brainhops transformation type.
    def register(func: tx.Callable) -> tx.Callable:
        _TO_OME[brainhops_type] = func
        return func

    return register


def to_ome(
    transform: Transformation, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    """Map a brainhops transformation to an OME-Zarr coordinate transformation.

    The result is the JSON of one OME coordinate transformation.
    `storage_perm` reorders each parameter from the brainhops axis order
    into the stored OME order. A transformation that reduces to a per-axis
    scale and translation is written in that leanest form. Any other affine
    is written as a full affine, and a sequence is written as a sequence of
    the mapped children, so a rotation, a shear, or a composed placement is
    kept.

    A transformation that does not reduce to an affine, such as a field, is
    refused with an
    [`OmeMappingError`][brainhops.io.transformations.zarr._map.OmeMappingError].
    """
    kind = type(transform)
    best_distance, best = float("inf"), None
    for registered, func in _TO_OME.items():
        dist = distance(kind, registered)
        if dist < best_distance:
            best_distance, best = dist, func
    if best is not None and best_distance < float("inf"):
        return best(transform, storage_perm, ndim)
    return _affine_to_ome(transform, storage_perm, ndim)


def scale_translation_from_affine(
    matrix: tx.Any, ndim: int
) -> tx.Optional[tx.Tuple[np.ndarray, np.ndarray]]:
    """Return the per-axis scale and translation of a diagonal affine.

    `matrix` is the compact ``(n, n + 1)`` affine matrix, or `None` for the
    identity. The result is a pair of vectors of length `ndim`: the scale
    read from the diagonal of the linear block, and the translation read
    from the last column. The identity decomposes to a scale of ones and a
    translation of zeros.

    The result is `None` when the matrix has an off-diagonal term, such as a
    rotation or a shear, since such an affine is not a per-axis scale and
    translation and must instead be written in full.
    """
    if matrix is None:
        return np.ones(ndim), np.zeros(ndim)
    matrix = np.asarray(matrix, dtype=float)
    linear = matrix[:, :-1]
    rows, cols = linear.shape[0], linear.shape[1]
    if rows != cols:
        return None
    off_diagonal = linear * (1 - np.eye(rows))
    if bool((np.abs(off_diagonal) > 1e-8).any()):
        return None
    return np.diag(linear), matrix[:, -1]


def needs_rich_version(entry: tx.Dict[str, tx.Any]) -> bool:
    """Whether an OME coordinate transformation needs a richer version.

    A per-axis scale and translation, and a sequence of them, are
    expressible in every OME-NGFF version. A rotation, an affine, or an axis
    map requires OME-NGFF 0.6 or later. This inspects the transformation
    and its children and reports whether the richer version is required.
    """
    if entry.get("type") in _RICH_TYPES:
        return True
    return any(
        needs_rich_version(child) for child in entry.get("transformations", [])
    )


def _matrix_to_list(matrix: tx.Any) -> tx.List[tx.List[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix)]


def _scale_translation_entry(
    scale: tx.Any, translation: tx.Any
) -> tx.Dict[str, tx.Any]:
    # The leanest OME encoding of a per-axis scale and translation: a scale
    # alone when the translation is zero, and a sequence of the scale and
    # the translation otherwise.
    scale = np.asarray(scale, dtype=float)
    translation = np.asarray(translation, dtype=float)
    scale_entry = {
        "type": "scale",
        "scale": [float(value) for value in scale],
    }  # type: tx.Dict[str, tx.Any]
    if not bool((translation != 0).any()):
        return scale_entry
    translation_entry = {
        "type": "translation",
        "translation": [float(value) for value in translation],
    }
    return {
        "type": "sequence",
        "transformations": [scale_entry, translation_entry],
    }


def _affine_to_ome(
    transform: Transformation, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    # The default writer for a transformation that reduces to an affine. A
    # diagonal affine is written as a scale and a translation; any other
    # affine is written in full.
    affine = transform.to(Affine, error=None)
    if affine is None:
        raise OmeMappingError(
            "This image is placed by a transformation that is not an affine, "
            "so it cannot be written as OME-Zarr multiscale metadata."
        )
    matrix = permute_affine(affine.matrix, storage_perm)
    decomposed = scale_translation_from_affine(matrix, ndim)
    if decomposed is None:
        return {"type": "affine", "affine": _matrix_to_list(matrix)}
    scale, translation = decomposed
    return _scale_translation_entry(scale, translation)


@_to_ome_for(Identity)
def _(
    transform: tx.Any, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    scale, translation = scale_translation_from_affine(None, ndim)
    return _scale_translation_entry(scale, translation)


@_to_ome_for(Scaling)
def _(
    transform: tx.Any, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    if transform.scale is None:
        scale, translation = scale_translation_from_affine(None, ndim)
        return _scale_translation_entry(scale, translation)
    return {
        "type": "scale",
        "scale": permute_vector(transform.scale, storage_perm),
    }


@_to_ome_for(Translation)
def _(
    transform: tx.Any, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    if transform.translation is None:
        scale, translation = scale_translation_from_affine(None, ndim)
        return _scale_translation_entry(scale, translation)
    return {
        "type": "translation",
        "translation": permute_vector(transform.translation, storage_perm),
    }


@_to_ome_for(Rotation)
def _(
    transform: tx.Any, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    affine = transform.to(Affine, error=None)
    if affine is None:
        raise OmeMappingError(
            "This rotation has no matrix, so it cannot be written as OME-Zarr "
            "metadata."
        )
    linear = permute_linear(
        np.asarray(affine.matrix, dtype=float)[:, :-1], storage_perm
    )
    return {"type": "rotation", "rotation": _matrix_to_list(linear)}


@_to_ome_for(Sequence)
def _(
    transform: tx.Any, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    children = transform.transformations or []
    return {
        "type": "sequence",
        "transformations": [
            to_ome(child, storage_perm, ndim) for child in children
        ],
    }


# `Affine` and `Linear` both go through the affine writer, which decides
# between the lean and the full form. Registering `Affine` keeps the
# dispatch exact for an affine rather than routing it through the fallback.
@_to_ome_for(Affine)
def _(
    transform: tx.Any, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    return _affine_to_ome(transform, storage_perm, ndim)


@_to_ome_for(Linear)
def _(
    transform: tx.Any, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    return _affine_to_ome(transform, storage_perm, ndim)
