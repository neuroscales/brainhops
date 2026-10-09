"""Mapping of coordinate transformations between OME-Zarr and brainhops.

Each OME coordinate transformation is read as a brainhops transformation:

| OME-Zarr                         | brainhops                             |
| -------------------------------- | ------------------------------------- |
| `identity`                       | [`Identity`][]                        |
| `scale`                          | [`Scaling`][]                         |
| `translation`                    | [`Translation`][]                     |
| `rotation`                       | [`Rotation`][]                        |
| `affine`                         | [`Affine`][]                          |
| `mapAxis`                        | [`Permutation`][] or [`Projection`][] |
| `displacements`, `coordinates`   | the field read from its node          |
| `sequence`                       | [`Sequence`][] of the mapped children |

On write, a per-axis scale and translation is written in that lean form, and
any other affine in full, so that rotations and shears are kept.

Both directions dispatch with a [`Function`][] to the converter registered for
the nearest supertype, and a tie between incomparable types goes to the
earliest registration. A new kind is supported by registering a converter.
"""

import itertools

import numpy as np
import typing_extensions as tx
from abczarr.ome.v0_6 import transformations as _ot
from bagof.dispatchers import Function, NoMethodError

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
    """A coordinate transformation that cannot be mapped.

    The error is raised for an OME kind that brainhops does not read, and for a
    brainhops transformation that OME-Zarr cannot express.
    """


# ----------------------------------------------------------------------
#   axis reordering
# ----------------------------------------------------------------------
# Parameters are stored in the OME axis order and read into the brainhops
# order, by permuting matrix rows and linear columns or vector entries.


def permute_vector(
    values: tx.Sequence[float], perm: tx.Sequence[int]
) -> tx.List[float]:
    """Reorder a per-axis vector by a permutation."""
    return [float(values[p]) for p in perm]


def permute_affine(matrix: tx.Any, perm: tx.Sequence[int]) -> np.ndarray:
    """Reorder the axes of a compact affine by a permutation.

    The rows (output axes), the linear columns (input axes) and the entries of
    the translation are reordered, and the translation column stays last.
    """
    matrix = np.asarray(matrix, dtype=float)
    perm = list(perm)
    linear = matrix[:, :-1][np.ix_(perm, perm)]
    translation = matrix[:, -1][perm]
    return np.concatenate([linear, translation[:, None]], axis=1)


def permute_linear(matrix: tx.Any, perm: tx.Sequence[int]) -> np.ndarray:
    """Reorder the rows and columns of a square matrix by a permutation."""
    matrix = np.asarray(matrix, dtype=float)
    perm = list(perm)
    return matrix[np.ix_(perm, perm)]


def _invert_perm(perm: tx.Sequence[int]) -> tx.List[int]:
    # perm[i] is the stored index at brainhops position i, so the inverse maps
    # a stored index to its brainhops position.
    inverse = [0] * len(perm)
    for position, stored in enumerate(perm):
        inverse[stored] = position
    return inverse


def _map_axis_transform(
    mapping: tx.Sequence[int], perm: tx.Sequence[int], ndim: int
) -> Transformation:
    # A bijective axis map is a permutation; a map that names a subset of the
    # input axes is a projection that drops the others.
    mapping = list(mapping)
    inverse = _invert_perm(perm)
    if sorted(mapping) == list(range(ndim)):
        # Output axis o (in brainhops order) is stored axis perm[o], and the
        # input axis that it names is mapped back by the inverse permutation.
        return Permutation(permutation=[inverse[mapping[p]] for p in perm])
    if mapping == sorted(mapping) and set(mapping) <= set(range(ndim)):
        # An increasing subset drops the omitted input axes; the dropped axes
        # are reported in brainhops order, and no axis is created.
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
_from_ome_fn: Function = Function("from_ome")
_from_order = itertools.count()


def _from_ome(ome_type: type) -> tx.Callable:
    # Dispatch on the OME transform only: the other arguments are matched by
    # `object`. Decreasing priorities make the earliest registration win a tie.
    def register(func: tx.Callable) -> tx.Callable:
        _from_ome_fn.register(
            (ome_type, object, object, object),
            priority=-next(_from_order),
        )(func)
        return func

    return register


def from_ome(
    transform: tx.Any,
    perm: tx.Sequence[int],
    ndim: int,
    read_field: tx.Optional[tx.Callable] = None,
) -> Transformation:
    """Map an OME coordinate transformation to a brainhops transformation.

    Parameters
    ----------
    transform
        The OME coordinate transformation.
    perm
        The permutation from the stored OME axis order to the brainhops order.
    ndim
        The number of spatial dimensions.
    read_field
        A callable that reads the displacement or coordinate field from the
        node that a field transformation names. It is only needed for field
        transformations.

    Raises
    ------
    OmeMappingError
        If brainhops does not read this kind of transformation.
    """
    try:
        return _from_ome_fn(transform, perm, ndim, read_field)
    except NoMethodError:
        name = getattr(transform, "type", type(transform).__name__)
        raise OmeMappingError(
            "This OME-Zarr image is placed by a "
            f"{name!r} coordinate transformation, which brainhops cannot yet "
            "read as an image geometry."
        ) from None


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
# Writers return the JSON of one OME coordinate transformation, with its
# parameters reordered by storage_perm into the stored OME order.

# OME types that only the richer OME-NGFF versions carry: scale and translation
# exist in every version, but the others do not.
_RICH_TYPES = frozenset(
    {"rotation", "affine", "mapAxis", "displacements", "coordinates"}
)

_to_ome_fn: Function = Function("to_ome")
_to_order = itertools.count()


def _to_ome_for(brainhops_type: type) -> tx.Callable:
    # Dispatch on the transformation only: the other arguments are matched by
    # `object`. Decreasing priorities make the earliest registration win a tie.
    def register(func: tx.Callable) -> tx.Callable:
        _to_ome_fn.register(
            (brainhops_type, object, object),
            priority=-next(_to_order),
        )(func)
        return func

    return register


def to_ome(
    transform: Transformation, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    """Map a brainhops transformation to one OME coordinate transformation.

    A per-axis scale and translation is written in its lean form, any other
    affine is written in full, and a sequence is written as a sequence of its
    mapped children. The parameters are reordered by `storage_perm`, from the
    brainhops order to the stored OME order.

    Raises
    ------
    OmeMappingError
        If the transformation does not reduce to an affine, as a field does
        not.
    """
    # A type with no registered ancestor falls back to the affine writer.
    try:
        return _to_ome_fn(transform, storage_perm, ndim)
    except NoMethodError:
        return _affine_to_ome(transform, storage_perm, ndim)


def scale_translation_from_affine(
    matrix: tx.Any, ndim: int
) -> tx.Optional[tx.Tuple[np.ndarray, np.ndarray]]:
    """Return the per-axis scale and translation of a diagonal affine.

    `matrix` is a compact affine, or `None` for the identity, which gives a
    scale of ones and a translation of zeros. `None` is returned when the
    linear block is not square or has an off-diagonal term (beyond a tolerance
    of 1e-8), such as a rotation or a shear, so that the affine must be written
    in full.
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
    """Return whether an OME transformation needs a richer OME-NGFF version.

    Scales, translations and sequences of them exist in every version.
    Rotations, affines, axis maps, displacements and coordinates need version
    0.6 or later. The entry and its children are checked recursively.
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
    # Leanest form: a scale alone, or a sequence of a scale and a translation.
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
    # Diagonal affines become a scale and translation; others stay affines.
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


# Affine is registered explicitly, like Linear, so that dispatch selects the
# affine writer exactly instead of through the fallback.
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
