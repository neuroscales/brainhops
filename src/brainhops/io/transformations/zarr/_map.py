"""Mapping of coordinate transformations between OME-Zarr and brainhops.

OME-Zarr places an image, and relates coordinate systems to each other,
with coordinate transformations. Their metadata is parsed by abczarr and
normalized to OME-NGFF 0.6, the version that adopted RFC-5, before this
module maps each transformation to a brainhops transformation:

| OME-Zarr                       | brainhops                             |
| ------------------------------ | ------------------------------------- |
| `identity`                     | [`Identity`][]                        |
| `scale`                        | [`Scaling`][]                         |
| `translation`                  | [`Translation`][]                     |
| `rotation`                     | [`Rotation`][]                        |
| `affine`                       | [`Affine`][]                          |
| `mapAxis`                      | [`Permutation`][]                     |
| `projectAxis`                  | [`Projection`][]                      |
| `displacements`, `coordinates` | the field read from the node it names |
| `sequence`                     | [`Sequence`][] of the mapped children |
| `bijection`                    | [`Bijection`][]                       |
| `byDimension`                  | [`Affine`][] or subspace parts        |

A `byDimension` whose parts are all affine is read as one [`Affine`][].
Otherwise each part becomes a [`SubspaceTransformation`][] of a
[`Sequence`][], which is exact only when each part writes the axes that
it reads, so any other such `byDimension` is refused.

The draft type `inverseOf` is not read. RFC-5 removed it before version
0.6 was released (ome/ngff `rfc/5/responses/2/index.md`, "Remove
`inverseOf` transformation", at d844a6a), and abczarr refuses to
normalize a draft that uses it, so such metadata is refused with the
error of abczarr.

Both formats read a transformation in the same direction. OME-NGFF 0.6
defines a transformation as a function from points of its `input`
coordinate system to points of its `output` coordinate system (ngff-spec
branch `0.6` at 89da1da, `index.md`, section "coordinateTransformations
metadata"), and a brainhops transformation maps points of its `input`
system to points of its `output` system.

The parameters of an OME transformation follow the order of the axes of
its coordinate systems. Entry `i` of a scale or a translation belongs to
axis `i`, and a matrix acts on a column vector whose first entry is the
first axis (same file, sections "Additional details" and "Matrix
transformations"). For an image, this is the order of the array
dimensions, such as `(t, c, z, y, x)`. brainhops sorts the axes of each
coordinate system into its own order, as described in
[`brainhops.io.images.zarr`][]. The mapping therefore needs to know, for
both the input and the output system of a transformation, the stored
position of each brainhops axis. This list is called an order below. The
columns and the input entries of a parameter are reordered by the order
of the input system, and its rows and output entries by the order of the
output system. When the two systems sort their axes differently, as two
systems with axes `(z, y, x)` and `(x, y, z)` do, a type that cannot
express the reordering, such as a scale, is followed by a
[`Permutation`][], so that the mapping stays exact.

On write, an affine that only scales and translates each axis is written
as a `scale` and a `translation`. Any other affine is written in full as an
`affine`, so that rotations and shears are kept. The dataset of a level
only holds a scale and a translation in version 0.6, which
[`to_dataset_ome`][] writes.

In both directions, a [`Function`][] selects the converter that is
registered for the nearest supertype of the transformation. When two
incomparable types tie, the converter that was registered first is used. A
new kind of transformation is supported by registering a converter for it.
"""

import numpy as np
import typing_extensions as tx
from abczarr.ome.v0_6 import transformations as _ot
from bagof.dispatchers import Function, NoMethodError

from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    Identity,
    Linear,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
)
from brainhops.errors import ConversionError
from brainhops.io.transformations.zarr import _node


class OmeMappingError(ValueError):
    """A coordinate transformation that cannot be mapped.

    The error is raised for an OME kind that brainhops does not read, and for a
    brainhops transformation that OME-Zarr cannot express.
    """


class Reading(tx.NamedTuple):
    """What the mapping of a transformation may need to read.

    The parameters of a transformation may be stored in an array of the
    group instead of inline, and a field transformation names the node of
    its field. A transformation inside a sequence may also name the
    coordinate system that it maps to.
    """

    node: tx.Optional[tx.Any] = None
    """The group against which the paths of the metadata are resolved."""

    read_field: tx.Optional[tx.Callable] = None
    """The callable that reads a displacement or coordinate field.

    It is called as `read_field(transform, kind, source, target)`, where
    `kind` is `"displacements"` or `"coordinates"` and `source` and
    `target` are the orders of the input and output systems.
    """

    orders: tx.Mapping[str, tx.Sequence[int]] = {}
    """The order of each named coordinate system of the metadata."""


# ----------------------------------------------------------------------
#   axis reordering
# ----------------------------------------------------------------------
# The writer stores parameters from the brainhops axis order into the
# stored OME order by permuting the rows and the linear columns of a
# matrix, or the entries of a vector.


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


# ----------------------------------------------------------------------
#   read: OME-Zarr coordinate transformation -> brainhops transformation
# ----------------------------------------------------------------------
_from_ome_fn: Function = Function("from_ome")
_from_ome = _from_ome_fn.register


def from_ome(
    transform: tx.Any,
    source: tx.Sequence[int],
    target: tx.Sequence[int],
    reading: tx.Optional[Reading] = None,
) -> Transformation:
    """Map an OME coordinate transformation to a brainhops transformation.

    The orders say where each brainhops axis is stored. Element `k` of
    `source` is the position, among the axes of the input system as OME
    lists them, of the axis that brainhops places at position `k`, as
    [`to_canonical`][brainhops.io.images.zarr._axisorder.to_canonical]
    returns it. `target` says the same for the output system. Both are
    permutations of all the axes of their system.

    Parameters
    ----------
    transform : Any
        The OME coordinate transformation, normalized to version 0.6.
    source : sequence of int
        The order of the input coordinate system.
    target : sequence of int
        The order of the output coordinate system.
    reading : Reading, optional
        The group, the field reader and the named orders that the
        transformation may need. Without it, nothing can be read from a
        group.

    Returns
    -------
    Transformation
        The brainhops transformation, without input and output systems.

    Raises
    ------
    OmeMappingError
        If brainhops does not read this kind of transformation, or if its
        parameters do not fit its coordinate systems.
    """
    if reading is None:
        reading = Reading()
    try:
        return _from_ome_fn(transform, list(source), list(target), reading)
    except NoMethodError:
        name = getattr(transform, "type", type(transform).__name__)
        raise OmeMappingError(
            "This OME-Zarr image is placed by a "
            f"{name!r} coordinate transformation, which brainhops cannot yet "
            "read as an image geometry."
        ) from None


def in_stored_order(
    xform: Transformation,
    source: tx.Sequence[int],
    target: tx.Sequence[int],
) -> Transformation:
    """Place a transformation that acts on the stored axis orders.

    Some transformations, such as a `projectAxis` or an OME-Zarr field,
    are easiest to express in the order in which OME lists the axes of
    their systems. The result applies the permutation from the brainhops
    order of the input system to its stored order, then `xform`, then the
    permutation from the stored order of the output system to its
    brainhops order. A permutation that would change nothing is left out.

    Parameters
    ----------
    xform : Transformation
        The transformation, from the stored order of the input system to
        the stored order of the output system.
    source : sequence of int
        The order of the input system, as in [`from_ome`][].
    target : sequence of int
        The order of the output system, as in [`from_ome`][].

    Returns
    -------
    Transformation
        `xform` alone, or a [`Sequence`][] that surrounds it with
        permutations.
    """
    position = _positions(source)
    parts = []  # type: tx.List[Transformation]
    if not _is_identity_order(source):
        to_stored = [position[stored] for stored in range(len(source))]
        parts.append(Permutation(permutation=to_stored))
    parts.append(xform)
    if not _is_identity_order(target):
        parts.append(Permutation(permutation=list(target)))
    if len(parts) == 1:
        return parts[0]
    return Sequence(parts)


@_from_ome
def _(
    transform: _ot.Identity,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    _check_same_dimension(transform, source, target)
    return _relabel(Identity(), source, target)


@_from_ome
def _(
    transform: _ot.Scale,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    _check_same_dimension(transform, source, target)
    values = _parameter(transform, "scale", reading, ndim=1)
    _check_entries(transform, values, source)
    return _relabel(
        Scaling(scale=permute_vector(values, source)), source, target
    )


@_from_ome
def _(
    transform: _ot.Translation,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    _check_same_dimension(transform, source, target)
    values = _parameter(transform, "translation", reading, ndim=1)
    _check_entries(transform, values, source)
    return _relabel(
        Translation(translation=permute_vector(values, source)),
        source,
        target,
    )


@_from_ome
def _(
    transform: _ot.Affine,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    # The matrix has one row per output axis and one column per input axis,
    # followed by the translation. A full homogeneous matrix, whose last row
    # is (0, ..., 0, 1), is also accepted.
    matrix = _affine_rows(
        _parameter(transform, "affine", reading, ndim=2), len(target)
    )
    width = len(source)
    if matrix.shape[1] != width + 1:
        raise OmeMappingError(
            f"This OME affine has {matrix.shape[1]} columns, which does not "
            f"fit an input coordinate system of {width} axes."
        )
    columns = list(source) + [width]
    return Affine(matrix=matrix[np.ix_(list(target), columns)])


@_from_ome
def _(
    transform: _ot.Rotation,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    _check_same_dimension(transform, source, target)
    matrix = np.asarray(
        _parameter(transform, "rotation", reading, ndim=2), dtype=float
    )
    width = len(source)
    if matrix.shape != (width, width):
        raise OmeMappingError(
            f"This OME rotation has the shape {matrix.shape}, which does "
            f"not fit a coordinate system of {width} axes."
        )
    return _relabel(
        Rotation(matrix=permute_linear(matrix, source)), source, target
    )


@_from_ome
def _(
    transform: _ot.MapAxis,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    # Entry `i` of `mapAxis` is the input axis that output axis `i` carries,
    # which is also what a brainhops permutation stores.
    mapping = [int(index) for index in transform.mapAxis]
    if sorted(mapping) == list(range(len(mapping))):
        _check_same_dimension(transform, source, target)
        if len(mapping) != len(source):
            raise OmeMappingError(
                f"This OME mapAxis {mapping} does not fit coordinate "
                f"systems of {len(source)} axes."
            )
        position = _positions(source)
        try:
            permutation = [position[mapping[stored]] for stored in target]
        except (IndexError, KeyError):
            raise OmeMappingError(
                f"This OME mapAxis {mapping} does not fit coordinate "
                f"systems of {len(source)} axes."
            ) from None
        return Permutation(permutation=permutation)
    # A subset of the input axes was never a valid mapAxis, so it is
    # refused, and `projectAxis` is the spelling of a drop
    # (spec/design/compute-api-and-open-systems.md, section A.8).
    raise OmeMappingError(
        f"This OME mapAxis {mapping} is not a permutation of the input "
        "axes, which OME-NGFF 0.6 requires. A transformation that drops "
        "axes is spelled projectAxis."
    )


@_from_ome
def _(
    transform: _ot.ProjectAxis,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    # Both lists hold stored positions. The projection is built in the
    # stored order and surrounded by the permutations to and from it.
    dropped = [int(index) for index in transform.droppedInputs or []]
    created = [int(index) for index in transform.createdOutputs or []]
    _check_indices(transform, "droppedInputs", dropped, len(source))
    _check_indices(transform, "createdOutputs", created, len(target))
    if len(target) != len(source) - len(dropped) + len(created):
        raise OmeMappingError(
            f"This OME projectAxis drops {len(dropped)} and creates "
            f"{len(created)} axes, which does not map {len(source)} input "
            f"axes onto {len(target)} output axes."
        )
    return in_stored_order(
        Projection(dropped=dropped, created=created), source, target
    )


@_from_ome
def _(
    transform: _ot.Displacements,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    _check_same_dimension(transform, source, target)
    return _read_field(transform, "displacements", source, target, reading)


@_from_ome
def _(
    transform: _ot.Coordinates,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    return _read_field(transform, "coordinates", source, target, reading)


@_from_ome
def _(
    transform: _ot.Sequence,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    # The spaces between two children are unnamed unless a child names its
    # output system. Any order is exact for such a space, as long as both
    # children that meet there use the same one, so the order of the input
    # system is kept while the number of axes stays the same.
    children = list(transform.transformations)
    mapped = []  # type: tx.List[Transformation]
    current = list(source)
    for position, child in enumerate(children):
        if position == len(children) - 1:
            following = list(target)
        else:
            following = _intermediate_order(child, current, reading)
        mapped.append(from_ome(child, current, following, reading))
        current = following
    return Sequence(mapped)


@_from_ome
def _(
    transform: _ot.Bijection,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    _check_same_dimension(transform, source, target)
    forward = from_ome(transform.forward, source, target, reading)
    backward = from_ome(transform.inverse, target, source, reading)
    return Bijection(forward=forward, backward=backward)


@_from_ome
def _(
    transform: _ot.ByDimension,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    # Each part reads the input axes and writes the output axes that it
    # lists, in its own order, and every output axis is written once.
    parts = []  # type: tx.List[tx.Tuple[tx.List[int], tx.List[int], tx.Any]]
    written = []  # type: tx.List[int]
    for item in transform.transformations:
        inputs = [int(i) for i in item.inputAxes or []]
        outputs = [int(i) for i in item.outputAxes or []]
        inner = item.transformation
        if not isinstance(inner, _ot.CoordinateTransformation) or not (
            inputs and outputs
        ):
            raise OmeMappingError(
                "A part of this OME byDimension does not name its "
                "transformation, its input axes or its output axes."
            )
        parts.append((inputs, outputs, inner))
        written.extend(outputs)
    width_in = len(source)
    width_out = len(target)
    if sorted(written) != list(range(width_out)):
        raise OmeMappingError(
            "This OME byDimension does not write every output axis exactly "
            "once, as OME-NGFF 0.6 requires."
        )
    if any(i >= width_in for inputs, _, _ in parts for i in inputs):
        raise OmeMappingError(
            "This OME byDimension reads an input axis that its input "
            "coordinate system does not have."
        )
    mapped = [
        (
            inputs,
            outputs,
            from_ome(
                child,
                list(range(len(inputs))),
                list(range(len(outputs))),
                reading,
            ),
        )
        for inputs, outputs, child in parts
    ]
    affine = _assembled_affine(mapped, source, target, width_in, width_out)
    if affine is not None:
        return affine
    return _subspaces(mapped, source, target)


def _read_field(
    transform: tx.Any,
    kind: str,
    source: tx.List[int],
    target: tx.List[int],
    reading: Reading,
) -> Transformation:
    if reading.read_field is None:
        raise OmeMappingError(
            "A field transformation can only be read from a group."
        )
    return reading.read_field(transform, kind, source, target)


# ----------------------------------------------------------------------
#   write: brainhops transformation -> OME-Zarr coordinate transformation
# ----------------------------------------------------------------------
# Writers return the JSON of one OME coordinate transformation, with its
# parameters reordered by storage_perm into the stored OME order.

# These OME types exist only in the richer OME-NGFF versions, whereas scale
# and translation exist in every version.
_RICH_TYPES = frozenset(
    {"rotation", "affine", "mapAxis", "displacements", "coordinates"}
)

_to_ome_fn: Function = Function("to_ome")
_to_ome_for = _to_ome_fn.register


def to_ome(
    transform: Transformation, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    """Map a brainhops transformation to one OME coordinate transformation.

    An affine that only scales and translates each axis is written in the
    lean form of a `scale`, followed by a `translation` when the translation
    is not zero. Any other affine is written in full, and a sequence is
    written as a sequence of its mapped children. The parameters are
    reordered by `storage_perm`, from the brainhops axis order to the stored
    OME order.

    Raises
    ------
    OmeMappingError
        If the transformation does not reduce to an affine, which is the
        case for a field.
    """
    # A type with no registered ancestor falls back to the affine writer.
    try:
        return _to_ome_fn(transform, storage_perm, ndim)
    except NoMethodError:
        return _affine_to_ome(transform, storage_perm, ndim)


def to_dataset_ome(
    transform: Transformation, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Optional[tx.Dict[str, tx.Any]]:
    """Map the geometry of a level to the transformation of its dataset.

    OME-NGFF 0.6 restricts the transformation of a dataset to a single
    `scale` or `identity`, or a `sequence` of a scale followed by a
    translation (ngff-spec tag `0.6` at 89da1da, `index.md`, section
    "coordinateTransformations metadata", "Constraints"). A geometry whose
    mapping is already of that form is returned as it is. Any other
    geometry that only scales and translates each axis, such as a
    translation followed by a scale, is written as a scale followed by a
    translation.

    Parameters
    ----------
    transform : Transformation
        The geometry of the level, from its voxels to the intrinsic system.
    storage_perm : sequence of int
        The permutation from the brainhops axis order to the stored order.
    ndim : int
        The number of axes of the level.

    Returns
    -------
    dict or None
        The JSON of the dataset transformation, or `None` when the geometry
        is not diagonal, such as a rotation or a shear, or when it flips an
        axis, since a dataset only holds a positive scale.

    Raises
    ------
    OmeMappingError
        If the geometry cannot be written as OME metadata at all, as for a
        field.
    """
    entry = to_ome(transform, storage_perm, ndim)
    if is_dataset_transform(entry):
        return entry
    affine = transform.to(Affine, on_error={ConversionError: None})
    if affine is None:
        return None
    matrix = permute_affine(affine.matrix, storage_perm)
    decomposed = scale_translation_from_affine(matrix, ndim)
    if decomposed is None:
        return None
    scale, translation = decomposed
    # A flip makes the entry an affine, which a dataset cannot hold.
    entry = _scale_translation_entry(scale, translation)
    return entry if is_dataset_transform(entry) else None


def diagonal_signs(
    transform: Transformation, ndim: int
) -> tx.Optional[np.ndarray]:
    """Return the signs of the scale of a diagonal geometry.

    A negative sign flips its axis. A flip cannot be written in the dataset
    of a level, which only holds a positive scale, so the writer moves the
    flip that every level shares to the placement of the pyramid, with
    [`unflipped`][].

    Parameters
    ----------
    transform : Transformation
        The geometry of a level, in the brainhops axis order.
    ndim : int
        The number of axes of the level.

    Returns
    -------
    ndarray or None
        One sign per axis, `-1.0` or `1.0`, or `None` when the geometry is
        not an affine that only scales and translates each axis.
    """
    affine = transform.to(Affine, on_error={ConversionError: None})
    if affine is None:
        return None
    decomposed = scale_translation_from_affine(affine.matrix, ndim)
    if decomposed is None:
        return None
    scale, _ = decomposed
    return np.where(scale < 0, -1.0, 1.0)


def unflipped(
    transform: Transformation, signs: tx.Sequence[float], ndim: int
) -> Affine:
    """Remove a flip from a geometry that only scales and translates.

    With `D` the diagonal matrix of `signs`, the result `G` satisfies
    `transform = D @ G`, so the geometry is the flip applied after `G`.

    Parameters
    ----------
    transform : Transformation
        The geometry of a level, as accepted by [`diagonal_signs`][].
    signs : sequence of float
        The signs of the flip, as [`diagonal_signs`][] returns them.
    ndim : int
        The number of axes of the level.

    Returns
    -------
    Affine
        The geometry without the flip, with the systems of `transform`.
    """
    matrix = np.asarray(transform.to(Affine).matrix, dtype=float)
    flipped = np.asarray(signs, dtype=float)[:, None] * matrix
    return Affine(
        matrix=flipped,
        input=getattr(transform, "input", None),
        output=getattr(transform, "output", None),
    )


def flip(signs: tx.Sequence[float]) -> Affine:
    """Return the linear transformation that flips the axes of negative sign.

    Parameters
    ----------
    signs : sequence of float
        One sign per axis, as [`diagonal_signs`][] returns them.

    Returns
    -------
    Affine
        The diagonal affine of `signs`, without translation. It is written
        as an OME `affine`, never as a negative `scale`.
    """
    signs = np.asarray(signs, dtype=float)
    matrix = np.concatenate([np.diag(signs), np.zeros((len(signs), 1))], 1)
    return Affine(matrix=matrix)


def is_dataset_transform(entry: tx.Mapping[str, tx.Any]) -> bool:
    """Return whether an OME transform is allowed on a 0.6 dataset.

    The prose of ngff-spec 0.6 asks for a scale followed by a translation,
    which ome-zarr-models-py also checks (`v06/multiscales.py`, at
    c75f227), although the schema of version 0.6 accepts either order. The
    stricter order is required here, so that what brainhops writes passes
    every reader.
    """
    kind = entry.get("type")
    if kind in ("identity", "scale"):
        return True
    if kind != "sequence":
        return False
    steps = [step.get("type") for step in entry.get("transformations", [])]
    return steps == ["scale", "translation"]


def scale_translation_from_affine(
    matrix: tx.Any, ndim: int
) -> tx.Optional[tx.Tuple[np.ndarray, np.ndarray]]:
    """Return the per-axis scale and translation of a diagonal affine.

    The argument `matrix` is a compact affine, or `None` for the identity,
    whose scale is all ones and whose translation is all zeros. The function
    returns `None` when the linear block is not square, or when it has an
    off-diagonal term larger than 1e-8 in absolute value, as a rotation or a
    shear does. Such an affine must be written in full.
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
    # The leanest form is a scale alone when the translation is zero, and
    # otherwise a sequence of a scale and a translation. The schema of
    # OME-NGFF 0.6 requires a positive scale, and a flip is never written as
    # a negative one: a scale with an entry that is not positive is written
    # as the equivalent affine (spec/design/format-records.md, item 135).
    scale = np.asarray(scale, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if bool((scale <= 0).any()):
        matrix = np.concatenate([np.diag(scale), translation[:, None]], 1)
        return {"type": "affine", "affine": _matrix_to_list(matrix)}
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
    # A diagonal affine is written as a scale and a translation, and any
    # other affine is written as an affine.
    affine = transform.to(Affine, on_error={ConversionError: None})
    if affine is None:
        raise OmeMappingError(
            "This image is placed by a transformation that is not an affine, "
            "so it cannot be written as OME-Zarr multiscale metadata."
        )
    # A projection writes the values of its created axes in the translation
    # of its matrix, and a free axis, whose value is NaN, has none to write.
    if np.any(np.isnan(np.asarray(affine.matrix, dtype=float))):
        raise OmeMappingError(
            "This transformation depends on an axis that a projection "
            "creates free, which has no value to write."
        )
    # Every entry maps the axes of the image onto as many axes, so an
    # element that adds or removes axes, such as a projection inside a
    # sequence that is not computed, would lose rows or columns when its
    # matrix is reordered.
    shape = np.shape(affine.matrix)
    if shape != (len(storage_perm), len(storage_perm) + 1):
        raise OmeMappingError(
            f"This transformation maps {shape[1] - 1} axes onto {shape[0]} "
            f"axes, whereas each transformation of an image of "
            f"{len(storage_perm)} axes is written as a map of these axes "
            f"onto as many axes. A sequence whose elements add or remove "
            f"axes can be written once it is computed."
        )
    matrix = permute_affine(affine.matrix, storage_perm)
    decomposed = scale_translation_from_affine(matrix, ndim)
    if decomposed is None:
        return {"type": "affine", "affine": _matrix_to_list(matrix)}
    scale, translation = decomposed
    return _scale_translation_entry(scale, translation)


@_to_ome_for
def _(
    transform: Identity, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    scale, translation = scale_translation_from_affine(None, ndim)
    return _scale_translation_entry(scale, translation)


@_to_ome_for
def _(
    transform: Scaling, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    if transform.scale is None:
        scale, translation = scale_translation_from_affine(None, ndim)
        return _scale_translation_entry(scale, translation)
    scale = permute_vector(transform.scale, storage_perm)
    return _scale_translation_entry(scale, np.zeros(len(scale)))


@_to_ome_for
def _(
    transform: Translation, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    if transform.translation is None:
        scale, translation = scale_translation_from_affine(None, ndim)
        return _scale_translation_entry(scale, translation)
    return {
        "type": "translation",
        "translation": permute_vector(transform.translation, storage_perm),
    }


@_to_ome_for
def _(
    transform: Rotation, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    affine = transform.to(Affine, on_error={ConversionError: None})
    if affine is None:
        raise OmeMappingError(
            "This rotation has no matrix, so it cannot be written as OME-Zarr "
            "metadata."
        )
    linear = permute_linear(
        np.asarray(affine.matrix, dtype=float)[:, :-1], storage_perm
    )
    return {"type": "rotation", "rotation": _matrix_to_list(linear)}


@_to_ome_for
def _(
    transform: Sequence, storage_perm: tx.Sequence[int], ndim: int
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
@_to_ome_for
def _(
    transform: Affine, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    return _affine_to_ome(transform, storage_perm, ndim)


@_to_ome_for
def _(
    transform: Linear, storage_perm: tx.Sequence[int], ndim: int
) -> tx.Dict[str, tx.Any]:
    return _affine_to_ome(transform, storage_perm, ndim)


# ----------------------------------------------------------------------
#   helpers of the reader
# ----------------------------------------------------------------------


def _positions(order: tx.Sequence[int]) -> tx.Dict[int, int]:
    # Map each stored position to the brainhops position of its axis.
    return {int(stored): position for position, stored in enumerate(order)}


def _is_identity_order(order: tx.Sequence[int]) -> bool:
    return list(order) == list(range(len(order)))


def _check_same_dimension(
    transform: tx.Any, source: tx.Sequence[int], target: tx.Sequence[int]
) -> None:
    if len(source) != len(target):
        raise OmeMappingError(
            f"This OME {transform.type} maps {len(source)} axes onto "
            f"{len(target)} axes, although it keeps the number of axes."
        )


def _check_entries(
    transform: tx.Any, values: np.ndarray, source: tx.Sequence[int]
) -> None:
    if values.shape[0] != len(source):
        raise OmeMappingError(
            f"This OME {transform.type} has {values.shape[0]} entries, "
            f"which does not fit a coordinate system of {len(source)} axes."
        )


def _check_indices(
    transform: tx.Any, name: str, indices: tx.Sequence[int], count: int
) -> None:
    # OME-NGFF 0.6 requires valid, unique and zero-based axis indices.
    if len(set(indices)) != len(indices):
        raise OmeMappingError(
            f"The {name} {list(indices)} of this OME {transform.type} "
            "repeat an axis."
        )
    if any(not 0 <= index < count for index in indices):
        raise OmeMappingError(
            f"The {name} {list(indices)} of this OME {transform.type} name "
            f"an axis that a coordinate system of {count} axes lacks."
        )


def _relabel(
    xform: Transformation,
    source: tx.Sequence[int],
    target: tx.Sequence[int],
) -> Transformation:
    # `xform` is expressed in the order of the input system on both of its
    # sides. Its output axis `j` is therefore the stored output axis
    # `source[j]`, and the permutation that follows it moves that axis to
    # the brainhops position that `target` gives it.
    position = _positions(source)
    permutation = [position[int(stored)] for stored in target]
    if _is_identity_order(permutation):
        return xform
    return Sequence([xform, Permutation(permutation=permutation)])


def _parameter(
    transform: tx.Any, name: str, reading: Reading, ndim: int
) -> np.ndarray:
    # A parameter is given inline, or as an array of the group named by
    # `path`. The path is resolved against the group that holds the
    # metadata, which is how the field paths are resolved too. Version 0.6
    # only allows `path` for an affine or a rotation, and the reader of an
    # image refuses it for a scale or a translation, except in the drafts
    # 0.6.dev1 and 0.6.dev2, which allowed it (spec/design/format-records.md,
    # item 134).
    # abczarr marks an absent field with a sentinel that is false.
    values = getattr(transform, name, None)
    if not values:
        path = getattr(transform, "path", None)
        if not isinstance(path, str):
            raise OmeMappingError(
                f"This OME {transform.type} gives neither its {name!r} "
                "parameters nor the path of an array that holds them."
            )
        if reading.node is None:
            raise OmeMappingError(
                f"This OME {transform.type} stores its parameters in the "
                f"array {path!r}, which can only be read from a group."
            )
        try:
            values = _node.follow_path(reading.node, path)[...]
        except Exception as error:
            raise OmeMappingError(
                f"The parameters of this OME {transform.type} could not be "
                f"read from the array {path!r}. {error}"
            ) from error
    values = np.asarray(values, dtype=float)
    if values.ndim != ndim:
        raise OmeMappingError(
            f"The parameters of this OME {transform.type} have "
            f"{values.ndim} dimensions instead of {ndim}."
        )
    return values


def _affine_rows(matrix: np.ndarray, rows: int) -> np.ndarray:
    # The metadata stores the rows of the output axes. A full homogeneous
    # matrix, whose extra last row is (0, ..., 0, 1), is accepted as well.
    if matrix.shape[0] == rows + 1 and _is_homogeneous_row(matrix[-1]):
        return matrix[:-1]
    if matrix.shape[0] != rows:
        raise OmeMappingError(
            f"This OME affine has {matrix.shape[0]} rows, which does not fit "
            f"an output coordinate system of {rows} axes."
        )
    return matrix


def _is_homogeneous_row(row: np.ndarray) -> bool:
    expected = np.zeros(row.shape[0])
    expected[-1] = 1.0
    return bool(np.array_equal(row, expected))


def _intermediate_order(
    child: tx.Any, current: tx.Sequence[int], reading: Reading
) -> tx.List[int]:
    # A child that names a declared output system takes the order of that
    # system. Otherwise the space keeps the order of the previous space
    # when it has the same number of axes, and the stored order when the
    # number of axes changes.
    output = getattr(child, "output", None)
    name = getattr(output, "name", None)
    in_group = getattr(output, "path", None) is None
    if isinstance(name, str) and in_group and name in reading.orders:
        return list(reading.orders[name])
    count = _output_count(child, len(current), reading)
    if count == len(current):
        return list(current)
    return list(range(count))


def _output_count(child: tx.Any, count: int, reading: Reading) -> int:
    # Return the number of axes that `child` maps `count` input axes onto.
    # A coordinates transformation is assumed to keep the number of axes,
    # and its reader checks that its field agrees.
    if isinstance(child, _ot.Affine):
        matrix = _parameter(child, "affine", reading, ndim=2)
        square = matrix.shape == (count + 1, count + 1)
        if square and _is_homogeneous_row(matrix[-1]):
            return count
        return int(matrix.shape[0])
    if isinstance(child, _ot.ProjectAxis):
        dropped = child.droppedInputs or []
        created = child.createdOutputs or []
        return count - len(dropped) + len(created)
    if isinstance(child, _ot.MapAxis):
        return len(child.mapAxis)
    if isinstance(child, _ot.ByDimension):
        return sum(
            len(item.outputAxes or []) for item in child.transformations
        )
    if isinstance(child, _ot.Sequence):
        for inner in child.transformations:
            count = _output_count(inner, count, reading)
        return count
    return count


def _part_matrix(
    part: Transformation, inputs: int, outputs: int
) -> tx.Optional[np.ndarray]:
    # Return the compact matrix of one part of a byDimension, or `None` when
    # the part is not affine.
    if isinstance(part, Identity):
        if inputs != outputs:
            return None
        return np.eye(outputs, inputs + 1)
    affine = part.to_affine(on_error=None)
    if affine is None:
        return None
    matrix = np.asarray(affine.matrix, dtype=float)
    if matrix.shape != (outputs, inputs + 1):
        return None
    return matrix


def _assembled_affine(
    parts: tx.Sequence[tx.Tuple[tx.List[int], tx.List[int], Transformation]],
    source: tx.Sequence[int],
    target: tx.Sequence[int],
    width_in: int,
    width_out: int,
) -> tx.Optional[Affine]:
    # Each affine part fills the rows of its output axes and the columns of
    # its input axes, in the stored order, and the full matrix is then
    # reordered into the brainhops order. `None` is returned as soon as one
    # part is not affine.
    full = np.zeros((width_out, width_in + 1))
    for inputs, outputs, part in parts:
        matrix = _part_matrix(part, len(inputs), len(outputs))
        if matrix is None:
            return None
        full[np.ix_(outputs, inputs)] = matrix[:, :-1]
        full[outputs, -1] = matrix[:, -1]
    columns = list(source) + [width_in]
    return Affine(matrix=full[np.ix_(list(target), columns)])


def _subspaces(
    parts: tx.Sequence[tx.Tuple[tx.List[int], tx.List[int], Transformation]],
    source: tx.Sequence[int],
    target: tx.Sequence[int],
) -> Transformation:
    # A part that is not affine is embedded as a subspace transformation,
    # which leaves the other axes unchanged. The parts then form an exact
    # sequence only when each one reads the axes that it writes, since they
    # then act on disjoint sets of axes.
    if len(source) != len(target):
        raise OmeMappingError(
            "This OME byDimension has a part that is not affine and changes "
            "the number of axes, which brainhops cannot read."
        )
    source_position = _positions(source)
    target_position = _positions(target)
    embedded = []  # type: tx.List[Transformation]
    for inputs, outputs, part in parts:
        input_axes = [source_position[i] for i in inputs]
        output_axes = [target_position[o] for o in outputs]
        if sorted(input_axes) != sorted(output_axes):
            raise OmeMappingError(
                "This OME byDimension has a part that is not affine and "
                "writes other axes than those it reads, which brainhops "
                "cannot read."
            )
        embedded.append(
            SubspaceTransformation(
                transformation=part,
                input_axes=input_axes,
                output_axes=output_axes,
            )
        )
    if len(embedded) == 1:
        return embedded[0]
    return Sequence(embedded)
