"""The kinds of NRRD axes that hold vectors and tensors, and their frames.

A NRRD file tells what the samples along each axis are with the `kinds`
field, and some kinds say that the samples are the components of a
vector, a point or a matrix. The components are measured in a
coordinate system, which brainhops calls the frame of the values and
NRRD calls the measurement frame. The `measurement frame` field is a
square matrix, written one column at a time, whose columns are the
basis vectors of that system expressed in the world space of the file,
so that it maps the components of a vector to its components in the
world space.

The reader gives each of these kinds the type of axis that it names:

| Kind                         | Axis                              |
| ---------------------------- | --------------------------------- |
| `vector`, `2-vector`,        | `vector`, of valence `"vector"`   |
| `3-vector`, `4-vector`       |                                   |
| `covariant-vector`, `normal` | `vector`, of valence `"covector"` |
| `3-gradient`, `3-normal`     | `vector`, of valence `"covector"` |
| `point`                      | `point`                           |
| `2D-symmetric-matrix`        | `tensor`, layout `"symmetric"`    |
| `3D-symmetric-matrix`        | `tensor`, layout `"symmetric"`    |
| `2D-matrix`, `3D-matrix`     | `tensor`, layout `"C"`            |

A tensor has two indices of valence `"vector"`, because NRRD does not
say how a matrix changes with the frame, and because the measurement
frame acts on both indices of a diffusion tensor in this way. The order
of the components is that of the specification: `Mxx Mxy Myy` and
`Mxx Mxy Mxz Myy Myz Mzz` for the symmetric matrices, which is the
`"symmetric"` layout, and the rows one after the other for the full
matrices, which is the `"C"` layout. A component axis whose length does
not match the dimension of the frame, such as a `vector` axis of twenty
entries in a three-dimensional space, holds no components of a vector
in that frame and is read as a channel axis.

The masked kinds (`2D-masked-symmetric-matrix`,
`3D-masked-symmetric-matrix`, `2D-masked-matrix` and `3D-masked-matrix`)
store a mask value before the components of a matrix. An axis of the
data model holds either independent values or the components of one
object, and a reader does not change the data, so these axes are read as
channel axes. The other kinds of values, such as colours, `complex`,
`quaternion` and `list`, are channel axes too.
"""

# The design is recorded in spec/design/vector-and-tensor-images.md.

__all__ = [
    "MeasuredFrame",
    "WrittenKind",
    "component_axis",
    "embedded",
    "frame_size",
    "gathered",
    "has_world",
    "header_frames",
    "measured_frame",
    "measurement_change",
    "same_matrix",
    "same_reading",
    "written_kind",
]

import numpy as np
import typing_extensions as tx

from brainhops.backends import get_array_backend
from brainhops.datamodel._axes.components import _layout_entries
from brainhops.datamodel._frames import (
    FrameSlot,
    component_axes,
    component_count,
    pair_axes,
)
from brainhops.datamodel._transformations.compute.multiscale import (
    _as_affine,
)
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.enums import Valence
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Transformation
from brainhops.io.common._frames import restricted_frame

from ._raw import NrrdRaw

_VECTOR_KINDS: tx.Dict[str, tx.Tuple[Valence, tx.Optional[int]]] = {
    "vector": (Valence.vector, None),
    "2-vector": (Valence.vector, 2),
    "3-vector": (Valence.vector, 3),
    "4-vector": (Valence.vector, 4),
    "covariant-vector": (Valence.covector, None),
    "normal": (Valence.covector, None),
    "3-gradient": (Valence.covector, 3),
    "3-normal": (Valence.covector, 3),
}
"""The valence of the kinds of vectors, and the length that they need."""

_TENSOR_KINDS: tx.Dict[str, tx.Tuple[int, str]] = {
    "2d-symmetric-matrix": (2, "symmetric"),
    "2d-matrix": (2, "C"),
    "3d-symmetric-matrix": (3, "symmetric"),
    "3d-matrix": (3, "C"),
}
"""The dimension of the frame and the layout of the kinds of matrices."""

_MATRIX_SIZES = {"symmetric": {2: 3, 3: 6}, "C": {2: 4, 3: 9}}
"""The number of components of each layout of a matrix."""


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def has_world(record: NrrdRaw, nspace: int) -> bool:
    """Return whether a header places its samples in a world space.

    The header needs a space, or a space dimension, the directions of
    its spatial axes, and at least one spatial axis, as the reader of
    the geometry requires.
    """
    return bool(
        record.space_directions is not None
        and record.space_dimension
        and nspace
    )


def frame_size(record: NrrdRaw, nspace: int) -> int:
    """Return the number of axes of the frame of the vectors of a header.

    The frame of a header with a world space has the spatial axes of
    that space, which are all its axes but time. The frame of any other
    header has the spatial axes of the array.
    """
    if not has_world(record, nspace):
        return nspace
    space = record.space or ""
    return record.space_dimension - (1 if space.endswith("-time") else 0)


def component_axis(
    kind: tx.Optional[str],
    name: str,
    size: int,
    ndim: int,
    **options: tx.Any,
) -> tx.Optional[Axis]:
    """Return the component axis that a kind names, or `None`.

    Parameters
    ----------
    kind : str, optional
        The kind of the axis in the header, lower-cased.
    name : str
        The name of the axis.
    size : int
        The number of samples along the axis.
    ndim : int
        The number of axes of the frame of the values, as
        [`frame_size`][] gives it.
    **options : Any
        Other fields of the axis, such as its unit.

    Returns
    -------
    Axis or None
        The component axis, or `None` when the kind names no component
        axis, or when the length of the axis does not match the frame.
    """
    if kind in _VECTOR_KINDS:
        valence, length = _VECTOR_KINDS[kind]
        if size != ndim or (length is not None and size != length):
            return None
        return Axis(name, "vector", valence=valence, **options)
    if kind == "point":
        if size != ndim:
            return None
        return Axis(name, "point", **options)
    if kind in _TENSOR_KINDS:
        n, layout = _TENSOR_KINDS[kind]
        if n != ndim or size != _MATRIX_SIZES[layout][n]:
            return None
        return Axis(name, "tensor", layout=layout, **options)
    return None


def header_frames(
    record: NrrdRaw,
    preferred: tx.Optional[Transformation],
    world: bool,
    axes: tx.Optional[tx.Sequence[Axis]],
    shape: tx.Tuple[int, ...],
) -> tx.Dict[int, FrameSlot]:
    """Return the frame of each component axis, as a header states it.

    The frame of a header with a measurement frame `M` is the preferred
    transformation `A`, which maps the array to the world space,
    followed by the linear map `M⁻¹` from the world space to the
    measurement frame. It is `A` itself when `M` is the identity, so that
    the values are already in the frame of the preferred transformation.

    The specification gives no frame to a header without a measurement
    frame. ITK writes vectors and tensors in the world space of the
    header without a measurement frame, and ITK and Slicer read them in
    that space, so the frame is then `A`: the world space of a header
    with a world space, or the physical space that the spacings of a
    header without a world space give.

    A measurement frame that is not a square matrix of the dimension of
    the world space, or that cannot be inverted accurately, gives a
    foreign frame,
    a coordinate system named `"measurement"` that no transformation of
    the image reaches. So does a measurement frame in a header without a
    world space, since nothing then relates the array to that space.

    Parameters
    ----------
    record : NrrdRaw
        The header.
    preferred : Transformation, optional
        The preferred transformation that the header decodes to.
    world : bool
        Whether `preferred` maps the array to a world space, rather than
        to the physical space of the spacings.
    axes : sequence of Axis, optional
        The axes of the array system.
    shape : tuple of int
        The shape of the data.

    Returns
    -------
    dict of int to slot
        The frame of each component axis, or an empty mapping when the
        image has no component axis.
    """
    components = component_axes(axes)
    if not components or preferred is None:
        return {}
    frame: FrameSlot = preferred
    if "measurement frame" in record.fields:
        frame = _measured_frame(record, preferred, world)
    if not isinstance(frame, Transformation):
        return {i: frame for i in components}
    return {i: restricted_frame(frame, axes, i, shape[i]) for i in components}


def _measured_frame(
    record: NrrdRaw, preferred: Transformation, world: bool
) -> FrameSlot:
    """Return the map from the array to the measurement frame of a header.

    The measurement frame maps its coordinates to those of the world, so
    the map is its inverse after the preferred transformation, or a
    foreign frame when the header does not give it.
    """
    n = len(list(getattr(preferred.output, "axes", None) or []))
    try:
        matrix = record.measurement_frame
    except Exception:  # noqa: BLE001
        matrix = None
    if (
        not world
        or matrix is None
        or matrix.shape != (n, n)
        or not np.all(np.isfinite(matrix))
        or np.linalg.cond(matrix) >= 1 / np.finfo(float).eps
    ):
        return _measurement_system(preferred.output)
    return measurement_change(preferred, matrix)


def _measurement_system(space: CoordinateSystem) -> CoordinateSystem:
    """Return the coordinate system of a measurement frame.

    Its axes are those of the world space, without their anatomical
    orientation, which the measurement frame does not keep.
    """
    return CoordinateSystem(
        [
            Axis(axis.name, axis.type, unit=getattr(axis, "unit", None))
            for axis in getattr(space, "axes", None) or []
        ],
        name="measurement",
    )


def measurement_change(
    world: Transformation, matrix: np.ndarray
) -> Transformation:
    """Return the frame `M⁻¹ ∘ A` that a measurement frame `M` records.

    `world` is the map `A` from the array to the world space, and `M`
    acts on all its outputs. The frame is `A` itself when `M` is the
    identity.
    """
    n = len(matrix)
    if np.array_equal(matrix, np.eye(n)):
        return world
    change = Affine(
        matrix=np.concatenate([np.linalg.inv(matrix), np.zeros((n, 1))], 1),
        input=world.output,
        output=_measurement_system(world.output),
    )
    return change @ world


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


class WrittenKind(tx.NamedTuple):
    """The kind under which NRRD writes a component axis."""

    kind: str
    """The kind, spelled as the specification spells it."""

    gather: tx.Optional[np.ndarray]
    """For each written component, the entry of the axis that holds it.

    An entry of `-1` is a component that the layout of the axis leaves
    out, which is zero. `None` means that the entries are written in
    their order.
    """


def written_kind(axis: Axis, size: int, ndim: int) -> tx.Optional[WrittenKind]:
    """Return the kind under which NRRD writes a component axis, or `None`.

    A vector, or a displacement, of valence `"vector"` is written as a
    `vector`, and a vector of valence `"covector"` as a
    `covariant-vector`. The coordinates of points are written as a
    `point`. A tensor whose two indices are of valence `"vector"` is
    written as a symmetric matrix when its layout stores no two
    components whose indices are permutations of each other, and as a
    full matrix otherwise. The components are then written in the order
    of the specification, so another layout of a symmetric matrix is
    reordered, and the components that a layout leaves out, such as the
    components off the diagonal of the `"diagonal"` layout, are written
    as zeros.

    Parameters
    ----------
    axis : Axis
        The component axis.
    size : int
        The number of entries along the axis.
    ndim : int
        The number of axes of the frame.

    Returns
    -------
    WrittenKind or None
        The kind and the order of the components, or `None` when the
        reader would not read the axis back with the same type. This is
        the case of a vector whose length is not the dimension of the
        frame, of a tensor of another rank or valence, and of a tensor in
        a frame of another dimension than two or three.
    """
    type_ = axis.type
    if type_ in ("vector", "displacement"):
        if size != ndim:
            return None
        if axis.valences == (Valence.vector,):
            return WrittenKind("vector", None)
        return WrittenKind("covariant-vector", None)
    if type_ in ("point", "coordinate"):
        return WrittenKind("point", None) if size == ndim else None
    if type_ != "tensor" or ndim not in (2, 3):
        return None
    if tuple(axis.valences) != (Valence.vector, Valence.vector):
        return None
    try:
        entries = axis.entries(ndim, size)
    except ValueError:
        return None
    sorted_entries = [tuple(sorted(entry)) for entry in entries]
    full = len(set(sorted_entries)) < len(sorted_entries)
    layout = "C" if full else "symmetric"
    unpack = axis._unpack(ndim, size)
    gather = np.array(
        [
            unpack[np.ravel_multi_index(entry, (ndim, ndim))]
            for entry in _layout_entries(layout, ndim, 2)
        ],
        dtype=int,
    )
    if np.array_equal(gather, np.arange(size)):
        gather = None
    name = "matrix" if full else "symmetric-matrix"
    return WrittenKind(f"{ndim}D-{name}", gather)


def gathered(values: tx.Any, axis: int, gather: np.ndarray) -> tx.Any:
    """Return the values with the entries along an axis taken in order.

    An entry of `-1` in `gather` takes the value zero.
    """
    xp = get_array_backend(values)
    moved = xp.moveaxis(values, axis, -1)
    if np.any(gather < 0):
        zeros = xp.zeros_like(moved[..., :1])
        moved = xp.concatenate([moved, zeros], axis=-1)
        gather = np.where(gather < 0, moved.shape[-1] - 1, gather)
    return xp.moveaxis(moved[..., gather], -1, axis)


class MeasuredFrame(tx.NamedTuple):
    """The measurement frame that records a frame of the values."""

    matrix: np.ndarray
    """The square matrix `M` from the frame to the world space."""

    rows: tx.List[int]
    """The outputs of the world map that the components follow, in order."""


def measured_frame(
    slot: tx.Optional[Transformation],
    world: Transformation,
    axes: tx.Sequence[Axis],
    index: int,
    size: int,
) -> tx.Optional[MeasuredFrame]:
    """Return the measurement frame that records the frame of an axis.

    The frame `F` of the values and the map `A` from the array to the
    world space give the matrix `M = D A · (D F)⁻¹`, which maps the
    components of a vector in `F` to its components in the world space.
    The values are then recorded as they are, with `M` as the
    measurement frame, because the reader reads the frame `M⁻¹ ∘ A`,
    whose Jacobian is `D F`. The coordinates of a point also change with
    the translation of the frame, so a point axis is recorded only when
    `F` is `M⁻¹ ∘ A` exactly.

    Parameters
    ----------
    slot : Transformation, optional
        The frame of the values, `None` for the array system.
    world : Transformation
        The map from the array to the world space, restricted to the
        components of the axis.
    axes : sequence of Axis
        The axes of the array system.
    index : int
        The index of the component axis.
    size : int
        The number of entries along the axis.

    Returns
    -------
    MeasuredFrame or None
        The measurement frame, or `None` when one of the two frames is
        not affine, or when the frame of a point axis is not `M⁻¹ ∘ A`.

    Raises
    ------
    ValueError
        If the two frames do not take the components from the same axes
        of the array, so that no matrix relates them.
    """
    ncomponents = component_count(axes, index, size, [world])
    source, target, cols = pair_axes(slot, world, axes, ncomponents)
    if not target.affine or (source is not None and not source.affine):
        return None
    rows = target.rows
    world_jacobian = target.linear[np.ix_(rows, cols)]
    if slot is world:
        matrix = np.eye(ncomponents)
    elif source is None:
        matrix = world_jacobian
    else:
        jacobian = source.linear[np.ix_(source.rows, cols)]
        if np.array_equal(jacobian, world_jacobian):
            matrix = np.eye(ncomponents)
        else:
            matrix = world_jacobian @ np.linalg.inv(jacobian)
    points = axes[index].valences is None
    if points and slot is not world:
        if not _points_agree(slot, world, matrix, len(axes)):
            return None
    return MeasuredFrame(matrix, list(rows))


def _points_agree(
    slot: tx.Optional[Transformation],
    world: Transformation,
    matrix: np.ndarray,
    ndim: int,
) -> bool:
    """Return whether a frame of points is `M⁻¹ ∘ A`, translation included."""
    if slot is None:
        return False
    first, second = _as_affine(slot), _as_affine(world)
    if first is None or second is None:
        return False
    a = _widened(np.asarray(first.matrix, dtype=float), ndim)
    b = _widened(np.asarray(second.matrix, dtype=float), ndim)
    if a.shape != b.shape or b.shape[0] != len(matrix):
        return False
    return same_matrix(a, np.linalg.solve(matrix, b))


def _widened(matrix: np.ndarray, ndim: int) -> np.ndarray:
    """Return an affine matrix with a column for each of `ndim` inputs."""
    missing = ndim + 1 - matrix.shape[1]
    if missing <= 0:
        return matrix
    linear = np.pad(matrix[:, :-1], ((0, 0), (0, missing)))
    return np.concatenate([linear, matrix[:, -1:]], axis=1)


def same_reading(first: str, second: str, size: int) -> bool:
    """Return whether the reader reads two kinds as the same axis.

    The kinds of vectors of the same valence are read alike when the
    axis has the length that each of them requires, such as `vector`
    and `3-vector` for an axis of three entries. The kinds of matrices
    are read alike only when they are the same kind. Other kinds are
    compared by name, regardless of case.
    """
    first, second = first.lower(), second.lower()
    if first in _VECTOR_KINDS and second in _VECTOR_KINDS:
        valence, length = _VECTOR_KINDS[first]
        other, needed = _VECTOR_KINDS[second]
        return (
            valence == other
            and length in (None, size)
            and needed in (None, size)
        )
    return first == second


def same_matrix(first: np.ndarray, second: np.ndarray) -> bool:
    """Return whether two matrices are the same, up to rounding."""
    if first.shape != second.shape:
        return False
    scale = max(float(np.abs(second).max(initial=0.0)), 1.0)
    return bool(np.allclose(first, second, rtol=0, atol=1e-12 * scale))


def embedded(measured: MeasuredFrame, noutputs: int) -> np.ndarray:
    """Return a measurement frame over all the outputs of the world map.

    The outputs that the components do not follow, such as time, keep
    their coordinates.
    """
    full = np.eye(noutputs)
    rows = measured.rows
    full[np.ix_(rows, rows)] = measured.matrix
    return full
