"""Frames of the values of images, and the changes of frame.

An image whose axes hold the components of vectors or tensors records,
for each data axis, the frame in which these components are expressed
(see `spec/design/vector-and-tensor-images.md`). This module holds the
helpers that read, validate, compose and change these frames. The public
interface is
[`SingleScaleImage.frames`][brainhops.datamodel.images.SingleScaleImage.frames]
and
[`SingleScaleImage.reframe`][brainhops.datamodel.images.SingleScaleImage.reframe].
"""

from collections.abc import Mapping

import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core.typing import ArrayProtocol, Shape
from brainhops.backends import backend, get_array_backend, resolve_backend

from ._sugar import get_axes
from ._transformations.compute.multiscale import (
    _as_affine,
    _as_affine_ignoring_fields,
)
from .axes import Axis, AxisSequence
from .content import _batched, _inv
from .systems import CoordinateSystem
from .transformations import CartesianField, Sequence, Transformation

FrameSlot: tx.TypeAlias = tx.Union[Transformation, CoordinateSystem, None]
"""The frame of the values along one axis.

`None` is the array system of the image, a transformation maps the array
system to the frame, and a coordinate system names a foreign frame.
"""

FrameTarget: tx.TypeAlias = tx.Union[int, str, Transformation, None]
"""A frame to re-express values in.

An integer selects an entry of `transformations`, a string selects the
most preferred entry whose output system has that name, `None` is the
array system, and a transformation maps the array system to the frame.
"""

FrameTargets: tx.TypeAlias = tx.Union[
    FrameTarget, tx.Mapping[tx.Union[int, str], FrameTarget]
]
"""One frame for every content axis, or a frame for each named axis."""

ReframeMethod: tx.TypeAlias = tx.Union[
    tx.Literal["jacobian", "rotation"],
    tx.Callable[[ArrayProtocol], ArrayProtocol],
]
"""What is applied to the values when their frame changes."""


# ----------------------------------------------------------------------
#   AXES OF THE ARRAY SYSTEM
# ----------------------------------------------------------------------


def array_axes(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Optional[AxisSequence]:
    """Return the `ndim` axes of an array system, or `None` if unknown."""
    axes = get_axes(system)
    if axes.is_variadic:
        if axes == [...]:
            return None
        axes = axes.expand(ndim)
    if len(axes) != ndim:
        return None
    return axes


def content_axes(axes: tx.Optional[tx.Sequence[Axis]]) -> tx.List[int]:
    """Return the indices of the axes that have a content."""
    return [
        i
        for i, axis in enumerate(axes or ())
        if getattr(axis, "content", None) is not None
    ]


# ----------------------------------------------------------------------
#   READING THE FRAMES
# ----------------------------------------------------------------------


def normalize_frames(
    raw: tx.Any, axes: tx.Optional[tx.Sequence[Axis]], ndim: int
) -> tx.Tuple[FrameSlot, ...]:
    """Return one frame slot per data axis.

    Parameters
    ----------
    raw : tuple or mapping or None
        The frames as they were given: an empty tuple, a tuple of one slot
        per data axis, or a mapping from axis names or indices to slots.
    axes : sequence of Axis, optional
        The axes of the array system, or `None` if they are unknown.
    ndim : int
        The number of data axes.

    Returns
    -------
    tuple of (Transformation or CoordinateSystem or None)
        One slot per data axis.

    Raises
    ------
    TypeError
        If a slot is neither a transformation, a coordinate system nor
        `None`.
    ValueError
        If the tuple does not have one slot per data axis, or if a slot
        is set on an axis that has no content.
    KeyError
        If a mapping names an axis that does not exist.
    """
    if raw is None or (not isinstance(raw, Mapping) and len(raw) == 0):
        return (None,) * ndim
    if isinstance(raw, Mapping):
        slots: tx.List[FrameSlot] = [None] * ndim
        for key, value in raw.items():
            slots[axis_index(key, axes, ndim)] = value
    else:
        slots = list(raw)
        if len(slots) != ndim:
            raise ValueError(
                f"The image has {ndim} axes, so its frames must have one "
                f"slot per axis, not {len(slots)}."
            )
    with_content = set(content_axes(axes))
    for i, slot in enumerate(slots):
        if slot is None:
            continue
        if not isinstance(slot, (Transformation, CoordinateSystem)):
            raise TypeError(
                f"The frame of axis {i} must be a transformation, a "
                f"coordinate system or None, not a {type(slot).__name__}."
            )
        if i not in with_content:
            raise ValueError(
                f"Axis {i} has no content, so its values have no frame. "
                f"Set the content of the axis before setting its frame."
            )
    return tuple(slots)


def axis_index(
    key: tx.Union[int, str], axes: tx.Optional[tx.Sequence[Axis]], ndim: int
) -> int:
    """Return the index of an axis given by its index or its name.

    Raises
    ------
    KeyError
        If no axis has that name or index.
    """
    if isinstance(key, str):
        names = [getattr(axis, "name", None) for axis in axes or ()]
        if key not in names:
            raise KeyError(f"The image has no axis named {key!r}.")
        return names.index(key)
    index = int(key)
    if not -ndim <= index < ndim:
        raise KeyError(f"The image has no axis {index}.")
    return index % ndim


def resolve_target(
    target: FrameTarget, transformations: tx.Sequence[Transformation]
) -> tx.Optional[Transformation]:
    """Return the transformation that a frame target stands for.

    `None` stands for the array system. An image without transformations
    is placed in its array system, so `-1` then also stands for it.

    Raises
    ------
    KeyError
        If a string names no output system of `transformations`.
    IndexError
        If an integer is not the index of a transformation.
    TypeError
        If the target is of another type.
    """
    if target is None or isinstance(target, Transformation):
        return target
    if isinstance(target, bool):
        raise TypeError("A frame cannot be a boolean.")
    if isinstance(target, int):
        if not transformations and target == -1:
            return None
        return transformations[target]
    if isinstance(target, str):
        for xform in reversed(transformations):
            if getattr(xform.output, "name", None) == target:
                return xform
        raise KeyError(f"No transformation ends in a system named {target!r}.")
    raise TypeError(
        f"A frame is an index, a system name, a transformation or None, "
        f"not a {type(target).__name__}."
    )


def frames_as_backend(raw: tx.Any, **options: tx.Any) -> tx.Any:
    """Convert the transformations among frames to another backend.

    The frames are returned as they are when no transformation changes.
    """
    if not raw:
        return raw
    if isinstance(raw, Mapping):
        items = list(raw.items())
    else:
        items = list(enumerate(raw))
    changed = False
    converted = []
    for key, slot in items:
        new = slot
        if isinstance(slot, Transformation):
            new = slot.as_backend(**options)
            changed = changed or new is not slot
        converted.append((key, new))
    if not changed:
        return raw
    if isinstance(raw, Mapping):
        return dict(converted)
    return tuple(slot for _, slot in converted)


def with_content(
    system: tx.Optional[CoordinateSystem],
    source: tx.Optional[CoordinateSystem],
    source_axes: tx.Sequence[Axis],
    ndim: int,
) -> tx.Optional[CoordinateSystem]:
    """Return an array system whose axes have the content of other axes.

    When an image is resliced, the array system of the result is that of
    the target geometry, which may describe its channels without their
    content. The content of each axis of `source_axes` is copied onto the
    axis at the same position in `system`. When `system` is unknown,
    `source` is used instead. The system is returned as it is when no
    axis changes.
    """
    axes = array_axes(system, ndim)
    if axes is None:
        return source
    new_axes = list(axes)
    for i in content_axes(source_axes):
        content = source_axes[i].content
        if getattr(new_axes[i], "content", None) != content:
            new_axes[i] = replace(new_axes[i], content=content)
    if all(new is old for new, old in zip(new_axes, axes)):
        return system
    return replace(system, axes=new_axes)


# ----------------------------------------------------------------------
#   COMPOSING THE FRAMES
# ----------------------------------------------------------------------


def compose_frames(
    slots: tx.Sequence[FrameSlot],
    keep: tx.Sequence[tx.Optional[int]],
    voxel_map: Transformation,
    new_axes: tx.Optional[tx.Sequence[Axis]],
) -> tx.Tuple[FrameSlot, ...]:
    """Return the frames after the voxels are mapped by `voxel_map`.

    Parameters
    ----------
    slots : sequence of slots
        The frames of the original image, one per data axis.
    keep : sequence of int or None
        For each axis of the new image, the axis of the original image
        that it comes from, or `None` for a new axis.
    voxel_map : Transformation
        The map from the voxels of the new image to those of the original.
    new_axes : sequence of Axis, optional
        The axes of the array system of the new image.

    Returns
    -------
    tuple of slots
        One slot per axis of the new image. A transformation is composed
        with `voxel_map`, a foreign frame is kept, and the array system
        stays the array system when `voxel_map` does not change the
        components of a vector.
    """
    with_content = set(content_axes(new_axes))
    unchanged: tx.Optional[bool] = None
    out: tx.List[FrameSlot] = []
    for i, old in enumerate(keep):
        slot = None if old is None else slots[old]
        if i not in with_content:
            out.append(None)
        elif isinstance(slot, Transformation):
            out.append(slot @ voxel_map)
        elif slot is None:
            if unchanged is None:
                unchanged = _keeps_components(voxel_map)
            out.append(None if unchanged else voxel_map)
        else:
            out.append(slot)
    return tuple(out)


def _keeps_components(voxel_map: Transformation) -> bool:
    """Return whether a map leaves the voxel components of vectors alone.

    This is the case when the map is affine and its linear part is the
    identity on the axes that are not content axes, as for a crop or a
    translation. Composing the array system with the map would also be
    correct, so a map that this function cannot read returns `False`.
    """
    affine = _as_affine(voxel_map)
    if affine is None:
        return False
    linear = np.asarray(affine.matrix, dtype=float)[:, :-1]
    inputs = array_axes(voxel_map.input, linear.shape[1])
    outputs = array_axes(voxel_map.output, linear.shape[0])
    if inputs is None or outputs is None:
        return False
    cols = [c for c in range(linear.shape[1]) if c not in content_axes(inputs)]
    rows = [
        r for r in range(linear.shape[0]) if r not in content_axes(outputs)
    ]
    if len(rows) != len(cols):
        return False
    block = linear[np.ix_(rows, cols)]
    return bool(np.array_equal(block, np.eye(len(rows))))


# ----------------------------------------------------------------------
#   CHANGING THE FRAMES
# ----------------------------------------------------------------------
# The components of a vector follow some axes of the system of its frame,
# in order. These axes, and the axes of the array along which the frame is
# differentiated, are read from the structure of the frame rather than from
# the types of the axes, so that the rules do not assume that the vectors
# are spatial.


class FrameAxes(tx.NamedTuple):
    """The axes that relate the components of a vector to a frame."""

    rows: tx.List[int]
    """The axes of the system of the frame that the components follow."""

    cols: tx.List[int]
    """The axes of the array that the frame maps onto `rows`."""

    linear: np.ndarray
    """The linear part of the frame, or of its affine part."""

    affine: bool
    """Whether the frame is affine, so that `linear` is its Jacobian."""


def frame_axes(
    frame: Transformation, axes: tx.Sequence[Axis], ncomponents: int
) -> FrameAxes:
    """Return the axes that relate `ncomponents` components to a frame.

    The structure of the frame is read from its linear part, or from the
    linear part of its affine part, in which every field is taken as the
    identity. An output of the frame is coupled to an input when the
    entry between them is not zero.

    - The axes of the system of the frame that the components follow are
      the outputs that are not coupled to a content axis of the array. If
      there are more of them than components, as when the frame carries a
      time axis along, only the outputs that share one axis type are kept,
      when exactly one type has as many axes as there are components.
    - The axes of the array that the frame maps onto them are the inputs,
      other than content axes, that are coupled to these outputs and to no
      other output that is not coupled to a content axis.

    Raises
    ------
    ValueError
        If the frame is not affine even when its fields are ignored, or if
        it does not tell which of its axes the components follow.
    """
    exact = _as_affine(frame)
    affine = exact if exact is not None else _as_affine_ignoring_fields(frame)
    if affine is None:
        raise ValueError(
            "The frame is not affine even when its fields are ignored, so "
            "the axes that the components of the values follow cannot be "
            "read from it."
        )
    linear = np.asarray(affine.matrix, dtype=float)[:, :-1]
    coupled = _coupled(linear)
    contents = content_axes(axes)
    candidates = [
        r for r in range(linear.shape[0]) if not coupled[r, contents].any()
    ]
    rows = _component_rows(
        candidates, frame.output, linear.shape[0], ncomponents
    )
    others = [r for r in candidates if r not in rows]
    cols = [
        c
        for c in range(linear.shape[1])
        if c not in contents
        and coupled[rows, c].any()
        and not coupled[others, c].any()
    ]
    if len(cols) != ncomponents:
        raise ValueError(
            f"The frame maps {len(cols)} axes of the array onto the "
            f"{ncomponents} axes that the components follow, so it does not "
            f"define a change of frame for them."
        )
    return FrameAxes(rows, cols, linear, exact is not None)


def _coupled(linear: np.ndarray) -> np.ndarray:
    """Return which entries of a linear map are not zero.

    An entry is taken as zero when it is negligible next to the largest
    entry, so that a product of matrices does not couple axes by rounding.
    """
    magnitude = np.abs(linear)
    scale = magnitude.max() if magnitude.size else 0.0
    return magnitude > 1e-12 * (scale or 1.0)


def _component_rows(
    candidates: tx.List[int],
    system: tx.Optional[CoordinateSystem],
    noutputs: int,
    ncomponents: int,
) -> tx.List[int]:
    """Return the outputs of a frame that the components follow.

    Raises
    ------
    ValueError
        If the candidates are not as many as the components, and no single
        axis type groups as many of them.
    """
    if len(candidates) == ncomponents:
        return candidates
    outputs = array_axes(system, noutputs)
    if outputs is not None:
        groups: tx.Dict[tx.Any, tx.List[int]] = {}
        for r in candidates:
            kind = getattr(outputs[r], "type", None)
            if kind is not None:
                groups.setdefault(kind, []).append(r)
        matching = [g for g in groups.values() if len(g) == ncomponents]
        if len(matching) == 1:
            return matching[0]
    raise ValueError(
        f"The frame has {len(candidates)} axes besides the content axes, "
        f"and nothing tells which {ncomponents} of them the components of "
        f"the values follow. A frame whose system has exactly "
        f"{ncomponents} axes besides the content axes, for example one that "
        f"ends with a projection, removes the ambiguity."
    )


def pair_axes(
    source: tx.Optional[Transformation],
    target: tx.Optional[Transformation],
    axes: tx.Sequence[Axis],
    ncomponents: int,
) -> tx.Tuple[tx.Optional[FrameAxes], tx.Optional[FrameAxes], tx.List[int]]:
    """Return the axes of two frames, and the array axes that they share.

    The Jacobians of the two frames are only comparable when they are
    taken along the same axes of the array. The array system has no
    structure of its own, so it takes the axes of the other frame.

    Raises
    ------
    ValueError
        If the two frames take the components from different axes of the
        array.
    """
    infos = [
        None if frame is None else frame_axes(frame, axes, ncomponents)
        for frame in (source, target)
    ]
    known = [info for info in infos if info is not None]
    if len(known) == 2 and known[0].cols != known[1].cols:
        raise ValueError(
            f"The two frames take the components from different axes of "
            f"the array ({known[0].cols} and {known[1].cols}), so the "
            f"values cannot be re-expressed from one to the other."
        )
    cols = known[0].cols if known else []
    return infos[0], infos[1], cols


def jacobian(
    frame: tx.Optional[Transformation],
    info: tx.Optional[FrameAxes],
    cols: tx.Sequence[int],
    shape: Shape,
    like: ArrayProtocol,
) -> tx.Optional[ArrayProtocol]:
    """Return the Jacobian of a frame at the voxels.

    Parameters
    ----------
    frame : Transformation, optional
        A map from the array system to a frame. `None` is the identity.
    info : FrameAxes, optional
        The axes of the frame, from [`frame_axes`][].
    cols : sequence of int
        The axes of the array along which the frame is differentiated.
    shape : tuple of int
        The shape of the data, with the content axes reduced to 1.
    like : ArrayProtocol
        An array in the backend of the values.

    Returns
    -------
    ArrayProtocol or None
        `None` for the identity, a NumPy matrix of shape `(n, n)` for an
        affine frame, and otherwise an array of shape `(*shape, n, n)` in
        the backend of `like`.
    """
    if frame is None:
        return None
    rows = info.rows
    if info.affine:
        return info.linear[np.ix_(rows, cols)]
    # The frame is not affine, so it is evaluated at the voxels and
    # differentiated along each of `cols`, with central differences inside
    # and one-sided differences at the edges. A field cannot be
    # differentiated along an axis of one voxel, so it is taken to be
    # constant along such an axis, and the column of the Jacobian is that of
    # the affine part of the frame.
    grid = CartesianField(
        shape=tuple(shape), input=frame.input, output=frame.input
    )
    with backend(resolve_backend(like, "native")):
        coords = Sequence(transformations=[grid, frame]).compute().field
    xp = get_array_backend(coords)
    columns = []
    for c in cols:
        if shape[c] > 1:
            columns.append(xp.gradient(coords, axis=c)[..., rows])
        else:
            column = xp.asarray(info.linear[rows, c].astype(coords.dtype))
            columns.append(
                xp.broadcast_to(column, coords.shape[:-1] + (len(rows),))
            )
    return xp.stack(columns, axis=-1)


def change_matrix(
    source: tx.Optional[ArrayProtocol],
    target: tx.Optional[ArrayProtocol],
    method: ReframeMethod,
    ndim: int,
) -> ArrayProtocol:
    """Return the matrix that re-expresses values from one frame in another.

    The Jacobian of the change of frame is `D F′ · (D F)⁻¹`, where
    `source` is `D F` and `target` is `D F′`. The method decides what is
    applied: the Jacobian, its rotation, or the result of a callable.
    """
    eye = np.eye(ndim)
    target = eye if target is None else target
    if source is None:
        matrix = target
    else:
        inverse = _batched(_inv, source)
        xp = get_array_backend(inverse)
        matrix = xp.matmul(_backend_of(target, inverse), inverse)
    if method == "jacobian":
        return matrix
    if method == "rotation":
        return _batched(_rotation, matrix)
    if callable(method):
        return method(matrix)
    raise ValueError(
        f"The method of a change of frame is 'jacobian', 'rotation' or a "
        f"callable, not {method!r}."
    )


def _backend_of(
    matrix: ArrayProtocol, reference: ArrayProtocol
) -> ArrayProtocol:
    """Return a NumPy matrix in the backend of `reference`."""
    if isinstance(matrix, np.ndarray) and not isinstance(
        reference, np.ndarray
    ):
        return get_array_backend(reference).asarray(matrix)
    return matrix


def _rotation(matrix: ArrayProtocol) -> ArrayProtocol:
    """Return the rotation of the polar decomposition of stacked matrices.

    A matrix `J = U S Vᵀ` has the polar decomposition `J = R P` with
    `R = U Vᵀ`. `R` is orthogonal, and it includes a reflection when `J`
    has a negative determinant.
    """
    xp = get_array_backend(matrix)
    u, _, vt = xp.linalg.svd(matrix)
    return xp.matmul(u, vt)


def squeeze_axis(matrix: ArrayProtocol, axis: int, ndim: int) -> ArrayProtocol:
    """Remove a content axis of size 1 from the batch of a matrix field.

    A constant matrix of shape `(n, n)` is returned as it is.
    """
    if matrix.ndim == 2:
        return matrix
    index = tuple(0 if i == axis else slice(None) for i in range(ndim))
    return matrix[index]
