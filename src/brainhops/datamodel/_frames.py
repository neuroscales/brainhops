"""Frames of the values of images, and the changes of frame.

An image whose axes hold the components of points, vectors or tensors
records, for each data axis, the frame in which these components are
expressed (see `spec/design/vector-and-tensor-images.md`). This module
holds the helpers that read, validate, compose and change these frames.
The public interface is
[`SingleScaleImage.frames`][brainhops.datamodel.images.SingleScaleImage.frames]
and
[`SingleScaleImage.reframe`][brainhops.datamodel.images.SingleScaleImage.reframe].
"""

from collections.abc import Mapping

import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core.backends import array_backend, backend, resolve_backend
from brainhops._core.typing import ArrayProtocol, Shape

from ._axes.components import _batched, _inv, is_component_type
from ._sugar import get_axes
from .axes import Axis, AxisSequence
from .enums import ReframeMethod, ReframeMethodLike
from .systems import CoordinateSystem
from .transformations import (
    CartesianField,
    Permutation,
    Projection,
    Sequence,
    Transformation,
)

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
"""One frame for every component axis, or a frame for each named axis."""

ReframeMethodOrCallable: tx.TypeAlias = tx.Union[
    ReframeMethodLike,
    tx.Callable[[ArrayProtocol], ArrayProtocol],
]
"""What is applied to the values when their frame changes.

A [`ReframeMethod`][brainhops.datamodel.enums.ReframeMethod] or its
name, or a callable that receives the Jacobian of the change of frame
and returns the matrix to apply.
"""


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


def component_axes(axes: tx.Optional[tx.Sequence[Axis]]) -> tx.List[int]:
    """Return the indices of the component axes among `axes`."""
    return [
        i
        for i, axis in enumerate(axes or ())
        if is_component_type(getattr(axis, "type", None))
    ]


def is_point_axis(axis: Axis) -> bool:
    """Return whether the entries of an axis are coordinates of points."""
    return getattr(axis, "valences", None) is None


# ----------------------------------------------------------------------
#   READING THE FRAMES
# ----------------------------------------------------------------------


def normalize_frames(
    raw: tx.Any, axes: tx.Optional[tx.Sequence[Axis]], shape: Shape
) -> tx.Tuple[FrameSlot, ...]:
    """Return one frame slot per data axis.

    A transformation that has more axes than the components of the
    values is restricted to the axes that the components follow, as
    [`explicit_frame`][] does, so that every slot that is returned
    corresponds exactly to its components.

    Parameters
    ----------
    raw : tuple or mapping or None
        The frames as they were given: an empty tuple, a tuple of one slot
        per data axis, or a mapping from axis names or indices to slots.
    axes : sequence of Axis, optional
        The axes of the array system, or `None` if they are unknown.
    shape : tuple of int
        The shape of the data.

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
        If the tuple does not have one slot per data axis, if a slot is
        set on an axis that is not a component axis, or if a
        transformation does not tell which of its axes the components
        follow.
    KeyError
        If a mapping names an axis that does not exist.
    """
    ndim = len(shape)
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
    components = set(component_axes(axes))
    for i, slot in enumerate(slots):
        if slot is None:
            continue
        if not isinstance(slot, (Transformation, CoordinateSystem)):
            raise TypeError(
                f"The frame of axis {i} must be a transformation, a "
                f"coordinate system or None, not a {type(slot).__name__}."
            )
        if i not in components:
            raise ValueError(
                f"Axis {i} is not a component axis, so its values have no "
                f"frame. Give the axis a type such as 'vector' before "
                f"setting its frame."
            )
        if isinstance(slot, Transformation):
            ncomponents = component_count(axes, i, shape[i], [slot])
            slots[i] = explicit_frame(slot, axes, ncomponents)
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


def with_components(
    system: tx.Optional[CoordinateSystem],
    source: tx.Optional[CoordinateSystem],
    source_axes: tx.Sequence[Axis],
    ndim: int,
) -> tx.Optional[CoordinateSystem]:
    """Return an array system that has the component axes of another.

    When an image is resliced, the array system of the result is that of
    the target geometry, which may describe its components as channels.
    Each component axis of `source_axes` replaces the axis at the same
    position in `system`, and keeps the name of that axis when it has
    one. When `system` is unknown, `source` is used instead. The system
    is returned as it is when no axis changes.
    """
    axes = array_axes(system, ndim)
    if axes is None:
        return source
    new_axes = list(axes)
    for i in component_axes(source_axes):
        axis = source_axes[i]
        name = getattr(new_axes[i], "name", None)
        if name is not None and name != axis.name:
            axis = replace(axis, name=name)
        if new_axes[i] != axis:
            new_axes[i] = axis
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
    shape: Shape,
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
    shape : tuple of int
        The shape of the new image.

    Returns
    -------
    tuple of slots
        One slot per axis of the new image. A transformation is composed
        with `voxel_map`, and a foreign frame is kept. The array system
        stays the array system when `voxel_map` does not change the
        components, and becomes `voxel_map` otherwise, restricted to the
        axes that the components follow when the types of its axes tell
        them.
    """
    components = set(component_axes(new_axes))
    unchanged: tx.Dict[bool, bool] = {}
    out: tx.List[FrameSlot] = []
    for i, old in enumerate(keep):
        slot = None if old is None else slots[old]
        if i not in components:
            out.append(None)
        elif isinstance(slot, Transformation):
            out.append(slot @ voxel_map)
        elif slot is None:
            points = is_point_axis(new_axes[i])
            if points not in unchanged:
                unchanged[points] = _keeps_values(voxel_map, points)
            if unchanged[points]:
                out.append(None)
                continue
            # The components of the array system follow the axes that the
            # other frame of a change selects, so the map is restricted
            # only when the types of its axes tell these axes.
            try:
                ncomponents = component_count(
                    new_axes, i, shape[i], [voxel_map]
                )
                out.append(explicit_frame(voxel_map, new_axes, ncomponents))
            except ValueError:
                out.append(voxel_map)
        else:
            out.append(slot)
    return tuple(out)


def _keeps_values(voxel_map: Transformation, points: bool) -> bool:
    """Return whether a map leaves the values in the array system alone.

    The voxel components of a vector do not change when the map is affine
    and its linear part is the identity on the axes that are not
    component axes, as for a crop or a translation. The voxel
    coordinates of a point also need the map to have no translation.
    Composing the array system with the map would also be correct, so a
    map that this function cannot read returns `False`.
    """
    affine = voxel_map.to_affine(error=None)
    if affine is None or affine.matrix is None:
        return False
    matrix = np.asarray(affine.matrix, dtype=float)
    linear = matrix[:, :-1]
    inputs = array_axes(voxel_map.input, linear.shape[1])
    outputs = array_axes(voxel_map.output, linear.shape[0])
    if inputs is None or outputs is None:
        return False
    cols = [
        c for c in range(linear.shape[1]) if c not in component_axes(inputs)
    ]
    rows = [
        r for r in range(linear.shape[0]) if r not in component_axes(outputs)
    ]
    if len(rows) != len(cols):
        return False
    block = linear[np.ix_(rows, cols)]
    if not np.array_equal(block, np.eye(len(rows))):
        return False
    return not points or not np.any(matrix[rows, -1])


# ----------------------------------------------------------------------
#   THE STRUCTURE OF A FRAME
# ----------------------------------------------------------------------
# The components of a vector follow some axes of the system of its frame,
# in order. These axes, and the axes of the array along which the frame is
# differentiated, are read from the structure of the frame rather than from
# the types of the axes, so that the rules do not assume that the vectors
# are spatial. A frame may end with projections and permutations, which
# select and reorder the axes of the map before them.


class _Structure(tx.NamedTuple):
    """The linear structure of a frame."""

    head: tx.Optional[Transformation]
    """The frame without its final projections and permutations."""

    linear: np.ndarray
    """The linear part of `head`, or of its affine part."""

    affine: bool
    """Whether `head` is affine, so that `linear` is its Jacobian."""

    select: np.ndarray
    """The matrix of the final projections and permutations."""

    @property
    def outputs(self) -> np.ndarray:
        """The linear part of the whole frame."""
        return self.select @ self.linear


def _structure(frame: Transformation, ndim: int) -> tx.Optional[_Structure]:
    """Return the linear structure of a frame from an array of `ndim` axes.

    Every field of the frame is taken as the identity. `None` is
    returned when the frame is not affine even then.
    """
    parts = (
        list(frame.transformations or [])
        if isinstance(frame, Sequence)
        else [frame]
    )
    tail: tx.List[Transformation] = []
    while parts and isinstance(parts[-1], (Projection, Permutation)):
        tail.insert(0, parts.pop())
    if not parts:
        head = None
        linear = np.eye(ndim)
        exact = True
    else:
        head = parts[0] if len(parts) == 1 else Sequence(transformations=parts)
        affine = head.to_affine(error=None)
        exact = affine is not None and affine.matrix is not None
        if affine is None or affine.matrix is None:
            affine = head.to_affine(ignore=True, error=None)
        if affine is None or affine.matrix is None:
            return None
        linear = np.asarray(affine.matrix, dtype=float)[:, :-1]
    select = np.eye(linear.shape[0])
    for part in tail:
        select = _selection(part, select.shape[0]) @ select
    return _Structure(head, linear, exact, select)


def _selection(part: Transformation, ninputs: int) -> np.ndarray:
    """Return the matrix of a projection or a permutation."""
    if isinstance(part, Permutation):
        permutation = part.permutation
        if permutation is None:
            return np.eye(ninputs)
        return np.eye(ninputs)[np.asarray(permutation, dtype=int)]
    dropped = {int(i) for i in np.asarray(part.dropped).ravel()}
    created = {int(i) for i in np.asarray(part.created).ravel()}
    kept = iter(i for i in range(ninputs) if i not in dropped)
    noutputs = ninputs - len(dropped) + len(created)
    matrix = np.zeros((noutputs, ninputs))
    for row in range(noutputs):
        if row not in created:
            matrix[row, next(kept)] = 1.0
    return matrix


def _coupled(linear: np.ndarray) -> np.ndarray:
    """Return which entries of a linear map are not zero.

    An entry is taken as zero when it is negligible next to the largest
    entry, so that a product of matrices does not couple axes by rounding.
    """
    magnitude = np.abs(linear)
    scale = magnitude.max() if magnitude.size else 0.0
    return magnitude > 1e-12 * (scale or 1.0)


def _free_rows(linear: np.ndarray, components: tx.List[int]) -> tx.List[int]:
    """Return the outputs of a map that no component axis feeds."""
    coupled = _coupled(linear)
    return [
        r for r in range(linear.shape[0]) if not coupled[r, components].any()
    ]


def component_count(
    axes: tx.Sequence[Axis],
    index: int,
    size: int,
    frames: tx.Sequence[FrameSlot],
) -> int:
    """Return the number of axes of the frame that a component axis follows.

    The number of entries alone may not tell the size of the frame of a
    tensor, as when 36 entries are a full matrix of six axes or a
    symmetric matrix of eight axes, or when an explicit layout leaves
    the last axes of the frame out. The frames that are given are tried
    in order: the number of their outputs that no component axis feeds
    is the size of the frame when it agrees with the number of entries.
    The number of entries alone decides when no frame agrees.

    Raises
    ------
    ValueError
        If no frame agrees with the number of entries, and the number of
        entries alone does not tell the size of the frame.
    """
    axis = axes[index]
    components = component_axes(axes)
    for frame in frames:
        if not isinstance(frame, Transformation):
            continue
        structure = _structure(frame, len(axes))
        if structure is None:
            continue
        ndim = len(_free_rows(structure.outputs, components))
        try:
            return axis.frame_ndim(size, ndim)
        except ValueError:
            continue
    return axis.frame_ndim(size)


def explicit_frame(
    frame: Transformation, axes: tx.Sequence[Axis], ncomponents: int
) -> Transformation:
    """Restrict a frame to the axes that the components follow.

    A frame that is given by a user, such as the preferred transformation
    of an image whose system also has a time axis, may have more outputs
    than the components of the values, besides the outputs that the
    component axes of the array feed. The components then follow the
    outputs that share one axis type, provided that exactly one type has
    as many outputs as there are components. A projection that removes
    the other outputs is composed onto the frame, so that the slot that
    is stored corresponds exactly to the components. A frame that already
    corresponds, or whose structure cannot be read, is returned as it is.

    Raises
    ------
    ValueError
        If the frame has fewer outputs than components, or if the types
        of its outputs do not tell which of them the components follow.
    """
    structure = _structure(frame, len(axes))
    if structure is None:
        return frame
    final = structure.outputs
    candidates = _free_rows(final, component_axes(axes))
    if len(candidates) == ncomponents:
        return frame
    if len(candidates) > ncomponents:
        outputs = array_axes(frame.output, final.shape[0])
        groups: tx.Dict[tx.Any, tx.List[int]] = {}
        for r in candidates:
            kind = getattr(outputs[r], "type", None) if outputs else None
            if kind is not None:
                groups.setdefault(kind, []).append(r)
        matching = [g for g in groups.values() if len(g) == ncomponents]
        if len(matching) == 1:
            dropped = [r for r in candidates if r not in matching[0]]
            system = CoordinateSystem(
                [a for r, a in enumerate(outputs) if r not in dropped],
                name=getattr(frame.output, "name", None),
            )
            projection = Projection(
                dropped=dropped, input=frame.output, output=system
            )
            return projection @ frame
    raise ValueError(
        f"The frame has {len(candidates)} axes that no component axis of "
        f"the array feeds, and nothing tells which {ncomponents} of them "
        f"the components of the values follow. A frame whose system has "
        f"exactly {ncomponents} such axes, for example one that ends with "
        f"a projection, removes the ambiguity."
    )


class FrameAxes(tx.NamedTuple):
    """The axes that relate the components of a vector to a frame."""

    rows: tx.List[int]
    """The outputs of `head` that the components follow, in order."""

    cols: tx.List[int]
    """The axes of the array that the frame maps onto `rows`."""

    linear: np.ndarray
    """The linear part of `head`, or of its affine part."""

    affine: bool
    """Whether `head` is affine, so that `linear` is its Jacobian."""

    head: tx.Optional[Transformation]
    """The frame without its final projections and permutations."""


def frame_axes(
    frame: Transformation, axes: tx.Sequence[Axis], ncomponents: int
) -> FrameAxes:
    """Return the axes that relate `ncomponents` components to a frame.

    The structure of the frame is read from its linear part, or from the
    linear part of its affine part, in which every field is taken as the
    identity. An output of the frame is coupled to an input when the
    entry between them is not zero.

    - The outputs of the frame that no component axis of the array feeds
      must be exactly as many as the components, which follow them in
      order. No axis type is consulted.
    - The axes of the array that the frame maps onto them are the inputs,
      other than component axes, that are coupled to these outputs and to
      no other output that no component axis feeds. When the frame ends
      with projections and permutations, the outputs that the projections
      remove count among these other outputs, so that a time axis that
      the frame carries along is left out.

    Raises
    ------
    ValueError
        If the frame is not affine even when its fields are ignored, or if
        it does not correspond exactly to the components.
    """
    structure = _structure(frame, len(axes))
    if structure is None:
        raise ValueError(
            "The frame is not affine even when its fields are ignored, so "
            "the axes that the components of the values follow cannot be "
            "read from it."
        )
    components = component_axes(axes)
    final = _free_rows(structure.outputs, components)
    if len(final) != ncomponents:
        raise ValueError(
            f"The frame has {len(final)} axes that no component axis of the "
            f"array feeds, but the values have {ncomponents} components. "
            f"The frame of the values must have exactly one such axis per "
            f"component."
        )
    rows = []
    for r in final:
        source = np.flatnonzero(structure.select[r])
        if len(source) != 1:
            raise ValueError(
                "The frame ends with an embedding that creates an axis "
                "that the components follow, so the frame cannot be "
                "inverted."
            )
        rows.append(int(source[0]))
    linear = structure.linear
    coupled = _coupled(linear)
    others = [r for r in _free_rows(linear, components) if r not in rows]
    cols = [
        c
        for c in range(linear.shape[1])
        if c not in components
        and coupled[rows, c].any()
        and not coupled[others, c].any()
    ]
    if len(cols) != ncomponents:
        raise ValueError(
            f"The frame maps {len(cols)} axes of the array onto the "
            f"{ncomponents} axes that the components follow, so it does not "
            f"define a change of frame for them."
        )
    return FrameAxes(rows, cols, linear, structure.affine, structure.head)


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
        The shape of the data, with the component axes reduced to 1.
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
    # the affine part of the frame. The final projections and permutations
    # of the frame only select its outputs, which `rows` already does.
    head = info.head
    grid = CartesianField(
        shape=tuple(shape), input=head.input, output=head.input
    )
    with backend(resolve_backend(like, "native")):
        coords = Sequence(transformations=[grid, head]).compute().field
    xp = array_backend(coords)
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
    method: ReframeMethodOrCallable,
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
        xp = array_backend(inverse)
        matrix = xp.matmul(_backend_of(target, inverse), inverse)
    if method == ReframeMethod.jacobian:
        return matrix
    if method == ReframeMethod.rotation:
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
        return array_backend(reference).asarray(matrix)
    return matrix


def _rotation(matrix: ArrayProtocol) -> ArrayProtocol:
    """Return the rotation of the polar decomposition of stacked matrices.

    A matrix `J = U S Vᵀ` has the polar decomposition `J = R P` with
    `R = U Vᵀ`. `R` is orthogonal, and it includes a reflection when `J`
    has a negative determinant.
    """
    xp = array_backend(matrix)
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
