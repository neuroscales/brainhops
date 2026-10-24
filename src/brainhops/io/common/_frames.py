"""Values of vector and tensor axes re-expressed in the frame of a format.

An image whose axes hold the components of vectors or tensors records the
frame of these components (see
[`SingleScaleImage.frames`][brainhops.datamodel.images.SingleScaleImage.frames]).
A file format stores the components in a frame of its own, such as the
world of the affine that a NIfTI file writes, so a writer re-expresses
the values in that frame before it encodes them. The rules are these:

- a change of frame through affine maps is exact, and it is part of the
  encoding of the format, so it is applied silently;
- a change of frame through a map that is not affine is computed by
  finite differences, so the written values are approximations;
- an axis whose frame is foreign, or whose type the format cannot store
  in a frame of its own, is written with its values as they are, and the
  file loses what the values are;
- the coordinates of points in another frame than that of the format
  are written as they are, because re-expressing them needs the inverse
  of a map, which is not computed.

The last three cases lose information about the data, so they are refused
unless `lossy=True` is given, as a lossy conversion of a transformation
is. Readers never re-express values: they record the frame of the file.
Values whose frame is the very frame in which the format stores them,
even a foreign one that a reader decoded, are written as they are
without a loss.
"""

__all__ = ["ARRAY_SYSTEM", "restricted_frame", "values_in_format_frames"]

import typing_extensions as tx

from brainhops.datamodel._frames import (
    FrameSlot,
    _free_rows,
    _structure,
    array_axes,
    component_axes,
    component_count,
    explicit_frame,
    is_point_axis,
)
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Projection, Transformation
from brainhops.errors import LossyConversionError


class _ArraySystem:
    """The type of [`ARRAY_SYSTEM`][]."""

    def __repr__(self) -> str:
        return "ARRAY_SYSTEM"


ARRAY_SYSTEM = _ArraySystem()
"""The marker of the array system, as the frame of a format.

A frame slot of `None` already means the array system, and the callable
of [`FormatFrame`][] returns `None` when the format cannot store the
values, so it returns this marker when the format stores them in the
array system, as a file without geometry does.
"""

FormatFrame: tx.TypeAlias = tx.Callable[
    [int, Axis], tx.Union[FrameSlot, _ArraySystem, None]
]
"""The frame in which a format stores the values along an axis.

The callable receives the index of a component axis and the axis, and
returns the transformation from the array system to the frame of the
format, [`ARRAY_SYSTEM`][] when the format stores the values in the
array system, or `None` when the format cannot store these values in a
frame. It may also return the very slot of the axis, even a foreign
coordinate system, to say that the format stores the values in that
frame as they are, as for values that a reader decoded in a frame of its
own. A slot of `None` is the array system, so the marker says the same
for it.
"""


FormatCaveat: tx.TypeAlias = tx.Callable[[int, FrameSlot], tx.Optional[str]]
"""What a format assumes when it re-expresses the values along an axis.

Some changes of frame are exact computations that rest on an assumption
about the data, such as an identification of two coordinate systems.
The callable receives the index of a component axis and the frame of
its values, and returns a clause that states the assumption, which
completes "the values along axis ...", or `None` when the change
assumes nothing.
"""


def values_in_format_frames(
    image: SingleScaleImage,
    frame_of: FormatFrame,
    lossy: bool = False,
    caveat: tx.Optional[FormatCaveat] = None,
) -> tx.Optional[tx.Tuple[tx.Any, tx.Tuple[FrameSlot, ...]]]:
    """Return the values of an image in the frames of a format.

    Parameters
    ----------
    image : SingleScaleImage
        The image to write.
    frame_of : callable
        The frame in which the format stores the values along an axis,
        as described by [`FormatFrame`][].
    lossy : bool, default=False
        Whether the values may be written approximately, or without the
        frame that says what they are.
    caveat : callable, optional
        What a change of frame assumes that the values do not tell, as
        described by [`FormatCaveat`][]. Such a change is refused as a
        loss unless `lossy` is true.

    Returns
    -------
    tuple of (array, tuple of slots) or None
        The data with its values re-expressed, and the new frames, or
        `None` when no value changes frame.

    Raises
    ------
    LossyConversionError
        If the format cannot store the frame of some values, or stores
        them only approximately, and `lossy` is false.
    """
    axes = image._array_axes()
    components = component_axes(axes)
    if not components:
        return None
    slots = image.frames
    targets: tx.Dict[int, tx.Optional[Transformation]] = {}
    losses = []
    for i in components:
        name = repr(axes[i].name or i)
        slot = slots[i]
        target = frame_of(i, axes[i])
        # The marker stands for the array system, whose slot is `None`,
        # which `None` as a result cannot say, since it means that the
        # format cannot store the values.
        array = target is ARRAY_SYSTEM
        if array:
            target = None
        if (array or target is not None) and slot is target:
            continue
        storable = array or isinstance(target, Transformation)
        if not storable or isinstance(slot, CoordinateSystem):
            losses.append(
                f"the format cannot store the frame of the values along "
                f"axis {name}, which would be written as they are"
            )
            continue
        if is_point_axis(axes[i]):
            # Re-expressing points needs the inverse of a map, which
            # `reframe` does not compute.
            losses.append(
                f"the coordinates of the points along axis {name} cannot "
                f"be re-expressed in the frame of the format, and would be "
                f"written as they are"
            )
            continue
        note = None if caveat is None else caveat(i, slot)
        if note is not None:
            losses.append(f"the values along axis {name} {note}")
        if not _is_affine(slot, len(axes)) or not _is_affine(
            target, len(axes)
        ):
            losses.append(
                f"the values along axis {name} would be re-expressed "
                f"through a map that is not affine, by finite differences"
            )
        targets[i] = target
    if losses and not lossy:
        raise LossyConversionError(
            f"Writing this image loses information about its values: "
            f"{'; '.join(losses)}. Pass lossy=True to write it anyway."
        )
    if not targets:
        return None
    reframed = SingleScaleImage.reframe(image, targets)
    return reframed.data, reframed.frames


def restricted_frame(
    frame: Transformation,
    axes: tx.Sequence[Axis],
    index: int,
    size: int,
) -> FrameSlot:
    """Restrict the world of a file to the components of an axis.

    The world of a file such as a NIfTI or a NRRD file has spatial axes,
    which the components of a vector follow, and may have a time axis,
    which they do not. The frame is restricted to the spatial axes as
    `SingleScaleImage.frames` restricts it, by composing a projection
    that removes the other axes. A field of fewer components than spatial
    axes, such as a two-dimensional field, holds the components along the
    first spatial axes, as ITK reads it. A frame that has fewer axes than
    there are components cannot hold them, and its output system is
    returned instead, as a foreign frame.

    Parameters
    ----------
    frame : Transformation
        The map from the array system to the frame of the file.
    axes : sequence of Axis
        The axes of the array system.
    index : int
        The index of the component axis.
    size : int
        The number of entries along the component axis.

    Returns
    -------
    Transformation or CoordinateSystem
        The restricted frame, or the foreign frame.
    """
    try:
        count = component_count(axes, index, size, [frame])
        return explicit_frame(frame, axes, count)
    except ValueError:
        pass
    structure = _structure(frame, len(axes))
    outputs = None
    if structure is not None:
        outputs = array_axes(frame.output, structure.outputs.shape[0])
    if outputs is None:
        return frame.output
    free = _free_rows(structure.outputs, component_axes(axes))
    try:
        count = axes[index].frame_ndim(size)
    except ValueError:
        return frame.output
    if len(free) < count:
        return frame.output
    dropped = free[count:]
    system = CoordinateSystem(
        [axis for r, axis in enumerate(outputs) if r not in dropped],
        name=getattr(frame.output, "name", None),
    )
    projection = Projection(dropped=dropped, input=frame.output, output=system)
    return projection @ frame


def _is_affine(slot: tx.Optional[Transformation], ndim: int) -> bool:
    """Return whether a frame is the array system or an affine map.

    A frame may end with projections and permutations, which select its
    outputs, so the frame is read through its structure, as `reframe`
    reads it, rather than converted to a single affine.
    """
    if slot is None:
        return True
    structure = _structure(slot, ndim)
    return structure is not None and structure.affine
