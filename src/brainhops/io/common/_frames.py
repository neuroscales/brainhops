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
  file loses what the values are.

The last two cases lose information about the data, so they are refused
unless `lossy=True` is given, as a lossy conversion of a transformation
is. Readers never re-express values: they record the frame of the file.
"""

__all__ = ["values_in_format_frames"]

import typing_extensions as tx

from brainhops.datamodel._frames import (
    FrameSlot,
    _structure,
    component_axes,
    same_frame,
)
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Transformation
from brainhops.errors import LossyConversionError

FormatFrame: tx.TypeAlias = tx.Callable[
    [int, Axis], tx.Optional[Transformation]
]
"""The frame in which a format stores the values along an axis.

The callable receives the index of a component axis and the axis, and
returns the transformation from the array system to the frame of the
format, or `None` when the format cannot store these values in a frame.
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
    targets: tx.Dict[int, Transformation] = {}
    losses = []
    for i in components:
        name = repr(axes[i].name or i)
        slot = slots[i]
        target = frame_of(i, axes[i])
        if target is None or isinstance(slot, CoordinateSystem):
            losses.append(
                f"the format cannot store the frame of the values along "
                f"axis {name}, which would be written as they are"
            )
            continue
        if same_frame(slot, target):
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
