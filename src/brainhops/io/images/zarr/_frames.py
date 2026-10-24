"""The frames in which OME-Zarr stores the components of vectors.

An axis of an OME-Zarr image may hold, at each voxel, the components
of a vector or a tensor. The reader records the coordinate system in
which they are expressed, called their frame, for each level, and the
writer re-expresses the values of each level in the frame that the
store states for it. The section "Vectors and tensors" of
[`brainhops.io.images.zarr`][] describes these frames. This module
builds them.

The frame of a level is the map from its voxels to the system onto
which the levels map, followed by a permutation that puts the axes of
that system in the order of the store, because brainhops sorts the
axes of a system into its own order. The frame is then restricted to
the axes that the components follow, as
[`SingleScaleImage.frames`][brainhops.datamodel.images.SingleScaleImage.frames]
restricts the frames that a user gives, so that a time axis is left
out.
"""

# The design is recorded in spec/design/vector-and-tensor-images.md,
# sections 7 and 9.

import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel._frames import (
    FrameSlot,
    array_axes,
    component_axes,
    component_count,
    explicit_frame,
    is_point_axis,
)
from brainhops.datamodel.axes import Axis, is_component_type
from brainhops.datamodel.enums import AxisType
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Permutation,
    Transformation,
)

DecodedFrames: tx.TypeAlias = tx.Tuple[
    Transformation, tx.Mapping[int, FrameSlot]
]
"""The geometry of a level as it was read, and the frames it implies."""


def stored_order_frame(
    geometry: Transformation,
    axes: tx.Sequence[Axis],
    storage: tx.Sequence[int],
) -> Transformation:
    """Return the map from the voxels to a system in the stored order.

    brainhops sorts the axes of a coordinate system into its own order,
    with the spatial axes first, as x, y and z. OME-Zarr lists the
    components of a vector in the order in which it lists the axes,
    such as z, y and x. The frame of the components is therefore the
    geometry of the level, followed by the permutation that puts the
    axes of its output system back in the order of the store.

    Parameters
    ----------
    geometry : Transformation
        The map from the voxels of a level to the system onto which
        the levels map, in the brainhops order.
    axes : sequence of Axis
        The axes of that system, in the brainhops order.
    storage : sequence of int
        The permutation into the order of the store: element `i` is
        the position, in the brainhops order, of stored axis `i`.

    Returns
    -------
    Transformation
        The geometry, followed by the permutation.
    """
    output = geometry.output
    stored = CoordinateSystem(
        [axes[p] for p in storage], name=getattr(output, "name", None)
    )
    permutation = Permutation(
        permutation=list(storage), input=output, output=stored
    )
    return permutation @ geometry


def component_frames(
    frame: Transformation,
    axes: tx.Sequence[Axis],
    shape: tx.Sequence[int],
) -> tx.Dict[int, tx.Optional[Transformation]]:
    """Restrict a frame to the components of each component axis.

    The components follow the outputs of the frame that no component
    axis feeds. When there are more of these outputs than components,
    the frame is restricted to the outputs of one type, as
    [`explicit_frame`][brainhops.datamodel._frames.explicit_frame]
    does, so that a time axis is left out.

    Parameters
    ----------
    frame : Transformation
        The map from the array system to the frame of the format.
    axes : sequence of Axis
        The axes of the array system.
    shape : sequence of int
        The shape of the data.

    Returns
    -------
    dict of int to Transformation or None
        The restricted frame of each component axis, or `None` when
        the number of components does not match the axes of the frame.
    """
    frames: tx.Dict[int, tx.Optional[Transformation]] = {}
    for index in component_axes(axes):
        try:
            count = component_count(axes, index, shape[index], [frame])
            frames[index] = explicit_frame(frame, axes, count)
        except ValueError:
            frames[index] = None
    return frames


def level_frames(
    geometry: Transformation,
    axes: tx.Sequence[Axis],
    shape: tx.Sequence[int],
    storage: tx.Sequence[int],
    foreign: CoordinateSystem,
) -> tx.Dict[int, FrameSlot]:
    """Return the frame of each component axis of a level that is read.

    The coordinates of a coordinate field are in the foreign frame
    `foreign`. The components of the other component axes are in the
    system onto which the level maps, in the order of the store. When
    the number of components of an axis does not match the axes of
    that system, the store states no frame for them, so their frame is
    that system, as a foreign frame.

    Parameters
    ----------
    geometry : Transformation
        The map from the voxels of the level to the system onto which
        the levels map.
    axes : sequence of Axis
        The axes of the array system of the level.
    shape : sequence of int
        The shape of the data of the level.
    storage : sequence of int
        The permutation into the order of the store, as in
        [`stored_order_frame`][].
    foreign : CoordinateSystem
        The frame of the coordinates of a coordinate field.

    Returns
    -------
    dict of int to slot
        The frame of each component axis.
    """
    output = array_axes(geometry.output, len(axes))
    if output is None:
        return {i: geometry.output for i in component_axes(axes)}
    frame = stored_order_frame(geometry, output, storage)
    frames: tx.Dict[int, FrameSlot] = {}
    for index, slot in component_frames(frame, axes, shape).items():
        if axes[index].type == AxisType.coordinate:
            frames[index] = foreign
        elif slot is None:
            frames[index] = geometry.output
        else:
            frames[index] = slot
    return frames


class OmeFrames:
    """The frame in which OME-Zarr stores the values of a level.

    An instance is the callable that
    [`values_in_format_frames`][brainhops.io.common._frames.values_in_format_frames]
    receives for one level. The frame of a component axis is the map
    from the voxels of the level to the system onto which the written
    levels map, in the order of the store, restricted to the
    components. The callable returns the slot of the axis itself when
    the slot is the frame that the level decoded from its store and the
    geometry of the level is written unchanged, so that the values are
    written as they are. It returns `None`, which the writer reports as
    a loss, for the coordinates of a coordinate field in any other
    frame, since the format cannot state their frame, and for the
    coordinates of points in another frame than that of the format,
    since re-expressing points needs the inverse of a map. It also
    returns `None` for an axis that the store lists with a type that is
    not a component type, such as `channel`, whose values are then
    written as they are.

    Parameters
    ----------
    image : SingleScaleImage
        The level to write.
    geometry : Transformation
        The map from the voxels of the level to the system onto which
        the written levels map.
    axes : sequence of Axis
        The axes of that system, in the brainhops order, as they are
        written.
    storage : sequence of int
        The permutation into the order of the store, as in
        [`stored_order_frame`][].
    decoded : tuple, optional
        The geometry of the level and its frames as they were read, or
        `None` for a level that was not read from a store.
    """

    def __init__(
        self,
        image: tx.Any,
        geometry: Transformation,
        axes: tx.Sequence[Axis],
        storage: tx.Sequence[int],
        decoded: tx.Optional[DecodedFrames] = None,
    ) -> None:
        self.slots = image.frames
        self.axes = list(axes)
        self.own: tx.Dict[int, FrameSlot] = {}
        if decoded is not None and decoded[0] is geometry:
            self.own = dict(decoded[1])
        array = image._array_axes()
        self.targets: tx.Dict[int, tx.Optional[Transformation]] = {}
        if (
            array is not None
            and component_axes(array)
            and len(axes) == len(array)
        ):
            # The store lists the axes that are written, so the frame
            # ends in a system of these axes, whose types tell which
            # outputs the components follow.
            system = CoordinateSystem(
                list(axes), name=getattr(geometry.output, "name", None)
            )
            frame = stored_order_frame(
                replace(geometry, output=system), axes, storage
            )
            shape = tuple(image.shape)
            self.targets = component_frames(frame, array, shape)

    def __call__(self, index: int, axis: Axis) -> tx.Optional[FrameSlot]:
        """Return the frame of the values along an axis, or `None`."""
        slot = self.slots[index]
        if index in self.own and slot is self.own[index]:
            return slot
        written = self.axes[index] if index < len(self.axes) else None
        if not is_component_type(getattr(written, "type", None)):
            # The store lists this axis with another type, such as a
            # channel, so it states no frame, and the values are written
            # as they are.
            return None
        if axis.type == AxisType.coordinate:
            return None
        target = self.targets.get(index)
        if is_point_axis(axis) and slot is not target:
            return None
        return target
