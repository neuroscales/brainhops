"""Conversion of OME-Zarr axes to brainhops axes."""

from collections.abc import Mapping

import typing_extensions as tx

from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    PointAxis,
    SpaceAxis,
    TensorAxis,
    TimeAxis,
    VectorAxis,
)

_AXIS_TYPES = {
    "space": SpaceAxis,
    "time": TimeAxis,
    "channel": ChannelAxis,
    "displacement": DisplacementAxis,
    "coordinate": CoordinateAxis,
    "point": PointAxis,
    "vector": VectorAxis,
    "tensor": TensorAxis,
}


def to_axis(ome_axis: tx.Any) -> Axis:
    """Return the axis that an OME-Zarr axis describes.

    The OME axis is a mapping or an object with attributes. A known OME
    type selects the matching subclass of `Axis`, which fixes the type
    of the axis. An unknown type is stored in the `type` field of a
    plain `Axis`.

    Parameters
    ----------
    ome_axis : Any
        The OME axis, with its `name`, `type`, `unit` and `discrete`
        entries or attributes.

    Returns
    -------
    Axis
        The axis.
    """
    if isinstance(ome_axis, Mapping):
        type_ = ome_axis.get("type")
        name = ome_axis.get("name")
        unit = ome_axis.get("unit")
        discrete = ome_axis.get("discrete")
    else:
        type_ = getattr(ome_axis, "type", None)
        name = getattr(ome_axis, "name", None)
        unit = getattr(ome_axis, "unit", None)
        discrete = getattr(ome_axis, "discrete", None)
    cls = _AXIS_TYPES.get(type_, Axis)
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if unit is not None:
        kwargs["unit"] = unit
    if discrete is not None:
        kwargs["discrete"] = discrete
    if cls is Axis and type_ is not None:
        kwargs["type"] = type_
    return cls(**kwargs)
