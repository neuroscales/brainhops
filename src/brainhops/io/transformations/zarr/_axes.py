"""Map OME-Zarr axes onto brainhops axis types."""

# stdlib
from collections.abc import Mapping

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    SpatialAxis,
    TimeAxis,
)

_AXIS_TYPES = {
    "space": SpatialAxis,
    "time": TimeAxis,
    "channel": ChannelAxis,
    "displacement": DisplacementAxis,
    "coordinate": CoordinateAxis,
}


def _to_axis(ome_axis: tx.Any) -> Axis:
    # Build a brainhops `Axis` from one OME-Zarr axis. The OME axis is
    # either a mapping or an object with `type`, `name`, `unit` and
    # `discrete` attributes. A recognized OME axis type selects the
    # matching `Axis` subclass, which fixes the type; any other type is
    # carried on a plain `Axis`. The remaining attributes are copied when
    # they are present.
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
