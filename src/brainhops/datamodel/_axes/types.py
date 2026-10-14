__all__ = [
    "SpaceAxis",
    "TimeAxis",
    "ChannelAxis",
    "DisplacementAxis",
    "CoordinateAxis",
    "OrientedAxis",
    "OrientedTimeAxis",
    "OrientedSpaceAxis",
]

import typing_extensions as tx
from bagof.magic import Inherit, NoRepr

from ..units import IndexUnit, SpaceUnit, TimeUnit
from .base import Axis
from .dispatch import is_not_none


class SpaceAxis(Axis, on={"type": "space"}):
    """Axis that measures a spatial dimension.

    The unit is a space unit, an index unit, or `None`. Calling `Axis`
    builds a `SpaceAxis` whenever the type is `"space"`, whatever the
    unit, so `Axis(type="space", unit="s")` fails rather than building a
    generic `Axis`.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"x"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    unit: Inherit[tx.Optional[tx.Union[SpaceUnit, IndexUnit]]]
    type: NoRepr[...]


class TimeAxis(Axis, on={"type": "time"}):
    """Axis that measures time, in a time unit or an index unit.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"t"`.
    type : {"time"}, default="time"
        Pinned axis type.
    unit : TimeUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    unit: Inherit[tx.Optional[tx.Union[TimeUnit, IndexUnit]]]
    type: NoRepr[...]


class ChannelAxis(Axis, on={"type": "channel"}):
    """Axis that enumerates channels, such as colours or features.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"channel"}, default="channel"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class DisplacementAxis(Axis, on={"type": "displacement"}):
    """Axis that enumerates the components of displacement vectors.

    A displacement field has its spatial axes and one displacement axis.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"displacement"}, default="displacement"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class CoordinateAxis(Axis, on={"type": "coordinate"}):
    """Axis that enumerates the components of coordinate vectors.

    A coordinate field has its spatial axes and one coordinate axis.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"coordinate"}, default="coordinate"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class OrientedAxis(Axis, on={"orientation": is_not_none}):
    """Axis that carries an orientation.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"x"`.
    type : AxisType | str, optional
        Type of the axis, such as `"space"` or `"time"`.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation, required
        Meaningful direction of the axis, such as left to right.
    """


# Each of these classes derives from two registered classes, so it is
# selected when the conditions of both parents hold, without an `on=`
# condition of its own.


class OrientedSpaceAxis(SpaceAxis, OrientedAxis):
    """Spatial axis with an orientation.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"x"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation, required
        Meaningful direction of the axis, such as left to right.
    """


class OrientedTimeAxis(TimeAxis, OrientedAxis):
    """Time axis with an orientation.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"t"`.
    type : {"time"}, default="time"
        Pinned axis type.
    unit : TimeUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation, required
        Meaningful direction of the axis, such as forward.
    """
