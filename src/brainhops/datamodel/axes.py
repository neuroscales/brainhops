"""Axes of a coordinate system, such as a spatial or a time axis."""

__all__ = [
    "Axis",
    "SpaceAxis",
    "TimeAxis",
    "ChannelAxis",
    "DisplacementAxis",
    "CoordinateAxis",
    "R",
    "LR",
    "rightToLeftAxis",
    "RightToLeftAxis",
    "L",
    "RL",
    "leftToRightAxis",
    "LeftToRightAxis",
    "A",
    "PA",
    "anteriorToPosteriorAxis",
    "AnteriorToPosteriorAxis",
    "P",
    "AP",
    "posteriorToAnteriorAxis",
    "PosteriorToAnteriorAxis",
    "S",
    "IS",
    "inferiorToSuperiorAxis",
    "InferiorToSuperiorAxis",
    "I",
    "SI",
    "superiorToInferiorAxis",
    "SuperiorToInferiorAxis",
    "vector_axis",
    "AxisError",
]
# dependencies
import typing_extensions as tx

# core
from brainhops._core.typing import NoRepr

# locals
from .base import DataModelBase
from .enums import AnatomicalOrientationValue, AxisType
from .orientation import (
    AnteriorToPosterior,
    InferiorToSuperior,
    LeftToRight,
    Orientation,
    PosteriorToAnterior,
    RightToLeft,
    SuperiorToInferior,
)
from .units import SampleUnit, SpaceUnit, TimeUnit, Unit

# --- Dispatch helpers -------------------------------------------------


def _is_space_unit(unit: tx.Optional[Unit]) -> bool:
    """Whether `unit` is a unit of space, unspecified, or the sample.

    A spatial axis sampled on a grid measures its coordinates in samples
    (see [`SampleUnit`][]), so the sample unit belongs on a `SpaceAxis` as
    much as a millimetre does -- it says the axis indexes an array rather
    than that it stopped being spatial.
    """
    return unit is None or isinstance(unit, (SpaceUnit, SampleUnit))


def _is_time_unit(unit: tx.Optional[Unit]) -> bool:
    """Whether `unit` is a unit of time, unspecified, or the sample."""
    return unit is None or isinstance(unit, (TimeUnit, SampleUnit))


def _is_not_none(obj: tx.Any) -> bool:
    """Whether `obj` is not `None`."""
    return obj is not None


def _is_anatomical(orientation: tx.Optional[Orientation]) -> bool:
    """Whether `orientation` is an anatomical orientation."""
    return getattr(orientation, "type", None) == "anatomical"


def _has_value(value: str) -> tx.Callable[[tx.Optional[Orientation]], bool]:
    """
    Whether `orientation` is an anatomical orientation with the given value.
    """

    if not isinstance(value, AnatomicalOrientationValue):
        try:
            value = AnatomicalOrientationValue(value)
        except ValueError:
            ...
        try:
            value = AnatomicalOrientationValue[value]
        except KeyError:
            ...

    def _check(orientation: tx.Optional[Orientation]) -> bool:
        return getattr(orientation, "value", None) == value

    return _check


# --- API --------------------------------------------------------------


class Axis(DataModelBase, polymorphic=True):
    """One axis of a coordinate system or of a grid.

    An axis names its type, such as `"space"` or `"time"`, and may carry
    a unit, an orientation, and whether it is discrete.
    """

    name: tx.Optional[str] = None
    """The name of the axis, such as `"x"` or `"y"`."""

    type: tx.Optional[tx.Union[AxisType, str]] = None
    """The type of the axis, such as `"space"` or `"time"`."""

    unit: tx.Optional[Unit] = None
    """The unit in which coordinates are measured along the axis."""

    discrete: tx.Optional[bool] = None
    """
    Whether the axis is discrete, as opposed to continuous.

    Here discrete means that the axis has no particular order, and
    therefore does not come with a continuous coordinates system.
    """

    orientation: tx.Optional[Orientation] = None
    """
    The orientation of the axis, if it has one.

    This is useful to indicate that the direction of coordinates along
    the axis has a particular meaning, such as `"left-to-right"` for an
    anatomical axis.
    """

    def __pre_init__(self, arguments: tx.Any) -> None:
        # `...` is not a placeholder for an axis: a coordinate system
        # with a fixed number of dimensions lists every one of them. As
        # the first positional argument it would otherwise be read as
        # the axis name, and refused with a message about strings.
        if arguments.name is Ellipsis:
            raise TypeError("`...` is not an axis")


class SpaceAxis(Axis, on={"type": "space", "unit": _is_space_unit}):
    """An axis that measures a spatial dimension."""

    unit: tx.Optional[tx.Union[SpaceUnit, SampleUnit]] = None
    type: NoRepr[tx.Literal["space"]] = "space"


class TimeAxis(Axis, on={"type": "time", "unit": _is_time_unit}):
    """An axis that measures time."""

    unit: tx.Optional[tx.Union[TimeUnit, SampleUnit]] = None
    type: NoRepr[tx.Literal["time"]] = "time"


class ChannelAxis(Axis, on={"type": "channel"}):
    """An axis that enumerates channels, such as color or feature channels."""

    type: NoRepr[tx.Literal["channel"]] = "channel"


class DisplacementAxis(Axis, on={"type": "displacement"}):
    """An axis that carries the components of a displacement vector.

    A field of displacements names its spatial axes together with exactly
    one axis of this type. That axis enumerates the displacement
    components stored at each grid point.
    """

    type: NoRepr[tx.Literal["displacement"]] = "displacement"


class CoordinateAxis(Axis, on={"type": "coordinate"}):
    """An axis that carries the components of a coordinate vector.

    A field of coordinates names its spatial axes together with exactly
    one axis of this type. That axis enumerates the coordinate components
    stored at each grid point.
    """

    type: NoRepr[tx.Literal["coordinate"]] = "coordinate"


class OrientedAxis(Axis, on={"orientation": _is_not_none}):
    """An axis that carries an orientation."""


class OrientedTimeAxis(
    TimeAxis, OrientedAxis, on={"orientation": _is_not_none, "type": "time"}
):
    """A time axis that carries an orientation."""


class OrientedSpaceAxis(
    SpaceAxis, OrientedAxis, on={"orientation": _is_not_none, "type": "space"}
):
    """A spatial axis that carries an orientation."""


class AnatomicalAxis(OrientedSpaceAxis, on={"orientation": _is_anatomical}):
    """An axis that carries an anatomical orientation."""


Axis.register_polymorph(AnatomicalAxis, on={"orientation": _is_anatomical})


class LeftToRightAxis(
    AnatomicalAxis, on={"orientation": _has_value("left-to-right")}
):
    """A spatial axis oriented from left to right."""

    name: str = "left-to-right"
    orientation: NoRepr[LeftToRight] = LeftToRight()


class RightToLeftAxis(
    AnatomicalAxis, on={"orientation": _has_value("right-to-left")}
):
    """A spatial axis oriented from right to left."""

    name: str = "right-to-left"
    orientation: NoRepr[RightToLeft] = RightToLeft()


class AnteriorToPosteriorAxis(
    AnatomicalAxis, on={"orientation": _has_value("anterior-to-posterior")}
):
    """A spatial axis oriented from anterior to posterior."""

    name: str = "anterior-to-posterior"
    orientation: NoRepr[AnteriorToPosterior] = AnteriorToPosterior()


class PosteriorToAnteriorAxis(
    AnatomicalAxis, on={"orientation": _has_value("posterior-to-anterior")}
):
    """A spatial axis oriented from posterior to anterior."""

    name: str = "posterior-to-anterior"
    orientation: NoRepr[PosteriorToAnterior] = PosteriorToAnterior()


class InferiorToSuperiorAxis(
    AnatomicalAxis, on={"orientation": _has_value("inferior-to-superior")}
):
    """A spatial axis oriented from inferior to superior."""

    name: str = "inferior-to-superior"
    orientation: NoRepr[InferiorToSuperior] = InferiorToSuperior()


class SuperiorToInferiorAxis(
    AnatomicalAxis, on={"orientation": _has_value("superior-to-inferior")}
):
    """A spatial axis oriented from superior to inferior."""

    name: str = "superior-to-inferior"
    orientation: NoRepr[SuperiorToInferior] = SuperiorToInferior()


# Aliases
AxisLR: tx.TypeAlias = LeftToRightAxis
AxisRL: tx.TypeAlias = RightToLeftAxis
AxisAP: tx.TypeAlias = AnteriorToPosteriorAxis
AxisPA: tx.TypeAlias = PosteriorToAnteriorAxis
AxisIS: tx.TypeAlias = InferiorToSuperiorAxis
AxisSI: tx.TypeAlias = SuperiorToInferiorAxis


R = LR = leftToRightAxis = LeftToRightAxis()
"""Canonical instance of a left-to-right anatomical axis."""

L = RL = rightToLeftAxis = RightToLeftAxis()
"""Canonical instance of a right-to-left anatomical axis."""

A = PA = posteriorToAnteriorAxis = PosteriorToAnteriorAxis()
"""Canonical instance of a posterior-to-anterior anatomical axis."""

P = AP = anteriorToPosteriorAxis = AnteriorToPosteriorAxis()
"""Canonical instance of an anterior-to-posterior anatomical axis."""

S = IS = inferiorToSuperiorAxis = InferiorToSuperiorAxis()
"""Canonical instance of an inferior-to-superior anatomical axis."""

I = SI = superiorToInferiorAxis = SuperiorToInferiorAxis()
"""Canonical instance of a superior-to-inferior anatomical axis."""

# Rx = leftToRightAxis = LeftToRightAxis(name="x")
# Lx = rightToLeftAxis = RightToLeftAxis(name="x")
# Ay = posteriorToAnteriorAxis = PosteriorToAnteriorAxis(name="y")
# Py = anteriorToPosteriorAxis = AnteriorToPosteriorAxis(name="y")
# Sz = inferiorToSuperiorAxis = InferiorToSuperiorAxis(name="z")
# Iz = superiorToInferiorAxis = SuperiorToInferiorAxis(name="z")


# --- IO helpers --------------------------------------------------------


class AxisError(ValueError):
    """Raised when a list of axes cannot be read as a vector field.

    A vector field names its grid axes together with exactly one vector
    axis, of type `displacement` or `coordinate`. A list that names no
    vector axis, more than one, or one of each type, is refused with this
    error.
    """


def vector_axis(
    axes: tx.Optional[tx.Sequence[tx.Any]],
) -> tx.Tuple[str, int]:
    """Return the type and position of the single vector axis of a field.

    A vector field lists every grid axis together with exactly one vector
    axis. The vector axis is either a `displacement` axis or a
    `coordinate` axis, and it carries the components of the vector stored
    at each grid point. This function returns a pair of the vector axis
    type, one of `"displacement"` or `"coordinate"`, and its index in
    `axes`.

    A list of axes that names no vector axis, more than one vector axis,
    or both a displacement axis and a coordinate axis, is refused with an
    [`AxisError`][brainhops.datamodel.axes.AxisError].
    """
    axes = list(axes or [])
    displacement = [
        i
        for i, a in enumerate(axes)
        if getattr(a, "type", None) == "displacement"
    ]
    coordinate = [
        i
        for i, a in enumerate(axes)
        if getattr(a, "type", None) == "coordinate"
    ]
    if len(displacement) == 1 and not coordinate:
        return "displacement", displacement[0]
    if len(coordinate) == 1 and not displacement:
        return "coordinate", coordinate[0]
    if displacement and coordinate:
        raise AxisError(
            "These axes name both a displacement axis and a coordinate "
            "axis, which cannot be read as one field. Store the "
            "displacement axes and the coordinate axes as separate fields."
        )
    if len(displacement) > 1 or len(coordinate) > 1:
        raise AxisError(
            "These axes name more than one vector axis. A field must "
            "carry exactly one axis of type displacement or coordinate."
        )
    raise AxisError(
        "These axes name no displacement axis and no coordinate axis, so "
        "the vector components cannot be identified. A field must carry "
        "exactly one axis of type displacement or coordinate."
    )
