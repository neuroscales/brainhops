__all__ = [
    "Axis",
    "SpatialAxis",
    "TimeAxis",
    "ChannelAxis",
    "DisplacementAxis",
    "CoordinateAxis",
    "AxisError",
    "vector_axis",
    "R",
    "rightToLeftAxis",
    "RightToLeftAxis",
    "L",
    "leftToRightAxis",
    "LeftToRightAxis",
    "A",
    "anteriorToPosteriorAxis",
    "AnteriorToPosteriorAxis",
    "P",
    "posteriorToAnteriorAxis",
    "PosteriorToAnteriorAxis",
    "S",
    "inferiorToSuperiorAxis",
    "InferiorToSuperiorAxis",
    "I",
    "superiorToInferiorAxis",
    "SuperiorToInferiorAxis",
]
# dependencies
import typing_extensions as tx

# core
from brainhops._core.typing import HiddenConst

# locals
from .base import DataModelBase
from .orientation import (
    AnteriorToPosterior,
    InferiorToSuperior,
    LeftToRight,
    Orientation,
    PosteriorToAnterior,
    RightToLeft,
    SuperiorToInferior,
)
from .units import SpaceUnit, TimeUnit, Unit


class Axis(DataModelBase):
    name: tx.Optional[str] = None
    type: tx.Optional[str] = None
    unit: tx.Optional[Unit] = None
    discrete: tx.Optional[bool] = None
    orientation: tx.Optional[Orientation] = None


class SpatialAxis(Axis):
    unit: tx.Optional[SpaceUnit] = SpaceUnit("millimeter")
    type: HiddenConst[str] = "space"


class TimeAxis(Axis):
    unit: tx.Optional[TimeUnit] = TimeUnit("second")
    type: HiddenConst[str] = "time"


class ChannelAxis(Axis):
    type: HiddenConst[str] = "channel"


class DisplacementAxis(Axis):
    """An axis that carries the components of a displacement vector.

    A field of displacements names its spatial axes together with exactly
    one axis of this type. That axis enumerates the displacement
    components stored at each grid point.
    """

    type: HiddenConst[str] = "displacement"


class CoordinateAxis(Axis):
    """An axis that carries the components of a coordinate vector.

    A field of coordinates names its spatial axes together with exactly
    one axis of this type. That axis enumerates the coordinate components
    stored at each grid point.
    """

    type: HiddenConst[str] = "coordinate"


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


class LeftToRightAxis(SpatialAxis):
    name: str = "left-to-right"
    orientation: HiddenConst[LeftToRight] = LeftToRight()


class RightToLeftAxis(SpatialAxis):
    name: str = "right-to-left"
    orientation: HiddenConst[RightToLeft] = RightToLeft()


class AnteriorToPosteriorAxis(SpatialAxis):
    name: str = "anterior-to-posterior"
    orientation: HiddenConst[AnteriorToPosterior] = AnteriorToPosterior()


class PosteriorToAnteriorAxis(SpatialAxis):
    name: str = "posterior-to-anterior"
    orientation: HiddenConst[PosteriorToAnterior] = PosteriorToAnterior()


class InferiorToSuperiorAxis(SpatialAxis):
    name: str = "inferior-to-superior"
    orientation: HiddenConst[InferiorToSuperior] = InferiorToSuperior()


class SuperiorToInferiorAxis(SpatialAxis):
    name: str = "superior-to-inferior"
    orientation: HiddenConst[SuperiorToInferior] = SuperiorToInferior()


R = leftToRightAxis = LeftToRightAxis()
L = rightToLeftAxis = RightToLeftAxis()
A = posteriorToAnteriorAxis = PosteriorToAnteriorAxis()
P = anteriorToPosteriorAxis = AnteriorToPosteriorAxis()
S = inferiorToSuperiorAxis = InferiorToSuperiorAxis()
I = superiorToInferiorAxis = SuperiorToInferiorAxis()

# Rx = leftToRightAxis = LeftToRightAxis(name="x")
# Lx = rightToLeftAxis = RightToLeftAxis(name="x")
# Ay = posteriorToAnteriorAxis = PosteriorToAnteriorAxis(name="y")
# Py = anteriorToPosteriorAxis = AnteriorToPosteriorAxis(name="y")
# Sz = inferiorToSuperiorAxis = InferiorToSuperiorAxis(name="z")
# Iz = superiorToInferiorAxis = SuperiorToInferiorAxis(name="z")
