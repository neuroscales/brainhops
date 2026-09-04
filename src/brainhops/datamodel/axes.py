__all__ = [
    "Axis",
    "SpatialAxis",
    "TimeAxis",
    "ChannelAxis",
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
    type: HiddenConst[str] = "spatial"


class TimeAxis(Axis):
    unit: tx.Optional[TimeUnit] = TimeUnit("second")
    type: HiddenConst[str] = "time"


class ChannelAxis(Axis):
    type: HiddenConst[str] = "channel"


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


def same_axis_type(a1: Axis, a2: Axis) -> bool:
    """
    Check whether two axes are of a matching type.

    Two axes match if they have the same `type`. For `"spatial"` axes,
    the `name` is also compared to make sure the corrispond to the same
    spacial axes.

    Parameters
    ----------
    a1, a2 : Axis
        The axes to compare.

    Returns
    -------
    bool
        `True` if the axes are considered to be of the same type
        (and, for spatial axes, orientation), `False` otherwise.
    """
    if a1.type == a2.type:
        if a1.type != "spatial":
            return True
        return set(a1.orientation.value.split("-")) == set(
            a2.orientation.value.split("-")
        )
    return False
