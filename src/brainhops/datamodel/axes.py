__all__ = [
    "Axis", "SpatialAxis", "TimeAxis", "ChannelAxis",
    "R", "rightToLeftAxis", "RightToLeftAxis",
    "L", "leftToRightAxis", "LeftToRightAxis",
    "A", "anteriorToPosteriorAxis", "AnteriorToPosteriorAxis",
    "P", "posteriorToAnteriorAxis", "PosteriorToAnteriorAxis",
    "S", "inferiorToSuperiorAxis", "InferiorToSuperiorAxis",
    "I", "superiorToInferiorAxis", "SuperiorToInferiorAxis",
]
# dependencies
from typing_extensions import Optional

# internals
from brainhops._core.typing import HiddenConst

# locals
from .base import DataModelBase
from .orientation import (
    Orientation,
    LeftToRight, RightToLeft,
    AnteriorToPosterior, PosteriorToAnterior,
    InferiorToSuperior, SuperiorToInferior
)
from .units import Unit, SpaceUnit, TimeUnit


class Axis(DataModelBase):
    name: Optional[str] = None
    type: Optional[str] = None
    unit: Optional[Unit] = None
    discrete: Optional[bool] = None
    orientation: Optional[Orientation] = None


class SpatialAxis(Axis):
    unit: Optional[SpaceUnit] = SpaceUnit("millimeter")
    type: HiddenConst[str] = "spatial"
    axes: str = "None"
    direction: int = 1


class TimeAxis(Axis):
    unit: Optional[TimeUnit] = TimeUnit("second")
    type: HiddenConst[str] = "time"


class ChannelAxis(Axis):
    type: HiddenConst[str] = "channel"


class LeftToRightAxis(SpatialAxis):
    name: str = "left-to-right"
    orientation: HiddenConst[LeftToRight] = LeftToRight()
    axis: str = "x"
    direction: int = 1


class RightToLeftAxis(SpatialAxis):
    name: str = "right-to-left"
    orientation: HiddenConst[RightToLeft] = RightToLeft()
    axis: str = "x"
    direction: int = -1


class AnteriorToPosteriorAxis(SpatialAxis):
    name: str = "anterior-to-posterior"
    orientation: HiddenConst[AnteriorToPosterior] = AnteriorToPosterior()
    axis: str = "y"
    direction: int = 1


class PosteriorToAnteriorAxis(SpatialAxis):
    name: str = "posterior-to-anterior"
    orientation: HiddenConst[PosteriorToAnterior] = PosteriorToAnterior()
    axis: str = "y"
    direction: int = -1


class InferiorToSuperiorAxis(SpatialAxis):
    name: str = "inferior-to-superior"
    orientation: HiddenConst[InferiorToSuperior] = InferiorToSuperior()
    axis: str = "z"
    direction: int = 1


class SuperiorToInferiorAxis(SpatialAxis):
    name: str = "superior-to-inferior"
    orientation: HiddenConst[SuperiorToInferior] = SuperiorToInferior()
    axis: str = "z"
    direction: int = -1


R = leftToRightAxis = LeftToRightAxis()
L = rightToLeftAxis = RightToLeftAxis()
A = posteriorToAnteriorAxis = PosteriorToAnteriorAxis()
P = anteriorToPosteriorAxis = AnteriorToPosteriorAxis()
S = inferiorToSuperiorAxis = InferiorToSuperiorAxis()
I = superiorToInferiorAxis = SuperiorToInferiorAxis()

Rx = leftToRightAxis = LeftToRightAxis(name="x")
Lx = rightToLeftAxis = RightToLeftAxis(name="x")
Ay = posteriorToAnteriorAxis = PosteriorToAnteriorAxis(name="y")
Py = anteriorToPosteriorAxis = AnteriorToPosteriorAxis(name="y")
Sz = inferiorToSuperiorAxis = InferiorToSuperiorAxis(name="z")
Iz = superiorToInferiorAxis = SuperiorToInferiorAxis(name="z")
