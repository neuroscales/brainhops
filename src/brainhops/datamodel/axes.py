__all__ = [
    "Axis", "SpatialAxis", "TimeAxis", "ChannelAxis",
    "R", "rightToLeftAxis", "RightToLeftAxis",
    "L", "leftToRightAxis", "LeftToRightAxis",
    "A", "anteriorToPosteriorAxis", "AnteriorToPosteriorAxis",
    "P", "posteriorToAnteriorAxis", "PosteriorToAnteriorAxis",
    "S", "inferiorToSuperiorAxis", "InferiorToSuperiorAxis",
    "I", "superiorToInferiorAxis", "SuperiorToInferiorAxis",
]
# externals
from typing_extensions import Optional

# internals
from .base import DataModelBase
from .orientation import (
    Orientation,
    LeftToRight, RightToLeft,
    AnteriorToPosterior, PosteriorToAnterior,
    InferiorToSuperior, SuperiorToInferior
)
from .typing import HiddenConst
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


class TimeAxis(Axis):
    unit: Optional[TimeUnit] = TimeUnit("second")
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

Rx = leftToRightAxis = LeftToRightAxis(name="x")
Lx = rightToLeftAxis = RightToLeftAxis(name="x")
Ay = posteriorToAnteriorAxis = PosteriorToAnteriorAxis(name="y")
Py = anteriorToPosteriorAxis = AnteriorToPosteriorAxis(name="y")
Sz = inferiorToSuperiorAxis = InferiorToSuperiorAxis(name="z")
Iz = superiorToInferiorAxis = SuperiorToInferiorAxis(name="z")
