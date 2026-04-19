__all__ = [
    "Axis",
    "SpatialAxis",
    "TimeAxis",
    "ChannelAxis",
    "LeftToRightAxis",
    "RightToLeftAxis",
    "AnteriorToPosteriorAxis",
    "PosteriorToAnteriorAxis",
    "InferiorToSuperiorAxis",
    "SuperiorToInferiorAxis",
    "R", "L", "P", "A", "S", "I"
]
from typing_extensions import Optional

from .struct import SpecializedStruct
from .orientation import (
    Orientation, 
    LeftToRight, RightToLeft, 
    AnteriorToPosterior, PosteriorToAnterior, 
    InferiorToSuperior, SuperiorToInferior
)
from .typing import HiddenConst


class Axis(SpecializedStruct):
    name: Optional[str] = None
    type: Optional[str] = None
    unit: Optional[str] = None
    discrete: Optional[bool] = None
    orientation: Optional[Orientation] = None


class SpatialAxis(Axis):
    unit: str = "mm"
    type: HiddenConst[str] = "spatial"


class TimeAxis(Axis):
    unit: str = "s"
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
P = anteriorToPosteriorAxis = AnteriorToPosteriorAxis()
A = posteriorToAnteriorAxis = PosteriorToAnteriorAxis()
S = inferiorToSuperiorAxis = InferiorToSuperiorAxis()
I = superiorToInferiorAxis = SuperiorToInferiorAxis()