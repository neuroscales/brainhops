from typing_extensions import Optional

from .struct import SpecializedStruct
from .orientation import (
    Orientation, 
    LeftToRight, RightToLeft, 
    AnteriorToPosterior, PosteriorToAnterior, 
    InferiorToSuperior, SuperiorToInferior
)
from .typing import ConstHidden


class Axis(SpecializedStruct):
    name: Optional[str] = None
    type: Optional[str] = None
    unit: Optional[str] = None
    discrete: Optional[bool] = None
    orientation: Optional[Orientation] = None


class SpatialAxis(Axis):
    unit: str = "mm"
    type: ConstHidden[str] = "spatial"


class TimeAxis(Axis):
    unit: str = "s"
    type: ConstHidden[str] = "time"


class ChannelAxis(Axis):
    type: ConstHidden[str] = "channel"


class LeftToRightAxis(SpatialAxis):
    name: str = "left-to-right"
    orientation: ConstHidden[LeftToRight] = LeftToRight()


class RightToLeftAxis(SpatialAxis):
    name: str = "right-to-left"
    orientation: ConstHidden[RightToLeft] = RightToLeft()


class AnteriorToPosteriorAxis(SpatialAxis):
    name: str = "anterior-to-posterior"
    orientation: ConstHidden[AnteriorToPosterior] = AnteriorToPosterior()


class PosteriorToAnteriorAxis(SpatialAxis):
    name: str = "posterior-to-anterior"
    orientation: ConstHidden[PosteriorToAnterior] = PosteriorToAnterior()


class InferiorToSuperiorAxis(SpatialAxis):
    name: str = "inferior-to-superior"
    orientation: ConstHidden[InferiorToSuperior] = InferiorToSuperior()


class SuperiorToInferiorAxis(SpatialAxis):
    name: str = "superior-to-inferior"
    orientation: ConstHidden[SuperiorToInferior] = SuperiorToInferior()


R = leftToRightAxis = LeftToRightAxis()
L = rightToLeftAxis = RightToLeftAxis()
P = anteriorToPosteriorAxis = AnteriorToPosteriorAxis()
A = posteriorToAnteriorAxis = PosteriorToAnteriorAxis()
S = inferiorToSuperiorAxis = InferiorToSuperiorAxis()
I = superiorToInferiorAxis = SuperiorToInferiorAxis()