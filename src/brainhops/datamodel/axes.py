from typing_extensions import Optional

from brainhops.struct import ClassVar
from .struct import SpecializedStruct
from .orientation import Orientation


class Axis(SpecializedStruct):
    name: Optional[str] = None
    type: Optional[str] = None
    unit: Optional[str] = None
    discrete: Optional[bool] = None
    orientation: Optional[Orientation] = None


class SpatialAxis(Axis):
    unit: str = "mm"
    type: ClassVar[str] = "spatial"


class TimeAxis(Axis):
    unit: str = "s"
    type: ClassVar[str] = "time"


class ChannelAxis(Axis):
    type: ClassVar[str] = "channel"


class LeftToRightAxis(SpatialAxis):
    name: str = "left-to-right"
    orientation: ClassVar[Orientation] = Orientation("left-to-right")


class RightToLeftAxis(SpatialAxis):
    name: str = "right-to-left"
    orientation: ClassVar[Orientation] = Orientation("right-to-left")


class AnteriorToPosteriorAxis(SpatialAxis):
    name: str = "anterior-to-posterior"
    orientation: ClassVar[Orientation] = Orientation("anterior-to-posterior")


class PosteriorToAnteriorAxis(SpatialAxis):
    name: str = "posterior-to-anterior"
    orientation: ClassVar[Orientation] = Orientation("posterior-to-anterior")


class InferiorToSuperiorAxis(SpatialAxis):
    name: str = "inferior-to-superior"
    orientation: ClassVar[Orientation] = Orientation("inferior-to-superior")


class SuperiorToInferiorAxis(SpatialAxis):
    name: str = "superior-to-inferior"
    orientation: ClassVar[Orientation] = Orientation("superior-to-inferior")


R = leftToRightAxis = LeftToRightAxis()
L = rightToLeftAxis = RightToLeftAxis()
P = anteriorToPosteriorAxis = AnteriorToPosteriorAxis()
A = posteriorToAnteriorAxis = PosteriorToAnteriorAxis()
S = inferiorToSuperiorAxis = InferiorToSuperiorAxis()
I = superiorToInferiorAxis = SuperiorToInferiorAxis()