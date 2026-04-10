from typing import Optional

from brainhops.struct import ClassVar
from .struct import SpecializedStruct


class Orientation(SpecializedStruct):
    type: Optional[str] = None
    value: Optional[str] = None


class AnatomicalOrientation(Orientation):
    type: ClassVar[str] = "anatomical"


class LeftToRight(AnatomicalOrientation):
    value: ClassVar[str] = "left-to-right"


class RightToLeft(AnatomicalOrientation):
    value: ClassVar[str] = "right-to-left"


class AnteriorToPosterior(AnatomicalOrientation):
    value: ClassVar[str] = "anterior-to-posterior"


class PosteriorToAnterior(AnatomicalOrientation):
    value: ClassVar[str] = "posterior-to-anterior"


class InferiorToSuperior(AnatomicalOrientation):
    value: ClassVar[str] = "inferior-to-superior"


class SuperiorToInferior(AnatomicalOrientation):
    value: ClassVar[str] = "superior-to-inferior"


L = leftToRight = LeftToRight()
R = rightToLeft = RightToLeft()
A = anteriorToPosterior = AnteriorToPosterior()
P = posteriorToAnterior = PosteriorToAnterior()
I = inferiorToSuperior = InferiorToSuperior()
S = superiorToInferior = SuperiorToInferior()