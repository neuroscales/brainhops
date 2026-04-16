import typing_extensions as _tx

from .struct import SpecializedStruct
from .typing import HiddenConst


class Orientation(SpecializedStruct):
    type: _tx.Optional[str] = None
    value: _tx.Optional[str] = None


class AnatomicalOrientation(Orientation):
    type: HiddenConst[str] = "anatomical"


class LeftToRight(AnatomicalOrientation):
    value: HiddenConst[str] = "left-to-right"


class RightToLeft(AnatomicalOrientation):
    value: HiddenConst[str] = "right-to-left"


class AnteriorToPosterior(AnatomicalOrientation):
    value: HiddenConst[str] = "anterior-to-posterior"


class PosteriorToAnterior(AnatomicalOrientation):
    value: HiddenConst[str] = "posterior-to-anterior"


class InferiorToSuperior(AnatomicalOrientation):
    value: HiddenConst[str] = "inferior-to-superior"


class SuperiorToInferior(AnatomicalOrientation):
    value: HiddenConst[str] = "superior-to-inferior"


L = leftToRight = LeftToRight()
R = rightToLeft = RightToLeft()
A = anteriorToPosterior = AnteriorToPosterior()
P = posteriorToAnterior = PosteriorToAnterior()
I = inferiorToSuperior = InferiorToSuperior()
S = superiorToInferior = SuperiorToInferior()