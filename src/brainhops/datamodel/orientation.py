import typing_extensions as _tx

from .struct import SpecializedStruct
from .typing import ConstHidden


class Orientation(SpecializedStruct):
    type: _tx.Optional[str] = None
    value: _tx.Optional[str] = None


class AnatomicalOrientation(Orientation):
    type: ConstHidden[str] = "anatomical"


class LeftToRight(AnatomicalOrientation):
    value: ConstHidden[str] = "left-to-right"


class RightToLeft(AnatomicalOrientation):
    value: ConstHidden[str] = "right-to-left"


class AnteriorToPosterior(AnatomicalOrientation):
    value: ConstHidden[str] = "anterior-to-posterior"


class PosteriorToAnterior(AnatomicalOrientation):
    value: ConstHidden[str] = "posterior-to-anterior"


class InferiorToSuperior(AnatomicalOrientation):
    value: ConstHidden[str] = "inferior-to-superior"


class SuperiorToInferior(AnatomicalOrientation):
    value: ConstHidden[str] = "superior-to-inferior"


L = leftToRight = LeftToRight()
R = rightToLeft = RightToLeft()
A = anteriorToPosterior = AnteriorToPosterior()
P = posteriorToAnterior = PosteriorToAnterior()
I = inferiorToSuperior = InferiorToSuperior()
S = superiorToInferior = SuperiorToInferior()