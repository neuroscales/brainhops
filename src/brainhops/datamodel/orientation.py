__all__ = [
    "Orientation",
    "AnatomicalOrientation",
    "R",
    "leftToRight",
    "LeftToRight",
    "L",
    "rightToLeft",
    "RightToLeft",
    "A",
    "posteriorToAnterior",
    "PosteriorToAnterior",
    "P",
    "anteriorToPosterior",
    "AnteriorToPosterior",
    "I",
    "superiorToInferior",
    "SuperiorToInferior",
    "S",
    "inferiorToSuperior",
    "InferiorToSuperior",
]
# dependencies
import typing_extensions as _tx

# internals
from brainhops._core.typing import HiddenConst

# locals
from .base import DataModelBase
from .enums import OrientationType


class Orientation(DataModelBase, doc=True):
    """"""

    type: _tx.Optional[OrientationType] = None
    value: _tx.Optional[str] = None


class AnatomicalOrientation(Orientation):
    type: HiddenConst[OrientationType] = OrientationType.anatomical


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


R = leftToRight = LeftToRight()
L = rightToLeft = RightToLeft()
A = posteriorToAnterior = PosteriorToAnterior()
P = anteriorToPosterior = AnteriorToPosterior()
I = superiorToInferior = SuperiorToInferior()
S = inferiorToSuperior = InferiorToSuperior()
