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
import typing_extensions as tx

# core
from brainhops._core.typing import HiddenConst

# locals
from .base import DataModelBase
from .enums import OrientationType


class Orientation(DataModelBase, doc=True):
    """Describes the orientation of an axis or a space.

    An orientation has a `type`, drawn from [`OrientationType`][], that
    names the frame of reference it belongs to. Its `value` identifies
    the specific orientation within that frame, for example
    `"left-to-right"` for an anatomical orientation.
    """

    type: tx.Optional[OrientationType] = None
    value: tx.Optional[str] = None


class AnatomicalOrientation(Orientation):
    """Describes the anatomical orientation of a spatial axis."""

    type: HiddenConst[OrientationType] = OrientationType.anatomical


class LeftToRight(AnatomicalOrientation):
    """Left-to-right orientation type."""

    value: HiddenConst[str] = "left-to-right"


class RightToLeft(AnatomicalOrientation):
    """Right-to-left orientation type."""

    value: HiddenConst[str] = "right-to-left"


class AnteriorToPosterior(AnatomicalOrientation):
    """Anterior-to-posterior orientation type."""

    value: HiddenConst[str] = "anterior-to-posterior"


class PosteriorToAnterior(AnatomicalOrientation):
    """Posterior-to-anterior orientation type."""

    value: HiddenConst[str] = "posterior-to-anterior"


class InferiorToSuperior(AnatomicalOrientation):
    """Inferior-to-superior orientation type."""

    value: HiddenConst[str] = "inferior-to-superior"


class SuperiorToInferior(AnatomicalOrientation):
    """Superior-to-inferior orientation type."""

    value: HiddenConst[str] = "superior-to-inferior"


R = leftToRight = LeftToRight()
"""Singleton left-to-right orientation."""

L = rightToLeft = RightToLeft()
"""Singleton right-to-left orientation."""

A = posteriorToAnterior = PosteriorToAnterior()
"""Singleton posterior-to-anterior orientation."""

P = anteriorToPosterior = AnteriorToPosterior()
"""Singleton anterior-to-posterior orientation."""

I = superiorToInferior = SuperiorToInferior()
"""Singleton superior-to-inferior orientation."""

S = inferiorToSuperior = InferiorToSuperior()
"""Singleton inferior-to-superior orientation."""
