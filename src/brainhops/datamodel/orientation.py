"""Orientations of an axis or a space, such as left-to-right."""

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
    """An orientation in the anatomical frame of reference."""

    type: HiddenConst[OrientationType] = OrientationType.anatomical


class LeftToRight(AnatomicalOrientation):
    """The anatomical orientation in which coordinates increase from the
    left of the subject toward the right."""

    value: HiddenConst[str] = "left-to-right"


class RightToLeft(AnatomicalOrientation):
    """The anatomical orientation in which coordinates increase from the
    right of the subject toward the left."""

    value: HiddenConst[str] = "right-to-left"


class AnteriorToPosterior(AnatomicalOrientation):
    """The anatomical orientation in which coordinates increase from the
    front of the subject toward the back."""

    value: HiddenConst[str] = "anterior-to-posterior"


class PosteriorToAnterior(AnatomicalOrientation):
    """The anatomical orientation in which coordinates increase from the
    back of the subject toward the front."""

    value: HiddenConst[str] = "posterior-to-anterior"


class InferiorToSuperior(AnatomicalOrientation):
    """The anatomical orientation in which coordinates increase from the
    bottom of the subject toward the top."""

    value: HiddenConst[str] = "inferior-to-superior"


class SuperiorToInferior(AnatomicalOrientation):
    """The anatomical orientation in which coordinates increase from the
    top of the subject toward the bottom."""

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
