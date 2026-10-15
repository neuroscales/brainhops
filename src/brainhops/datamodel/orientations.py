"""Orientations of axes and spaces, such as left-to-right."""

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
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr

from brainhops._core.singleton import singleton

from .base import DataModelBase
from .enums import AnatomicalOrientationValue, OrientationType


class Orientation(DataModelBase, polymorphic=True, kw_only=False, slots=True):
    """The orientation of an axis or a space.

    The `type` is the frame of reference (an [`OrientationType`][]), and the
    `value` is the specific orientation within that frame, such as
    `"left-to-right"`.

    Parameters
    ----------
    value : str, optional
        Name of the orientation, such as `"left-to-right"`.
    type : OrientationType, optional
        Frame of reference of the orientation, such as `"anatomical"`.
    """

    value: tx.Optional[str] = None
    """Name of the orientation, such as `"left-to-right"`."""

    type: tx.Optional[OrientationType] = None
    """Frame of reference of the orientation, such as `"anatomical"`."""


class AnatomicalOrientation(Orientation, on={"type": "anatomical"}):
    """An orientation in the anatomical frame of reference.

    Instances of `AnatomicalOrientation` are singletons.

    Parameters
    ----------
    value : str, optional
        Name of the orientation, such as `"left-to-right"`.
    type : literal["anatomical"], default="anatomical"
        Pinned orientation type.
    """

    value: Inherit[tx.Optional[AnatomicalOrientationValue]] = None
    type: NoRepr[...]


@singleton
@Orientation.register_polymorph(on={"value": "left-to-right"})
class LeftToRight(
    AnatomicalOrientation, on={"value": "left-to-right"}, frozen=True
):
    """Coordinates increase from left to right.

    Parameters
    ----------
    value : literal["left-to-right"], default="left-to-right"
        Pinned orientation.
    type : literal["anatomical"], default="anatomical"
        Pinned orientation type.
    """

    value: NoRepr[...]


@singleton
@Orientation.register_polymorph(on={"value": "right-to-left"})
class RightToLeft(
    AnatomicalOrientation, on={"value": "right-to-left"}, frozen=True
):
    """Coordinates increase from right to left.

    Parameters
    ----------
    value : literal["right-to-left"], default="right-to-left"
        Pinned orientation.
    type : literal["anatomical"], default="anatomical"
        Pinned orientation type.
    """

    value: NoRepr[...]


@singleton
@Orientation.register_polymorph(on={"value": "anterior-to-posterior"})
class AnteriorToPosterior(
    AnatomicalOrientation, on={"value": "anterior-to-posterior"}, frozen=True
):
    """Coordinates increase from front to back.

    Parameters
    ----------
    value : literal["anterior-to-posterior"], default="anterior-to-posterior"
        Pinned orientation.
    type : literal["anatomical"], default="anatomical"
        Pinned orientation type.
    """

    value: NoRepr[...]


@singleton
@Orientation.register_polymorph(on={"value": "posterior-to-anterior"})
class PosteriorToAnterior(
    AnatomicalOrientation, on={"value": "posterior-to-anterior"}, frozen=True
):
    """Coordinates increase from back to front.

    Parameters
    ----------
    value : literal["posterior-to-anterior"], default="posterior-to-anterior"
        Pinned orientation.
    type : literal["anatomical"], default="anatomical"
        Pinned orientation type.
    """

    value: NoRepr[...]


@singleton
@Orientation.register_polymorph(on={"value": "inferior-to-superior"})
class InferiorToSuperior(
    AnatomicalOrientation, on={"value": "inferior-to-superior"}, frozen=True
):
    """Coordinates increase from bottom to top.

    Parameters
    ----------
    value : literal["inferior-to-superior"], default="inferior-to-superior"
        Pinned orientation.
    type : literal["anatomical"], default="anatomical"
        Pinned orientation type.
    """

    value: NoRepr[...]


@singleton
@Orientation.register_polymorph(on={"value": "superior-to-inferior"})
class SuperiorToInferior(
    AnatomicalOrientation, on={"value": "superior-to-inferior"}, frozen=True
):
    """Coordinates increase from top to bottom.

    Parameters
    ----------
    value : literal["superior-to-inferior"], default="superior-to-inferior"
        Pinned orientation.
    type : literal["anatomical"], default="anatomical"
        Pinned orientation type.
    """

    value: NoRepr[...]


R = LR = leftToRight = LeftToRight()
"""Left-to-right orientation, also available as `R` and `LR`.

Each single-letter name refers to the end of the axis towards which
coordinates increase: `R` is the right end of a left-to-right axis.
"""

L = RL = rightToLeft = RightToLeft()
"""Right-to-left orientation, also available as `L` and `RL`."""

A = PA = posteriorToAnterior = PosteriorToAnterior()
"""Posterior-to-anterior orientation, also available as `A` and `PA`."""

P = AP = anteriorToPosterior = AnteriorToPosterior()
"""Anterior-to-posterior orientation, also available as `P` and `AP`."""

I = SI = superiorToInferior = SuperiorToInferior()
"""Superior-to-inferior orientation, also available as `I` and `SI`."""

S = IS = inferiorToSuperior = InferiorToSuperior()
"""Inferior-to-superior orientation, also available as `S` and `IS`."""
