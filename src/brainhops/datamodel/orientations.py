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
from bagof.magic import Narrow

from brainhops._core.typing import NoRepr

from .base import DataModelBase
from .enums import OrientationType

_T = tx.TypeVar("_T", bound=type)


def singleton(cls: _T) -> _T:
    """Make a class a singleton, so that only one instance of it can exist.

    Calling the class again returns the existing instance. Subclassing a
    singleton raises a TypeError, because a subclass would introduce a second
    variant of an object that is meant to be unique. The orientation classes
    in this module are also frozen: every axis with a given orientation holds
    the same instance, so mutating that instance would change all of them.
    """

    registry = {}

    cls.__orig_new__ = __orig__new__ = cls.__new__

    def __new__(cls: tx.Type[tx.Self], *args, **kwargs) -> tx.Self:
        obj = registry.get(cls)
        if obj is None:
            registry[cls] = obj = __orig__new__(cls, *args, **kwargs)
        return obj

    def __init_subclass__(sub: type, **kwargs) -> None:
        raise TypeError(
            f"{cls.__name__} is a singleton: it cannot be subclassed."
        )

    cls.__new__ = __new__
    cls.__init_subclass__ = classmethod(__init_subclass__)

    return cls


class Orientation(DataModelBase, polymorphic=True):
    """The orientation of an axis or a space.

    The `type` is the frame of reference (an [`OrientationType`][]), and the
    `value` is the specific orientation within that frame, such as
    `"left-to-right"`.
    """

    value: tx.Optional[str] = None
    type: tx.Optional[OrientationType] = None


class AnatomicalOrientation(Orientation, on={"type": "anatomical"}):
    """An orientation in the anatomical frame of reference."""

    type: NoRepr[Narrow[OrientationType]] = OrientationType.anatomical


@singleton
class LeftToRight(
    AnatomicalOrientation, on={"value": "left-to-right"}, frozen=True
):
    """Coordinates increase from left to right."""

    value: NoRepr[Narrow[str]] = "left-to-right"


@singleton
class RightToLeft(
    AnatomicalOrientation, on={"value": "right-to-left"}, frozen=True
):
    """Coordinates increase from right to left."""

    value: NoRepr[Narrow[str]] = "right-to-left"


@singleton
class AnteriorToPosterior(
    AnatomicalOrientation, on={"value": "anterior-to-posterior"}, frozen=True
):
    """Coordinates increase from front to back."""

    value: NoRepr[Narrow[str]] = "anterior-to-posterior"


@singleton
class PosteriorToAnterior(
    AnatomicalOrientation, on={"value": "posterior-to-anterior"}, frozen=True
):
    """Coordinates increase from back to front."""

    value: NoRepr[Narrow[str]] = "posterior-to-anterior"


@singleton
class InferiorToSuperior(
    AnatomicalOrientation, on={"value": "inferior-to-superior"}, frozen=True
):
    """Coordinates increase from bottom to top."""

    value: NoRepr[Narrow[str]] = "inferior-to-superior"


@singleton
class SuperiorToInferior(
    AnatomicalOrientation, on={"value": "superior-to-inferior"}, frozen=True
):
    """Coordinates increase from top to bottom."""

    value: NoRepr[Narrow[str]] = "superior-to-inferior"


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
