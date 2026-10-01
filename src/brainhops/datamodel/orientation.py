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
from brainhops._core.typing import NoRepr

# locals
from .base import DataModelBase
from .enums import OrientationType


def singleton(cls: tx.Type) -> tx.Callable[..., tx.Self]:
    """
    Make a class a singleton, so that only one instance of it can exist.
    """

    registry = {}

    cls.__orig_new__ = __orig__new__ = cls.__new__

    def __new__(cls: tx.Type[tx.Self], *args, **kwargs) -> tx.Self:
        obj = registry.get(cls)
        if obj is None:
            registry[cls] = obj = __orig__new__(cls, *args, **kwargs)
        return obj

    def __init_subclasses__(cls: tx.Type[tx.Self], *args, **kwargs) -> None:
        raise TypeError(f"Cannot subclass singleton class {cls.__name__}")

    cls.__new__ = __new__
    cls.__init___subclasses__ = __init_subclasses__


    return cls


class Orientation(DataModelBase, doc=True, polymorphic=True):
    """Describes the orientation of an axis or a space.

    An orientation has a `type`, drawn from [`OrientationType`][], that
    names the frame of reference it belongs to. Its `value` identifies
    the specific orientation within that frame, for example
    `"left-to-right"` for an anatomical orientation.
    """

    value: tx.Optional[str] = None
    type: tx.Optional[OrientationType] = None


class AnatomicalOrientation(Orientation, on={"type": "anatomical"}):
    """An orientation in the anatomical frame of reference."""

    type: NoRepr[OrientationType] = OrientationType.anatomical


@singleton
class LeftToRight(AnatomicalOrientation, on={"value": "left-to-right"}):
    """The anatomical orientation in which coordinates increase from the
    left of the subject toward the right."""

    value: NoRepr[str] = "left-to-right"


@singleton
class RightToLeft(AnatomicalOrientation, on={"value": "right-to-left"}):
    """The anatomical orientation in which coordinates increase from the
    right of the subject toward the left."""

    value: NoRepr[str] = "right-to-left"


@singleton
class AnteriorToPosterior(
    AnatomicalOrientation, on={"value": "anterior-to-posterior"}
):
    """The anatomical orientation in which coordinates increase from the
    front of the subject toward the back."""

    value: NoRepr[str] = "anterior-to-posterior"


@singleton
class PosteriorToAnterior(
    AnatomicalOrientation, on={"value": "posterior-to-anterior"}
):
    """The anatomical orientation in which coordinates increase from the
    back of the subject toward the front."""

    value: NoRepr[str] = "posterior-to-anterior"


@singleton
class InferiorToSuperior(
    AnatomicalOrientation, on={"value": "inferior-to-superior"}
):
    """The anatomical orientation in which coordinates increase from the
    bottom of the subject toward the top."""

    value: NoRepr[str] = "inferior-to-superior"


@singleton
class SuperiorToInferior(
    AnatomicalOrientation, on={"value": "superior-to-inferior"}
):
    """The anatomical orientation in which coordinates increase from the
    top of the subject toward the bottom."""

    value: NoRepr[str] = "superior-to-inferior"


R = LR = leftToRight = LeftToRight()
"""Singleton left-to-right orientation."""

L = RL = rightToLeft = RightToLeft()
"""Singleton right-to-left orientation."""

A = PA = posteriorToAnterior = PosteriorToAnterior()
"""Singleton posterior-to-anterior orientation."""

P = AP = anteriorToPosterior = AnteriorToPosterior()
"""Singleton anterior-to-posterior orientation."""

I = SI = superiorToInferior = SuperiorToInferior()
"""Singleton superior-to-inferior orientation."""

S = IS = inferiorToSuperior = InferiorToSuperior()
"""Singleton inferior-to-superior orientation."""
