import typing_extensions as tx

from ..enums import AnatomicalOrientationValue
from ..orientations import Orientation


def is_not_none(obj: tx.Any) -> bool:
    """True if input value is not `None`."""
    return obj is not None


def is_anatomical(orientation: tx.Optional[Orientation]) -> bool:
    """True if the orientation type is anatomical."""
    return getattr(orientation, "type", None) == "anatomical"


def has_orientation(
    value: str,
) -> tx.Callable[[tx.Optional[Orientation]], bool]:

    if not isinstance(value, AnatomicalOrientationValue):
        try:
            value = AnatomicalOrientationValue(value)
        except ValueError:
            ...
        try:
            value = AnatomicalOrientationValue[value]
        except KeyError:
            ...

    def _check(orientation: tx.Optional[Orientation]) -> bool:
        return getattr(orientation, "value", None) == value

    return _check
