import typing_extensions as tx

from .. import axes as _axes
from ..axes import Axis
from ..units import Unit, is_indexunit


def _finite(
    axes: tx.Optional[tx.Sequence],
) -> tx.Optional[tx.List[Axis]]:
    """
    Return the axes as a finite list.
    Return `None` if they are missing or variadic.
    """
    if axes is None:
        return None
    axes = list(axes)
    if any(axis is ... for axis in axes):
        return None
    return axes


def isnd(n: int) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """
    Generate an axes checker that returns true when they have a given length.
    """

    def _isnd(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _finite(axes)
        return closed is not None and len(closed) == n

    _isnd.__name__ = _isnd.__qualname__ = f"is{n}d"
    return _isnd


is2d = isnd(2)
is3d = isnd(3)


def _all(
    test: tx.Callable[[Axis], bool], name: str
) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """Build a predicate that holds when finite, non-empty axes pass `test`."""

    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _finite(axes)
        return bool(closed) and all(test(axis) for axis in closed)

    check.__name__ = check.__qualname__ = name
    return check


is_spatial = _all(lambda axis: axis.type == "space", "_is_spatial")
is_array = _all(lambda axis: is_indexunit(axis.unit), "_is_array")
MM = Unit("mm")
is_mm = _all(lambda axis: axis.unit == MM, "_is_mm")


def both(
    *tests: tx.Callable[[tx.Any], bool],
) -> tx.Callable[[tx.Any], bool]:
    def check(axes: tx.Any) -> bool:
        return all(test(axes) for test in tests)

    check.__name__ = check.__qualname__ = "_and_".join(
        test.__name__.lstrip("_") for test in tests
    )
    check.__name__ = check.__qualname__ = "_" + check.__name__
    return check


def is_anat(code: str) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """Build a predicate that compares the axes with the orientation `code`.

    The letters are those of [`brainhops.datamodel.axes`][], so `"RAS"` means
    left-to-right, posterior-to-anterior and inferior-to-superior axes, in that
    order. Only orientations are compared, not names or units.
    """
    expected = tuple(
        getattr(_axes, letter)().orientation.value for letter in code
    )

    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _finite(axes)
        if closed is None or len(closed) != len(expected):
            return False
        return all(
            getattr(getattr(axis, "orientation", None), "value", None) == value
            for axis, value in zip(closed, expected)
        )

    check.__name__ = check.__qualname__ = f"_is_{code}"
    return check
