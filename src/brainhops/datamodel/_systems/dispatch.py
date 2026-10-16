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


def _isnd(n: int) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """
    Generate an axes checker that returns true when they have a given length.
    """

    def isnd(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _finite(axes)
        return closed is not None and len(closed) == n

    isnd.__name__ = isnd.__qualname__ = f"_is{n}d"
    return isnd


_is2d = _isnd(2)
_is3d = _isnd(3)


def _all(
    test: tx.Callable[[Axis], bool], name: str
) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """Build a predicate that holds when finite, non-empty axes pass `test`."""

    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _finite(axes)
        return bool(closed) and all(test(axis) for axis in closed)

    check.__name__ = check.__qualname__ = name
    return check


_is_spatial = _all(lambda axis: axis.type == "space", "_is_spatial")
_is_array = _all(lambda axis: is_indexunit(axis.unit), "_is_array")
_MM = Unit("mm")
_is_mm = _all(lambda axis: axis.unit == _MM, "_is_mm")


def _both(
    *tests: tx.Callable[[tx.Any], bool],
) -> tx.Callable[[tx.Any], bool]:
    def check(axes: tx.Any) -> bool:
        return all(test(axes) for test in tests)

    check.__name__ = check.__qualname__ = "_and_".join(
        test.__name__.lstrip("_") for test in tests
    )
    check.__name__ = check.__qualname__ = "_" + check.__name__
    return check


def _is_anat(code: str) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
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
