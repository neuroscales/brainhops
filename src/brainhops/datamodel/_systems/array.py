__all__ = [
    "ArrayCoordinateSystem",
    "CArrayCoordinateSystem",
    "FArrayCoordinateSystem",
]

import typing_extensions as tx

from .base import CoordinateSystem


class ArrayCoordinateSystem(CoordinateSystem):
    """
    A coordinate system that indexes a multidimensional array.

    The coordinates count samples, so the default axes of the two- and
    three-dimensional subclasses carry an [`IndexUnit`][]. Dispatch never
    selects the class itself from a parent class. When the class is called
    directly with two or three axes, however, it builds the matching subclass
    with a fixed number of axes, and when it is called with `order="C"` or
    `order="F"`, it builds the matching ordered subclass.

    Parameters
    ----------
    axes : sequence[Axis | {...}], optional
        The axes, in order, where `...` stands for zero or more axes
        about which nothing is known.
    name : str, default="array"
        The name of the system, if any.
    order : literal["C", "F"], optional
        The memory order of the indexed array, or `None` when unspecified.
    """

    name: ... = "array"

    order: tx.Optional[tx.Literal["C", "F"]] = None
    """
    The memory order of the indexed array, or `None` when unspecified.

    With `"C"` the last axis varies fastest, and with `"F"` the first
    one does.
    """


class CArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "C"}):
    """
    A C-ordered array system, whose last axis varies fastest in memory.

    Parameters
    ----------
    axes : sequence[Axis | {...}], optional
        The axes, in order, where `...` stands for zero or more axes
        about which nothing is known.
    name : str, default="carray"
        The name of the system, if any.
    order : literal["C"], default="C"
        The pinned memory order.
    """

    name: ... = "carray"


class FArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "F"}):
    """
    An F-ordered array system, whose first axis varies fastest in memory.

    Parameters
    ----------
    axes : sequence[Axis | {...}], optional
        The axes, in order, where `...` stands for zero or more axes
        about which nothing is known.
    name : str, default="farray"
        The name of the system, if any.
    order :lLiteral["F"], default="F"
        The pinned memory order.
    """

    name: ... = "farray"
