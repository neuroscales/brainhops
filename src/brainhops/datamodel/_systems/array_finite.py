__all__ = [
    "ArrayCoordinateSystem2D",
    "ArrayCoordinateSystem3D",
    "CArrayCoordinateSystem2D",
    "CArrayCoordinateSystem3D",
    "FArrayCoordinateSystem2D",
    "FArrayCoordinateSystem3D",
]

from bagof.magic import Inherit

from .array import (
    ArrayCoordinateSystem,
    CArrayCoordinateSystem,
    FArrayCoordinateSystem,
)
from .dispatch import _both, _is2d, _is3d, _is_array
from .factories import _ArrayAxesFactory
from .finite import CoordinateSystem2D, CoordinateSystem3D, _2Axes, _3Axes


# ArrayCoordinateSystem has no constraint of its own, so its 2-D subclass is
# registered with it by hand: called with two axes of any kind,
# ArrayCoordinateSystem builds an ArrayCoordinateSystem2D. The `on=`
# constraint covers the other route, from CoordinateSystem2D, which requires
# axes that count samples.
@ArrayCoordinateSystem.register_polymorph(axes=_is2d)
class ArrayCoordinateSystem2D(
    CoordinateSystem2D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is2d, _is_array)},
):
    """An array coordinate system with two axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis], default=(Axis(), Axis())
        The two array axes, in order.
    name : str, default="array"
        The name of the system, if any.
    order : Literal["C", "F"], optional
        The memory order of the indexed array, or `None` when unspecified.
    """

    axes: Inherit[_2Axes] = _ArrayAxesFactory(2)


@ArrayCoordinateSystem.register_polymorph(axes=_is3d)
class ArrayCoordinateSystem3D(
    CoordinateSystem3D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is3d, _is_array)},
):
    """An array coordinate system with three axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis, Axis], default=(Axis(), Axis(), Axis())
        The three array axes, in order.
    name : str, default="array"
        The name of the system, if any.
    order : Literal["C", "F"], optional
        The memory order of the indexed array, or `None` when unspecified.
    """

    axes: Inherit[_3Axes] = _ArrayAxesFactory(3)


# Each class below inherits from a class with a fixed number of axes and
# from an ordered class. It needs no constraint of its own, because it is
# selected when the constraints of both parents hold.
class CArrayCoordinateSystem2D(CoordinateSystem2D, CArrayCoordinateSystem):
    """A C-ordered array coordinate system with two axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis], default=(Axis(), Axis())
        The two array axes, in C order.
    name : str, default="carray"
        The name of the system, if any.
    order : literal["C"], default="C"
        The pinned memory order.
    """

    axes: Inherit[_2Axes] = _ArrayAxesFactory(2)


class CArrayCoordinateSystem3D(CoordinateSystem3D, CArrayCoordinateSystem):
    """A C-ordered array coordinate system with three axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis, Axis], default=(Axis(), Axis(), Axis())
        The three array axes, in C order.
    name : str, default="carray"
        The name of the system, if any.
    order : literal["C"], default="C"
        The pinned memory order.
    """

    axes: Inherit[_3Axes] = _ArrayAxesFactory(3)


class FArrayCoordinateSystem2D(CoordinateSystem2D, FArrayCoordinateSystem):
    """An F-ordered array coordinate system with two axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis], default=(Axis(), Axis())
        The two array axes, in F order.
    name : str, default="farray"
        The name of the system, if any.
    order : literal["F"], default="F"
        The pinned memory order.
    """

    axes: Inherit[_2Axes] = _ArrayAxesFactory(2)


class FArrayCoordinateSystem3D(CoordinateSystem3D, FArrayCoordinateSystem):
    """An F-ordered array coordinate system with three axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis, Axis], default=(Axis(), Axis(), Axis())
        The three array axes, in F order.
    name : str, default="carray"
        The name of the system, if any.
    order : literal["F"], default="F"
        The pinned memory order.
    """

    axes: Inherit[_3Axes] = _ArrayAxesFactory(3)
