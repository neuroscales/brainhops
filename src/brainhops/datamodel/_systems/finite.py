__all__ = ["CoordinateSystem2D", "CoordinateSystem3D"]
import typing_extensions as tx
from bagof.magic import Factory, Inherit

from ..axes import AxisTuple
from .base import AXIS, CoordinateSystem
from .dispatch import is2d, is3d

_2Axes: tx.TypeAlias = AxisTuple[AXIS, AXIS]
_3Axes: tx.TypeAlias = AxisTuple[AXIS, AXIS, AXIS]


class CoordinateSystem2D(CoordinateSystem, on={"axes": is2d}):
    """A coordinate system with exactly two axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis], default=(Axis(), Axis())
        The two axes, in order.
    name : str, optional
        The name of the system, if any.
    """

    axes: Inherit[_2Axes] = Factory()


class CoordinateSystem3D(CoordinateSystem, on={"axes": is3d}):
    """A coordinate system with exactly three axes.

    Parameters
    ----------
    axes : tuple[Axis, Axis, Axis], default=(Axis(), Axis(), Axis())
        The three axes, in order.
    name : str, optional
        The name of the system, if any.
    """

    axes: Inherit[_3Axes] = Factory()
