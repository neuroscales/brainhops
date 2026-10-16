__all__ = ["PhysicalCoordinateSystem"]
from .._sugar import get_axes
from ..units import is_indexunit, is_physicalunit
from .base import CoordinateSystem


class PhysicalCoordinateSystem(CoordinateSystem):
    """
    A coordinate system whose coordinates measure physical quantities.

    Each explicitly listed axis has a physical unit, such as mm or s, or
    a unit that is not specified (`None`), but never an [`IndexUnit`][].
    Reversing an axis therefore flips the sign of its coordinates,
    whereas reversing a sampled axis shifts its origin. The axis types
    already check that each unit is of the right kind, so this class
    only rejects index units.

    The class has no constraint of its own, so dispatch never selects it
    from a parent class. Called directly with matching axes, it can
    still build one of its subclasses, such as [`RASmm`][], [`LPSmm`][]
    or [`RSAmm`][], which require millimetres on every axis.

    !!! example
        ```pycon
        >>> PhysicalCoordinateSystem(axes=[SpaceAxis(unit="mm"), ...]).ndim
        >>> PhysicalCoordinateSystem(axes=[R(), A(unit="cm")]).ndim
        2
        >>> mm = [R(unit="mm"), A(unit="mm"), S(unit="mm")]
        >>> type(PhysicalCoordinateSystem(axes=mm)).__name__
        'RASmm'
        >>> type(CoordinateSystem(axes=[R(), A(), S()])).__name__
        'RASCoordinateSystem'
        ```

    Parameters
    ----------
    axes : sequence[Axis | {...}], optional
        The axes, in order, where `...` stands for zero or more axes
        about which nothing is known. All axes must have physical units.
    name : str, optional
        The name of the system, if any.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        name = type(self).__name__
        for axis in get_axes(self):
            if axis is ...:
                continue
            unit = getattr(axis, "unit", None)
            if unit is None or is_physicalunit(unit):
                continue
            what = (
                f"counts samples (its unit is {unit!r}), which says it "
                "indexes an array"
                if is_indexunit(unit)
                else f"carries {unit!r}, which measures nothing"
            )
            raise ValueError(
                f"{name} is a physical coordinate system, so none of its "
                f"axes counts samples: each is measured in a physical "
                f"unit, or in one not yet given (`None`). The axis "
                f"{axis.name or axis.type!r} {what}."
            )
