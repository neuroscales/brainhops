__all__ = ["Axis", "AxisLike"]

import typing_extensions as tx
from bagof.magic import HideIfDefault, NotKwOnly, fields

from ..base import DataModelBase
from ..enums import AxisType
from ..orientations import Orientation
from ..units import Unit


class Axis(DataModelBase, polymorphic=True, eq=True, repr=HideIfDefault):
    """One axis of a coordinate system or of a grid.

    An axis records what is known about one dimension of a space: its
    name, type, unit and orientation, and whether it is discrete. A field
    that is `None` is unknown, so `Axis()` is a placeholder about which
    nothing is known. Subclasses fix some of the fields. A [`SpaceAxis`][],
    for example, is known to be spatial, although its unit is unknown by
    default.

    Two axes are equal when they have the same class and equal fields.
    [`compatible_with`][] tests the looser condition that two axes could
    describe the same axis.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"x"`.
    type : AxisType | str, optional
        Type of the axis, such as `"space"` or `"time"`.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    name: NotKwOnly[tx.Optional[str]] = None
    """Name of the axis, such as `"x"`."""

    type: NotKwOnly[tx.Optional[tx.Union[AxisType, str]]] = None
    """Type of the axis, such as `"space"` or `"time"`."""

    unit: NotKwOnly[tx.Optional[Unit]] = None
    """Unit of the coordinates along the axis."""

    discrete: tx.Optional[bool] = None
    """
    Whether the axis is discrete, meaning that its elements have no
    specific order and should not be interpolated.
    """

    orientation: tx.Optional[Orientation] = None
    """Meaningful direction of the axis, such as left to right."""

    def __pre_init__(self, arguments: tx.Any) -> None:
        # Fixed-size systems such as `CoordinateSystem3D` forbid `...`.
        # Without this check, the error would be an obscure conversion
        # message.
        if arguments.name is Ellipsis:
            raise TypeError("`...` is not an axis")

    def compatible_with(self, other: "Axis") -> bool:
        """Return whether two axes could describe the same axis.

        Two axes are compatible when all their fields have either the
        same value **or** at least one of them has the value `None`.

        The relation is symmetric but not transitive: `Axis()` is
        compatible with `Axis(name="x")` and `Axis(name="y")`, but these
        two are not compatible with each other.

        !!! example
            ```pycon
            >>> Axis(name="x").compatible_with(Axis(name="x", unit="mm"))
            True
            >>> Axis(name="x").compatible_with(Axis(name="y"))
            False
            >>> SpaceAxis(unit="mm").compatible_with(Axis(unit="micrometer"))
            False
            ```

        Parameters
        ----------
        other : Axis
            The axis to compare with.

        Returns
        -------
        bool
            Whether no field is set on both sides to different values.

        Raises
        ------
        TypeError
            If `other` is not an [`Axis`][].
        """
        if not isinstance(other, Axis):
            raise TypeError(
                f"An axis is compatible only with another Axis, not with "
                f"{type(other).__name__}."
            )
        return not _conflicts(self, other)

    def merge_with(self, other: "Axis") -> "Axis":
        """Combine two compatible axes.

        The input axes must be compatible in the sense of
        [`compatible_with`][]. The resulting axis has all non-`None`
        values found on either axis.

        When both axes have different `Axis` subtypes, the resulting axis
        has the type of the most derived class. For example, an [`Axis`][]
        merged with a [`SpaceAxis`][] results in a [`SpaceAxis`][].

        !!! example
            ```pycon
            >>> Axis(name="x").merge_with(SpaceAxis(unit="micrometer"))
            SpaceAxis(name='x', unit='micrometer')
            >>> Axis(name="x").merge_with(Axis(name="y"))
            Traceback (most recent call last):
              ...
            ValueError: Cannot merge axes that disagree on name: 'x' != 'y'.
            ```

        Parameters
        ----------
        other : Axis
            Another description of the same axis.

        Returns
        -------
        Axis
            A new axis with every field known to either side.

        Raises
        ------
        TypeError
            If `other` is not an [`Axis`][].
        ValueError
            If the axes disagree on a field, or if neither class derives from
            the other.
        """
        if not isinstance(other, Axis):
            raise TypeError(
                f"An axis merges only with another Axis, not with "
                f"{type(other).__name__}."
            )
        # The more derived class can hold every field, and `_merge_with` reads
        # the fields of its first argument, so the instance of the more
        # derived class goes first.
        if isinstance(self, type(other)):
            cls, first, second = type(self), self, other
        elif isinstance(other, type(self)):
            cls, first, second = type(other), other, self
        else:
            # A field disagreement, such as LR against RL, is reported in
            # preference to the lack of a common class.
            _refuse_conflicts(self, other)
            raise ValueError(
                f"Cannot merge a {type(self).__name__} with a "
                f"{type(other).__name__}: neither class derives from the "
                f"other, so no class holds what both describe."
            )
        return cls(**_merge_with(first, second))


AxisLike: tx.TypeAlias = tx.Union[Axis, str]
"""
A value that is a [`Axis`][] or that can be converted to one.
"""


# --- Helpers ----------------------------------------------------------


def _field_names(axis: Axis) -> tx.List[str]:
    return [field.name for field in fields(type(axis))]


def _conflicts(
    first: Axis, second: Axis
) -> tx.List[tx.Tuple[str, tx.Any, tx.Any]]:
    # Return the fields that are set on both axes to different values.
    names = _field_names(first)
    names += [name for name in _field_names(second) if name not in names]
    conflicts = []
    for name in names:
        mine = getattr(first, name, None)
        theirs = getattr(second, name, None)
        if mine is not None and theirs is not None and mine != theirs:
            conflicts.append((name, mine, theirs))
    return conflicts


def _refuse_conflicts(first: Axis, second: Axis) -> None:
    conflicts = _conflicts(first, second)
    if conflicts:
        name, mine, theirs = conflicts[0]
        raise ValueError(
            f"Cannot merge axes that disagree on {name}: "
            f"{mine!r} != {theirs!r}."
        )


def _merge_with(first: Axis, second: Axis) -> tx.Dict[str, tx.Any]:
    _refuse_conflicts(first, second)
    kwargs = {}
    for field in fields(type(first)):
        if not (field.init and field.kw):
            # Class constants, such as `SpaceAxis.type`, already agree because
            # no conflict was found.
            continue
        value = getattr(first, field.name, None)
        if value is None:
            value = getattr(second, field.name, None)
        kwargs[field.public_name] = value
    return kwargs
