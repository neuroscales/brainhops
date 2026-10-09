"""Axes of coordinate systems, such as spatial and time axes."""

__all__ = [
    "Axis",
    "SpaceAxis",
    "TimeAxis",
    "ChannelAxis",
    "DisplacementAxis",
    "CoordinateAxis",
    "R",
    "LR",
    "RightToLeftAxis",
    "L",
    "RL",
    "LeftToRightAxis",
    "A",
    "PA",
    "AnteriorToPosteriorAxis",
    "P",
    "AP",
    "PosteriorToAnteriorAxis",
    "S",
    "IS",
    "InferiorToSuperiorAxis",
    "I",
    "SI",
    "SuperiorToInferiorAxis",
]
import typing_extensions as tx
from bagof.magic import fields

from brainhops._core.typing import NoRepr

from .base import DataModelBase
from .enums import AnatomicalOrientationValue, AxisType
from .orientations import (
    AnteriorToPosterior,
    InferiorToSuperior,
    LeftToRight,
    Orientation,
    PosteriorToAnterior,
    RightToLeft,
    SuperiorToInferior,
)
from .units import IndexUnit, SpaceUnit, TimeUnit, Unit

# ======================================================================
#
#                               B A S E
#
# ======================================================================


class Axis(DataModelBase, polymorphic=True):
    """One axis of a coordinate system or of a grid.

    A field that is `None` is unknown, so `Axis()` is a placeholder about
    which nothing is known. A subclass fixes some fields: a [`SpaceAxis`][]
    is known to be spatial, although its unit is unknown by default.

    Two axes are equal when they have the same class and equal fields.
    [`compatible_with`][] tests whether they could be the same axis.
    """

    name: tx.Optional[str] = None
    """Name of the axis, such as `"x"`."""

    type: tx.Optional[tx.Union[AxisType, str]] = None
    """Type of the axis, such as `"space"` or `"time"`."""

    unit: tx.Optional[Unit] = None
    """Unit of the coordinates along the axis."""

    discrete: tx.Optional[bool] = None
    """Whether the axis is discrete, so that its positions have no order."""

    orientation: tx.Optional[Orientation] = None
    """Meaningful direction of the axis, such as left to right."""

    def __pre_init__(self, arguments: tx.Any) -> None:
        # Fixed-size systems such as `CoordinateSystem3D` forbid `...`; without
        # this check the error is an obscure conversion message.
        if arguments.name is Ellipsis:
            raise TypeError("`...` is not an axis")

    def compatible_with(self, other: "Axis") -> bool:
        """Return whether two axes could describe the same axis.

        Two axes are compatible when every field set on both has the same
        value, so `Axis()` is compatible with every axis. The relation is
        symmetric but not transitive: `Axis()` fits both `Axis(name="x")` and
        `Axis(name="y")`, which do not fit each other.

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
        """Combine two descriptions of the same axis.

        Each field takes the value set on either axis, and the axes must be
        compatible in the sense of [`compatible_with`][]. The result has the
        more derived class: an [`Axis`][] merged with a [`SpaceAxis`][] gives a
        [`SpaceAxis`][].

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
        # the fields of its first argument, so that instance goes first.
        if isinstance(self, type(other)):
            cls, first, second = type(self), self, other
        elif isinstance(other, type(self)):
            cls, first, second = type(other), other, self
        else:
            # Report a field disagreement, such as LR against RL, before the
            # lack of a common class.
            _refuse_conflicts(self, other)
            raise ValueError(
                f"Cannot merge a {type(self).__name__} with a "
                f"{type(other).__name__}: neither class derives from the "
                f"other, so no class holds what both describe."
            )
        return cls(**_merge_with(first, second))


def _field_names(axis: Axis) -> tx.List[str]:
    return [field.name for field in fields(type(axis))]


def _conflicts(
    first: Axis, second: Axis
) -> tx.List[tx.Tuple[str, tx.Any, tx.Any]]:
    # Fields set on both axes to different values.
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


# ======================================================================
#
#                             B Y   T Y P E
#
# ======================================================================


def _is_not_none(obj: tx.Any) -> bool:
    return obj is not None


def _is_anatomical(orientation: tx.Optional[Orientation]) -> bool:
    return getattr(orientation, "type", None) == "anatomical"


def _has_value(value: str) -> tx.Callable[[tx.Optional[Orientation]], bool]:

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


class SpaceAxis(Axis, on={"type": "space"}):
    """Axis that measures a spatial dimension.

    The unit is a space unit, an index unit, or `None`. The class is
    selected on the type and not on the unit, so `Axis(type="space",
    unit="s")` fails rather than building a generic `Axis`.
    """

    unit: tx.Optional[tx.Union[SpaceUnit, IndexUnit]] = None
    type: NoRepr[tx.Literal["space"]] = "space"


class TimeAxis(Axis, on={"type": "time"}):
    """Axis that measures time, in a time unit or an index unit."""

    unit: tx.Optional[tx.Union[TimeUnit, IndexUnit]] = None
    type: NoRepr[tx.Literal["time"]] = "time"


class ChannelAxis(Axis, on={"type": "channel"}):
    """Axis that enumerates channels, such as colours or features."""

    type: NoRepr[tx.Literal["channel"]] = "channel"


class DisplacementAxis(Axis, on={"type": "displacement"}):
    """Axis that enumerates the components of displacement vectors.

    A displacement field has its spatial axes and one displacement axis.
    """

    type: NoRepr[tx.Literal["displacement"]] = "displacement"


class CoordinateAxis(Axis, on={"type": "coordinate"}):
    """Axis that enumerates the components of coordinate vectors.

    A coordinate field has its spatial axes and one coordinate axis.
    """

    type: NoRepr[tx.Literal["coordinate"]] = "coordinate"


class OrientedAxis(Axis, on={"orientation": _is_not_none}):
    """Axis that carries an orientation."""


# These classes derive from two registered classes, so they are selected on
# both conditions without an `on=` of their own.


class OrientedTimeAxis(TimeAxis, OrientedAxis):
    """Time axis with an orientation."""


class OrientedSpaceAxis(SpaceAxis, OrientedAxis):
    """Spatial axis with an orientation."""


# ======================================================================
#
#                          A N A T O M I C A L
#
# ======================================================================


# An anatomical orientation implies a spatial axis, but every parent requires
# `type='space'`, so the class is also registered with the root on the
# orientation alone.
@Axis.register_polymorph(on={"orientation": _is_anatomical})
class AnatomicalAxis(
    OrientedSpaceAxis,
    on={"orientation": _is_anatomical},
):
    """Spatial axis with an anatomical orientation.

    An anatomical orientation selects this class even without a type. The
    unit may be an index unit, for a voxel axis along that direction.
    """


class LeftToRightAxis(
    AnatomicalAxis, on={"orientation": _has_value("left-to-right")}
):
    """Spatial axis oriented from left to right."""

    name: str = "left-to-right"
    orientation: NoRepr[LeftToRight] = LeftToRight()


class RightToLeftAxis(
    AnatomicalAxis, on={"orientation": _has_value("right-to-left")}
):
    """Spatial axis oriented from right to left."""

    name: str = "right-to-left"
    orientation: NoRepr[RightToLeft] = RightToLeft()


class AnteriorToPosteriorAxis(
    AnatomicalAxis, on={"orientation": _has_value("anterior-to-posterior")}
):
    """Spatial axis oriented from anterior to posterior."""

    name: str = "anterior-to-posterior"
    orientation: NoRepr[AnteriorToPosterior] = AnteriorToPosterior()


class PosteriorToAnteriorAxis(
    AnatomicalAxis, on={"orientation": _has_value("posterior-to-anterior")}
):
    """Spatial axis oriented from posterior to anterior."""

    name: str = "posterior-to-anterior"
    orientation: NoRepr[PosteriorToAnterior] = PosteriorToAnterior()


class InferiorToSuperiorAxis(
    AnatomicalAxis, on={"orientation": _has_value("inferior-to-superior")}
):
    """Spatial axis oriented from inferior to superior."""

    name: str = "inferior-to-superior"
    orientation: NoRepr[InferiorToSuperior] = InferiorToSuperior()


class SuperiorToInferiorAxis(
    AnatomicalAxis, on={"orientation": _has_value("superior-to-inferior")}
):
    """Spatial axis oriented from superior to inferior."""

    name: str = "superior-to-inferior"
    orientation: NoRepr[SuperiorToInferior] = SuperiorToInferior()


# The aliases are classes because axes are mutable and a shared instance would
# leak between systems. Use `R(unit='mm')` and `isinstance(axis, R)`.

R: tx.TypeAlias = LeftToRightAxis
LR: tx.TypeAlias = LeftToRightAxis
"""Aliases of [`LeftToRightAxis`][].

Coordinates along this axis increase toward the right.
"""

L: tx.TypeAlias = RightToLeftAxis
RL: tx.TypeAlias = RightToLeftAxis
"""Aliases of [`RightToLeftAxis`][].

Coordinates along this axis increase toward the left.
"""

A: tx.TypeAlias = PosteriorToAnteriorAxis
PA: tx.TypeAlias = PosteriorToAnteriorAxis
"""Aliases of [`PosteriorToAnteriorAxis`][].

Coordinates along this axis increase toward the front.
"""

P: tx.TypeAlias = AnteriorToPosteriorAxis
AP: tx.TypeAlias = AnteriorToPosteriorAxis
"""Aliases of [`AnteriorToPosteriorAxis`][].

Coordinates along this axis increase toward the back.
"""

S: tx.TypeAlias = InferiorToSuperiorAxis
IS: tx.TypeAlias = InferiorToSuperiorAxis
"""Aliases of [`InferiorToSuperiorAxis`][].

Coordinates along this axis increase toward the top.
"""

I: tx.TypeAlias = SuperiorToInferiorAxis
SI: tx.TypeAlias = SuperiorToInferiorAxis
"""Aliases of [`SuperiorToInferiorAxis`][].

Coordinates along this axis increase toward the bottom.
"""
