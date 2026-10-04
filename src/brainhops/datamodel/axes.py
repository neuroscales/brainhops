"""Axes of a coordinate system, such as a spatial or a time axis."""

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
    "vector_axis",
    "AxisError",
]
# dependencies
import typing_extensions as tx
from bagof.magic import fields

# core
from brainhops._core.typing import NoRepr

# locals
from .base import DataModelBase
from .enums import AnatomicalOrientationValue, AxisType
from .orientation import (
    AnteriorToPosterior,
    InferiorToSuperior,
    LeftToRight,
    Orientation,
    PosteriorToAnterior,
    RightToLeft,
    SuperiorToInferior,
)
from .units import IndexUnit, SpaceUnit, TimeUnit, Unit

# --- Dispatch helpers -------------------------------------------------


def _is_not_none(obj: tx.Any) -> bool:
    """Whether `obj` is not `None`."""
    return obj is not None


def _is_anatomical(orientation: tx.Optional[Orientation]) -> bool:
    """Whether `orientation` is an anatomical orientation."""
    return getattr(orientation, "type", None) == "anatomical"


def _has_value(value: str) -> tx.Callable[[tx.Optional[Orientation]], bool]:
    """
    Whether `orientation` is an anatomical orientation with the given value.
    """

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


# --- API --------------------------------------------------------------


class Axis(DataModelBase, polymorphic=True):
    """One axis of a coordinate system or of a grid.

    An axis names its type, such as `"space"` or `"time"`, and may carry
    a unit, an orientation, and whether it is discrete.

    Each field that is `None` is unknown. A plain `Axis()`, whose fields
    are all `None`, is an axis about which nothing is known. It is the
    placeholder that fills a position no description covers. An instance
    of a subclass that sets a field, such as a [`SpaceAxis`][], whose
    type is `"space"`, is not unknown -- although its unit, which it
    leaves unspecified by default, is.

    Equality (`==`) is strict: two axes are equal when they are of the
    same class and every field is equal. [`compatible_with`][]
    is the looser question of whether two descriptions could be of the
    same axis.
    """

    name: tx.Optional[str] = None
    """The name of the axis, such as `"x"` or `"y"`."""

    type: tx.Optional[tx.Union[AxisType, str]] = None
    """The type of the axis, such as `"space"` or `"time"`."""

    unit: tx.Optional[Unit] = None
    """The unit in which coordinates are measured along the axis."""

    discrete: tx.Optional[bool] = None
    """
    Whether the axis is discrete, as opposed to continuous.

    Here discrete means that the axis has no particular order, and
    therefore does not come with a continuous coordinates system.
    """

    orientation: tx.Optional[Orientation] = None
    """
    The orientation of the axis, if it has one.

    This is useful to indicate that the direction of coordinates along
    the axis has a particular meaning, such as `"left-to-right"` for an
    anatomical axis.
    """

    def __pre_init__(self, arguments: tx.Any) -> None:
        # `...` is not a placeholder for an axis: a coordinate system
        # with a fixed number of dimensions lists every one of them. As
        # the first positional argument it would otherwise be read as
        # the axis name, and refused with a message about strings.
        if arguments.name is Ellipsis:
            raise TypeError("`...` is not an axis")

    def compatible_with(self, other: "Axis") -> bool:
        """Whether `self` and `other` could describe the same axis.

        Two axes are compatible when every field that is set (not `None`)
        on both of them is equal. A field that is `None` on either side
        is unknown there, and matches anything. In particular, the
        unknown `Axis()` is compatible with every axis.

        The relation is symmetric, but it is not transitive: `Axis()` is
        compatible with both `Axis(name="x")` and `Axis(name="y")`, which
        are not compatible with each other.

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
            Whether no field is known on both sides with different values.

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
        """Combine what `self` and `other` know about the same axis.

        Each field of the result is the value set on either side, or
        `None` if neither side sets it. The two axes must be
        [`compatible_with`][]: a field that both set must be set
        to the same value.

        The result is an instance of the more derived of the two classes,
        so merging an [`Axis`][] with a [`SpaceAxis`][] gives a
        [`SpaceAxis`][]. Merging with the unknown `Axis()` returns an
        axis equal to the other side.

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
            The other description of the same axis.

        Returns
        -------
        Axis
            A new axis that carries every field known on either side.

        Raises
        ------
        TypeError
            If `other` is not an [`Axis`][].
        ValueError
            If a field is set on both sides to different values, or if
            neither class derives from the other, so that no class can
            hold what both sides know.
        """
        if not isinstance(other, Axis):
            raise TypeError(
                f"An axis merges only with another Axis, not with "
                f"{type(other).__name__}."
            )
        conflicts = _conflicts(self, other)
        if conflicts:
            name, mine, theirs = conflicts[0]
            raise ValueError(
                f"Cannot merge axes that disagree on {name}: "
                f"{mine!r} != {theirs!r}."
            )
        if isinstance(other, type(self)):
            cls = type(other)
        elif isinstance(self, type(other)):
            cls = type(self)
        else:
            raise ValueError(
                f"Cannot merge a {type(self).__name__} with a "
                f"{type(other).__name__}: neither class derives from the "
                f"other, so no class holds what both describe."
            )
        kwargs = {}
        for field in fields(cls):
            if not (field.init and field.kw):
                # A constant of the class, such as the type of a
                # `SpaceAxis`. No conflict was found, so it already
                # agrees with the value on the other side, if any.
                continue
            value = getattr(self, field.name, None)
            if value is None:
                value = getattr(other, field.name, None)
            kwargs[field.public_name] = value
        return cls(**kwargs)


def _field_names(axis: Axis) -> tx.List[str]:
    return [field.name for field in fields(type(axis))]


def _conflicts(
    first: Axis, second: Axis
) -> tx.List[tx.Tuple[str, tx.Any, tx.Any]]:
    # The fields set on both axes to different values, as
    # `(name, first value, second value)`.
    names = _field_names(first)
    names += [name for name in _field_names(second) if name not in names]
    conflicts = []
    for name in names:
        mine = getattr(first, name, None)
        theirs = getattr(second, name, None)
        if mine is not None and theirs is not None and mine != theirs:
            conflicts.append((name, mine, theirs))
    return conflicts


class SpaceAxis(Axis, on={"type": "space"}):
    """An axis that measures a spatial dimension.

    Its unit is a unit of space, an index unit (the axis indexes an
    array), or
    unspecified (`None`, the default). Any other unit is refused by the
    type of the field. The unit is not what an axis is selected on, so
    `Axis(type="space", unit="s")` builds a `SpaceAxis`, which refuses the
    second, rather than quietly falling back to a generic `Axis`.
    """

    unit: tx.Optional[tx.Union[SpaceUnit, IndexUnit]] = None
    type: NoRepr[tx.Literal["space"]] = "space"


class TimeAxis(Axis, on={"type": "time"}):
    """An axis that measures time.

    Its unit is a unit of time, an index unit (the axis indexes an
    array), or
    unspecified (`None`, the default). Any other unit is refused.
    """

    unit: tx.Optional[tx.Union[TimeUnit, IndexUnit]] = None
    type: NoRepr[tx.Literal["time"]] = "time"


class ChannelAxis(Axis, on={"type": "channel"}):
    """An axis that enumerates channels, such as color or feature channels."""

    type: NoRepr[tx.Literal["channel"]] = "channel"


class DisplacementAxis(Axis, on={"type": "displacement"}):
    """An axis that carries the components of a displacement vector.

    A field of displacements names its spatial axes together with exactly
    one axis of this type. That axis enumerates the displacement
    components stored at each grid point.
    """

    type: NoRepr[tx.Literal["displacement"]] = "displacement"


class CoordinateAxis(Axis, on={"type": "coordinate"}):
    """An axis that carries the components of a coordinate vector.

    A field of coordinates names its spatial axes together with exactly
    one axis of this type. That axis enumerates the coordinate components
    stored at each grid point.
    """

    type: NoRepr[tx.Literal["coordinate"]] = "coordinate"


class OrientedAxis(Axis, on={"orientation": _is_not_none}):
    """An axis that carries an orientation."""


# The two classes below inherit from two registered classes, so bagof
# selects them on what both parents stand for -- a time (or spatial) axis
# that carries an orientation -- with no `on=` of their own.


class OrientedTimeAxis(TimeAxis, OrientedAxis):
    """A time axis that carries an orientation."""


class OrientedSpaceAxis(SpaceAxis, OrientedAxis):
    """A spatial axis that carries an orientation."""


# An anatomical orientation says that the axis runs through space, so a
# generic `Axis(orientation=R())` -- which names no type -- is read as a
# spatial axis. That is a step the class statement cannot express (the
# classes it registers with all ask for `type="space"`), so it is
# registered with the root by hand, on the orientation alone: an axis of
# another type with an anatomical orientation is a contradiction, and
# building it as an anatomical axis refuses it. An anatomical axis whose
# unit is an index unit is a spatial axis of a voxel grid that points in
# that direction.
@Axis.register_polymorph(on={"orientation": _is_anatomical})
class AnatomicalAxis(
    OrientedSpaceAxis,
    on={"orientation": _is_anatomical},
):
    """An axis that carries an anatomical orientation.

    `Axis(orientation=...)` with an anatomical orientation builds one of
    these even when it names no type: an anatomical direction is a
    direction in space. Its unit may still be an index unit, for a voxel axis
    that points in that direction.
    """


class LeftToRightAxis(
    AnatomicalAxis, on={"orientation": _has_value("left-to-right")}
):
    """A spatial axis oriented from left to right."""

    name: str = "left-to-right"
    orientation: NoRepr[LeftToRight] = LeftToRight()


class RightToLeftAxis(
    AnatomicalAxis, on={"orientation": _has_value("right-to-left")}
):
    """A spatial axis oriented from right to left."""

    name: str = "right-to-left"
    orientation: NoRepr[RightToLeft] = RightToLeft()


class AnteriorToPosteriorAxis(
    AnatomicalAxis, on={"orientation": _has_value("anterior-to-posterior")}
):
    """A spatial axis oriented from anterior to posterior."""

    name: str = "anterior-to-posterior"
    orientation: NoRepr[AnteriorToPosterior] = AnteriorToPosterior()


class PosteriorToAnteriorAxis(
    AnatomicalAxis, on={"orientation": _has_value("posterior-to-anterior")}
):
    """A spatial axis oriented from posterior to anterior."""

    name: str = "posterior-to-anterior"
    orientation: NoRepr[PosteriorToAnterior] = PosteriorToAnterior()


class InferiorToSuperiorAxis(
    AnatomicalAxis, on={"orientation": _has_value("inferior-to-superior")}
):
    """A spatial axis oriented from inferior to superior."""

    name: str = "inferior-to-superior"
    orientation: NoRepr[InferiorToSuperior] = InferiorToSuperior()


class SuperiorToInferiorAxis(
    AnatomicalAxis, on={"orientation": _has_value("superior-to-inferior")}
):
    """A spatial axis oriented from superior to inferior."""

    name: str = "superior-to-inferior"
    orientation: NoRepr[SuperiorToInferior] = SuperiorToInferior()


# Aliases
AxisLR: tx.TypeAlias = LeftToRightAxis
AxisRL: tx.TypeAlias = RightToLeftAxis
AxisAP: tx.TypeAlias = AnteriorToPosteriorAxis
AxisPA: tx.TypeAlias = PosteriorToAnteriorAxis
AxisIS: tx.TypeAlias = InferiorToSuperiorAxis
AxisSI: tx.TypeAlias = SuperiorToInferiorAxis


# Short names. These are the classes, not instances: an axis is mutable,
# so a module-level instance would be shared by every system that took it.
# Build one where it is needed -- `R()`, `R(unit="mm")`, `R(name="x")` --
# and test with `isinstance(axis, R)`.

R = LR = LeftToRightAxis
"""A left-to-right anatomical axis (coordinates increase toward the right)."""

L = RL = RightToLeftAxis
"""A right-to-left anatomical axis (coordinates increase toward the left)."""

A = PA = PosteriorToAnteriorAxis
"""A posterior-to-anterior anatomical axis (increasing toward the front)."""

P = AP = AnteriorToPosteriorAxis
"""An anterior-to-posterior anatomical axis (increasing toward the back)."""

S = IS = InferiorToSuperiorAxis
"""An inferior-to-superior anatomical axis (increasing toward the top)."""

I = SI = SuperiorToInferiorAxis
"""A superior-to-inferior anatomical axis (increasing toward the bottom)."""


# --- IO helpers --------------------------------------------------------


class AxisError(ValueError):
    """Raised when a list of axes cannot be read as a vector field.

    A vector field names its grid axes together with exactly one vector
    axis, of type `displacement` or `coordinate`. A list that names no
    vector axis, more than one, or one of each type, is refused with this
    error.
    """


def vector_axis(
    axes: tx.Optional[tx.Sequence[tx.Any]],
) -> tx.Tuple[str, int]:
    """Return the type and position of the single vector axis of a field.

    A vector field lists every grid axis together with exactly one vector
    axis. The vector axis is either a `displacement` axis or a
    `coordinate` axis, and it carries the components of the vector stored
    at each grid point. This function returns a pair of the vector axis
    type, one of `"displacement"` or `"coordinate"`, and its index in
    `axes`.

    A list of axes that names no vector axis, more than one vector axis,
    or both a displacement axis and a coordinate axis, is refused with an
    [`AxisError`][brainhops.datamodel.axes.AxisError].
    """
    axes = list(axes or [])
    displacement = [
        i
        for i, a in enumerate(axes)
        if getattr(a, "type", None) == "displacement"
    ]
    coordinate = [
        i
        for i, a in enumerate(axes)
        if getattr(a, "type", None) == "coordinate"
    ]
    if len(displacement) == 1 and not coordinate:
        return "displacement", displacement[0]
    if len(coordinate) == 1 and not displacement:
        return "coordinate", coordinate[0]
    if displacement and coordinate:
        raise AxisError(
            "These axes name both a displacement axis and a coordinate "
            "axis, which cannot be read as one field. Store the "
            "displacement axes and the coordinate axes as separate fields."
        )
    if len(displacement) > 1 or len(coordinate) > 1:
        raise AxisError(
            "These axes name more than one vector axis. A field must "
            "carry exactly one axis of type displacement or coordinate."
        )
    raise AxisError(
        "These axes name no displacement axis and no coordinate axis, so "
        "the vector components cannot be identified. A field must carry "
        "exactly one axis of type displacement or coordinate."
    )
