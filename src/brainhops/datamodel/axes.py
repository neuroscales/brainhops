"""Axes of a coordinate system, such as a spatial or a time axis."""

__all__ = [
    "Axis",
    "SpatialAxis",
    "TimeAxis",
    "ChannelAxis",
    "DisplacementAxis",
    "CoordinateAxis",
    "AxisError",
    "vector_axis",
    "R",
    "rightToLeftAxis",
    "RightToLeftAxis",
    "L",
    "leftToRightAxis",
    "LeftToRightAxis",
    "A",
    "anteriorToPosteriorAxis",
    "AnteriorToPosteriorAxis",
    "P",
    "posteriorToAnteriorAxis",
    "PosteriorToAnteriorAxis",
    "S",
    "inferiorToSuperiorAxis",
    "InferiorToSuperiorAxis",
    "I",
    "superiorToInferiorAxis",
    "SuperiorToInferiorAxis",
]
# dependencies
import typing_extensions as tx
from bagof.magic import fields

# core
from brainhops._core.typing import HiddenConst

# locals
from .base import DataModelBase
from .orientation import (
    AnteriorToPosterior,
    InferiorToSuperior,
    LeftToRight,
    Orientation,
    PosteriorToAnterior,
    RightToLeft,
    SuperiorToInferior,
)
from .units import SpaceUnit, TimeUnit, Unit


class Axis(DataModelBase):
    """One axis of a coordinate system or of a grid.

    An axis names its type, such as `"space"` or `"time"`, and may carry
    a unit, an orientation, and whether it is discrete.

    Each field that is `None` is unknown. A plain `Axis()`, whose fields
    are all `None`, is an axis about which nothing is known. It is the
    placeholder that fills a position no description covers. An instance
    of a subclass that sets a field, such as a [`SpatialAxis`][], whose
    type is `"space"` and whose unit defaults to millimetres, is not
    unknown.

    Equality (`==`) is strict: two axes are equal when they are of the
    same class and every field is equal. [`compatible_with`][]
    is the looser question of whether two descriptions could be of the
    same axis.
    """

    name: tx.Optional[str] = None
    type: tx.Optional[str] = None
    unit: tx.Optional[Unit] = None
    discrete: tx.Optional[bool] = None
    orientation: tx.Optional[Orientation] = None

    def compatible_with(self, other: "Axis") -> bool:
        """Whether `self` and `other` could describe the same axis.

        Two axes are compatible when every field that is set (not `None`)
        on both of them is equal. A field that is `None` on either side
        is unknown there, and matches anything. In particular, the
        unknown `Axis()` is compatible with every axis.

        The relation is symmetric, but it is not transitive: `Axis()` is
        compatible with both `Axis(name="x")` and `Axis(name="y")`, which
        are not compatible with each other.

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

        !!! example
            ```pycon
            >>> Axis(name="x").compatible_with(Axis(name="x", unit="mm"))
            True
            >>> Axis(name="x").compatible_with(Axis(name="y"))
            False
            >>> SpatialAxis().compatible_with(Axis(unit="micrometer"))
            False
            ```
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
        so merging an [`Axis`][] with a [`SpatialAxis`][] gives a
        [`SpatialAxis`][]. Merging with the unknown `Axis()` returns an
        axis equal to the other side.

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

        !!! example
            ```pycon
            >>> Axis(name="x").merge_with(SpatialAxis(unit="micrometer"))
            SpatialAxis(name='x', unit='micrometer')
            >>> Axis(name="x").merge_with(Axis(name="y"))
            Traceback (most recent call last):
              ...
            ValueError: Cannot merge axes that disagree on name: 'x' != 'y'.
            ```
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
                # `SpatialAxis`. No conflict was found, so it already
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


class SpatialAxis(Axis):
    """An axis that measures a spatial dimension."""

    unit: tx.Optional[SpaceUnit] = SpaceUnit("millimeter")
    type: HiddenConst[str] = "space"


class TimeAxis(Axis):
    """An axis that measures time."""

    unit: tx.Optional[TimeUnit] = TimeUnit("second")
    type: HiddenConst[str] = "time"


class ChannelAxis(Axis):
    """An axis that enumerates channels, such as color or feature channels."""

    type: HiddenConst[str] = "channel"


class DisplacementAxis(Axis):
    """An axis that carries the components of a displacement vector.

    A field of displacements names its spatial axes together with exactly
    one axis of this type. That axis enumerates the displacement
    components stored at each grid point.
    """

    type: HiddenConst[str] = "displacement"


class CoordinateAxis(Axis):
    """An axis that carries the components of a coordinate vector.

    A field of coordinates names its spatial axes together with exactly
    one axis of this type. That axis enumerates the coordinate components
    stored at each grid point.
    """

    type: HiddenConst[str] = "coordinate"


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


class LeftToRightAxis(SpatialAxis):
    """A spatial axis oriented from left to right."""

    name: str = "left-to-right"
    orientation: HiddenConst[LeftToRight] = LeftToRight()


class RightToLeftAxis(SpatialAxis):
    """A spatial axis oriented from right to left."""

    name: str = "right-to-left"
    orientation: HiddenConst[RightToLeft] = RightToLeft()


class AnteriorToPosteriorAxis(SpatialAxis):
    """A spatial axis oriented from anterior to posterior."""

    name: str = "anterior-to-posterior"
    orientation: HiddenConst[AnteriorToPosterior] = AnteriorToPosterior()


class PosteriorToAnteriorAxis(SpatialAxis):
    """A spatial axis oriented from posterior to anterior."""

    name: str = "posterior-to-anterior"
    orientation: HiddenConst[PosteriorToAnterior] = PosteriorToAnterior()


class InferiorToSuperiorAxis(SpatialAxis):
    """A spatial axis oriented from inferior to superior."""

    name: str = "inferior-to-superior"
    orientation: HiddenConst[InferiorToSuperior] = InferiorToSuperior()


class SuperiorToInferiorAxis(SpatialAxis):
    """A spatial axis oriented from superior to inferior."""

    name: str = "superior-to-inferior"
    orientation: HiddenConst[SuperiorToInferior] = SuperiorToInferior()


R = leftToRightAxis = LeftToRightAxis()
L = rightToLeftAxis = RightToLeftAxis()
A = posteriorToAnteriorAxis = PosteriorToAnteriorAxis()
P = anteriorToPosteriorAxis = AnteriorToPosteriorAxis()
S = inferiorToSuperiorAxis = InferiorToSuperiorAxis()
I = superiorToInferiorAxis = SuperiorToInferiorAxis()

# Rx = leftToRightAxis = LeftToRightAxis(name="x")
# Lx = rightToLeftAxis = RightToLeftAxis(name="x")
# Ay = posteriorToAnteriorAxis = PosteriorToAnteriorAxis(name="y")
# Py = anteriorToPosteriorAxis = AnteriorToPosteriorAxis(name="y")
# Sz = inferiorToSuperiorAxis = InferiorToSuperiorAxis(name="z")
# Iz = superiorToInferiorAxis = SuperiorToInferiorAxis(name="z")
