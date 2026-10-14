__all__ = [
    "CoordinateSystem",
    # re-export axis sequences from systems
    "AxisSequence",
    "AxisList",
    "AxisTuple",
]

import typing_extensions as tx
from bagof.magic import HideIfDefault, NotKwOnly, replace

from ..axes import Axis, AxisList, AxisSequence, AxisTuple
from ..base import DataModelBase
from .decorators import Axes, bind_axes_default

if tx.TYPE_CHECKING:
    from types import EllipsisType as _Ellipsis

else:
    _Ellipsis: tx.TypeAlias = type(Ellipsis)
    # This is the type of `...`, which Python 3.10 names
    # `types.EllipsisType`.

AXIS = tx.TypeVar("AXIS", bound=Axis, default=Axis)
_EllipsisOr: tx.TypeAlias = tx.Union[AXIS, _Ellipsis]

_VariadicAxes: tx.TypeAlias = AxisList[_EllipsisOr[AXIS]]


@bind_axes_default
class CoordinateSystem(
    DataModelBase, polymorphic=True, eq=True, repr=HideIfDefault
):
    """
    A coordinate system, which gives coordinates in a space their meaning.

    A coordinate system lists its axes in order and describes each of
    them with properties such as a name and a unit. The system itself
    can also have a name.

    !!! note "Variadic systems"
        The axes are an [`AxisList`][] that holds at most one `...`,
        standing for zero or more axes about which nothing is known.
        A system with `...` is _variadic_, since its number of axes is
        unknown, and a system without it is _finite_:

        * `[..., TimeAxis()]` says that the last axis is time, and
          nothing is known about the others.
        * `[Axis(name="x"), ...]` says that the first axis is x.
        * `[...]`, the default, says nothing about the axes. `axes=None`
          is read as if the axes were not given, so
          `CoordinateSystem(axes=None) == CoordinateSystem()`.

        Fixed-arity classes such as [`CoordinateSystem3D`][] are always
        finite. They store their axes as an immutable [`AxisTuple`][],
        whose type fixes the number and class of the axes.

        Only finite systems are dispatched:
        `CoordinateSystem(axes=[x, y])` is a [`CoordinateSystem2D`][],
        whereas `CoordinateSystem(axes=[x, ...])` stays a
        `CoordinateSystem` until [`expand`][] makes it finite.

    !!! note "Equality"
        Two systems are equal when they have the same class, the same
        name and equal axes. A system never equals `None`, not even
        `CoordinateSystem()`. As the input or output of a transformation,
        `None` means that no system is given and lets the context decide,
        whereas `CoordinateSystem()` is a system and is kept as given.
        [`compatible_with`][] asks the looser question of whether two
        systems could describe the same space.

    Parameters
    ----------
    axes : sequence[Axis | {...}], optional
        The axes, in order, where `...` stands for zero or more axes
        about which nothing is known.
    name : str, optional
        The name of the system, if any.
    """

    axes: NotKwOnly[Axes[_VariadicAxes]] = [...]
    """
    The axes, in order.
    `None` is automatically converted to `[...]`, which says nothing
    about the axes.
    """

    name: tx.Optional[str] = None
    """The name of the system, if any."""

    order: tx.Optional[tx.Literal["C", "F"]] = None
    """The memory order, which is only valid for an array system."""

    # --- construction -------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        bind_axes_default(cls)

    # --- validation ---------------------------------------------------

    def __post_init__(self) -> None:
        if self.axes is None:
            self.axes = [...]
        if sum(a is ... for a in self.axes) > 1:
            raise ValueError(
                "The axes of a coordinate system hold at most one `...`, "
                "which stands for all the axes about which nothing is known."
            )
        from .array import ArrayCoordinateSystem

        if self.order is not None and not isinstance(
            self, ArrayCoordinateSystem
        ):
            raise ValueError(
                f"A {type(self).__name__} indexes no array, so it has no "
                f"memory order: order={self.order!r}."
            )

    # --- properties ---------------------------------------------------

    @property
    def ndim(self) -> tx.Optional[int]:
        """The number of axes, or `None` for an open system.

        !!! example
            ```pycon
            >>> CoordinateSystem(axes=[Axis(), Axis()]).ndim
            2
            >>> CoordinateSystem(axes=[Axis(), ...]).ndim is None
            True
            >>> CoordinateSystem().ndim is None
            True
            ```
        """
        return self.axes.ndim

    # --- operations ---------------------------------------------------

    def expand(self, ndim: int) -> tx.Self:
        """Return the closed system of `ndim` axes that this system describes.

        The `...` is replaced with unknown axes of the type that the class
        declares, as [`AxisSequence.expand`][] does. The class is then called
        again with the new axes and the other fields unchanged, so the result
        may be a subclass. For example, a [`SpatialCoordinateSystem`][] closed
        to three axes is a [`SpatialCoordinateSystem3D`][]. A system that is
        already closed is returned unchanged.

        !!! example
            ```pycon
            >>> CoordinateSystem(axes=[Axis(name="x"), ...]).expand(4)
            CoordinateSystem(axes=[Axis(name='x'), Axis(), Axis(), Axis()])
            >>> CoordinateSystem().expand(2)
            CoordinateSystem2D(axes=(Axis(), Axis()))
            ```

        Parameters
        ----------
        ndim : int
            The number of axes.

        Returns
        -------
        CoordinateSystem
            The closed system.

        Raises
        ------
        ValueError
            If `ndim` is smaller than the number of explicit axes of an open
            system, or differs from the number of axes of a closed system.
        TypeError
            If `ndim` is not an integer.
        """
        expanded = self.axes.expand(ndim)
        if not self.axes.is_variadic:
            return self
        return replace(self, axes=expanded)

    def restrict(
        self, refs: tx.Iterable[tx.Union[int, str]]
    ) -> "CoordinateSystem":
        """Return the system formed by a selection of the axes.

        Each reference selects an axis by position or by name, as in
        [`AxisSequence.restrict`][], and the selected axes follow the order of
        `refs`. A position covered by the `...` of an open system gives an
        unknown axis. The result describes another space,
        so neither the class nor the name is carried over: the result is
        whatever `CoordinateSystem(axes=...)` builds from the selected axes.

        !!! example
            ```pycon
            >>> x, y, z = Axis(name="x"), Axis(name="y"), Axis(name="z")
            >>> CoordinateSystem(axes=[x, y, z]).restrict(["z", 0])
            CoordinateSystem2D(axes=(Axis(name='z'), Axis(name='x')))
            >>> CoordinateSystem(axes=[x, ...]).restrict([0])
            CoordinateSystem(axes=[Axis(name='x')])
            ```

        Parameters
        ----------
        refs : iterable of int or str
            The positions or names of the axes to keep.

        Returns
        -------
        CoordinateSystem
            A closed system with one axis per reference.

        Raises
        ------
        ValueError, IndexError, TypeError
            As raised by [`AxisSequence.restrict`][].
        """
        return CoordinateSystem(axes=self.axes.restrict(refs))

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> "CoordinateSystem":
        """Return the system of a larger space that contains this system's
        axes.

        The method is the inverse of [`restrict`][]. As in
        [`AxisSequence.embed`][], axis `j` of this system is placed at position
        `positions[j]`, and every other position holds an unknown axis. As with
        [`restrict`][], neither the class nor the name is carried over.

        !!! example
            ```pycon
            >>> x = Axis(name="x")
            >>> CoordinateSystem(axes=[x]).embed([1])
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Ellipsis])
            >>> CoordinateSystem(axes=[x]).embed([1], ndim=4)
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Axis(), Axis()])
            ```

        Parameters
        ----------
        positions : iterable of int
            The non-negative position of each axis in the larger space.
        ndim : int, optional
            The number of axes of the larger space. If omitted, the result is
            open and ends with `...` after the last embedded axis.

        Returns
        -------
        CoordinateSystem
            A closed system if `ndim` is given, and an open one otherwise.

        Raises
        ------
        ValueError, TypeError
            As raised by [`AxisSequence.embed`][].
        """
        return CoordinateSystem(axes=self.axes.embed(positions, ndim=ndim))

    def compatible_with(self, other: tx.Optional["CoordinateSystem"]) -> bool:
        """Return whether two systems could describe the same space.

        Only the axes are compared, with [`AxisSequence.compatible_with`][]:
        the systems are compatible when some choice of the axes that each `...`
        stands for makes them match pairwise under [`Axis.compatible_with`][].
        Unlike `==`, an unknown axis matches any axis, and `None` is compatible
        with every system. The relation is symmetric but not transitive.

        !!! example
            ```pycon
            >>> x, t = SpaceAxis(name="x"), TimeAxis()
            >>> CoordinateSystem(axes=[x, ...]).compatible_with(
            ...     CoordinateSystem(axes=[x, Axis(), t])
            ... )
            True
            >>> CoordinateSystem(axes=[..., t]).compatible_with(
            ...     CoordinateSystem(axes=[x])
            ... )
            False
            ```

        Raises
        ------
        TypeError
            If `other` is neither a coordinate system nor `None`.
        """
        if other is not None and not isinstance(other, CoordinateSystem):
            raise TypeError(
                f"A coordinate system is compatible only with another "
                f"CoordinateSystem or None, not with {type(other).__name__}."
            )
        other_axes = getattr(other, "axes", AxisList([...]))
        return self.axes.compatible_with(other_axes)
