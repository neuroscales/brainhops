# stdlib
import re

# dependencies
import typing_extensions as tx

# brainhops
from brainhops.errors import AxisError, OnError, raise_or_warn

# internals
from ._axes.components import COMPONENT_TYPES
from .axes import Axis, AxisList, AxisSequence

if tx.TYPE_CHECKING:
    from .systems import CoordinateSystemLike


def get_axes(
    system: tx.Optional["CoordinateSystemLike"],
    ndim: tx.Optional[int] = None,
    on_error: OnError = "raise",
) -> AxisSequence:
    """Return an axis sequence, treating missing axes as unknown.

    Parameters
    ----------
    system : CoordinateSystemLike | Sequence[Axis] | None
        A coordinate system, its axes, or `None`.
    ndim : int, optional
        If provided and the input system is variadic, expand to `ndim`
        axes.
    on_error : {"raise", "warn", "ignore"} | Exception | Any, default="raise"
        What to do when the system does not have `ndim` axes, as
        [`raise_or_warn`][brainhops.errors.raise_or_warn] describes. Any
        other value, such as `None`, is returned instead of the axes.

    Returns
    -------
    AxisSequence
        The stored axes.
        Missing systems and `axes=None` give a fresh `AxisList([...])`,
        without changing the stored value. An empty sequence stays empty.
        Axes stored as a plain tuple or list, as the default of a system
        class may be, are returned as an `AxisList` of the same axes.
    """
    if isinstance(system, AxisSequence):
        axes = system
    elif isinstance(system, (list, tuple)):
        # A plain sequence of axes, as a CoordinateSystemLike may be.
        axes = system
    else:
        axes = getattr(system, "axes", None)
    if axes is None:
        axes = AxisList([...])
    elif not isinstance(axes, AxisSequence):
        axes = AxisList(list(axes))
    if ndim is not None:
        if axes.is_variadic:
            axes = axes.expand(ndim)
        if len(axes) != ndim:
            if on_error == "ignore":
                return axes
            msg = f"Expected {ndim} axes, but got {len(axes)}."
            return raise_or_warn(
                msg, on_error=on_error, default=(ValueError, UserWarning)
            )
    return axes


_AxisFilter: tx.TypeAlias = tx.Union[
    tx.Callable[[Axis], bool],
    tx.Mapping[str, tx.Any],
]


def _make_filter(filter: _AxisFilter) -> tx.Callable[[Axis], bool]:
    if callable(filter):
        return filter

    def _make_one_filter(target: tx.Any) -> tx.Callable[[tx.Any], bool]:

        if isinstance(target, re.Pattern):

            def _one_filter(value: tx.Any) -> bool:
                return (
                    isinstance(value, str)
                    and target.fullmatch(value) is not None
                )

        elif tx.get_origin(target) is tx.Literal:
            targets = tx.get_args(target)

            def _one_filter(value: tx.Any) -> bool:
                return value in targets

        else:

            def _one_filter(value: tx.Any) -> bool:
                return value == target

        return _one_filter

    filters = {name: _make_one_filter(value) for name, value in filter.items()}

    def _filter(axis: Axis) -> bool:
        return all(f(getattr(axis, k)) for k, f in filters.items())

    return _filter


def filter_axes(
    filter: _AxisFilter,
    axes: tx.Optional["CoordinateSystemLike"],
    ndim: tx.Optional[int] = None,
) -> tx.Iterator[Axis]:
    """Return the axes of a system that satisfy a filter.

    Parameters
    ----------
    filter : callable | dict[str, Any]
        A predicate that receives an axis and returns whether it is kept,
        or a dictionary of attribute names and values.
    axes : CoordinateSystemLike
        A coordinate system, its axes, or `None`.
    ndim : int, optional
        If provided and the input system is variadic, expand to `ndim`
        axes.

    Returns
    -------
    Iterator[Axis]
        The indices and axes that satisfy the filter.
    """
    axes = get_axes(axes, ndim)
    filter = _make_filter(filter)
    return (axis for axis in axes if axis is not ... and filter(axis))


def enumfilter_axes(
    filter: _AxisFilter,
    axes: tx.Optional["CoordinateSystemLike"],
    ndim: tx.Optional[int] = None,
) -> tx.Iterator[tx.Tuple[int, Axis]]:
    """Return the indices and axes of a system that satisfy a filter.

    Parameters
    ----------
    filter : callable | dict[str, Any]
        A predicate that receives an axis and returns whether it is kept,
        or a dictionary of attribute names and values.
    axes : CoordinateSystemLike
        A coordinate system, its axes, or `None`.
    ndim : int, optional
        If provided and the input system is variadic, expand to `ndim`
        axes.

    Returns
    -------
    Iterator[Tuple[int, Axis]]
        The indices and axes that satisfy the filter.
    """
    axes = get_axes(axes, ndim)
    filter = _make_filter(filter)
    return (
        (i, axis)
        for i, axis in enumerate(axes)
        if axis is not ... and filter(axis)
    )


def axis_index(
    axes: tx.Optional["CoordinateSystemLike"],
    key: tx.Union[int, str],
    ndim: tx.Optional[int] = None,
) -> int:
    """Return the index of an axis given by its index or its name.

    Parameters
    ----------
    axes : CoordinateSystemLike
        A coordinate system, its axes, or `None`.
    key : int or str
        The index or name of the axis.
    ndim : int, optional
        If provided and the input system is variadic, expand to `ndim`
        axes.

    Returns
    -------
    int
        The index of the axis in `axes`.

    Raises
    ------
    KeyError
        If no axis has that name or index.
    """
    axes = get_axes(axes, ndim)

    if isinstance(key, str):
        try:
            return axes.index(key)
        except ValueError as e:
            raise KeyError(f"The image has no axis named {key!r}.") from e

    index = int(key)
    ndim = len(axes)
    if not -ndim <= index < ndim:
        raise KeyError(f"The image has no axis {index}.")
    return index % ndim


def enum_component_axes(
    axes: tx.Optional["CoordinateSystemLike"],
) -> tx.Iterator[tx.Tuple[int, Axis]]:
    """Return the indices and the component axes among `axes`."""
    axes = get_axes(axes)
    return (
        (i, axis) for i, axis in enumerate(axes) if is_component_axis(axis)
    )


def is_component_type(value: tx.Any) -> bool:
    """Return whether an axis type describes a component axis.

    The readers and the writers of files group these axes with the
    channels, and only these axes can have a frame.

    Parameters
    ----------
    value : AxisType or str or None
        The type of an axis, such as `"vector"`. Any other value, such as
        `None`, gives `False`.

    Returns
    -------
    bool
        Whether the entries along an axis of this type are the components
        of a point, a vector or a tensor.
    """
    return isinstance(value, str) and value in COMPONENT_TYPES


def is_component_axis(axis: Axis) -> bool:
    """
    Return whether an axis holds the components of a point, vector or tensor.

    Parameters
    ----------
    axis : Axis
        The axis to test.

    Returns
    -------
    bool
        Whether the entries along the axis are the components of a point,
        a vector or a tensor.
    """
    # The unknown axes of a variadic system are an ellipsis, with no type.
    return is_component_type(getattr(axis, "type", None))


def is_point_axis(axis: Axis) -> bool:
    """Return whether the entries of an axis are coordinates of points."""
    return getattr(axis, "variances", None) is None


def get_vector_axis(
    axes: tx.Optional["CoordinateSystemLike"],
) -> tx.Tuple[str, int]:
    """Return the type and index of the vector axis of a field.

    A vector field has its grid axes and exactly one vector axis, of type
    `"displacement"` or `"coordinate"`, which holds the vector components.

    Parameters
    ----------
    axes : sequence or None
        The axes of the field. `None` counts as no axes.

    Returns
    -------
    tuple of (str, int)
        The type of the vector axis and its index in `axes`.

    Raises
    ------
    AxisError
        If there is no vector axis, more than one, or one of each type.
    """
    axes = get_axes(axes)
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
