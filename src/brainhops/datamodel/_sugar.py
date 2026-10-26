import typing_extensions as tx

from brainhops.errors import AxisError

from .axes import AxisList, AxisSequence

if tx.TYPE_CHECKING:
    from .systems import CoordinateSystem


def get_axes(
    system: tx.Union["CoordinateSystem", AxisSequence, None],
) -> AxisSequence:
    """Return an axis sequence, treating missing axes as unknown.

    Parameters
    ----------
    system : CoordinateSystem | AxisSequence | None
        A coordinate system, its axes, or `None`.

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
        return system
    axes = getattr(system, "axes", None)
    if axes is None:
        return AxisList([...])
    if not isinstance(axes, AxisSequence):
        return AxisList(list(axes))
    return axes


def vector_axis(
    axes: tx.Optional[tx.Sequence[tx.Any]],
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
