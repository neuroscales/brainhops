import typing_extensions as tx

from brainhops.errors import AxisError

from .systems import AxisList, CoordinateSystem


def get_axes(system: tx.Optional[CoordinateSystem]) -> AxisList:
    """Return the axes of a coordinate system.

    Parameters
    ----------
    system : CoordinateSystem or None
        A coordinate system, or `None`.

    Returns
    -------
    AxisList
        The axes of the system, or `[...]` (unknown axes) when `system`
        is `None` or has no `axes` attribute.
    """
    return getattr(system, "axes", AxisList([...]))


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
