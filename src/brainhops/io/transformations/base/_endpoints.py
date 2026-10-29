"""Checks that the endpoints of a transformation are the expected systems.

The classes of [`affines`][brainhops.io.transformations.base.affines] and
[`fields`][brainhops.io.transformations.base.fields] are bound to the
voxel, RAS and LPS systems that their names state. When such a class is
built, or a format writes a transformation as one of its kinds, these
functions check that the endpoints of the transformation are compatible
with those systems.
"""

import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.errors import IncompatibleSystemError


def check_endpoint(
    t: _xforms.Transformation, name: str, expected: _systems.CoordinateSystem
) -> None:
    """
    Refuse an endpoint of `t` that cannot be the system its class names.

    The endpoint must be compatible with `expected`, in the sense of
    [`CoordinateSystem.compatible_with`][brainhops.datamodel.systems.CoordinateSystem.compatible_with],
    with two allowances. First, the names of the axes are not compared,
    because a name only labels an axis. For example, `RASmm` calls its
    axes `x`, `y` and `z`, whereas `RASCoordinateSystem` calls them by
    their orientation. Second, a system with fewer axes is compared with
    the first axes of `expected`, because the world of a 2-D image is
    made of the first two axes of RAS or LPS.

    Parameters
    ----------
    t : Transformation
        The transformation whose endpoint is checked.
    name : str
        The endpoint to check, either `"input"` or `"output"`.
    expected : CoordinateSystem
        The system that the class of `t` is bound to on that side.

    Raises
    ------
    IncompatibleSystemError
        If the endpoint is not compatible with `expected`.
    """
    system = getattr(t, name)
    if compatible(system, expected):
        return
    raise IncompatibleSystemError(
        f"The {name} of a {type(t).__name__} is the {expected.name} "
        f"system, and {system!r} is not compatible with it. A "
        f"transformation between other systems is not relabelled as a "
        f"{type(t).__name__}: bridge its systems first, or keep it as a "
        f"plain transformation."
    )


def compatible(
    system: tx.Optional[_systems.CoordinateSystem],
    expected: _systems.CoordinateSystem,
) -> bool:
    """Tell whether an endpoint can be the system that a class names.

    The names of the axes are not compared, and a system with fewer axes
    is compared with the first axes of `expected`, as
    [`check_endpoint`][] explains.

    Parameters
    ----------
    system : CoordinateSystem, optional
        The endpoint, or `None` when it is not known, which is accepted.
    expected : CoordinateSystem
        The system that the class is bound to on that side.

    Returns
    -------
    bool
        Whether the endpoint is compatible with `expected`.
    """
    if system is None:
        return True
    axes = [
        axis if axis is ... else _unnamed(axis) for axis in get_axes(system)
    ]
    wanted = [_unnamed(axis) for axis in get_axes(expected)]
    if all(axis is not ... for axis in axes):
        wanted = wanted[: len(axes)]
    wanted = _systems.CoordinateSystem(axes=wanted)
    return wanted.compatible_with(_systems.CoordinateSystem(axes=axes))


def _unnamed(axis: Axis) -> Axis:
    # Rebuild the axis without its name. An anatomical axis cannot be
    # left unnamed, so it takes the name of its orientation, and every
    # axis with that orientation that is rebuilt here takes the same
    # name.
    return Axis(
        type=axis.type,
        unit=axis.unit,
        discrete=axis.discrete,
        orientation=axis.orientation,
    )
