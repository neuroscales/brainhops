"""Format-independent affine transformations between the standard voxel,
RAS and LPS coordinate systems.

Each class is bound to the systems it names, and checks that its endpoints
are compatible with them. The names of the axes are not compared, and an
endpoint with fewer axes is compared with the first axes of the system, as
the world of a 2-D image is the first two axes of RAS or LPS. A
transformation between other systems is therefore not relabelled as one of
these classes: `Scaling(input=LPSmm(), output=LPSmm()).to(VoxelToRAS)`
raises an
[`IncompatibleSystemError`][brainhops.errors.IncompatibleSystemError]. An
endpoint that is not known, or has axes that are not known, is accepted.
"""

import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.axes import Axis
from brainhops.errors import IncompatibleSystemError


class VoxelToRAS(_xforms.Affine):
    """Affine transformation from voxel coordinates to RAS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        _check_endpoint(self, "input", _VOXEL)
        _check_endpoint(self, "output", _RAS)


class RASToVoxel(_xforms.Affine):
    """Affine transformation from RAS millimetres to voxel coordinates."""

    _input: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()
    _output: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )

    # Both classes map the same spaces in opposite directions, so each is the
    # inverse of the other. Declaring the pairing once, on the class defined
    # second, links the two classes both ways.
    _reverseof: tx.ClassVar[type] = VoxelToRAS

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        _check_endpoint(self, "input", _RAS)
        _check_endpoint(self, "output", _VOXEL)


class VoxelToLPS(_xforms.Affine):
    """Affine transformation from voxel coordinates to LPS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        _check_endpoint(self, "input", _VOXEL)
        _check_endpoint(self, "output", _LPS)


class LPSToVoxel(_xforms.Affine):
    """Affine transformation from LPS millimetres to voxel coordinates."""

    _input: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()
    _output: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )

    _reverseof: tx.ClassVar[type] = VoxelToLPS

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        _check_endpoint(self, "input", _LPS)
        _check_endpoint(self, "output", _VOXEL)


class RASToRAS(_xforms.Affine):
    """Affine transformation from one RAS world space to another.

    A typical example is the result of a registration, which maps the world
    coordinates of one image to those of another. The class is a data model of
    its own rather than a plain
    [`Affine`][brainhops.datamodel.transformations.Affine], so that an affine
    with arbitrary endpoints is never written to a format that can only store
    RAS-to-RAS transformations.
    """

    _input: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        _check_endpoint(self, "input", _RAS)
        _check_endpoint(self, "output", _RAS)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------

_VOXEL = _systems.VoxelCoordinateSystem()
"""The voxels of a grid."""

_RAS = _systems.RASmm()
"""The RAS world, in millimetres."""

_LPS = _systems.LPSmm()
"""The LPS world, in millimetres."""


def _check_endpoint(
    t: _xforms.Transformation, name: str, expected: _systems.CoordinateSystem
) -> None:
    """
    Refuse an endpoint of `t` that cannot be the system its class names.

    The endpoint (`name` is `"input"` or `"output"`) must be compatible with
    `expected`, in the sense of
    [`CoordinateSystem.compatible_with`][brainhops.datamodel.systems.CoordinateSystem.compatible_with],
    with two allowances. The names of the axes are not compared: a name
    labels an axis, and `RASmm` calls its axes `x`, `y` and `z` where
    `RASCoordinateSystem` calls them by their orientation. And a system
    with fewer axes is compared with the first axes of `expected`, as the
    world of a 2-D image is the first two axes of RAS or LPS.

    Raises
    ------
    IncompatibleSystemError
        If the endpoint is not compatible with `expected`.
    """
    system = getattr(t, name)
    if _compatible(system, expected):
        return
    raise IncompatibleSystemError(
        f"The {name} of a {type(t).__name__} is the {expected.name} "
        f"system, and {system!r} is not compatible with it. A "
        f"transformation between other systems is not relabelled as a "
        f"{type(t).__name__}: bridge its systems first, or keep it as a "
        f"plain transformation."
    )


def _compatible(
    system: tx.Optional[_systems.CoordinateSystem],
    expected: _systems.CoordinateSystem,
) -> bool:
    if system is None:
        return True
    axes = [axis if axis is ... else _unnamed(axis) for axis in system.axes]
    wanted = [_unnamed(axis) for axis in expected.axes]
    if all(axis is not ... for axis in axes):
        wanted = wanted[: len(axes)]
    wanted = _systems.CoordinateSystem(axes=wanted)
    return wanted.compatible_with(_systems.CoordinateSystem(axes=axes))


def _unnamed(axis: Axis) -> Axis:
    # The axis without its name. An anatomical axis cannot be unnamed and
    # takes the name of its orientation, which every axis of that
    # orientation rebuilt here takes too.
    return Axis(
        type=axis.type,
        unit=axis.unit,
        discrete=axis.discrete,
        orientation=axis.orientation,
    )
