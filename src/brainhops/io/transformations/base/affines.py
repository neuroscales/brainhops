"""Format-independent affine transformations between the standard voxel,
RAS and LPS coordinate systems.

Each class is bound to the coordinate systems that its name states, and
it checks, when it is built, that its endpoints are compatible with
those systems. The names of the axes are not compared. An endpoint with
fewer axes than the system is compared with the first axes of the
system, because the world of a 2-D image is made of the first two axes
of RAS or LPS. An endpoint that is not known, or whose axes are not
known, is accepted.

The check means that a transformation between other systems cannot be
given one of these classes while keeping its parameters. For example,
`Scaling(input=LPSmm(), output=LPSmm()).to(VoxelToRAS)` raises an
[`IncompatibleSystemError`][brainhops.errors.IncompatibleSystemError].
"""

import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

from ._endpoints import check_endpoint


class VoxelToRAS(_xforms.Affine):
    """Affine transformation from voxel coordinates to RAS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        check_endpoint(self, "input", _systems.VoxelCoordinateSystem())
        check_endpoint(self, "output", _systems.RASmm())


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
        check_endpoint(self, "input", _systems.RASmm())
        check_endpoint(self, "output", _systems.VoxelCoordinateSystem())


class VoxelToLPS(_xforms.Affine):
    """Affine transformation from voxel coordinates to LPS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        check_endpoint(self, "input", _systems.VoxelCoordinateSystem())
        check_endpoint(self, "output", _systems.LPSmm())


class LPSToVoxel(_xforms.Affine):
    """Affine transformation from LPS millimetres to voxel coordinates."""

    _input: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()
    _output: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )

    _reverseof: tx.ClassVar[type] = VoxelToLPS

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        check_endpoint(self, "input", _systems.LPSmm())
        check_endpoint(self, "output", _systems.VoxelCoordinateSystem())


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
        check_endpoint(self, "input", _systems.RASmm())
        check_endpoint(self, "output", _systems.RASmm())
