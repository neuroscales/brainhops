"""Format-independent affine transformations between the standard voxel,
RAS and LPS coordinate systems.
"""

import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class VoxelToRAS(_xforms.Affine):
    """Affine transformation from voxel coordinates to RAS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()


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


class VoxelToLPS(_xforms.Affine):
    """Affine transformation from voxel coordinates to LPS millimetres."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()


class LPSToVoxel(_xforms.Affine):
    """Affine transformation from LPS millimetres to voxel coordinates."""

    _input: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()
    _output: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )

    _reverseof: tx.ClassVar[type] = VoxelToLPS


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
