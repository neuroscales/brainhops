"""Coordinate systems, from unitless arrays to anatomical spaces.

A coordinate system describes the axes of a space. Calling
[`CoordinateSystem`][] does not always build a plain `CoordinateSystem`: it
builds the most specific subclass that the given axes describe, and this
selection of a subclass is called dispatch below. Each class declares, in its
`on=` constraint, the axes that it stands for:

| The axes are...                               | ...so the system is        |
| --------------------------------------------- | -------------------------- |
| two or three of anything                      | `CoordinateSystem2D`/`3D`  |
| all spatial                                   | `SpatialCoordinateSystem*` |
| two or three, all measured in samples         | `ArrayCoordinateSystem*`   |
| spatial *and* measured in samples             | `Pixel`/`VoxelCoordinate…` |
| oriented right, anterior, superior (in order) | `RASCoordinateSystem`      |
| ... and in millimetres                        | `RASmm`                    |

The same rules apply to LPS and RSA systems. Some classes inherit from two
classes that dispatch can select. For example, `SpatialCoordinateSystem3D`
inherits from `CoordinateSystem3D` and from `SpatialCoordinateSystem`. Such a
class has no constraint of its own and is selected when the constraints of
both parents hold.

The memory order of an array cannot be read from its axes, so the C-ordered
and F-ordered variants are selected on the `order` field together with the
axes. The field holds `"C"`, `"F"`, or `None` when the order is unspecified:

| `order=`, and the axes are...           | ...so the system is            |
| --------------------------------------- | ------------------------------ |
| `"C"` / `"F"`, anything                 | `C`/`FArrayCoordinateSystem`   |
| ... two or three                        | `C`/`FArrayCoordinateSystem2D/3D` |
| ... two, spatial                        | `C`/`FPixelCoordinateSystem`   |
| ... three, spatial                      | `C`/`FVoxelCoordinateSystem`   |
| `"F"`, oriented R, A, S                 | `FRASCoordinateSystem`         |
| `"C"`, oriented S, A, R (a C-ordered    | `CRASCoordinateSystem`         |
| grid lists its axes z, y, x)            |                                |

Every class accepts `order` and passes it on, so `CoordinateSystem(axes=<RAS
axes>, order="F")` builds an `FRASCoordinateSystem`. Only array systems have an
order, so `RASmm(order="F")` raises a ValueError.

Every row of these tables states something about all the axes of a system, so
only closed systems are dispatched. An open system is a system whose
[`AxisList`][] holds `...`. Such a system does not know all its axes, so it is
built as the class it was called as: `CoordinateSystem(axes=[x, ...])` is not
two-dimensional. Once [`CoordinateSystem.expand`][] has closed the system, the
closed system is dispatched normally.
"""

from ._systems import *  # noqa: F403
from ._systems import __all__  # noqa: F401
