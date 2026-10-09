"""
Readers and writers for transformations stored in NIfTI files.

## Fields of RAS displacements and of RAS coordinates

The NIfTI-1 standard (`nifti1.h`) gives vector fields two intent codes with
different meanings. `DISPVECT` (1006) is "specifically for displacements" and
is read and written by [`NiftiRASDisplacementField`][]. `VECTOR` (1007) is "for
any other type of vector" and is read and written by
[`NiftiRASCoordinatesField`][].

A displacement field maps `x` to `x + u(x)`, while a coordinates field maps
each voxel to the position that its vector holds. Reading one kind as the other
shifts every point by the world position of its voxel, so the intent code
decides. `DISPVECT` fields are read as RAS displacements in millimetres, as ITK
5.4 and later read them. Coordinates are written under `VECTOR` with the intent
name `"Mapping"`, which is what SPM writes for its `y_` deformations, a field
of the same kind.

`POINTSET` (1008) would describe coordinates better, but the standard ties it
to a flat list of points (`dim[2] = dim[3] = dim[4] = 1`), so readers,
brainhops included, would misread a grid written with it.

!!! warning "Coordinates fields written by older versions of brainhops"
    Older versions of brainhops wrote RAS coordinates under `DISPVECT`, with no
    marker. Such a file is byte for byte a standard displacement field, so it
    is read as one, and guessing from the values is refused. The file is read
    correctly with `load(path, hint="nifti.coordinates")` or
    `NiftiRASCoordinatesField.from_file(path)`, and saving it rewrites it under
    the `VECTOR` intent.
"""

__all__ = [
    "NiftiBasedTransformation",
    "NiftiRASCoordinatesField",
    "NiftiRASDisplacementField",
    "NiftiRASToVoxel",
    "NiftiVoxelToRAS",
]

from .affines import NiftiRASToVoxel, NiftiVoxelToRAS
from .base import NiftiBasedTransformation
from .fields import NiftiRASCoordinatesField, NiftiRASDisplacementField

# The converters into these formats register themselves when this module
# is imported, here, so that importing the formats makes `t.to(Format)`
# work. The SPM converters import theirs from it too, and rely on it.
from . import _converters  # noqa: E402, F401  isort: skip
