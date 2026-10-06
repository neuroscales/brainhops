"""
Readers and writers for transformations stored in NIfTI files.

Fields of RAS displacements and of RAS coordinates
--------------------------------------------------
The NIfTI-1 standard (`nifti1.h`) gives a field of vectors two intent
codes, and they do not mean the same map:

- `DISPVECT` (1006), "specifically for displacements", is read and
  written by [`NiftiRASDisplacementField`][].
- `VECTOR` (1007), "for any other type of vector", is read and written
  by [`NiftiRASCoordinatesField`][].

A displacement field maps `x -> x + u(x)`; a coordinates field maps a
voxel to the position its vector holds. Reading one as the other moves
every point by the whole world position of its voxel, so the intent code
decides between them.

`DISPVECT` is read as RAS displacements in millimetres, as ITK 5.4 and
later reads it. A field of coordinates is written as `VECTOR`, with the
intent name `"Mapping"`, which is what SPM writes for its `y_`
deformations -- the same kind of field. `POINTSET` (1008) would name the
content better ("the vector value at each voxel is really a spatial
coordinate"), but the standard ties it to a flat list of points
(`dim[2] = dim[3] = dim[4] = 1`), and readers -- brainhops included --
lay out its axes as one, so a grid written with it would be misread.

!!! warning "Coordinates fields written by brainhops before this change"
    Older versions of brainhops wrote fields of RAS *coordinates* with
    the `DISPVECT` intent and nothing else to mark them. Such a file is
    byte-for-byte a standard displacement field, so it is now read as
    one. Nothing in the file tells the two apart, and guessing from the
    values would be just that, so read them explicitly:
    `load(path, hint="nifti.coordinates")`, or
    `NiftiRASCoordinatesField.from_file(path)`. Saving the result
    rewrites it with the `VECTOR` intent.
"""

__all__ = [
    "NiftiBasedTransformation",
    "NiftiRASCoordinatesField",
    "NiftiRASDisplacementField",
    "NiftiRASToVoxel",
    "NiftiVoxelToRAS",
    "ReadOnlyNiftiBasedTransformation",
    "converters",
]

from .affines import NiftiRASToVoxel, NiftiVoxelToRAS
from .base import NiftiBasedTransformation, ReadOnlyNiftiBasedTransformation
from .fields import NiftiRASCoordinatesField, NiftiRASDisplacementField

# The converters into these formats register themselves on import.
from . import converters  # noqa: E402, F401  isort: skip
