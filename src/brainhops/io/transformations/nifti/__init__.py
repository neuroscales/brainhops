"""
Readers and writers for transformations stored in NIfTI files.

Each transformation holds the header of its file as
[`NiftiMetadata`][brainhops.io.common.nifti.NiftiMetadata], whose record
is a [`NiftiRaw`][brainhops.io.common.nifti.NiftiRaw], and the array of
the file as it is stored, in `rawdata`. The two classes of the header
are defined in [`brainhops.io.common.nifti`][] and exported here as
well. An affine keeps its matrix in the header, and the `data` of a
field is a lazy array of its file, which reads the voxels only when a
computation needs them. A field keeps its header and its array in a
[`NiftiImage`][brainhops.io.images.nifti.NiftiImage], through which it
is read and written. A field that is read and written again without
changes is written as its file stored it.

## Fields of RAS displacements and of RAS coordinates

The NIfTI-1 standard (`nifti1.h`) gives vector fields two intent codes with
different meanings. `DISPVECT` (1006) is "specifically for displacements" and
is read and written by [`NiftiRASDisplacementField`][]. `VECTOR` (1007) is "for
any other type of vector" and is read and written by
[`NiftiRASCoordinateField`][].

A displacement field maps `x` to `x + u(x)`, while a coordinate field maps
each voxel to the position that its vector holds. Reading one kind as the other
shifts every point by the world position of its voxel, so the intent code
decides. `DISPVECT` fields are read as RAS displacements in millimetres, as ITK
5.4 and later read them. Coordinates are written under `VECTOR` with the intent
name `"Mapping"`, which is what SPM writes for its `y_` deformations, a field
of the same kind.

`POINTSET` (1008) would describe coordinates better, but the standard ties it
to a flat list of points (`dim[2] = dim[3] = dim[4] = 1`), so readers,
brainhops included, would misread a grid written with it.
"""

__all__ = [
    "NiftiBackedTransformation",
    "NiftiMetadata",
    "NiftiRASCoordinateField",
    "NiftiRASDisplacementField",
    "NiftiRASToVoxel",
    "NiftiRaw",
    "NiftiVoxelToRAS",
]

from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw

from ._affines import NiftiRASToVoxel, NiftiVoxelToRAS
from ._base import NiftiBackedTransformation
from ._fields import NiftiRASCoordinateField, NiftiRASDisplacementField

# The converters into these formats register themselves when this module
# is imported, here, so that importing the formats makes `t.to(Format)`
# work. The SPM converters import theirs from it too, and rely on it.
from . import _converters  # noqa: E402, F401  isort: skip


# -- deprecated names ----------------------------------------------------
# The old spellings of the coordinate fields (#547) still import, with a
# DeprecationWarning, for one release.
from brainhops._core.deprecation import (  # noqa: E402
    renamed_attributes as _renamed_attributes,
)

__getattr__ = _renamed_attributes(
    __name__,
    {
        "NiftiRASCoordinatesField": "NiftiRASCoordinateField",
    },
)
