"""
Reading and writing of AFNI datasets.

This module holds what all AFNI formats, images and transformations
alike, share: the header, the geometry, and the decoding and encoding of
voxel values. All of these formats derive from [`AfniFormat`][]. The
conventions follow the AFNI sources (`README.attributes`, `3ddata.h`).

A dataset is a pair of files. `prefix+view.HEAD` holds text attributes
and `prefix+view.BRIK` holds the voxel values, possibly compressed. The
view is `orig` (scanner space), `acpc`, or `tlrc` (Talairach or any
other template). The NIML (XML) variant of the header is not
supported.

The BRIK stores `DATASET_RANK[1]` sub-bricks (time points, statistics
or warp components) one after the other, with `x` varying fastest,
which gives a Fortran-ordered `(nx, ny, nz, nvals)` array. Each
sub-brick has its own type (`BRICK_TYPES`) and, optionally, a scale
factor (`BRICK_FLOAT_FACS`).

The world space is DICOM space, that is LPS in millimetres. The
`ORIENT_SPECIFIC`, `ORIGIN` and `DELTA` attributes define a signed
permutation from voxels to DICOM coordinates, the cardinal matrix, which
is recomputed on read rather than taken from `IJK_TO_DICOM`. An oblique
dataset also stores its true matrix in `IJK_TO_DICOM_REAL`. AFNI
programs compute on the cardinal grid, while `3dAFNItoNIFTI` stores the
true matrix as the NIfTI sform.
"""

__all__ = [
    "AfniFormat",
    "AfniParser",
    "AfniHeader",
]

from ._header import AfniHeader
from ._parsers import AfniFormat, AfniParser
