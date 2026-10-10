"""
Reading and writing of MGH and MGZ files.

MGH is the big-endian volume format of FreeSurfer, and MGZ is the same
format gzipped. A file holds a fixed header (version, dimensions, voxel
type, `goodRASFlag`, and the geometry described in
[`brainhops.io.common.freesurfer`][]), the voxels in Fortran order, an
optional footer of acquisition parameters, and optional trailing tags
such as the command history.

Everything but the voxels is held as an [`MghRaw`][] record, and
[`MghMetadata`][] reads and writes that record without the voxels.
[`MghImage`][brainhops.io.images.freesurfer.mgh.MghImage] holds such
metadata and a lazy array of the voxels. The header and the footer are
held in a nibabel header, and the tags as the bytes that the file
stores, so that a file that is read and written again without changes
keeps its bytes.

!!! warning "goodRASFlag"
    When the flag is not positive, FreeSurfer ignores the stored
    geometry and uses 1 mm voxels, coronal LIA cosines and a zero
    centre. nibabel's default cosines are LSP instead, which disagrees
    with FreeSurfer and with nibabel's own tkr matrix. The properties of
    [`MghRaw`][] follow FreeSurfer, while its nibabel header keeps the
    flag and the geometry that the file stores.
"""

__all__ = [
    "MghMetadata",
    "MghRaw",
]

from ._metadata import MghMetadata
from ._raw import MghRaw
