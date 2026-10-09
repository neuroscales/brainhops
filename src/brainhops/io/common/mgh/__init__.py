"""
Reading and writing of MGH and MGZ files.

MGH is the big-endian volume format of FreeSurfer, and MGZ is the same
format gzipped. A file holds a fixed header (version, dimensions, voxel
type, `goodRASFlag`, and the geometry described in
[`brainhops.io.common.freesurfer`][]), the voxels in Fortran order, an
optional footer of acquisition parameters, and optional trailing tags
such as the command history.

The header and voxels are read and written with nibabel. The trailing
tags and the `goodRASFlag`, which nibabel drops or resets to 1, are
read from the raw bytes so that a file round-trips.

!!! warning "goodRASFlag"
    When the flag is not positive, FreeSurfer ignores the stored
    geometry and uses 1 mm voxels, coronal LIA cosines and a zero
    centre. nibabel's default cosines are LSP instead, which disagrees
    with FreeSurfer and with nibabel's own tkr matrix. This module
    follows FreeSurfer.
"""

__all__ = [
    "MghReaderWriter",
]

from ._parsers import MghReaderWriter
