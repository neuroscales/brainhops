"""
The shared MGH/MGZ-reading and MGH/MGZ-writing machinery.

MGH is FreeSurfer's volume format, and MGZ is the same bytes gzipped.
A file holds, in order and big-endian:

1. a fixed 284-byte header: `version` (always 1), the four dimensions
   `width, height, depth, nframes`, the voxel `type`, `dof`,
   `goodRASFlag`, the voxel size `delta`, the direction cosines `Mdc` and
   the RAS centre of the volume `Pxyz_c` (see
   [`brainhops.io.common.freesurfer`][]);
2. the voxels, x fastest, then y, z and frames (F order);
3. an optional footer of MRI acquisition parameters: `TR` (ms),
   `flip_angle` (radians), `TE` (ms), `TI` (ms) and `FoV`;
4. optional trailing *tags* (the command line history, the talairach
   transform file name, ...).

The header and the voxels are read and written with `nibabel`
(`nibabel.freesurfer.mghformat`). Two pieces `nibabel` drops are read
here from the raw bytes, so that a file round-trips:

- the trailing tags, kept verbatim as bytes;
- `goodRASFlag`, which `nibabel` silently resets to 1 (see below).

!!! warning "`goodRASFlag`"
    When `goodRASFlag` is not positive, FreeSurfer ignores the voxel size,
    the direction cosines and the centre stored in the header and uses
    its defaults instead: 1 mm voxels, coronal LIA direction cosines and
    a zero centre. `nibabel` also resets the voxel size and the centre,
    but its default direction cosines are those of an LSP volume, which
    disagrees with FreeSurfer (and with `nibabel`'s own tkr matrix). The
    readers here follow FreeSurfer.
"""

__all__ = [
    "MghParser",
]

from ._parsers import MghParser
