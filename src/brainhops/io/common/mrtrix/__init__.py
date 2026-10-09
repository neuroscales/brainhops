"""
The shared MRtrix-reading and MRtrix-writing machinery behind every
MRtrix-based image and transformation format.

An MRtrix image is a text header followed by raw voxel data. The header
and the data are either in the same file (`.mif`, or `.mif.gz` when the
whole file is gzip-compressed) or in two files (`.mih` for the header,
and the data file it names, conventionally `.dat`).

The header is a list of `key: value` lines, between a first line that
reads `mrtrix image` and a last line that reads `END`::

    mrtrix image
    dim: 64,64,32,7
    vox: 2,2,2.5,nan
    layout: -0,-1,+2,+3
    datatype: Float32LE
    transform: 0.996, 0.087, 0, -61.2
    transform: -0.087, 0.996, 0, -70.5
    transform: 0, 0, 1, -40
    scaling: 0,1
    dw_scheme: 0,0,1,0
    dw_scheme: 0,1,0,1000
    file: . 552
    END

The conventions below were checked against the MRtrix3 sources
(`core/formats/mrtrix_utils.{h,cpp}`, `core/formats/mrtrix{,_gz}.cpp`,
`core/file/key_value.cpp`, `core/stride.h`, `core/raw.h`,
`core/header.cpp`, `core/transform.h`, `core/datatype.cpp`).

* **Comments.** Everything after a `#` on a line is a comment, and is
  dropped. A value cannot therefore contain a `#`.
* **Keys.** The compulsory keys (`dim`, `vox`, `layout`, `datatype`,
  `transform`, `scaling`) are matched case-insensitively. Any other key
  is kept as it is spelled. A key that appears on several lines (the
  three rows of `transform`, the rows of a `dw_scheme`, the entries of a
  `command_history`) has one value per line; the other keys are
  collected as their lines joined by newlines, which is how MRtrix holds
  them.
* **`dim`, `vox`.** The size and the voxel size of each axis. `vox` may
  hold fewer entries than `dim` (but at least three, or as many as
  `dim` when it has fewer), and may hold `nan` for an axis that has no
  physical size.
* **`layout`.** One signed integer per axis, such as `-0,-1,+2`. The
  absolute value is the *rank* of the axis in the file: the axis of
  rank 0 changes fastest, the one of rank 1 next, and so on. The sign
  says whether the voxels along the axis are stored in increasing
  (`+`) or decreasing (`-`) order of their index. Voxel `[0, 0, ...]`
  is therefore *not* the first value in the file when any axis is
  negative: it is `sum((size[i] - 1) * |stride[i]|)` values in, over
  the negative axes. `layout: +0,+1,+2` is a plain Fortran-ordered
  (x fastest) array.
* **`datatype`.** One of `Bit`, `Int8`, `UInt8`, `[U]Int{16,32,64}`,
  `Float{32,64}` and `CFloat{32,64}`, the multi-byte ones optionally
  suffixed with `LE` or `BE`. Without a suffix the byte order is the
  native one of the machine. `Bit` packs eight voxels per byte, the
  first one in the most significant bit.
* **`transform`.** Three rows of four numbers, the top of a `4x4`
  matrix that maps *voxel coordinates multiplied by the voxel sizes*
  (not plain voxel indices) to scanner coordinates, in millimetres,
  in RAS+ (x to the right, y to the front, z up), like a NIfTI sform.
  The voxel-to-scanner matrix is `transform @ diag(vox[:3], 1)`. MRtrix
  writes the three direction columns at unit length; one that is not is
  normalised and its length moved into the voxel size, which leaves the
  voxel-to-scanner matrix unchanged.
* **Missing transform.** When `transform` is absent (or not finite),
  MRtrix centres the field of view on the origin: the rotation is the
  identity, and the translation is `-0.5 * (size - 1) * vox` along
  each of the first three axes. It is *not* the identity.
* **`scaling`.** `offset,scale`: a stored value `v` means
  `offset + scale * v`.
* **`file`.** `file: <name> [<offset>]`. A name of `.` means the data
  follow the header in the same file, `offset` bytes from its start.
  Any other name is the data file, relative to the header's directory,
  and its offset defaults to zero.

This package holds what an image reader and a transformation reader
(an MRtrix warp, for instance) share: the header, the decoding of the
voxel data into an array whose axes are the header's axes, and the
encoding of such an array back into bytes.
"""

__all__ = [
    "MrtrixParser",
    "MrtrixHeader",
]

from ._header import MrtrixHeader
from ._parsers import MrtrixParser
