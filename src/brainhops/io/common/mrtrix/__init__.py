"""
Reading and writing of MRtrix images.

This module holds what the MRtrix image and transformation formats
share: the header, and the decoding and encoding of voxel data. An
MRtrix image is a text header followed by raw voxel values, either in
one file (`.mif`, or `.mif.gz` gzipped) or in two (a `.mih` header and
the data file it names). The header is a list of `key: value` lines
between `mrtrix image` and `END`, where `#` starts a comment:

```text
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
```

`layout` gives the storage rank and direction of each axis (see
[`parse_layout`][]). `transform` maps voxel coordinates multiplied by
the voxel sizes, not voxel indices, to scanner RAS millimetres; without
one, MRtrix centres the field of view on the origin rather than
assuming the identity. `scaling` gives `offset,scale`, by which a
stored value `v` means `offset + scale * v`. `file` names the data file
and an offset, where `.` means the header file itself. The conventions
follow the MRtrix3 sources.
"""

__all__ = [
    "MrtrixReaderWriter",
    "MrtrixHeader",
]

from ._header import MrtrixHeader
from ._parsers import MrtrixReaderWriter
