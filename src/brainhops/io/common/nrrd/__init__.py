"""Reading and writing machinery shared by all NRRD-based formats.

A NRRD file is a text header followed by the samples, which are either attached
in the same `.nrrd` file or detached in data files that a `.nhdr` header names:

```
NRRD0004
# Complete NRRD file format specification at:
# http://teem.sourceforge.net/nrrd/format.html
type: short
dimension: 4
space: left-posterior-superior
sizes: 3 64 64 32
space directions: none (2,0,0) (0,2,0) (0,0,2.5)
kinds: vector domain domain domain
endian: little
encoding: gzip
space origin: (-63,-63,-40)
measurement frame: (1,0,0) (0,1,0) (0,0,1)
DWMRI_b-value:=1000

<data>
```

The module follows the NRRD specification and supports every encoding, every
scalar type except `block`, and all three forms of the `data file` field. Raw
data in a single local file are memory-mapped when read from a path.

!!! note "Why not pynrrd"
    pynrrd supports neither `LIST` or pattern data files nor `hex`, and it
    always loads the data into memory.
"""

__all__ = [
    "NrrdParser",
    "NrrdHeader",
]

from ._header import NrrdHeader
from ._parsers import NrrdParser
