"""
The shared NRRD-reading and NRRD-writing machinery behind every NRRD-based
format.

A NRRD ("Nearly Raw Raster Data") file is a text header followed by the
sample values. The header and the values are either in the same file
(*attached*, `.nrrd`) or the header names the file(s) that hold them
(*detached*, `.nhdr` with its `.raw`, `.raw.gz`, ...).

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

The conventions below follow the format specification
(<https://teem.sourceforge.net/nrrd/format.html>).

* **Magic.** The first line is `NRRD000X`, with `X` the version (1-5).
* **Lines.** `field: value` lines (the field names are matched without
  regard to case, and the spellings without a space -- `datafile`,
  `byteskip`, `axismins`, `centerings`, ... -- are accepted), `key:=value`
  pairs (`\\n` and `\\\\` are escapes in both), and `#` comments. The
  header ends at the first empty line, after which an attached header's
  data start; a detached header may also end at the end of its file.
* **Types.** Every spelling of the specification (`short`, `int16`,
  `int16_t`, `signed short int`, ...) of the signed and unsigned 8 to 64
  bit integers, `float` and `double`. `block` is not supported.
* **Encodings.** `raw`, `ascii` (`txt`, `text`), `hex`, `gzip` (`gz`) and
  `bzip2` (`bz2`). `endian` is required for multi-byte types in a binary
  encoding.
* **Data files.** `data file: <name>`, relative to the header's
  directory unless absolute; `data file: <format> <min> <max> <step>
  [<subdim>]`, a `printf` pattern expanded over the inclusive range; and
  `data file: LIST [<subdim>]`, followed by one file name per line until
  the end of the header. Several files are read in order and concatenated,
  each holding as many samples.
* **Skips.** `line skip` lines are skipped first, in the file as stored.
  `byte skip` bytes are then skipped -- after decompression for `gzip` and
  `bzip2` (as `pynrrd` and teem do) -- and `byte skip: -1` means that the
  data are the last bytes of the (decompressed) file.

Reading an attached `raw` file, or a detached one with a single `raw` data
file, from a local path memory-maps the values, so nothing but the header
is read until they are indexed.

!!! note "Why not `pynrrd`"
    NRRD headers are simple text, and `pynrrd` covers fewer of them than
    this parser does: it reads neither the `LIST` nor the pattern forms
    of `data file`, nor the `hex` encoding, and always reads the values
    into memory. A dedicated parser keeps the dependencies to numpy and
    the standard library (`gzip`, `bz2`).
"""

__all__ = [
    "NrrdParser",
    "NrrdHeader",
]

from ._header import NrrdHeader
from ._parsers import NrrdParser
