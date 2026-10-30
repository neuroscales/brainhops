"""Reading of Analyze 7.5 images.

Analyze 7.5 is the format from which NIfTI-1 descends. Old data sets,
and the files that SPM99 and SPM2 write, still use it. An image is a
pair of files with the same name: a header file (`.hdr`) of 348 bytes,
and an image file (`.img`) that holds the voxels, with the first axis
varying fastest. A header whose magic number is that of NIfTI-1 (`ni1`)
belongs to a NIfTI pair, which the NIfTI readers handle, so the Analyze
readers refuse it.

The header is held as an [`AnalyzeRaw`][] record, and
[`AnalyzeMetadata`][] reads that record without the voxels.
[`AnalyzeImage`][brainhops.io.images.analyze.AnalyzeImage] holds such
metadata and a lazy array of the voxels. That lazy array is the one that
reads the voxels of NIfTI files, because both formats store their voxels
in the same way.

An Analyze header stores no orientation. The voxel-to-world matrix is
built from the voxel sizes and from the origin that SPM stores in the
header, with the first axis running from right to left, as SPM and
nibabel build it. The fields that SPM uses as a scale factor and as an
intercept scale the voxels, as nibabel scales them. Analyze files are
only read, and an image is saved to another format, such as NIfTI.
"""

__all__ = [
    "AnalyzeMetadata",
    "AnalyzeRaw",
]

from ._metadata import AnalyzeMetadata
from ._raw import AnalyzeRaw
