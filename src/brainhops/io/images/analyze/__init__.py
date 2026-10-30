"""Analyze 7.5 images.

Analyze 7.5 is the format from which NIfTI-1 descends, and old data sets
and the files that SPM99 and SPM2 write still use it. An image is a pair
of files: a header file (`.hdr`) and an image file (`.img`), which may
both be compressed with gzip. [`AnalyzeImage`][] reads such a pair from
the path of either file, and answers to the format hint `analyze`. A
header with the NIfTI-1 magic number belongs to a NIfTI pair, which
[`brainhops.io.images.nifti`][brainhops.io.images.nifti] reads instead.
Files are read with `nibabel` (`pip install brainhops[analyze]`).

```python
import brainhops.io as io

image = io.images.load("brain.hdr")  # an AnalyzeImage
image.data.shape                     # F order, as in NIfTI
image.transformation                 # voxel -> RAS world, in mm
image.transformations[0]             # voxel -> physical (voxel sizes)
image.metadata.raw.header            # the nibabel header
```

An image holds the header of its pair as [`AnalyzeMetadata`][], whose
record is an [`AnalyzeRaw`][], and its voxels in `rawdata`, a lazy array
that reads them only when `data` is first needed. Analyze stores no
orientation, so the voxel-to-world transformation follows SPM: the first
axis runs from right to left, and the origin of the world is the voxel
that SPM stores in the header, or else the centre of the grid. The
convention is described in
[`brainhops.io.common.analyze`][brainhops.io.common.analyze].

Analyze files are only read, because NIfTI stores the same voxels with a
complete geometry, and an image is saved to another format, such as
NIfTI.
"""

__all__ = ["AnalyzeImage", "AnalyzeMetadata", "AnalyzeRaw"]

from brainhops.io.common.analyze import AnalyzeMetadata, AnalyzeRaw

from ._image import AnalyzeImage
