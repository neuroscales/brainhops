"""Reading and writing machinery shared by all NIfTI-based formats.

The header of a NIfTI file is held as a [`NiftiRaw`][] record, and
[`NiftiMetadata`][] reads and writes that header alone, without the
voxels. [`NiftiImage`][brainhops.io.images.nifti.NiftiImage] and the
transformations of
[`brainhops.io.transformations.nifti`][brainhops.io.transformations.nifti]
hold such metadata and a lazy array of the voxels, and read and write
their files with [`NiftiReaderWriter`][].
"""

__all__ = [
    "NiftiMetadata",
    "NiftiRaw",
    "NiftiReaderWriter",
    "NiftiUnitWarning",
]

from ._metadata import NiftiMetadata
from ._parsers import NiftiReaderWriter
from ._raw import NiftiRaw
from ._units import NiftiUnitWarning
