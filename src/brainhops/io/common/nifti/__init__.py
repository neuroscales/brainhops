"""Reading and writing machinery shared by all NIfTI-based formats.

The header of a NIfTI file is held as a [`NiftiRaw`][] record, and
[`NiftiMetadata`][] reads and writes that header alone, without the
voxels. [`NiftiImage`][brainhops.io.images.nifti.NiftiImage] holds such
metadata and a lazy array of the voxels. The transformation formats
stored as NIfTI files are still built on [`NiftiReaderWriter`][], which
holds a nibabel image and its header.
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
