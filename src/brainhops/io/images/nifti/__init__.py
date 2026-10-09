"""Readers and writers for NIfTI images.

A [`NiftiImage`][] holds the header of its file as [`NiftiMetadata`][],
whose record is a [`NiftiRaw`][], and the voxels as stored in the file.
The two classes of the header are defined in
[`brainhops.io.common.nifti`][] and exported here as well.
"""

__all__ = ["NiftiImage", "NiftiMetadata", "NiftiRaw"]

from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw

from ._image import NiftiImage
