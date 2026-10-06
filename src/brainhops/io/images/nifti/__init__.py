"""Readers and writers for images stored in NIfTI files."""

__all__ = ["NiftiImage", "NiftiMetadata"]

from brainhops.io.base._nifti_metadata import NiftiMetadata

from ._image import NiftiImage
