"""Shared base of the transformations stored in NIfTI files."""

import typing_extensions as tx

from brainhops.io.common.nifti import NiftiReaderWriter
from brainhops.io.transformations.base import TransformationFormat


class NiftiBasedTransformation(TransformationFormat, NiftiReaderWriter):
    """
    Transformation stored in a NIfTI file.

    This class is abstract and is not registered as a format. Concrete NIfTI
    transformations inherit from it and register themselves.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
