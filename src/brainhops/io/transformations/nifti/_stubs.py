"""The light half of the transformations stored in NIfTI files.

A format that is registered before it is imported derives from a stub,
which declares what dispatch reads before the format is imported (see
[`brainhops.io.base._registry`][]). A format stored in a NIfTI file
answers to the hint `nifti` and to the NIfTI extensions, so its stub
derives from [`NiftiBasedTransformationStub`][], which declares them.
This module imports neither nibabel nor the data model.
"""

__all__ = ["NiftiBasedTransformationStub"]

import typing_extensions as tx

from brainhops.io.transformations.base._base import TransformationFormat


class NiftiBasedTransformationStub(TransformationFormat):
    """Hints and file names of the transformations stored in NIfTI files.

    [`NiftiBasedTransformation`][brainhops.io.transformations.nifti.NiftiBasedTransformation]
    derives from this class, and so does the stub of each format that
    derives from it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)
