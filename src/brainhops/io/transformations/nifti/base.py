"""The shared bases of the transformations stored in NIfTI files."""

# dependencies
import typing_extensions as tx

# internals
from brainhops.io.base.nifti import NiftiParser
from brainhops.io.transformations.base import (
    FileBasedTransformation,
    WritableFileBasedTransformation,
)


class NiftiBasedTransformation(WritableFileBasedTransformation, NiftiParser):
    """
    A transformation that is stored in a NIfTI file.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete NIfTI transformations inherit from
    it and register themselves.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")


class ReadOnlyNiftiBasedTransformation(FileBasedTransformation, NiftiParser):
    """
    A transformation that is read from a NIfTI file, and not written.

    Abstract, like [`NiftiBasedTransformation`][]. A format that inherits
    from it is registered for reading only, so `save` never offers it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
