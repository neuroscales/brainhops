# dependencies
import typing_extensions as tx

# internals
from brainhops.io.base.nifti import NiftiParser
from brainhops.io.transformations.base import WritableFileBasedTransformation


class NiftiBasedTransformation(WritableFileBasedTransformation, NiftiParser):
    """
    A transformation that is stored in a NIfTI file.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete NIfTI transformations inherit from
    it and register themselves.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
