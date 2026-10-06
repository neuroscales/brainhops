"""The shared base of every transformation stored in a NIfTI file."""

# dependencies
import typing_extensions as tx

# internals
from brainhops.io.base.nifti import NiftiMetadataField, NiftiParser
from brainhops.io.transformations.base import WritableFileBasedTransformation


class NiftiBasedTransformation(WritableFileBasedTransformation, NiftiParser):
    """
    A transformation that is stored in a NIfTI file.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete NIfTI transformations inherit from
    it and register themselves.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")

    # Declared again here, not only on `NiftiParser`: `Transformation`
    # comes before `NiftiParser` in the MRO of every NIfTI transformation,
    # and its generic `metadata` would otherwise win (and convert the
    # NIfTI metadata, header and all, into the generic `Metadata`).
    metadata: NiftiMetadataField
