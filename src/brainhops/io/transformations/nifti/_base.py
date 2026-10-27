"""The shared base of the transformations stored in NIfTI files."""

# stdlib

# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base.parsers import (
    WriterNotImplementedError,
)
from brainhops.io.common.nifti import (
    NiftiMetadata,
    NiftiRaw,
    NiftiReaderWriter,
)
from brainhops.io.common.nifti._data import _NiftiProxy
from brainhops.io.common.nifti._header import (
    _NiftiObject,
)
from brainhops.io.transformations.base import TransformationFormat

_NiftiHeader = tx.Union[nb.Nifti1Header, nb.Nifti2Header]
_NibabelImage = tx.Union[nb.Nifti1Image, nb.Nifti2Image]


class NiftiBasedTransformation(NiftiReaderWriter, TransformationFormat):
    """A transformation stored in a NIfTI file.

    A NIfTI transformation holds the header of its file as
    [`NiftiMetadata`][], and the array of the file as it is stored in
    `rawdata`. When the transformation is read from a file, `rawdata` is
    a lazy array, which reads the array of the file only when it is
    needed. Each
    concrete format decodes its `data`, and the transformation of the
    data model, from `rawdata` and from the header, and caches what it
    decodes.

    `rawdata` and the header describe the file as it was read, or as the
    transformation was last consolidated. Data and transformations that
    are assigned are pending until [`consolidate`][] encodes them into a
    copy, or until the transformation is written. A write encodes them in
    the same way, into a copy, and leaves the transformation as it was.

    When the transformation is consolidated, a copy of the header of its
    file is the base of the new header. A transformation whose data model
    was not changed keeps its array under that header, and one whose data
    model was changed encodes the model over the header. This class is
    abstract and is not registered as a format. Its subclasses implement
    [`consolidate`][], on which the writers rely.

    The metadata is the `metadata` field of the transformation of the
    data model. This class does not derive from that transformation, so
    it cannot take the field over: each concrete subclass does, with
    `_metadata: ... = Inherit("metadata", default=None)`, so that the
    field holds [`NiftiMetadata`][] and converts other metadata into it.
    """

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The array of the file as it is stored, or `None` without one.

    After a read, `rawdata` is a lazy array of the file, which applies
    the scaling of the header when it is read, as nibabel applies it. A
    transformation that keeps its data in the header, such as an affine,
    has no array. `rawdata` describes the file as it was read, or as the
    transformation was last consolidated, so assigning `data` leaves it
    as it is.
    """

    # --- writing ------------------------------------------------------

    def _record(self) -> tx.Optional[NiftiRaw]:
        """Return a copy of the record of the metadata, or `None`.

        The encoders build the new header over this copy, so they are free
        to change it.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        return self.metadata.raw.copy()

    def consolidate(self, *, inplace: bool = False, **options) -> tx.Self:
        """Encode the pending state into `rawdata` and the header.

        Each concrete format implements this method, on which the writers
        rely. Consolidating makes `rawdata` and the header describe the
        transformation, as a file written from it would, and nothing is
        pending afterwards. A transformation that has a header and nothing
        pending, consolidated without options, is left as it is.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        **options : Any
            The writer options of the concrete format.

        Returns
        -------
        Self
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        WriterNotImplementedError
            Always, since the base does not know what to encode.
        """
        raise WriterNotImplementedError(
            f"{type(self).__name__} does not know how to write itself to "
            f"NIfTI."
        )


# ----------------------------------------------------------------------
#   HELPERS OF THE CONCRETE FORMATS
# ----------------------------------------------------------------------


def _nibabel_parts(
    nifti: _NiftiObject,
) -> tx.Tuple[tx.Optional[ArrayProtocol], _NiftiHeader]:
    """Return the array and the header of a nibabel image or header.

    A header alone has no array. The array proxy of an image that nibabel
    read from a file is replaced by the lazy array that brainhops reads
    the same file with.

    Raises
    ------
    TypeError
        If `nifti` is neither a NIfTI image nor a NIfTI header.
    """
    if isinstance(nifti, nb.Nifti1Image):
        return _NiftiProxy.from_nibabel(nifti.dataobj), nifti.header
    if isinstance(nifti, nb.Nifti1Header):
        return None, nifti
    raise TypeError(f"Expected a NIfTI image or header, got {type(nifti)}")


def _consolidated_metadata(
    xform: NiftiBasedTransformation, record: NiftiRaw
) -> NiftiMetadata:
    """Return the metadata that holds the record of a consolidation.

    The new metadata, of the class of the metadata of the transformation,
    is decoded from the new record, so its vocabulary describes that
    record, and the metadata held before is never changed. A
    transformation without metadata gets new NIfTI metadata.
    """
    if xform.metadata is None:
        return NiftiMetadata.from_raw(record)
    return type(xform.metadata).from_raw(record)
