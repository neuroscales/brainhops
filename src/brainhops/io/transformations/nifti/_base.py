"""The shared base of the transformations stored in NIfTI files."""

# stdlib

# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base.parsers import (
    WriterError,
    WriterNotImplementedError,
)
from brainhops.io.common.nifti import (
    NiftiMetadata,
    NiftiRaw,
    NiftiReaderWriter,
)
from brainhops.io.common.nifti._internal_api import (
    NIFTI_INTENT_NAME_MAPPING,
    NIFTI_INTENT_VECTOR,
    NiftiObject,
    NiftiProxy,
    nifti_intent,
    nifti_intent_name,
)
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base._internal_api import (
    ImageBackedTransformation,
    consolidate_through,
)

# this format
from ._views import check_single_field

_NiftiHeader = tx.Union[nb.Nifti1Header, nb.Nifti2Header]
_NibabelImage = tx.Union[nb.Nifti1Image, nb.Nifti2Image]


class NiftiBackedTransformation(NiftiReaderWriter, TransformationFormat):
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

    def _consolidate(self, *, inplace: bool = False, **options) -> tx.Self:
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


class NiftiImageBackedTransformation(
    ImageBackedTransformation, NiftiBackedTransformation
):
    """A transformation whose file is a NIfTI image, which it keeps.

    A field of displacements or of coordinates is stored as a NIfTI
    image whose voxels hold vectors. A transformation of this class
    keeps that file in a private
    [`NiftiImage`][brainhops.io.images.nifti.NiftiImage], as
    [`ImageBackedTransformation`][brainhops.io.transformations.base._image_backed.ImageBackedTransformation]
    describes, and reads, consolidates and writes through that image.
    The readers take the options of the nibabel readers, `mmap` and
    `keep_file_open`. A concrete subclass consolidates through
    [`consolidate_nifti`][].

    A concrete subclass takes the `metadata` field of its data model
    over with
    `_metadata: ... = Inherit("metadata", default=None, var=True)`, so
    that the constructor converts other metadata into
    [`NiftiMetadata`][].
    """

    IMAGE = NiftiImage

    _rawdata: NoRepr[tx.Optional[ArrayProtocol]] = Inherit("rawdata", var=True)


# ----------------------------------------------------------------------
#   HELPERS OF THE CONCRETE FORMATS
# ----------------------------------------------------------------------


def consolidate_nifti(
    xform: NiftiImageBackedTransformation,
    image: NiftiImage,
    intent: tx.Mapping[str, tx.Any],
    changes: tx.Mapping[str, tx.Any],
    **options: tx.Any,
) -> NiftiImageBackedTransformation:
    """Consolidate a NIfTI field through the image of its pending state.

    The call is that of
    [`consolidate_through`][brainhops.io.transformations.base._image_backed.consolidate_through],
    and the consolidated image is refused when its array holds a field
    for each of several volumes, because the data model does not
    support such a batch of fields yet.

    Raises
    ------
    WriterError
        If the image has no data, or if its array holds a field for each
        of several volumes.
    """
    return consolidate_through(
        xform, image, intent, changes, check=_check_single, **options
    )


def nibabel_parts(
    nifti: NiftiObject,
) -> tx.Tuple[tx.Optional[ArrayProtocol], _NiftiHeader]:
    """Return the array and the header of a nibabel image or header.

    A header alone has no array. The array proxy of an image that
    nibabel read from a file is replaced by the lazy array that
    brainhops reads the same file with.

    Raises
    ------
    TypeError
        If `nifti` is neither a NIfTI image nor a NIfTI header.
    """
    if isinstance(nifti, nb.Nifti1Image):
        return NiftiProxy.from_nibabel(nifti.dataobj), nifti.header
    if isinstance(nifti, nb.Nifti1Header):
        return None, nifti
    raise TypeError(f"Expected a NIfTI image or header, got {type(nifti)}")


def consolidated_metadata(
    xform: NiftiBackedTransformation, record: NiftiRaw
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


def intent_overrides(
    code: int, name: str = "", params: tx.Sequence[float] = ()
) -> tx.Dict[str, tx.Any]:
    """Return the header overrides that give a header an intent.

    The overrides set the intent code and the intent name. The three
    intent parameters take the values in `params`, in order, and the
    parameters that `params` does not give are reset to zero, as
    nibabel's `set_intent` does.
    """
    values = (*params, 0.0, 0.0, 0.0)[:3]
    params = {f"intent_p{i}": v for i, v in enumerate(values, start=1)}
    return {"intent_code": code, "intent_name": name, **params}


def coordinates_intent(
    header: tx.Optional[NiftiObject],
) -> tx.Dict[str, tx.Any]:
    """Return the header overrides that mark a field of coordinates.

    A field of coordinates is written under the `VECTOR` intent (1007)
    with the SPM intent name `"Mapping"`. A header that already has this
    intent needs no override, so that its intent parameters are kept,
    and any other header, or no header, gets the overrides of
    [`intent_overrides`][].
    """
    intent = (nifti_intent(header), nifti_intent_name(header))
    if intent == (NIFTI_INTENT_VECTOR, NIFTI_INTENT_NAME_MAPPING):
        return {}
    return intent_overrides(NIFTI_INTENT_VECTOR, NIFTI_INTENT_NAME_MAPPING)


def _check_single(image: NiftiImage) -> None:
    check_single_field(image.rawdata, WriterError)
