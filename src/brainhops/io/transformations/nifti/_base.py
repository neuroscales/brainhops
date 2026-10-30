"""The shared base of the transformations stored in NIfTI files."""

# stdlib

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Field, Inherit, KwOnly, NoRepr, replace

# internals
from brainhops._core.properties import drop_cache
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.datamodel.transformations import Affine
from brainhops.io.base.parsers import (
    WriterError,
    WriterNotImplementedError,
)
from brainhops.io.common._internal_api import held_transformations
from brainhops.io.common.nifti import (
    NiftiMetadata,
    NiftiRaw,
    NiftiReaderWriter,
)
from brainhops.io.common.nifti._internal_api import (
    NIBABEL_OPTIONS,
    NiftiObject,
    NiftiProxy,
    split_options,
)
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.metadata._internal_api import requested_fields
from brainhops.io.transformations.base import TransformationFormat

# this format
from ._views import check_single_field

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


class NiftiImageBackedTransformation(NiftiBasedTransformation):
    """A transformation whose file is a NIfTI image, which it keeps.

    A field of displacements or of coordinates is stored as a NIfTI
    image whose voxels hold vectors. A transformation of this class
    keeps that file, as it was read or as the transformation was last
    consolidated, in a private
    [`NiftiImage`][brainhops.io.images.nifti.NiftiImage], and it reads,
    consolidates and writes through that image. The properties `rawdata`
    and `metadata` read the array and the metadata of the image, and
    assigning one of them replaces the image with a copy that holds the
    new value. The data and the chain of the data model are kept in the
    private fields of the transformation instead. They are pending until
    the transformation is consolidated, and they take precedence over
    the image. The design memo
    `spec/design/image-backed-transformations.md` explains these two
    tiers of state.

    The constructor takes the positional arguments of the data model,
    and the keywords `image`, `rawdata` and `metadata`. It applies the
    keywords in that order, so that a later keyword overrides an earlier
    one. `image` becomes the stored image, and `rawdata` and `metadata`
    replace the matching parts of it. Without `image`, the stored image
    is built from `rawdata` and `metadata`. The fields of the data model
    are stored last, and they are never given to the image, because the
    `data` and the `transformations` of a transformation do not mean the
    `data` and the `transformations` of its image. An image of another
    class than `NiftiImage` raises `TypeError`, and an image that has
    pending data or transformations is consolidated when it is given, so
    that the stored image always describes a file.

    A concrete subclass takes the `metadata` field of its data model
    over as a value that the constructor takes and does not keep, with
    `_metadata: ... = Inherit("metadata", default=None, var=True)`. It
    describes what it stores in the method `_pending_image`, on which
    [`consolidate`][] relies.
    """

    # The image is not converted, so that an image of another class is
    # refused rather than read as a NIfTI image.
    _image: KwOnly[NoRepr[tx.Optional[NiftiImage]]] = Field(
        default=None, convert=False
    )
    """The image that holds the file, which the constructor always sets.

    The image describes the file as it was read, or as the
    transformation was last consolidated, and it never holds pending
    data or transformations.
    """

    _rawdata: NoRepr[tx.Optional[ArrayProtocol]] = Inherit("rawdata", var=True)

    _DECODED_FROM_IMAGE: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """The cached views that are decoded from the image.

    They are dropped when the stored image is replaced, so that they are
    decoded again from the new image.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        image = self._image
        if image is None:
            image = NiftiImage()
        elif not isinstance(image, NiftiImage):
            raise TypeError(
                f"A {type(self).__name__} is stored in a NiftiImage, and "
                f"cannot be built from a {type(image).__name__}."
            )
        elif (
            image.data is not image.rawdata
            or held_transformations(image) is not None
        ):
            image = image.consolidate()
        parts = {
            name: arguments[name]
            for name in ("rawdata", "metadata")
            if arguments.get(name) is not None
        }
        self._image = replace(image, **parts) if parts else image
        post_init = getattr(super(), "__post_init__", None)
        if post_init is not None:
            post_init(arguments)

    # --- stored image -------------------------------------------------

    @property
    def rawdata(self) -> tx.Optional[ArrayProtocol]:
        """The array of the file as it is stored, or `None` without one.

        The array is the `rawdata` of the stored image. After a read, it
        is a lazy array of the file, which applies the scaling of the
        header when it is read. It describes the file as it was read, or
        as the transformation was last consolidated, so assigning `data`
        leaves it as it is. Assigning `rawdata` replaces the stored
        image with a copy that holds the new array.
        """
        return self._image.rawdata

    @rawdata.setter
    def rawdata(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._replace_image(replace(self._image, rawdata=value))

    @property
    def metadata(self) -> tx.Optional[NiftiMetadata]:
        """The metadata of the file, which holds its header, or `None`.

        The metadata is that of the stored image. A transformation built
        from its data alone has none. Assigning other metadata replaces
        the stored image with a copy that holds it, converted into
        [`NiftiMetadata`][], and drops the views decoded from the
        previous header.
        """
        return self._image.metadata

    @metadata.setter
    def metadata(self, value: tx.Optional[NiftiMetadata]) -> None:
        self._replace_image(replace(self._image, metadata=value))

    def _replace_image(self, image: NiftiImage) -> None:
        """Replace the stored image and drop the views decoded from it."""
        self._image = image
        drop_cache(self, *self._DECODED_FROM_IMAGE)

    # --- reading ------------------------------------------------------

    @classmethod
    def from_image(cls, image: NiftiImage, **options: tx.Any) -> tx.Self:
        """Build a transformation from the NIfTI image that stores it.

        The image becomes the stored image of the transformation, and
        the transformation decodes its data and its chain from that
        image when they are needed. The call is the same as
        `cls(image=image, **options)`.

        Parameters
        ----------
        image : NiftiImage
            The image, which is consolidated first if it has pending
            data or transformations.
        **options : Any
            The other fields of the transformation.

        Returns
        -------
        Self
            The transformation.

        Raises
        ------
        TypeError
            If the image is not a `NiftiImage`.
        """
        return cls(image=image, **options)

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs: tx.Any) -> tx.Self:
        """Read a transformation from the path of a NIfTI file.

        The file is read as a
        [`NiftiImage`][brainhops.io.images.nifti.NiftiImage], whose
        array is read lazily, and the transformation is built from that
        image with [`from_image`][].

        Parameters
        ----------
        filename : path-like
            The path of the file, which may be compressed.
        **kwargs : Any
            The options of the nibabel readers, `mmap` and
            `keep_file_open`, and the other fields of the
            transformation.

        Returns
        -------
        Self
            The transformation, whose array is read lazily.
        """
        reader = _reader_options(kwargs)
        image = NiftiImage.from_filename(filename, **reader)
        return cls.from_image(image, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs: tx.Any) -> tx.Self:
        """Read a transformation from an open NIfTI stream.

        The stream is read as a
        [`NiftiImage`][brainhops.io.images.nifti.NiftiImage], whose lazy
        array keeps the stream, so the stream must stay open while the
        array may be read. The transformation is built from that image
        with [`from_image`][].

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be compressed with gzip.
        **kwargs : Any
            The options of the nibabel readers, `mmap` and
            `keep_file_open`, and the other fields of the
            transformation.

        Returns
        -------
        Self
            The transformation, whose array is read lazily.
        """
        reader = _reader_options(kwargs)
        image = NiftiImage.from_fileobj(file, **reader)
        return cls.from_image(image, **kwargs)

    # --- writing ------------------------------------------------------

    def to_image(self) -> NiftiImage:
        """Return the image that a file of this transformation holds.

        The transformation is consolidated into a copy, as by
        [`consolidate`][], and the stored image of that copy is
        returned. The transformation itself is not changed.

        Returns
        -------
        NiftiImage
            The image, without pending data or transformations.
        """
        return self.consolidate()._image

    def _pending_image(
        self, options: tx.Dict[str, tx.Any]
    ) -> tx.Tuple[NiftiImage, tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
        """Return the image that holds the pending state of the format.

        A concrete format encodes its pending data, or its pending
        chain, into a copy of the stored image with `_image_with`, or
        returns the stored image when nothing is pending. It removes
        from `options` the options that are its own rather than the
        image's, such as the `log` option of a displacement field.

        Parameters
        ----------
        options : dict
            The options of the consolidation, which the method may
            change.

        Returns
        -------
        image : NiftiImage
            The image to consolidate.
        intent : dict
            The header fields that mark the format, such as the intent
            code, which are set when the image is encoded.
        changes : dict
            The fields that the consolidated transformation receives,
            such as `data=None`.

        Raises
        ------
        WriterNotImplementedError
            Always, since the base does not know what to encode.
        """
        raise WriterNotImplementedError(
            f"{type(self).__name__} does not know how to write itself to "
            f"NIfTI."
        )

    def _image_with(
        self, raw: ArrayProtocol, vox2ras: tx.Optional[np.ndarray] = None
    ) -> NiftiImage:
        """Return a copy of the stored image that holds pending data.

        The array `raw` is laid out as the file stores it. When the grid
        is given, as the (4, 4) voxel-to-RAS matrix `vox2ras`, it
        becomes the voxel-to-world transformation of the image, and the
        image encodes its geometry from it. Otherwise, the image keeps
        the geometry of its header.
        """
        if vox2ras is None:
            return replace(self._image, data=raw)
        grid = [Affine(matrix=np.asarray(vox2ras)[:3])]
        return replace(self._image, data=raw, transformations=grid)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        **options: tx.Any,
    ) -> tx.Self:
        # The format builds an image that holds its pending state, and
        # the image encodes it. The precedence on write, the loss report
        # and the choice between NIfTI-1 and NIfTI-2 are therefore those
        # of NiftiImage. A transformation that has an image and nothing
        # pending, consolidated without options, keeps the image, so it
        # is still saved byte for byte, and the codes of the format are
        # set only when the image is encoded again.
        image, intent, changes = self._pending_image(options)
        fields, overrides = split_options(options)
        if (
            image is self._image
            and like is None
            and not overrides
            and not requested_fields(self.metadata, fields)
        ):
            intent = {}
        # The image reports what it loses to a report of this format, on
        # which the policy in effect then acts once.
        name = type(self).__name__
        report = ConversionReport(source=name, target=name)
        options = {**intent, **options}
        image = image.consolidate(like=like, on_loss=report, **options)
        apply_loss_policy(report, stacklevel=2)
        check_single_field(image.rawdata, WriterError)
        if not inplace:
            return replace(self, image=image, **changes)
        self._replace_image(image)
        for name, value in changes.items():
            setattr(self, name, value)
        return self


# ----------------------------------------------------------------------
#   HELPERS OF THE CONCRETE FORMATS
# ----------------------------------------------------------------------


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


def intent_overrides(code: int, name: str = "") -> tx.Dict[str, tx.Any]:
    """Return the header overrides that give a header an intent.

    The overrides set the intent code and the intent name, and reset the
    three intent parameters to zero, as nibabel's `set_intent` does when
    it is given no parameter.
    """
    params = {f"intent_p{i}": 0.0 for i in (1, 2, 3)}
    return {"intent_code": code, "intent_name": name, **params}


def _reader_options(kwargs: tx.Dict[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    # The options of the nibabel readers go to the reader of the image,
    # and the others stay in `kwargs`, for the transformation.
    return {
        name: kwargs.pop(name) for name in NIBABEL_OPTIONS if name in kwargs
    }
