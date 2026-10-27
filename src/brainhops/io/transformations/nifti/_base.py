"""The shared base of the transformations stored in NIfTI files."""

# stdlib

# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops._core.path import is_local
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    WriterError,
    WriterNotImplementedError,
)
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._files import (
    write_nifti_file,
    write_nifti_stream,
)
from brainhops.io.common.nifti._header import (
    _NiftiObject,
    _written_nifti,
)
from brainhops.io.common.nifti._raw import (
    _sniffed_header,
    read_nifti_file,
    read_nifti_stream,
    reads_file,
)
from brainhops.io.transformations.base import TransformationFormat

_NiftiHeader = tx.Union[nb.Nifti1Header, nb.Nifti2Header]
_NibabelImage = tx.Union[nb.Nifti1Image, nb.Nifti2Image]


class NiftiBasedTransformation(
    BinaryFileReader, BinaryFileWriter, TransformationFormat
):
    """A transformation stored in a NIfTI file.

    A NIfTI transformation holds the header of its file as
    [`NiftiMetadata`][], and the array of the file as it is stored in
    `rawdata`. When the transformation is read from a file, `rawdata` is
    a nibabel proxy, which reads the array only when it is needed. Each
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

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The array of the file as it is stored, or `None` without one.

    After a read, `rawdata` is a nibabel proxy, which applies the
    scaling of the header when it is read. A transformation that keeps
    its data in the header, such as an affine, has no array. `rawdata`
    describes the file as it was read, or as the transformation was last
    consolidated, so assigning `data` leaves it as it is.
    """

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds this transformation.

        A stream that starts with a NIfTI header is scored with
        `_score_nibabel`. Any other stream is declined by
        [`NiftiRaw.sniff_fileobj`][], which raises the requested error.

        Parameters
        ----------
        file : IO
            The stream, which is read from its current position.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no NIfTI
            header.
        **kwargs : Any
            Options of the reader. The NIfTI version is read from
            `version`.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        header = _sniffed_header(file, kwargs.get("version"))
        if header is None:
            return NiftiRaw.sniff_fileobj(file, error=error, **kwargs)
        return cls._score_nibabel(header)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Return how well a valid NIfTI header matches this class.

        The header has already passed the check of its size and magic
        number, so the score says how likely the file holds the kind of
        transformation that this class builds. Each concrete format
        overrides this method, and the base returns `Confidence.MAYBE`.
        """
        return Confidence.MAYBE

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read a transformation from the path of a NIfTI file.

        A local file is handed to nibabel by name, so that nibabel opens
        it whenever the array is read and can memory-map it. A remote file
        is read into memory instead.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the nibabel readers, such as `mmap` and
            `keep_file_open`.

        Returns
        -------
        NiftiBasedTransformation
            The transformation, whose array is read lazily.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        """
        filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        if not is_local(filename):
            # The lazy array outlives this call, so it reads a remote file
            # from memory rather than from a stream that would be closed.
            return cls.from_bytes(filename.read_bytes(), **kwargs)
        raw, proxy = read_nifti_file(filename, **kwargs)
        return cls(rawdata=proxy, metadata=NiftiMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read a transformation from an open NIfTI stream.

        The proxy reads the array from the stream when it is needed, so
        the stream must stay open while the data may be read.

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be compressed with gzip.
        **kwargs : Any
            Options of the nibabel readers, such as `mmap` and
            `keep_file_open`.

        Returns
        -------
        NiftiBasedTransformation
            The transformation, whose array is read lazily.
        """
        raw, proxy = read_nifti_stream(file, **kwargs)
        return cls(rawdata=proxy, metadata=NiftiMetadata.from_raw(raw))

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """Build a transformation from a nibabel image or header.

        The array of an image becomes `rawdata` as it is, and a copy of
        its header becomes the record of the metadata. A header alone
        gives a transformation without an array.

        Parameters
        ----------
        nifti : nibabel.Nifti1Image or nibabel.Nifti1Header
            The nibabel image or header. NIfTI-2 images and headers are
            accepted too.
        **kwargs : Any
            Other fields of the transformation.

        Returns
        -------
        NiftiBasedTransformation
            The transformation.

        Raises
        ------
        TypeError
            If `nifti` is neither a NIfTI image nor a NIfTI header.
        """
        raw, header = _nibabel_parts(nifti)
        metadata = NiftiMetadata.from_raw(NiftiRaw(header=header.copy()))
        return cls(rawdata=raw, metadata=metadata, **kwargs)

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

    def to_nibabel(self, **options) -> _NibabelImage:
        """Build the nibabel image that a file written from this holds.

        The transformation is consolidated with the options, as by
        [`consolidate`][], into a copy, and the nibabel image holds the
        `rawdata` array and the header of that copy. The transformation itself
        is not changed.

        Parameters
        ----------
        **options : Any
            The writer options of [`consolidate`][], such as the template
            `like` and the header overrides.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.

        Raises
        ------
        WriterError
            If the transformation has no array to write.
        """
        return _nifti_written(self.consolidate(**options))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transformation to a path.

        The file is compressed when its name ends with `.gz`. The
        transformation is consolidated into a copy, which is written, and
        the transformation itself is not changed. A local file is replaced
        at once, so a transformation can be written over the file that it
        was read from. The transformation is then consolidated in place,
        so that it holds what the file holds (#426): its lazy array reads
        the new file, and its record is the header of that file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The writer options of [`consolidate`][].
        """
        if not reads_file(self.rawdata, filename):
            write_nifti_file(self.to_nibabel(**kwargs), filename)
            return
        self.consolidate(inplace=True, **kwargs)
        write_nifti_file(self.to_nibabel(), filename)
        # The lazy array of the old file describes how that file stored
        # the values, so the transformation reads the new file, holds the
        # record of its header, and drops the views decoded before.
        record, proxy = read_nifti_file(filename)
        self.rawdata = proxy
        self.metadata = type(self.metadata).from_raw(record)
        self.data = None

    def to_fileobj(self, file: tx.BinaryIO, **kwargs) -> None:
        """Write the transformation to a stream as an uncompressed file.

        Parameters
        ----------
        file : BinaryIO
            The stream.
        **kwargs : Any
            The writer options of [`consolidate`][].
        """
        write_nifti_stream(self.to_nibabel(**kwargs), file)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the uncompressed NIfTI encoding of the transformation.

        Parameters
        ----------
        **kwargs : Any
            The writer options of [`consolidate`][].

        Returns
        -------
        bytes
            The content of an uncompressed NIfTI file.
        """
        return self.to_nibabel(**kwargs).to_bytes()


# ----------------------------------------------------------------------
#   HELPERS OF THE CONCRETE FORMATS
# ----------------------------------------------------------------------


def _nibabel_parts(
    nifti: _NiftiObject,
) -> tx.Tuple[tx.Optional[ArrayProtocol], _NiftiHeader]:
    """Return the array and the header of a nibabel image or header.

    A header alone has no array.

    Raises
    ------
    TypeError
        If `nifti` is neither a NIfTI image nor a NIfTI header.
    """
    if isinstance(nifti, nb.Nifti1Image):
        return nifti.dataobj, nifti.header
    if isinstance(nifti, nb.Nifti1Header):
        return None, nifti
    raise TypeError(f"Expected a NIfTI image or header, got {type(nifti)}")


def _nifti_written(xform: NiftiBasedTransformation) -> _NibabelImage:
    """Return the nibabel image that writes a consolidated transformation.

    Raises
    ------
    WriterError
        If the transformation has no array to write.
    """
    if xform.rawdata is None:
        raise WriterError(
            f"This {type(xform).__name__} has no array, so there is "
            f"nothing to write."
        )
    return _written_nifti(xform.rawdata, xform.metadata.raw)


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
