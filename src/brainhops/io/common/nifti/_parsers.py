"""The reader and writer shared by the formats stored in NIfTI files."""

# stdlib
from io import BytesIO

# dependencies
import nibabel as nb
import typing_extensions as tx

# core
from brainhops._core import path
from brainhops._core.path import is_local, is_same_file
from brainhops._core.proxy import ArrayProxy
from brainhops._core.streams import is_seekable, preserve_position

# io
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    WriterError,
)

# this format
from ._data import _NiftiProxy
from ._files import _image_to_stream
from ._header import _NiftiObject, _written_nifti
from ._metadata import NiftiMetadata
from ._raw import NiftiRaw, _sniffed_header

_NibabelImage = tx.Union[nb.Nifti1Image, nb.Nifti2Image]

_NIBABEL_OPTIONS = ("mmap", "keep_file_open")
"""The reader options that a NIfTI format accepts, as nibabel's readers."""


class NiftiReaderWriter(BinaryFileReader, BinaryFileWriter):
    """The reader and writer of an object stored in a NIfTI file.

    [`NiftiImage`][brainhops.io.images.nifti.NiftiImage] and
    [`NiftiBasedTransformation`][brainhops.io.transformations.nifti.NiftiBasedTransformation]
    both store a header and an array in a NIfTI file, and both read and
    write it through nibabel. They list this class first among their
    bases, so that they share its reading and writing methods. Each of
    them holds the header as [`NiftiMetadata`][] in `metadata`, the array
    as the file stores it in `rawdata`, and encodes its pending state
    into them with `consolidate`.

    A file is read lazily. A local file is kept by its name, so that the
    lazy array opens it whenever the array is read, and memory-maps it
    when the file is not compressed. A remote file is read into memory,
    and a stream is kept by the lazy array, so it must stay open while
    the array may be read. A file is written through nibabel. A file
    is written from a copy that `consolidate` encodes, and the object
    itself is not changed, except when it is written over the file that
    its lazy array reads (#426).
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)

    # --- sniffing -----------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an object of this class.

        A stream that starts with a NIfTI header is scored by
        [`score_header`][]. Any other stream is declined by
        [`NiftiRaw.sniff_fileobj`][], which raises the requested error.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode and read from its current
            position, which is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no NIfTI
            header. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            The NIfTI `version` to look for, 1 or 2. Other options are
            ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.

        Raises
        ------
        SnifferContentError
            If the stream holds no NIfTI header and `error` is `True`.
        """
        header = _sniffed_header(file, kwargs.get("version"))
        if header is None:
            return NiftiRaw.sniff_fileobj(file, error=error, **kwargs)
        return cls.score_header(header)

    @classmethod
    def score_header(cls, header: _NiftiObject) -> float:
        """Return how well a valid NIfTI header matches this class.

        The header has already passed the check of its size and magic
        number, so the score says how likely the file holds the kind of
        object that this class builds. A format that can tell its files
        apart from the header, such as by its intent code, overrides this
        method. The default is `Confidence.MAYBE`.

        Parameters
        ----------
        header : nibabel.Nifti1Header or nibabel.Nifti2Header
            The header of the file.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return Confidence.MAYBE

    # --- reading ------------------------------------------------------

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read an object from the path of a NIfTI file.

        A local file is kept by its name, so that the lazy array opens it
        whenever the array is read, and memory-maps it when the file is
        not compressed. The lazy array outlives this call, so a remote
        file is read into memory and then read by [`from_bytes`][], rather
        than from a stream that would be closed.

        Parameters
        ----------
        filename : path-like
            The path of the file, which may be compressed.
        **kwargs : Any
            The reader options, which nibabel's readers also take.
            `mmap` tells whether the array of a local file that is not
            compressed is memory-mapped. `keep_file_open` is accepted
            and ignored, because the lazy array never keeps a file open.

        Returns
        -------
        Self
            The object, whose array is read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ValueError
            If `mmap` is not one of `True`, `False`, `"c"` and `"r"`.
        ParserExistsError
            If the path does not exist.
        """
        mmap = _mmap_option(kwargs)
        filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        if not is_local(filename):
            # The lazy array keeps the stream that it reads, and a stream
            # opened here would be closed when this method returns, so a
            # remote file is read into memory.
            return cls.from_bytes(filename.read_bytes(), **kwargs)
        with filename.open("rb") as stream:
            record = NiftiRaw.from_fileobj(stream)
        proxy = _NiftiProxy.from_header(
            str(filename), record.header, mmap=mmap
        )
        return cls(rawdata=proxy, metadata=NiftiMetadata.from_raw(record))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read an object from an open NIfTI stream.

        The record is read first, and the position of the stream is
        restored so that the lazy array reads the array from the same
        start. The lazy array keeps the stream, so the stream must stay
        open while the array may be read. A stream that cannot seek is
        read into memory first, and a stream is never memory-mapped.

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be compressed with gzip.
        **kwargs : Any
            The reader options, as for [`from_filename`][].

        Returns
        -------
        Self
            The object, whose array is read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ValueError
            If `mmap` is not one of `True`, `False`, `"c"` and `"r"`.
        """
        mmap = _mmap_option(kwargs)
        if not is_seekable(file):
            return cls.from_bytes(file.read(), **kwargs)
        start = file.tell()
        with preserve_position(file):
            record = NiftiRaw.from_fileobj(file)
        proxy = _NiftiProxy.from_header(file, record.header, start, mmap=mmap)
        return cls(rawdata=proxy, metadata=NiftiMetadata.from_raw(record))

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """Build an object from a nibabel image or header.

        The array of an image becomes `rawdata`, and a copy of its header
        becomes the record of the metadata. A header alone gives an object
        without an array. The array proxy of an image that nibabel read
        from a file is replaced by the lazy array that [`from_filename`][]
        builds for that file, and any other array is kept as it is.

        Parameters
        ----------
        nifti : nibabel.Nifti1Image or nibabel.Nifti1Header
            The nibabel image or header. NIfTI-2 images and headers are
            accepted too.
        **kwargs : Any
            Other fields of the object.

        Returns
        -------
        Self
            The object.

        Raises
        ------
        TypeError
            If `nifti` is neither a NIfTI image nor a NIfTI header.
        """
        if isinstance(nifti, nb.Nifti1Image):
            raw = _NiftiProxy.from_nibabel(nifti.dataobj)
            header = nifti.header
        elif isinstance(nifti, nb.Nifti1Header):
            raw, header = None, nifti
        else:
            raise TypeError(
                f"Expected a NIfTI image or header, got {type(nifti)}"
            )
        metadata = NiftiMetadata.from_raw(NiftiRaw(header=header.copy()))
        return cls(rawdata=raw, metadata=metadata, **kwargs)

    # --- writing ------------------------------------------------------

    def to_nibabel(self, **options) -> _NibabelImage:
        """Build the nibabel image that a file written from this object holds.

        The object is consolidated with the options into a copy, and the
        nibabel image holds the `rawdata` and the header of that copy.
        The object itself is not changed.

        Parameters
        ----------
        **options : Any
            The options of `consolidate`, such as the template `like`
            and the fields of the header.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.

        Raises
        ------
        WriterError
            If the object has no array to write.
        """
        written = self.consolidate(**options)
        if written.rawdata is None:
            raise WriterError(
                f"This {type(self).__name__} has no data, so there is "
                f"nothing to write."
            )
        return _written_nifti(written.rawdata, written.metadata.raw)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the object to a path.

        The object is consolidated into a copy, which is written as the
        base writer writes any file: a local file is replaced at once,
        and a name that ends with `.gz` compresses the file. An object can
        therefore be written over the file that its lazy array reads. It
        is then consolidated in place, so that it holds what the file
        holds (#426): its lazy array reads the new file, and its record is
        the header of that file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The options of `consolidate`.

        Raises
        ------
        WriterError
            If the object cannot be encoded.
        """
        raw = self.rawdata
        root = raw.root if isinstance(raw, ArrayProxy) else raw
        reading = isinstance(root, _NiftiProxy) and is_same_file(
            root.source, filename
        )
        if not reading:
            super().to_filename(filename, **kwargs)
            return
        self.consolidate(inplace=True, **kwargs)
        super().to_filename(filename)
        # The lazy array of the old file describes how that file stored
        # the array, so the object reads the new file, holds the record of
        # its header, and drops the views decoded before.
        written = type(self).from_filename(filename)
        self.rawdata = written.rawdata
        self.metadata = type(self.metadata).from_raw(written.metadata.raw)
        self.data = None

    def to_fileobj(self, file: tx.BinaryIO, **kwargs) -> None:
        """Write the object to a stream as an uncompressed NIfTI file.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        **kwargs : Any
            The options of `consolidate`.

        Raises
        ------
        WriterError
            If the object cannot be encoded.
        """
        _image_to_stream(self.to_nibabel(**kwargs), file)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the content of an uncompressed NIfTI file of the object.

        Parameters
        ----------
        **kwargs : Any
            The options of `consolidate`.

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        WriterError
            If the object cannot be encoded.
        """
        buffer = BytesIO()
        self.to_fileobj(buffer, **kwargs)
        return buffer.getvalue()


def _mmap_option(kwargs: tx.Mapping[str, tx.Any]) -> bool:
    """Return whether the reader options ask for a memory map.

    `mmap` is true by default, and the modes `"c"` and `"r"` that nibabel
    takes ask for one too. `keep_file_open` is ignored.

    Raises
    ------
    TypeError
        If an option is not one of the nibabel readers.
    ValueError
        If `mmap` is not one of `True`, `False`, `"c"` and `"r"`.
    """
    unknown = [name for name in kwargs if name not in _NIBABEL_OPTIONS]
    if unknown:
        raise TypeError(
            f"Unknown NIfTI reader option(s): {', '.join(unknown)}"
        )
    mmap = kwargs.get("mmap", True)
    if mmap not in (True, False, "c", "r"):
        raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
    return bool(mmap)
