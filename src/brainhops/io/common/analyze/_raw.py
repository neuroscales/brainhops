"""The record of an Analyze 7.5 image, which is its header."""

# stdlib
import warnings
from io import BytesIO

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory

# internals
from brainhops._core import path
from brainhops._core.magic import replace
from brainhops._core.streams import open_compressed, preserve_position
from brainhops.io.base._internal_api import Raw, checked_score
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
)
from brainhops.io.common.nifti._internal_api import has_nifti_magic

# this format
from ._files import analyze_files

ANALYZE_HEADER_SIZE = 348
"""The size of an Analyze 7.5 header, in bytes."""

DECLINE_MESSAGE = "Content is not an Analyze 7.5 header."
"""The message of the error raised when a sniffer declines a stream."""

NIFTI_MESSAGE = (
    "This header has the magic number of NIfTI-1, so it is read as a "
    "NIfTI file and not as an Analyze 7.5 file."
)
"""The message of the error raised when an Analyze reader meets NIfTI."""


class AnalyzeRaw(Raw, BinaryFileReader):
    """Header of an Analyze 7.5 image, as the header file stores it.

    An Analyze 7.5 image is a pair of files: a header file (`.hdr`) of
    348 bytes, and an image file (`.img`) that holds the voxels. NIfTI-1
    descends from this format and keeps the same header size, so a
    header that carries the NIfTI-1 magic number is a NIfTI header and
    is refused here. The record holds the header as nibabel parses it
    with the SPM2 interpretation of its unused fields
    (`nibabel.Spm2AnalyzeHeader`), which is also what `nibabel.load`
    uses for such a pair. Under that interpretation, the field
    `funused1` is a scale factor, `funused2` is an intercept, and the
    field `originator` holds the origin of the image in voxels.

    A record is read from the header file alone, so the voxels are never
    read. It is held by
    [`AnalyzeMetadata`][brainhops.io.common.analyze.AnalyzeMetadata].
    Analyze files cannot be written, so a record is never written
    either.
    """

    header: tx.Annotated[nb.Spm2AnalyzeHeader, Factory(nb.Spm2AnalyzeHeader)]
    """The header, as nibabel parses it.

    A new record holds an empty header.
    """

    def copy(self) -> tx.Self:
        """Return a copy of the record that can be changed freely.

        Returns
        -------
        AnalyzeRaw
            The copy, whose header is a copy of this header.
        """
        return replace(self, header=self.header.copy())

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a path names an Analyze image.

        The header file of the pair is tested, whichever file of the pair
        the path names, so that a path to the image file is recognized
        too. A path that names neither file of a pair is tested as it is.

        Parameters
        ----------
        filename : path-like
            The path of the header file or of the image file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the path names no Analyze
            image. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        try:
            filename, _ = analyze_files(filename)
        except (ParserContentError, TypeError):
            pass
        return super().sniff_filename(filename, error=error, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "AnalyzeRaw":
        """Read the header of the Analyze image that a path names.

        The header file of the pair is read, whichever file of the pair
        the path names. A path that names neither file of a pair is read
        as it is.

        Parameters
        ----------
        filename : path-like
            The path of the header file or of the image file.
        **kwargs : Any
            The options of [`from_fileobj`][].

        Returns
        -------
        AnalyzeRaw
            The record.
        """
        try:
            filename, _ = analyze_files(filename)
        except (ParserContentError, TypeError):
            pass
        return super().from_filename(filename, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an Analyze header.

        A stream that starts with the header size of Analyze, in either
        byte order, followed by a header that nibabel can parse and whose
        dimensions are valid, is very likely an Analyze header. A stream
        with the NIfTI-1 magic number is declined, because the NIfTI
        readers handle it. The stream may be compressed with gzip, and
        its position is restored.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no Analyze
            header. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            `Confidence.LIKELY` or `Confidence.NO`.

        Raises
        ------
        SnifferContentError
            If the stream holds no Analyze header and `error` is `True`.
        """
        header = sniffed_header(file)
        score = Confidence.NO if header is None else Confidence.LIKELY
        return checked_score(score, error, DECLINE_MESSAGE)

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, check: bool = True, **kwargs
    ) -> "AnalyzeRaw":
        """Read the header at the start of an Analyze header stream.

        The stream may be compressed with gzip, and its position is
        restored. Only the 348 bytes of the header are read.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        check : bool, default=True
            Whether nibabel checks the header, and fixes the fields that
            it can fix, such as a voxel size of zero.
        **kwargs : Any
            Refused, since the record has no other option.

        Returns
        -------
        AnalyzeRaw
            The record.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If the stream holds a NIfTI-1 header, or holds less than a
            header.
        """
        if kwargs:
            raise TypeError(
                f"Unknown Analyze header option(s): {', '.join(kwargs)}"
            )
        with preserve_position(file):
            head = open_compressed(file).read(ANALYZE_HEADER_SIZE)
        if has_nifti_magic(head, 1):
            raise ParserContentError(NIFTI_MESSAGE)
        if len(head) < ANALYZE_HEADER_SIZE:
            raise ParserContentError(
                "The stream ends before the end of an Analyze header."
            )
        read = nb.Spm2AnalyzeHeader.from_fileobj
        return cls(header=read(BytesIO(head), check=check))


def sniffed_header(file: tx.IO) -> tx.Optional[nb.Spm2AnalyzeHeader]:
    """Return the Analyze header at the start of a stream, or `None`.

    The header size and the NIfTI magic number are checked before nibabel
    parses the header, because nibabel parses any bytes and warns about
    the garbage in a file that is not an Analyze header. The header is
    then parsed without the warnings of nibabel, which a real read still
    shows, and its dimensions are checked. The position of the stream is
    restored.
    """
    try:
        with preserve_position(file):
            head = open_compressed(file).read(ANALYZE_HEADER_SIZE)
        if len(head) < ANALYZE_HEADER_SIZE or has_nifti_magic(head, 1):
            return None
        sizes = {
            int.from_bytes(head[:4], order) for order in ("little", "big")
        }
        if ANALYZE_HEADER_SIZE not in sizes:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            read = nb.Spm2AnalyzeHeader.from_fileobj
            header = read(BytesIO(head), check=False)
    except Exception:  # noqa: BLE001
        return None
    dim = np.asarray(header["dim"])
    ndim = int(dim[0])
    if not 1 <= ndim <= 7 or np.any(dim[1 : ndim + 1] < 1):
        return None
    return header
