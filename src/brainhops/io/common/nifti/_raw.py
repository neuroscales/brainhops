"""The record of a NIfTI file, which is its header."""

# stdlib
import warnings
from io import BytesIO

# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.magic import Factory

# internals
from brainhops._core._internal_api import (
    open_compressed,
    preserve_position,
    replace,
)
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._internal_api import Raw
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    SnifferContentError,
)

# this format
from ._files import (
    _NIFTI_HEADER_SIZES,
    _NIFTI_HEADERS,
    _has_nifti_magic,
    _nifti_version,
)

_NiftiHeader = tx.Union[nb.Nifti1Header, nb.Nifti2Header]
_NibabelImage = tx.Union[nb.Nifti1Image, nb.Nifti2Image]


class NiftiRaw(Raw, BinaryFileReader, BinaryFileWriter):
    """Header of a NIfTI file, as the file stores it.

    The record of a NIfTI file is its nibabel header, which also holds the
    extensions that follow the header in the file. A record is read from
    the start of a file without reading the voxels, and it is written as
    the header alone, so a record read from a file and written again gives
    back the first bytes of that file. The geometry, the data type and the
    other fields of the header are kept as they are stored, and only the
    readers of the NIfTI formats decode them.

    A record is held by
    [`NiftiMetadata`][brainhops.io.common.nifti.NiftiMetadata], which never
    changes it in place. A writer works on the copy that [`copy`][]
    returns.
    """

    header: tx.Annotated[_NiftiHeader, Factory(nb.Nifti1Header)]
    """The NIfTI-1 or NIfTI-2 header, with its extensions.

    A new record holds an empty NIfTI-1 header.
    """

    def copy(self) -> tx.Self:
        """Return a copy of the record that can be changed freely.

        The fields of the header are copied. The list of extensions is new,
        but the extensions themselves are shared, because nibabel copies a
        header in this way.

        Returns
        -------
        NiftiRaw
            The copy.
        """
        return replace(self, header=self.header.copy())

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        *,
        version: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a NIfTI header.

        A stream that starts with the header size and the magic number of
        NIfTI-1 or NIfTI-2 is a NIfTI file with certainty. The stream may be
        compressed with gzip, and its position is restored.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no NIfTI header.
            With `True`, the error is a `SnifferContentError`, and an
            exception class is raised instead when one is given.
        version : int, optional
            The NIfTI version to look for, 1 or 2. Both are tried by
            default.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            `Confidence.CERTAIN` or `Confidence.NO`.

        Raises
        ------
        SnifferContentError
            If the stream holds no NIfTI header and `error` is `True`.
        """
        if _sniffed_header(file, version) is not None:
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            if version is None:
                kind = "NIfTI-1 or NIfTI-2"
            else:
                kind = f"NIfTI-{version}"
            raise error(f"Content is not a valid {kind} file")
        return Confidence.NO

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, check: bool = True, **kwargs
    ) -> "NiftiRaw":
        """Read the header at the start of a NIfTI stream.

        The stream may be compressed with gzip. Only the header and its
        extensions are read, so the stream may end before the voxels.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        check : bool, default=True
            Whether nibabel checks the header.
        **kwargs : Any
            Refused, since the record has no other option.

        Raises
        ------
        TypeError
            If another option is given.
        """
        if kwargs:
            raise TypeError(
                f"Unknown NIfTI header option(s): {', '.join(kwargs)}"
            )
        read = _NIFTI_HEADERS[_nifti_version(file)].from_fileobj
        return cls(header=read(open_compressed(file), check=check))

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the header and its extensions as a NIfTI file stores them.

        The header is written from a copy, because nibabel sets the offset
        of the voxels in the header that it writes when the offset is zero.
        """
        buffer = BytesIO()
        self.header.copy().write_to(buffer)
        return buffer.getvalue()


def _sniffed_header(
    file: tx.IO, version: tx.Optional[int] = None
) -> tx.Optional[_NiftiHeader]:
    """Return the NIfTI header at the start of a stream, or `None`.

    The header size and the magic number are checked before nibabel parses
    the header, because nibabel parses any bytes and warns about the garbage
    in a file that is not NIfTI. The header is then parsed without its
    extensions and without the warnings of nibabel, which a real read still
    shows. The position of the stream is restored.
    """
    for candidate in (1, 2) if version is None else (version,):
        try:
            size = _NIFTI_HEADER_SIZES[candidate]
            with preserve_position(file):
                head = open_compressed(file).read(size)
            if not _has_nifti_magic(head, candidate):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                read = _NIFTI_HEADERS[candidate].from_fileobj
                return read(BytesIO(head), check=False)
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------
#   WHOLE FILES
# ----------------------------------------------------------------------
