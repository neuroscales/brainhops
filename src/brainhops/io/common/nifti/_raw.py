"""The record of a NIfTI file, and reading and writing whole files."""

# stdlib
import warnings
from io import BytesIO

# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.magic import Factory, Magic
from nibabel.arrayproxy import ArrayProxy

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed, preserve_position
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
    _NIFTI_IMAGES,
    _accepted,
    _has_nifti_magic,
    _image_to_stream,
    _is_local,
    _nifti_from_stream,
    _nifti_version,
    _save_nifti,
)

_NiftiHeader = tx.Union[nb.Nifti1Header, nb.Nifti2Header]
_NibabelImage = tx.Union[nb.Nifti1Image, nb.Nifti2Image]


class NiftiRaw(Magic, BinaryFileReader, BinaryFileWriter):
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

    def copy(self) -> "NiftiRaw":
        """Return a copy of the record that can be changed freely.

        The fields of the header are copied. The list of extensions is new,
        but the extensions themselves are shared, because nibabel copies a
        header in this way.

        Returns
        -------
        NiftiRaw
            The copy.
        """
        return NiftiRaw(header=self.header.copy())

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
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with a NIfTI header.

        The bytes are read as a stream with [`sniff_fileobj`][].
        """
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "NiftiRaw":
        """Read the header at the start of a NIfTI stream.

        The stream may be compressed with gzip. Only the header and its
        extensions are read, so the stream may end before the voxels. The
        keyword arguments that the nibabel reader of headers accepts, such
        as `check`, are passed to it, and the others are ignored.
        """
        read = _NIFTI_HEADERS[_nifti_version(file)].from_fileobj
        header = read(open_compressed(file), **_accepted(read, kwargs))
        return cls(header=header)

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


def read_nifti(
    file: path.FileLike, **kwargs
) -> tx.Tuple[NiftiRaw, ArrayProxy]:
    """Read the record and a lazy array of the voxels of a NIfTI file.

    The record is read from the header as the file stores it. The voxels
    are not read: nibabel returns a proxy, an array-like object that reads
    them when they are accessed. A local path is handed to nibabel by name,
    so that nibabel opens the file each time and can memory-map the
    voxels. A remote file is read into memory, because its stream is
    closed when this function returns. An open stream is kept by the
    proxy, so the caller must keep it open while the voxels may be read.

    Parameters
    ----------
    file : path or file object
        The NIfTI file, possibly compressed with gzip.
    **kwargs : Any
        Options of the nibabel readers, such as `mmap` and
        `keep_file_open`. Each reader receives the options that it
        accepts.

    Returns
    -------
    raw : NiftiRaw
        The record of the file.
    proxy : nibabel.arrayproxy.ArrayProxy
        The lazy array of the voxels, as stored, with the scaling of the
        header applied when it is read.
    """
    if isinstance(file, str):
        file = path.Path(file)
    if not isinstance(file, path.PathLike):
        return _read_nifti_stream(file, **kwargs)
    if not _is_local(file):
        with file.open("rb") as stream:
            return _read_nifti_stream(BytesIO(stream.read()), **kwargs)
    filename = str(file)
    with open(filename, "rb") as stream:
        raw = NiftiRaw.from_fileobj(stream, **kwargs)
    version = 2 if isinstance(raw.header, nb.Nifti2Header) else 1
    read = _NIFTI_IMAGES[version].from_filename
    image = read(filename, **_accepted(read, kwargs))
    return raw, image.dataobj


def _read_nifti_stream(
    fileobj: tx.BinaryIO, **kwargs
) -> tx.Tuple[NiftiRaw, ArrayProxy]:
    """Read the record and the proxy of a NIfTI stream.

    The record is read first, and the position of the stream is restored
    so that nibabel reads the same header before it builds the proxy.
    """
    with preserve_position(fileobj):
        raw = NiftiRaw.from_fileobj(fileobj, **kwargs)
    return raw, _nifti_from_stream(fileobj, **kwargs).dataobj


def write_nifti(image: _NibabelImage, file: path.FileLike) -> None:
    """Write a nibabel image to a path or a stream.

    nibabel writes the header first and the voxels after it. A path is
    compressed when its name ends with `.gz`, and a stream receives an
    uncompressed file.

    Parameters
    ----------
    image : nibabel.Nifti1Image or nibabel.Nifti2Image
        The image to write.
    file : path or file object
        Where to write the image.
    """
    if isinstance(file, (str, path.PathLike)):
        _save_nifti(image, file)
    else:
        _image_to_stream(image, file)
