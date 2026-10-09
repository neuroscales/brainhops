"""Reading and writing NIfTI files and streams through `nibabel`."""

# stdlib
import gzip
import inspect
from io import BytesIO
from urllib.parse import urlsplit

# dependencies
import nibabel as nb
import typing_extensions as tx

from brainhops._core import path
from brainhops._core.streams import open_compressed

# ----------------------------------------------------------------------
#   READING AND WRITING THROUGH A PATH
# ----------------------------------------------------------------------
#
# `nibabel` opens a file it is handed by name with the built-in `open`,
# so it takes any name for a local file: given `s3://bucket/x.nii` or
# `memory://x.nii`, it looks for a local file of that name. The helpers
# below are the one place in `io` that hands `nibabel` a path. A local
# path goes to `nibabel` by name, so the voxels are memory-mapped as
# before. A remote path is opened through its own backend, and `nibabel`
# is handed the open file.

# The protocols of a path on the local file system, as `bagof.paths`
# reports them: none, `file://` and `local://`.
_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})


def _is_local(file: path.FilenameLike) -> bool:
    """Whether a path names a file that `nibabel` can open by name."""
    return path.Path(file).protocol.lower() in _LOCAL_PROTOCOLS


# A NIfTI image or header class, by NIfTI version.
_NIFTI_IMAGES = {1: nb.Nifti1Image, 2: nb.Nifti2Image}
_NIFTI_HEADERS = {1: nb.Nifti1Header, 2: nb.Nifti2Header}


def _tell(fileobj: tx.IO) -> tx.Optional[int]:
    """The position of a stream, or `None` if it cannot seek back."""
    try:
        return fileobj.tell() if fileobj.seekable() else None
    except Exception:
        return None


# The size of the header, and where and what its magic string is, by
# NIfTI version. NIfTI-1 keeps it near the end of its header, NIfTI-2
# right after `sizeof_hdr`, followed by a DOS/Unix line-ending check.
_NIFTI_HEADER_SIZES = {1: 348, 2: 540}
_NIFTI_MAGICS = {
    1: (344, (b"n+1\0", b"ni1\0")),
    2: (4, (b"n+2\0\r\n\032\n", b"ni2\0\r\n\032\n")),
}


def _has_nifti_magic(head: bytes, version: int) -> bool:
    """
    Whether the first bytes of a (decompressed) file hold the header
    size and magic string of a NIfTI header of the given version.

    This is cheap and exact, so a file that is not a NIfTI is turned
    away before `nibabel` is asked to make sense of it.
    """
    size = _NIFTI_HEADER_SIZES[version]
    if len(head) < size:
        return False
    sizes = {int.from_bytes(head[:4], order) for order in ("little", "big")}
    if size not in sizes:
        return False
    offset, magics = _NIFTI_MAGICS[version]
    return any(head[offset : offset + len(m)] == m for m in magics)


def _nifti_version(fileobj: tx.BinaryIO) -> int:
    """
    The NIfTI version of an open, possibly gzipped, file object: 2 if
    its header size is that of NIfTI-2, else 1.

    The stream is left where it was. One that cannot seek back is not
    peeked at, and taken for NIfTI-1.
    """
    start = _tell(fileobj)
    if start is None:
        return 1
    head = open_compressed(fileobj).read(4)
    fileobj.seek(start)
    # `sizeof_hdr`, in either byte order.
    sizes = {int.from_bytes(head, order) for order in ("little", "big")}
    return 2 if 540 in sizes else 1


def _accepted(func: tx.Callable, kwargs: tx.Mapping[str, tx.Any]) -> dict:
    """
    The keyword arguments, of `kwargs`, that `func` accepts: all of them
    if it takes `**kwargs`.

    The options for reading a NIfTI file differ with how it is read: a
    file `nibabel` opens by name takes `mmap` and `keep_file_open`, a
    stream does not. Each `nibabel` call is handed the ones it knows.
    """
    Parameter = inspect.Parameter
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is Parameter.VAR_KEYWORD for p in parameters):
        return dict(kwargs)
    named = (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    names = {p.name for p in parameters if p.kind in named}
    return {key: value for key, value in kwargs.items() if key in names}


def _image_from_stream(
    image_class: type, fileobj: tx.BinaryIO, **kwargs
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """
    Build a NIfTI image from an open, uncompressed file object.

    `from_stream` arrived in nibabel 5.0; before, the stream goes in a
    file map, as `nibabel` 4's own `from_bytes` does.
    """
    if hasattr(image_class, "from_stream"):
        read = image_class.from_stream
        return read(fileobj, **_accepted(read, kwargs))
    file_map = image_class.make_file_map({"image": fileobj, "header": fileobj})
    read = image_class.from_file_map
    return read(file_map, **_accepted(read, kwargs))


def _image_to_stream(
    image: tx.Union[nb.Nifti1Image, nb.Nifti2Image], fileobj: tx.BinaryIO
) -> None:
    """
    Write a NIfTI image to an open file object, uncompressed.

    `to_stream` arrived in nibabel 5.0; before, the stream goes in a file
    map, as `nibabel` 4's own `to_bytes` does.
    """
    if hasattr(image, "to_stream"):
        image.to_stream(fileobj)
        return
    image.to_file_map(
        image.make_file_map({"image": fileobj, "header": fileobj})
    )


def _nifti_from_stream(
    fileobj: tx.BinaryIO, **kwargs
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """
    Build a NIfTI-1 or NIfTI-2 image from an open, possibly gzipped, file
    object.

    A stream has no name for `nibabel` to tell a `.gz` from, so the
    compression is sniffed from its magic bytes. The image's array proxy
    reads the voxels from `fileobj` lazily, so the caller keeps it open
    for as long as they may be read. On failure, the stream is put back
    where it was.

    `nibabel` cannot memory-map a stream: given `mmap`, it falls back to
    reading the voxels, and `keep_file_open` has no effect on an open
    file object.
    """
    image_class = _NIFTI_IMAGES[_nifti_version(fileobj)]
    start = _tell(fileobj)
    try:
        return _image_from_stream(
            image_class, open_compressed(fileobj), **kwargs
        )
    except Exception:
        if start is not None:
            fileobj.seek(start)
        raise


def _load_nifti(
    file: path.FilenameLike, **kwargs
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """
    Load a NIfTI image from a path, local or remote.

    A local path is handed to `nibabel`'s `from_filename`, which
    memory-maps the voxels and reads them only when asked. A remote path
    is opened through its own backend (universal-pathlib or
    cloudpathlib, through `bagof.paths`) and read into memory: the array
    proxy reads long after this returns, when the remote stream would be
    closed, and reading the voxels fetches them all anyway. Each is
    handed the `kwargs` it accepts; see `_nifti_from_stream`.
    """
    if _is_local(file):
        filename = str(path.Path(file))
        with open(filename, "rb") as f:
            image_class = _NIFTI_IMAGES[_nifti_version(f)]
        read = image_class.from_filename
        return read(filename, **_accepted(read, kwargs))
    with path.Path(file).open("rb") as f:
        buffer = BytesIO(f.read())
    return _nifti_from_stream(buffer, **kwargs)


def _load_nifti_header(
    file: path.FilenameLike,
) -> tx.Union[nb.Nifti1Header, nb.Nifti2Header]:
    """Read the header of a NIfTI file at a path, local or remote,
    without reading its voxels."""
    if _is_local(file):
        return _load_nifti(file).header
    with path.Path(file).open("rb") as f:
        header_class = _NIFTI_HEADERS[_nifti_version(f)]
        return header_class.from_fileobj(open_compressed(f))


def _save_nifti(
    image: tx.Union[nb.Nifti1Image, nb.Nifti2Image], file: path.FilenameLike
) -> None:
    """
    Write a NIfTI image to a path, local or remote, gzipped when its name
    ends in `.gz`.

    A local path is handed to `nibabel`'s `to_filename`, which picks the
    compression from the extension. A remote path is opened through its
    own backend, and the image written to the stream.
    """
    if _is_local(file):
        image.to_filename(str(path.Path(file)))
        return
    # The name is read from the URL's text: a backend may not know it,
    # and a query (`?token=...`) is not part of it. A stream is written
    # what it is given, so compression is ours to add. `open_compressed`
    # only reads (`indexed_gzip` cannot write), so this is `gzip`'s.
    compress = urlsplit(str(file)).path.lower().endswith(".gz")
    with path.Path(file).open("wb") as f:
        if compress:
            with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                _image_to_stream(image, gz)
        else:
            _image_to_stream(image, f)


def _like_header(like: tx.Any) -> tx.Optional[nb.Nifti1Header]:
    """
    Resolve a `like` template to the NIfTI header to copy fields from.

    The template may be a path to a NIfTI file, a `nibabel` image or
    header, or an object built by this package that carries a header. An
    object with no readable header resolves to `None`.
    """
    if like is None:
        return None
    if isinstance(like, (nb.Nifti1Header, nb.Nifti2Header)):
        return like
    if isinstance(like, (nb.Nifti1Image, nb.Nifti2Image)):
        return like.header
    header = getattr(like, "header", None)
    if header is not None:
        return header
    if isinstance(like, (str, path.PathLike)):
        return _load_nifti_header(like)
    return None
