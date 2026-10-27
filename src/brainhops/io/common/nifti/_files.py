"""Reading and writing NIfTI files and streams through `nibabel`."""

# stdlib
import inspect

# dependencies
import nibabel as nb
import typing_extensions as tx
from nibabel.arrayproxy import ArrayProxy

from brainhops._core.streams import open_compressed
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base._utils_proxies import NibabelProxy

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

_NIFTI_IMAGES = {1: nb.Nifti1Image, 2: nb.Nifti2Image}
_NIFTI_HEADERS = {1: nb.Nifti1Header, 2: nb.Nifti2Header}


def _tell(fileobj: tx.IO) -> tx.Optional[int]:
    """Return the stream position, or `None` if the stream cannot seek."""
    try:
        return fileobj.tell() if fileobj.seekable() else None
    except Exception:
        return None


# The NIfTI-1 magic sits near the end of the header, and the NIfTI-2 magic
# right after `sizeof_hdr`, followed by a line-ending check.
_NIFTI_HEADER_SIZES = {1: 348, 2: 540}
_NIFTI_MAGICS = {
    1: (344, (b"n+1\0", b"ni1\0")),
    2: (4, (b"n+2\0\r\n\032\n", b"ni2\0\r\n\032\n")),
}


def _has_nifti_magic(head: bytes, version: int) -> bool:
    """Return whether a decompressed header starts with the header size and
    magic of a NIfTI version.
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
    """Return the NIfTI version of a possibly compressed stream.

    The stream position is preserved. A stream that cannot seek is taken to be
    NIfTI-1.
    """
    start = _tell(fileobj)
    if start is None:
        return 1
    head = open_compressed(fileobj).read(4)
    fileobj.seek(start)
    # `sizeof_hdr` in either byte order.
    sizes = {int.from_bytes(head, order) for order in ("little", "big")}
    return 2 if 540 in sizes else 1


def _accepted(func: tx.Callable, kwargs: tx.Mapping[str, tx.Any]) -> dict:
    """Return the keyword arguments that `func` accepts.

    Files opened by name accept `mmap` and `keep_file_open`, whereas streams do
    not, so each nibabel call receives only the options it knows.
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
    """Read a NIfTI image from an uncompressed stream.

    Before nibabel 5.0, which added `from_stream`, the stream is passed in a
    file map.
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
    """Write a NIfTI image to an uncompressed stream.

    Before nibabel 5.0, which added `to_stream`, the stream is passed in a file
    map.
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
    """Read a NIfTI-1 or NIfTI-2 image from a possibly compressed stream.

    The voxels are read lazily, so the stream must stay open while they may be
    accessed. A stream cannot be memory-mapped. On failure, the stream position
    is restored.
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


# ----------------------------------------------------------------------
#   READING THE VOXELS
# ----------------------------------------------------------------------
# The views of the NIfTI-based formats convert between `rawdata`, the array as
# the file stores it, and `data`, the array that the data model exposes.
# They are defined beside each format, and they all read the stored array
# with the function below.


def _read(raw: ArrayProtocol) -> ArrayProtocol:
    """Return a lazy array of the voxels from a stored array.

    A nibabel proxy, which reads the voxels of a file, is wrapped in a
    lazy array of brainhops (see
    [`NibabelProxy`][brainhops.io.base._utils_proxies.NibabelProxy]), so
    that the views of the formats index and transpose it without reading
    anything. Any other array is returned as it is.
    """
    if not isinstance(raw, ArrayProxy):
        return raw
    return NibabelProxy(raw)
