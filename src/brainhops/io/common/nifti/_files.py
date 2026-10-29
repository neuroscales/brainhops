"""Writing NIfTI streams through `nibabel`, and recognising NIfTI headers."""

# stdlib

# dependencies
import nibabel as nb
import typing_extensions as tx

from brainhops._core.streams import open_compressed

# ----------------------------------------------------------------------
#   RECOGNISING AND WRITING STREAMS
# ----------------------------------------------------------------------
#
# `nibabel` writes a NIfTI file to a stream, which the writer of
# `NiftiReaderWriter` opens on a local or a remote path, compressed as
# its name asks. `nibabel` is never handed a path, because it would take
# `s3://bucket/x.nii` for the name of a local file. The voxels are not
# read through `nibabel` (see `_data.py`).

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
