"""Low-level MGH/MGZ helpers: raw header prefix, file names and
trailing tags."""

# dependencies
import typing_extensions as tx
from nibabel.freesurfer import mghformat as _mgh

# this format
from ._constants import (
    _MGH_TYPES,
    _PREFIX,
    MGH_FOOTER_SIZE,
)


def _read_prefix(fileobj: tx.BinaryIO) -> tx.Optional[tuple]:
    """
    Read the leading header fields of a decompressed stream, or `None` if
    the stream is too short.
    """
    raw = fileobj.read(_PREFIX.size)
    if len(raw) < _PREFIX.size:
        return None
    return _PREFIX.unpack(raw)


def _valid_prefix(prefix: tx.Optional[tuple]) -> bool:
    """Tell whether the leading fields describe an MGH volume."""
    if prefix is None:
        return False
    version, *dims, dtype, _dof, _flag = prefix
    return version == 1 and all(d > 0 for d in dims) and dtype in _MGH_TYPES


def _good_ras(prefix: tx.Optional[tuple]) -> tx.Optional[bool]:
    """
    Tell whether the `goodRASFlag` of the leading fields is positive, or
    `None` if they could not be read.
    """
    return None if prefix is None else prefix[-1] > 0


def _mgh_from_filename(
    filename: str,
    *,
    mmap: tx.Union[bool, str] = True,
    keep_file_open: tx.Optional[bool] = None,
) -> _mgh.MGHImage:
    """
    Load an image like nibabel's `MGHImage.from_filename`, but close the
    file the header is read from, which nibabel leaks.

    The voxels stay lazy, because the array proxy opens the file itself.
    """
    klass = _mgh.MGHImage
    if mmap not in (True, False, "c", "r"):
        raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
    file_map = klass.filespec_to_file_map(filename)
    holder = file_map["image"]
    with holder.get_prepare_fileobj("rb") as f:
        header = klass.header_class.from_fileobj(f)
    data = klass.ImageArrayProxy(
        holder.file_like,
        header.copy(),
        mmap=mmap,
        keep_file_open=keep_file_open,
    )
    return klass(data, header.get_affine(), header, file_map=file_map)


def _seekable(fileobj: tx.IO) -> bool:
    """Tell whether a stream can seek (no answer means no)."""
    try:
        return bool(fileobj.seekable())
    except Exception:
        return False


def _read_tags(stream: tx.BinaryIO, header: _mgh.MGHHeader) -> bytes:
    """
    Read the raw bytes after the footer of a decompressed stream.

    Offsets are counted from the start of the stream, as nibabel counts
    them.
    """
    offset = int(header.get_footer_offset()) + MGH_FOOTER_SIZE
    try:
        stream.seek(offset)
    except Exception:
        return b""
    return stream.read() or b""
