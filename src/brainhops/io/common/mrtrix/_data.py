"""MRtrix voxel data: decoding, encoding and locating it, and the lazy
array that reads it."""

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed, preserve_position
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base._utils_files import sibling as _sibling
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserExistsError,
    WriterError,
)

# this format
from ._codecs import (
    image_to_storage,
    storage_to_image,
)
from ._raw import MrtrixRaw

# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def decode_data(header: MrtrixRaw, buffer: tx.Any) -> np.ndarray:
    """
    Decode stored values into an array of header axes.

    The buffer holds the voxel data from their first byte. The result is a
    view of the buffer, except for `Bit` data, which are unpacked into a
    new boolean array. Intensity scaling is not applied.

    Raises
    ------
    ParserContentError
        If the buffer holds fewer values than the header requires.
    """
    count = header.count
    if header.is_bit:
        raw = np.frombuffer(buffer, dtype=np.uint8, count=header.nbytes)
        stored = np.unpackbits(raw, bitorder="big", count=count)
        stored = stored.astype(bool)
    else:
        if isinstance(buffer, np.ndarray):
            stored = buffer[: header.nbytes].view(header.dtype)
        else:
            stored = np.frombuffer(buffer, dtype=header.dtype, count=count)
    if stored.size < count:
        raise ParserContentError(
            f"The MRtrix data hold {stored.size} values, but the header "
            f"asks for {count}."
        )
    return storage_to_image(stored[:count], header.dim, header.layout)


def encode_data(header: MrtrixRaw, data: tx.Any) -> bytes:
    """
    Encode an array of header axes into the bytes of the file.

    Floats stored as integers are rounded.

    Raises
    ------
    WriterError
        If the shape of the array disagrees with the header.
    """
    array = np.asarray(data)
    if tuple(array.shape) != tuple(header.dim):
        raise WriterError(
            f"The data have shape {array.shape}, but the header says "
            f"{header.dim}."
        )
    stored = image_to_storage(array, header.layout)
    if header.is_bit:
        flat = np.ascontiguousarray(stored).reshape(-1).astype(bool)
        return np.packbits(flat, bitorder="big").tobytes()
    dtype = header.dtype
    if dtype.kind in "iu" and stored.dtype.kind in "fc":
        stored = np.round(np.real(stored))
    return np.ascontiguousarray(stored, dtype=dtype).tobytes()


def _is_gzip(file: tx.Any) -> bool:
    """Tell whether a path names a gzip file, judged by its content."""
    try:
        with _open_path(file) as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def _data_location(header: MrtrixRaw) -> tx.Tuple[str, int]:
    """
    Return the `(name, offset)` of the data, checked as MRtrix does.

    Raises
    ------
    ParserContentError
        If the header has no `file` entry, or an embedded image has no
        positive offset.
    """
    if header.file is None:
        raise ParserContentError(
            "The MRtrix header has no 'file' entry, so its data cannot be "
            "found."
        )
    name, offset = header.file
    if name == "." and offset <= 0:
        raise ParserContentError(
            "Invalid offset for an MRtrix image embedded in its header "
            f"file: {offset}"
        )
    return name, offset


# ----------------------------------------------------------------------
#   LAZY ARRAY
# ----------------------------------------------------------------------


def _scaled_dtype(record: MrtrixRaw) -> np.dtype:
    """Return the type of the values once the intensity scaling applies."""
    stored = np.dtype(bool) if record.is_bit else record.dtype
    if _is_scaled(record):
        return np.result_type(stored, np.float32)
    return stored


def _is_scaled(record: MrtrixRaw) -> bool:
    """Tell whether the intensity scaling of a header changes values."""
    scaling = record.scaling
    return bool(scaling) and tuple(scaling) != (0.0, 1.0)


class _MrtrixProxy:
    """A lazy array of the voxels of an MRtrix image.

    The proxy holds where the voxels lie, which is a path or the stream
    that the caller passed, and the record of the image, which says how
    they are stored. It reads them each time its values are needed and
    keeps no file open, but a stream given by the caller must stay open
    while the values may be read. The values are presented in the axis
    order of the header, whatever `layout` the file uses, and the
    intensity `scaling` of the header is applied when they are read, as
    nibabel's proxy applies the scaling of a NIfTI header.

    The voxels of a local file that is not compressed are memory-mapped
    when `mmap` is true, so that reading them gives a read-only view of
    the file.
    """

    def __init__(
        self,
        source: tx.Any,
        start: int,
        offset: int,
        record: MrtrixRaw,
        mmap: bool,
    ) -> None:
        self.source = source
        """The path of the file of the voxels, or the stream that holds it."""
        self.start = int(start)
        """The position of the file in a stream, before decompression."""
        self.offset = int(offset)
        """The position of the first voxel in the decompressed file."""
        self.record = record
        """The record of the image, which describes how voxels are stored."""
        self.mmap = bool(mmap)
        """Whether a local file that is not compressed is memory-mapped."""

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the image, read from the record."""
        return tuple(self.record.dim)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the image."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the values once the intensity scaling applies."""
        return _scaled_dtype(self.record)

    def read_stored(self) -> bytes:
        """Return the bytes of the voxels as the file stores them.

        Raises
        ------
        ParserContentError
            If the file holds fewer bytes than the record says.
        """
        nbytes = self.record.nbytes
        if isinstance(self.source, (str, os.PathLike, path.PathLike)):
            with _open_path(self.source) as f:
                content = _read_at(f, self.offset, nbytes)
        else:
            with preserve_position(self.source):
                self.source.seek(self.start)
                content = _read_at(self.source, self.offset, nbytes)
        if len(content) < nbytes:
            raise ParserContentError(
                f"The MRtrix data hold {len(content)} bytes, but the header "
                f"asks for {nbytes}."
            )
        return content

    def get_unscaled(self) -> np.ndarray:
        """Return the stored values in the axis order of the header."""
        local = None
        if isinstance(self.source, (str, os.PathLike, path.PathLike)):
            local = _local_path(self.source)
        nbytes = self.record.nbytes
        if local is not None and self.mmap and nbytes > 0:
            with open(local, "rb") as f:
                compressed = f.read(2) == b"\x1f\x8b"
            if not compressed:
                buffer = np.memmap(
                    local,
                    dtype=np.uint8,
                    mode="r",
                    offset=self.offset,
                    shape=(nbytes,),
                )
                return decode_data(self.record, buffer)
        return decode_data(self.record, self.read_stored())

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        values = self.get_unscaled()
        if _is_scaled(self.record):
            offset, scale = self.record.scaling
            values = offset + scale * np.asarray(values, dtype=self.dtype)
            values = values.astype(self.dtype, copy=False)
        return values if dtype is None else values.astype(dtype)

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        return np.asarray(self)[index]

    def __repr__(self) -> str:
        return f"_MrtrixProxy(shape={self.shape}, dtype={self.dtype})"


def _read_at(file: tx.BinaryIO, offset: int, nbytes: int) -> bytes:
    """Read bytes at an offset of a stream, after decompression."""
    stream = open_compressed(file)
    if stream is file:
        stream.seek(stream.tell() + offset)
    else:
        stream.read(offset)
    return stream.read(nbytes)


def read_mrtrix(
    file: tx.Any, mmap: bool = True
) -> tx.Tuple[MrtrixRaw, _MrtrixProxy]:
    """Read the record and a lazy array of the voxels of an MRtrix image.

    The file is a path or a stream, gzipped or not. The record is read
    from the header, and the voxels are not read. A separate data file,
    named by the header, is found next to the header file, so a stream
    must then have a name. The data file is checked to exist and, when it
    is a local file that is not compressed, to be long enough.

    Raises
    ------
    ParserExistsError
        If the path or the separate data file does not exist.
    ParserContentError
        If the content is not an MRtrix image, the data cannot be found
        from a stream without a name, or a local data file is too short.
    """
    if isinstance(file, str):
        file = path.Path(file)
    named = isinstance(file, (path.PathLike, os.PathLike))
    if named:
        if not path.exists(file):
            raise ParserExistsError(f"No such file: {file}")
        with _open_path(file) as f:
            record = MrtrixRaw.from_fileobj(f)
        origin, start = file, 0
    else:
        record = MrtrixRaw.from_fileobj(file)
        origin, start = getattr(file, "name", None), file.tell()
    name, offset = _data_location(record)
    if name == ".":
        source = file
    else:
        if not isinstance(origin, (str, bytes, os.PathLike, path.PathLike)):
            raise ParserContentError(
                f"The MRtrix header names a separate data file "
                f"({name!r}), which cannot be found from a stream that "
                f"has no file name. Read it from its path instead."
            )
        if isinstance(origin, bytes):
            origin = os.fsdecode(origin)
        source = _sibling(origin, name)
        start = 0
        if not path.exists(source):
            raise ParserExistsError(
                f"No such file: {source} (the data file named by the "
                f"MRtrix header {origin})"
            )
    if named or name != ".":
        local = _local_path(source)
        if local is not None and not _is_gzip(source):
            size = os.path.getsize(local)
            if size < offset + record.nbytes:
                raise ParserContentError(
                    f"The MRtrix data file {source} holds {size} bytes, but "
                    f"the header asks for {offset + record.nbytes}."
                )
    return record, _MrtrixProxy(source, start, offset, record, mmap)
