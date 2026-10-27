"""MRtrix voxel data: decoding, encoding and locating it, and the lazy
array that reads it."""

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr

# internals
from brainhops._core import path
from brainhops._core.properties import alias
from brainhops._core.proxy import ArrayProxy, ScaledProxy
from brainhops._core.streams import open_compressed
from brainhops._core.strided import StridedProxy
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
    _storage_shape,
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

    Parameters
    ----------
    header : MrtrixRaw
        The record that gives the shape, the type, the byte order and the
        layout of the stored data.
    data : array_like
        The voxels, with one axis for each axis of the header.

    Returns
    -------
    bytes
        The stored data.

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


class _MrtrixProxy(StridedProxy):
    """A lazy array of the stored voxels of an MRtrix image.

    The proxy holds where the voxels lie, which is a path or the stream
    that the caller passed, and the record of the image, which says how
    they are stored. It reads them each time its values are needed and
    keeps no file open, but a stream given by the caller must stay open
    while the values may be read. The values are presented in the axis
    order of the header, whatever `layout` the file uses. They are the
    stored values: when the header has an intensity `scaling`, the
    reader wraps the proxy in a
    [`ScaledProxy`][brainhops._core.proxy.ScaledProxy] that applies it,
    as the NIfTI reader does with the scaling of a NIfTI header.

    A read gets only the bytes that its selection needs (see
    [`StridedProxy`][]): the voxels of a local file that is not
    compressed are memory-mapped when `mmap` is true, and a compressed
    file is decompressed up to the end of the selection. The blocks that
    the proxy gives Dask are slabs of the axis that the file stores
    slowest. Packed `Bit` data cannot be read in part, so any read of
    such an image decodes the whole file.

    A reader builds the lazy array of the image, scaled or not, from
    its record with [`from_record`][].
    """

    # --- magic attributes ----

    record: KwOnly[NoRepr[MrtrixRaw]]
    """
    The record of the image, which describes how its voxels are stored.
    """

    # --- factory ---

    @classmethod
    def from_record(
        cls,
        source: tx.Any,
        start: int,
        offset: int,
        record: MrtrixRaw,
        mmap: bool,
    ) -> ArrayProxy:
        """
        Return the lazy array of the voxels that a record describes.

        The strided layout is read from the record: the stored shape and
        the reversed axes come from its `layout`. Packed `Bit` data are
        described as bytes, which only the whole file decodes. When the
        record has an intensity `scaling` that changes values, the proxy
        is wrapped in a
        [`ScaledProxy`][brainhops._core.proxy.ScaledProxy] that applies
        it.

        Parameters
        ----------
        source : path-like or file object
            The file, by its path, or a stream open in binary mode.
        start : int
            The position of the content in `source`, before
            decompression.
        offset : int
            The position of the first voxel in the decoded content.
        record : MrtrixRaw
            The record of the image.
        mmap : bool
            Whether a local file that is not compressed is memory-mapped.

        Returns
        -------
        _MrtrixProxy or ScaledProxy
            The lazy array of the voxels.
        """
        shape, position = _storage_shape(record.dim, record.layout)
        flips = [axis for axis, s in enumerate(record.layout) if s < 0]
        proxy = cls.contiguous(
            source,
            np.dtype(np.uint8) if record.is_bit else record.dtype,
            shape,
            offset=offset,
            order="C",
            axes=position,
            flips=flips,
            start=start,
            mmap=mmap,
            record=record,
        )
        if not _is_scaled(record):
            return proxy
        offset, scale = record.scaling
        return ScaledProxy(proxy, scale, offset)

    # --- properties ----

    @property
    def dtype(self) -> np.dtype:
        """The type of the stored values, which is `bool` for `Bit` data."""
        return np.dtype(bool) if self.record.is_bit else self.record.dtype

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape, or `None` for packed `Bit` data."""
        if self.record.is_bit:
            return None
        return super().chunks

    nbytes_stored = alias("record.nbytes", readonly=True)
    """The number of bytes of the stored voxels."""

    # --- methods ----

    def read_stored(self) -> bytes:
        """Return the bytes of the voxels as the file stores them.

        Raises
        ------
        ParserContentError
            If the file holds fewer bytes than the record says.
        """
        try:
            return super().read_stored()
        except ValueError as error:
            raise ParserContentError(
                f"The MRtrix data are shorter than the {self.record.nbytes} "
                f"bytes that the header asks for. {error}"
            ) from error

    def get_unscaled(self) -> np.ndarray:
        """Return all the stored values in the axis order of the header.

        The intensity scaling is not applied.
        """
        return decode_data(self.record, self.read_stored())

    def _reads_strided(self) -> bool:
        """Tell whether the voxels are read with strides.

        Packed `Bit` data are decoded from the whole file, so they are
        not.
        """
        return not self.record.is_bit

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        if not self.record.is_bit:
            try:
                return super()._read(index)
            except ValueError as error:
                raise ParserContentError(
                    f"The MRtrix data are shorter than the header asks "
                    f"for. {error}"
                ) from error
        return self.get_unscaled()[index]


def whole_file(raw: tx.Any) -> tx.Optional[_MrtrixProxy]:
    """Return the lazy array of a whole MRtrix file that an array reads.

    A writer copies the stored bytes of such a lazy array as they are.
    The array qualifies when it is the lazy array that the reader built
    for the file: the stored proxy itself, or the scaled proxy that
    applies the intensity scaling of its header. An index, a
    permutation, or a scaling that differs from the header is encoded
    like an array in memory.

    Parameters
    ----------
    raw : array
        The `rawdata` array of an object.

    Returns
    -------
    _MrtrixProxy or None
        The proxy of the stored voxels, or `None`.
    """
    scaled = isinstance(raw, ScaledProxy)
    proxy = raw.base if scaled else raw
    if not isinstance(proxy, _MrtrixProxy):
        return None
    if not scaled:
        return None if _is_scaled(proxy.record) else proxy
    if not _is_scaled(proxy.record):
        return None
    offset, scale = proxy.record.scaling
    if raw.slope is None or raw.inter is None:
        return None
    if raw.slope.size != 1 or raw.inter.size != 1:
        return None
    if float(raw.slope.item()) != scale or float(raw.inter.item()) != offset:
        return None
    return proxy


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
) -> tx.Tuple[MrtrixRaw, ArrayProxy]:
    """Read the record and a lazy array of the voxels of an MRtrix image.

    The file is a path or a stream, gzipped or not. The record is read
    from the header, and the voxels are not read. A separate data file,
    named by the header, is found next to the header file, so a stream
    must then have a name. The data file is checked to exist and, when it
    is a local file that is not compressed, to be long enough.

    Parameters
    ----------
    file : path-like or file object
        The path of the image, or a stream open in binary mode.
    mmap : bool, default=True
        Whether to memory-map a local data file that is not compressed.

    Returns
    -------
    raw : MrtrixRaw
        The record.
    proxy : array
        A lazy array of the voxels, with one axis for each axis of the
        header, and the intensity scaling of the header applied.

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
    return record, _MrtrixProxy.from_record(
        source, start, offset, record, mmap
    )
