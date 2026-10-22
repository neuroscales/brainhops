"""A lazy array of fixed-size values stored contiguously in a file.

Many formats store their voxels as one block of fixed-size values, at an
offset of a file that may be compressed: MRtrix, NRRD and AFNI images,
for example. [`StridedProxy`][] reads such a block. It differs from a
proxy that decodes the whole file because it reads only the bytes
that a selection needs, and because it tells Dask to split the array
into slabs of the axis that the file stores slowest, so that each block
of a Dask array reads one contiguous range of bytes.

The design is the third layer of the memo `spec/design/array-proxies.md`,
in a simpler form: the proxy reads the smallest range of whole slabs of
the slowest stored axis that holds the selection, rather than a merged
list of byte ranges.
"""

__all__ = ["StridedProxy"]

# stdlib
import math
import os

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.proxy import ArrayProxy
from brainhops._core.streams import open_compressed, preserve_position

_CHUNK_BYTES = 128 * 2**20
"""The size of a block when Dask is not installed to give its own."""


class StridedProxy(ArrayProxy):
    """A lazy array of fixed-size values stored in one block of a file.

    The file holds an array of shape `shape`, stored in C order (the last
    axis varies fastest) or in Fortran order (the first axis varies
    fastest), from the byte `offset` of the decoded stream. The stream is
    the content of `source` from the position `start`, decompressed when
    it is compressed with gzip, bzip2, xz or zstd. The proxy presents the
    stored array with its axes reordered by `axes` and the axes listed in
    `flips` reversed, which covers the layouts of MRtrix and NRRD, and
    applies a linear scaling `slope * value + inter` to the values that it
    reads.

    A read gets the smallest range of whole slabs of the slowest stored
    axis that holds the selection. A local file that is not compressed is
    memory-mapped instead, when `mmap` is true, which reads only the pages
    of the file that the selection touches. A compressed stream is
    decompressed from its start to the end of the range, and only the
    range is kept in memory. The blocks that the proxy prefers are slabs
    of the slowest stored axis, as many as fit in the block size of Dask,
    which is the option `array.chunk-size` of its configuration.

    The proxy keeps no file open between reads. When `source` is a stream,
    the caller must keep it open while the values may be read.

    Parameters
    ----------
    source : path-like or file object
        The file, by its path, or a stream open in binary mode.
    dtype : numpy.dtype
        The type of the stored values, with their byte order.
    shape : sequence of int
        The shape of the stored array, in the order of its axes.
    offset : int, default=0
        The position of the first value in the decoded stream.
    order : {"C", "F"}, default="C"
        The order in which the values are stored.
    axes : sequence of int, optional
        The stored axis that each presented axis is. By default, the
        presented axes are the stored axes.
    flips : sequence of int, default=()
        The presented axes that are reversed.
    slope : array_like, optional
        The factor of the scaling, a number or an array that broadcasts
        against the presented shape.
    inter : array_like, optional
        The offset of the scaling, a number or an array that broadcasts
        against the presented shape.
    start : int, default=0
        The position of the content in `source`, a path or a stream,
        before decompression.
    mmap : bool, default=True
        Whether a local file that is not compressed is memory-mapped.
    """

    def __init__(
        self,
        source: tx.Any,
        dtype: tx.Any,
        shape: tx.Sequence[int],
        offset: int = 0,
        order: str = "C",
        axes: tx.Optional[tx.Sequence[int]] = None,
        flips: tx.Sequence[int] = (),
        slope: tx.Any = None,
        inter: tx.Any = None,
        start: int = 0,
        mmap: bool = True,
    ) -> None:
        if order not in ("C", "F"):
            raise ValueError(f"order must be 'C' or 'F', not {order!r}.")
        shape = tuple(int(n) for n in shape)
        axes = tuple(range(len(shape))) if axes is None else tuple(axes)
        if sorted(axes) != list(range(len(shape))):
            raise ValueError(
                f"axes {axes} is not a permutation of the {len(shape)} "
                f"stored axes."
            )
        self.source = source
        """The file, by its path, or the stream that holds it."""
        self.stored_dtype = np.dtype(dtype)
        """The type of the stored values, with their byte order."""
        self.stored_shape = shape
        """The shape of the stored array, in the order of its axes."""
        self.offset = int(offset)
        """The position of the first value in the decoded stream."""
        self.order = order
        """The order in which the values are stored, `"C"` or `"F"`."""
        self.axes = axes
        """The stored axis that each presented axis is."""
        self.flips = tuple(sorted(set(int(i) for i in flips)))
        """The presented axes that are reversed."""
        self.slope = None if slope is None else np.asarray(slope)
        """The factor of the scaling, or `None`."""
        self.inter = None if inter is None else np.asarray(inter)
        """The offset of the scaling, or `None`."""
        self.start = int(start)
        """The position of the content in the source, before
        decompression."""
        self.mmap = bool(mmap)
        """Whether a local file that is not compressed is memory-mapped."""

    # ------------------------------------------------------------------
    #   DESCRIPTION
    # ------------------------------------------------------------------

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the presented array."""
        return tuple(self.stored_shape[axis] for axis in self.axes)

    @property
    def dtype(self) -> np.dtype:
        """The type of the values once the scaling applies.

        Values that are not scaled keep the stored type, with its byte
        order. A scaled integer becomes a float32, as a scaled float32
        stays one, whatever the type of the factors.
        """
        if not self.scaled:
            return self.stored_dtype
        return np.result_type(self.stored_dtype, np.float32)

    @property
    def scaled(self) -> bool:
        """Whether the scaling changes any value."""
        slope = self.slope
        inter = self.inter
        return not (
            (slope is None or bool(np.all(slope == 1)))
            and (inter is None or bool(np.all(inter == 0)))
        )

    @property
    def nbytes_stored(self) -> int:
        """The number of bytes of the stored array."""
        return math.prod(self.stored_shape) * self.stored_dtype.itemsize

    @property
    def base_chunks(self) -> tx.Tuple[int, ...]:
        """The block shape: slabs of the slowest stored axis.

        A block holds every value of the other axes, and as many slabs as
        fit in the block size of Dask, with at least one slab.
        """
        shape = self.stored_shape
        if not shape:
            return ()
        slowest = self._slowest
        slab = math.prod(n for i, n in enumerate(shape) if i != slowest)
        slab_bytes = max(1, slab * self.dtype.itemsize)
        count = max(1, _chunk_bytes() // slab_bytes)
        stored = list(shape)
        stored[slowest] = min(count, shape[slowest])
        return tuple(stored[axis] for axis in self.axes)

    @property
    def _slowest(self) -> int:
        # The stored axis that varies slowest.
        return 0 if self.order == "C" else len(self.stored_shape) - 1

    def __repr__(self) -> str:
        return (
            f"StridedProxy({_describe(self.source)}, shape={self.shape}, "
            f"dtype={self.dtype})"
        )

    # ------------------------------------------------------------------
    #   READING
    # ------------------------------------------------------------------

    def read_stored(self) -> bytes:
        """Return the bytes of the stored array as the file stores them.

        The bytes are those of the whole array, whatever part of it a view
        of the proxy selects. A writer copies them when it stores the same
        layout.

        Raises
        ------
        ValueError
            If the file holds fewer bytes than the array needs.
        """
        return self._read_bytes(0, self.nbytes_stored)

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        stored_index = [slice(None)] * len(self.stored_shape)
        for presented, entry in enumerate(index):
            axis = self.axes[presented]
            size = self.stored_shape[axis]
            if presented in self.flips:
                entry = _flip_entry(entry, size)
            stored_index[axis] = entry
        values = self._read_stored(tuple(stored_index))
        # The stored axes that remain, in their stored order, are put in
        # the presented order, and the flipped ones are reversed.
        kept = [
            axis
            for axis, entry in enumerate(stored_index)
            if not isinstance(entry, (int, np.integer))
        ]
        order = [
            kept.index(axis)
            for presented, axis in enumerate(self.axes)
            if axis in kept
        ]
        values = np.transpose(values, order)
        flips = []
        position = 0
        for presented, axis in enumerate(self.axes):
            if axis not in kept:
                continue
            if presented in self.flips:
                flips.append(position)
            position += 1
        if flips:
            values = np.flip(values, axis=tuple(flips))
        if self.scaled:
            values = self._scale(values, index)
        return values

    def _read_stored(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        # The values of a box of the stored array, in the stored axis order.
        shape = self.stored_shape
        dtype = self.stored_dtype
        local = _local(self.source)
        if (
            local is not None
            and self.mmap
            and self.nbytes_stored > 0
            and not _compressed(local, self.start)
        ):
            array = np.memmap(
                local,
                dtype=dtype,
                mode="r",
                offset=self.start + self.offset,
                shape=shape,
                order=self.order,
            )
            # A view of the map, which reads the pages of the box when its
            # values are used.
            return array[index]
        slowest = self._slowest
        entry = index[slowest]
        size = shape[slowest]
        if isinstance(entry, (int, np.integer)):
            low, high = int(entry), int(entry) + 1
            local_entry = 0
        else:
            low, high, step = entry.indices(size)
            high = max(low, high)
            local_entry = slice(0, high - low, step)
        slab = math.prod(n for i, n in enumerate(shape) if i != slowest)
        slab_bytes = slab * dtype.itemsize
        content = self._read_bytes(low * slab_bytes, (high - low) * slab_bytes)
        part = list(shape)
        part[slowest] = high - low
        array = np.frombuffer(content, dtype=dtype).reshape(
            part, order=self.order
        )
        local_index = list(index)
        local_index[slowest] = local_entry
        return np.array(array[tuple(local_index)])

    def _read_bytes(self, position: int, nbytes: int) -> bytes:
        # `nbytes` bytes of the decoded stream, from the first value plus
        # `position`.
        if nbytes <= 0:
            return b""
        if isinstance(self.source, (str, os.PathLike, path.PathLike)):
            with _open(self.source) as f:
                f.seek(self.start)
                content = _read_at(f, self.offset + position, nbytes)
        else:
            with preserve_position(self.source):
                self.source.seek(self.start)
                content = _read_at(self.source, self.offset + position, nbytes)
        if len(content) < nbytes:
            raise ValueError(
                f"{_describe(self.source)} ends before the {nbytes} bytes "
                f"that are read from position {self.offset + position}."
            )
        return content

    def _scale(
        self, values: np.ndarray, index: tx.Tuple[tx.Any, ...]
    ) -> np.ndarray:
        # The scaling, with its factors indexed as the values were.
        values = np.asarray(values, dtype=self.dtype)
        if self.slope is not None:
            values = values * _index_factor(self.slope, index, self.ndim)
        if self.inter is not None:
            values = values + _index_factor(self.inter, index, self.ndim)
        return values.astype(self.dtype, copy=False)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _chunk_bytes() -> int:
    """Return the number of bytes of a block that Dask is configured for."""
    try:
        import dask
        from dask.utils import parse_bytes
    except ImportError:
        return _CHUNK_BYTES
    size = dask.config.get("array.chunk-size", _CHUNK_BYTES)
    return int(parse_bytes(size)) if isinstance(size, str) else int(size)


def _flip_entry(entry: tx.Any, size: int) -> tx.Any:
    # The stored positions that an entry of a reversed axis selects: an
    # integer, or a slice with a positive step whose order is restored by
    # the reversal of the read values.
    if isinstance(entry, (int, np.integer)):
        return size - 1 - int(entry)
    start, stop, step = entry.indices(size)
    count = len(range(start, stop, step))
    if count == 0:
        return slice(0, 0, 1)
    last = start + (count - 1) * step
    return slice(size - 1 - last, size - start, step)


def _index_factor(
    factor: np.ndarray, index: tx.Tuple[tx.Any, ...], ndim: int
) -> np.ndarray:
    # A scaling factor that broadcasts against the presented array, indexed
    # as the presented array was: an axis of length one is kept as it is.
    if factor.ndim == 0:
        return factor
    factor = factor.reshape((1,) * (ndim - factor.ndim) + factor.shape)
    selection = []
    for axis, entry in enumerate(index):
        if factor.shape[axis] == 1:
            selection.append(
                0 if isinstance(entry, (int, np.integer)) else slice(None)
            )
        else:
            selection.append(entry)
    return factor[tuple(selection)]


def _local(source: tx.Any) -> tx.Optional[str]:
    # The local path of an existing regular file, or `None` for a stream
    # or a remote file.
    if not isinstance(source, (str, os.PathLike, path.PathLike)):
        return None
    try:
        local = os.fspath(source)
    except TypeError:
        return None
    if isinstance(local, bytes):
        local = os.fsdecode(local)
    return local if os.path.isfile(local) else None


def _compressed(local: str, start: int = 0) -> bool:
    # Whether the content of a local file from `start` is compressed,
    # judged by its magic bytes.
    with open(local, "rb") as f:
        f.seek(start)
        stream = open_compressed(f)
        return stream is not f


def _open(source: tx.Any) -> tx.BinaryIO:
    # A binary stream on a path, local or remote.
    local = _local(source)
    if local is not None:
        return open(local, "rb")
    return source.open("rb")


def _read_at(file: tx.BinaryIO, offset: int, nbytes: int) -> bytes:
    # Bytes at an offset of a stream, after decompression. A stream that is
    # not compressed is sought, and a decompressed one is sought as far as
    # it can be, which decompresses up to the offset without keeping it.
    stream = open_compressed(file)
    stream.seek(stream.tell() + offset)
    return stream.read(nbytes)


def _describe(source: tx.Any) -> str:
    if isinstance(source, (str, os.PathLike, path.PathLike)):
        return repr(str(source))
    return repr(source)
