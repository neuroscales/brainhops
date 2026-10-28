"""A lazy array of fixed-size values laid out with strides in a file.

Many formats store their voxels as fixed-size values at an offset of a
file that may be compressed: NIfTI, MGH, MRtrix, NRRD and AFNI images,
for example.
[`StridedProxy`][] reads such values. It differs from a proxy that
decodes the whole file because it reads only the bytes that a selection
needs, and because it tells Dask to split the array into slabs of the
axis with the largest stride, so that each block of a Dask array reads
one contiguous range of bytes. A format that also stores a scaling of
the values wraps the proxy in a
[`ScaledProxy`][brainhops._core.proxy.ScaledProxy].

Where each value lies is given as NumPy gives it for an array in
memory: the position of the first value, and one stride in bytes for
each axis, which is the distance between two neighbours along that
axis. A stride may be negative, which reverses the axis. The strides
describe C order, Fortran order, a permutation of the axes, reversed
axes and padding between rows or slices, without copying anything.

Transposing the proxy, and reshaping it when NumPy could reshape the
values without copying them, give another strided proxy of the same
bytes with other strides, so that the result reads only what it needs
as the proxy does.

The design is the third layer of the memo `spec/design/array-proxies.md`,
in a simpler form: the proxy reads the range of bytes that holds the
selected box, rather than a merged list of byte ranges.
"""

__all__ = ["StridedProxy"]

# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import ConvertTo

# internals
from brainhops._core.path import Path, as_path_or_stream, is_local_file
from brainhops._core.properties import alias
from brainhops._core.proxy import (
    ArrayProxy,
    StoredProxy,
    _permutation,
    _whole,
)
from brainhops._core.streams import open_compressed, preserve_position
from brainhops._core.typing import DTypeLike

_CHUNK_BYTES = 128 * 2**20
"""The size of a block when Dask is not installed to give its own."""


class StridedProxy(StoredProxy):
    """A lazy array of fixed-size values laid out with strides in a file.

    The values are those of the decoded stream, which is the content of
    `source` from the position `start`, decompressed when it is
    compressed with gzip, bzip2, xz or zstd. The value at the index
    `(i, j, ...)` starts at the byte `offset + i * strides[0] + j *
    strides[1] + ...` of that stream, as in a NumPy array. A format that
    stores a contiguous array in C or Fortran order, with its axes
    permuted or reversed, describes it with [`contiguous`][].

    A read gets the range of bytes that holds the selected box, from its
    first byte to its last one. A local file that is not compressed is
    memory-mapped instead, when `mmap` is true, which reads only the
    pages of the file that the selection touches. A compressed stream is
    decompressed from its start to the end of the range, and only the
    range is kept in memory. The blocks that the proxy prefers are slabs
    of the axis with the largest stride, as many as fit in the block size
    of Dask, which is the option `array.chunk-size` of its configuration.

    The proxy keeps no file open between reads. When `source` is a stream,
    the caller must keep it open while the values may be read.

    Its constructor is built from the fields below, as for any
    [`Magic`][bagof.magic.Magic] class, and it converts each argument to
    the type of its field, such as a list of sizes to a tuple of integers.
    The stored type and shape are given as `dtype=` and `shape=` and kept
    in the private fields `_dtype` and `_shape`, which the read-only
    aliases `dtype` and `shape` give, because the base class declares
    `dtype` and `shape` as abstract properties, which a field cannot
    implement.
    """

    source: tx.Annotated[
        tx.Union[Path, tx.BinaryIO], ConvertTo(as_path_or_stream)
    ]
    """
    The file, by its path, or a stream open in binary mode. A file name
    given as a string or a path-like object is converted to a
    [`Path`][brainhops._core.path.Path].
    """

    _dtype: np.dtype
    """The type of the stored values, with their byte order, given as
    `dtype=`."""

    _shape: tx.Tuple[int, ...]
    """The shape of the array, given as `shape=`."""

    strides: tx.Tuple[int, ...]
    """
    The distance in bytes between two neighbouring values along each axis.
    A negative stride reverses the axis.
    """

    offset: int = 0
    """
    The position in the decoded stream of the value whose index is zero
    along every axis.
    """

    start: int = 0
    """
    The position of the content in `source`, a path or a stream, before
    decompression.
    """

    mmap: bool = True
    """Whether a local file that is not compressed is memory-mapped."""

    def __post_init__(self) -> None:
        # The layout is checked before anything is read.
        if len(self.strides) != len(self._shape):
            raise ValueError(
                f"The proxy has {len(self._shape)} axes and "
                f"{len(self.strides)} strides."
            )
        if self._count and self._whole_span[0] < 0:
            raise ValueError(
                f"The strides {self.strides} reach before the start of the "
                f"stream from the offset {self.offset}."
            )

    @classmethod
    def contiguous(
        cls,
        source: tx.Any,
        dtype: DTypeLike,
        shape: tx.Sequence[int],
        offset: int = 0,
        order: tx.Literal["C", "F"] = "C",
        axes: tx.Optional[tx.Sequence[int]] = None,
        flips: tx.Sequence[int] = (),
        **kwargs: tx.Any,
    ) -> tx.Self:
        """
        Return the proxy of a contiguous array, presented with its axes moved.

        A contiguous array is stored without gaps, in C order (the last axis
        varies fastest) or in Fortran order (the first axis varies
        fastest), from the byte `offset` of the decoded stream. The proxy
        presents it with its axes reordered by `axes` and the axes listed
        in `flips` reversed, which covers the layouts of MRtrix and NRRD.

        Parameters
        ----------
        source : path-like or file object
            The file, by its path, or a stream open in binary mode.
        dtype : numpy.dtype
            The type of the stored values, with their byte order.
        shape : sequence of int
            The shape of the stored array, in the order of its axes.
        offset : int, default=0
            The position of the first stored value in the decoded stream.
        order : {"C", "F"}, default="C"
            The order in which the values are stored.
        axes : sequence of int, optional
            The stored axis that each presented axis is. By default, the
            presented axes are the stored axes.
        flips : sequence of int, default=()
            The presented axes that are reversed.
        **kwargs : Any
            The other fields of the proxy: `start`, `mmap`, and those of
            a subclass.

        Returns
        -------
        StridedProxy
            The proxy, of the class on which the method is called.

        Raises
        ------
        ValueError
            If `order` is neither `"C"` nor `"F"`, or `axes` is not a
            permutation of the stored axes.
        """
        if order not in ("C", "F"):
            raise ValueError(f"order must be 'C' or 'F', not {order!r}.")
        dtype = np.dtype(dtype)
        shape = tuple(map(int, shape))
        axes = tuple(range(len(shape))) if axes is None else tuple(axes)
        if sorted(axes) != list(range(len(shape))):
            raise ValueError(
                f"axes {axes} is not a permutation of the {len(shape)} "
                f"stored axes."
            )
        stored = _contiguous_strides(shape, dtype.itemsize, order)
        strides = [stored[axis] for axis in axes]
        presented = [shape[axis] for axis in axes]
        offset = int(offset)
        for axis in set(int(i) for i in flips):
            # A reversed axis starts at its last value and steps back.
            offset += (presented[axis] - 1) * strides[axis]
            strides[axis] = -strides[axis]
        return cls(source, dtype, presented, strides, offset, **kwargs)

    # ------------------------------------------------------------------
    #   DESCRIPTION
    # ------------------------------------------------------------------

    shape = alias("_shape", readonly=True)
    """The shape of the array."""

    dtype = alias("_dtype", readonly=True)
    """The type of the stored values, with their byte order."""

    @property
    def nbytes_stored(self) -> int:
        """The number of bytes from the first stored value to the last."""
        if not self._count:
            return 0
        low, high = self._whole_span
        return high - low

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        """The block shape: slabs of the axis with the largest stride.

        An axis of length one is not chosen, because it holds a single
        slab. A block holds every value of the other axes, and as many
        slabs as fit in the block size of Dask, with at least one slab.
        """
        shape = self._shape
        if not shape:
            return ()
        slowest = self._slowest
        slab = math.prod(n for i, n in enumerate(shape) if i != slowest)
        slab_bytes = max(1, slab * self.dtype.itemsize)
        count = max(1, _chunk_bytes() // slab_bytes)
        chunks = list(shape)
        chunks[slowest] = min(count, shape[slowest])
        return tuple(chunks)

    @property
    def _count(self) -> int:
        # The number of values of the array.
        return math.prod(self._shape)

    @property
    def _whole_span(self) -> tx.Tuple[int, int]:
        # The range of bytes that holds the whole array.
        return _span(
            self._shape,
            self.strides,
            self._dtype.itemsize,
            self.offset,
        )

    @property
    def _slowest(self) -> int:
        # The axis whose neighbours lie furthest apart in the stream. An
        # axis of length one has no neighbours, so its stride is ignored,
        # unless every axis has length one.
        reach = [
            abs(stride) if size > 1 else -1
            for size, stride in zip(self._shape, self.strides)
        ]
        return int(np.argmax(reach))

    # ------------------------------------------------------------------
    #   RESHAPING
    # ------------------------------------------------------------------

    def _reads_strided(self) -> bool:
        """Tell whether the values are read with the strides of the proxy.

        A subclass that decodes its values in another way, such as packed
        bits, returns false, and is then reshaped as any other proxy.
        """
        return True

    def transpose(self, *axes: tx.Any) -> StoredProxy:
        """Return the proxy with its axes in another order.

        The result is a strided proxy of the same bytes, whose shape and
        strides are permuted, so that it reads the bytes of a selection
        and gives Dask slabs of its slowest axis, as the proxy does. Its
        `root` is the root of the proxy, as for a reshape. A subclass
        that does not read its values with strides gives a
        [`PermutedProxy`][brainhops._core.proxy.PermutedProxy] instead.

        Parameters
        ----------
        *axes : int or sequence of int
            The permutation, given as integers or as one sequence. Axis
            `i` of the result is axis `axes[i]` of the proxy.

        Returns
        -------
        StoredProxy
            The transposed proxy.

        Raises
        ------
        ValueError
            If the axes are not a permutation of the axes of the proxy.
        """
        if not self._reads_strided():
            return super().transpose(*axes)
        perm = _permutation(axes, len(self._shape))
        if perm == tuple(range(len(perm))):
            return self
        return self._restrided(
            [self._shape[i] for i in perm], [self.strides[i] for i in perm]
        )

    def _reshaped(self, shape: tx.Tuple[int, ...]) -> ArrayProxy:
        """Return the proxy in another shape, by changing its strides.

        When NumPy could reshape the values of the proxy without copying
        them, the result is a strided proxy of the same bytes with the
        new shape and the strides that NumPy would give, and reading it
        reads only the bytes of a selection, as the proxy does. Its
        `root` is the root of the proxy, so that a writer still sees
        which file it reads, but it is not an instance of the class of
        the proxy, so that a writer does not copy its bytes as those of
        the whole file. Any other reshape gives a
        [`ReshapedProxy`][brainhops._core.proxy.ReshapedProxy].
        """
        box = _whole(self._shape)
        reshaped = self._reshaped_box(box, shape)
        if reshaped is None:
            return super()._reshaped(shape)
        return reshaped

    def _reshaped_box(
        self, box: tx.Tuple[tx.Any, ...], shape: tx.Tuple[int, ...]
    ) -> tx.Optional[StoredProxy]:
        """Return a box of the proxy in another shape, or `None`.

        The box is a strided array of the same bytes, with its own first
        value and strides, so it can be reshaped as the proxy is, when
        NumPy could do it without copying the values.
        """
        if not self._reads_strided() or not math.prod(shape):
            return None
        first, old_shape, old_strides = self._layout(box)
        strides = _restride(old_shape, old_strides, shape)
        if strides is None:
            return None
        return self._restrided(shape, strides, first)

    def _restrided(
        self,
        shape: tx.Sequence[int],
        strides: tx.Sequence[int],
        offset: tx.Optional[int] = None,
    ) -> "_RestridedProxy":
        """Return a strided proxy of the same bytes with other strides."""
        return _RestridedProxy(
            self.source,
            self._dtype,
            shape,
            strides,
            self.offset if offset is None else offset,
            start=self.start,
            mmap=self.mmap,
            base=self.root,
        )

    # ------------------------------------------------------------------
    #   READING
    # ------------------------------------------------------------------

    def read_stored(self) -> bytes:
        """Return the bytes from the first stored value to the last.

        A writer copies these bytes when it stores the same layout.

        Raises
        ------
        ValueError
            If the file holds fewer bytes than the array needs.
        """
        if not self._count:
            return b""
        low, _ = self._whole_span
        return self._read_bytes(low, self.nbytes_stored)

    def _layout(
        self, index: tx.Tuple[tx.Any, ...]
    ) -> tx.Tuple[int, tx.List[int], tx.List[int]]:
        # The position of the first value of a box, its shape and its
        # strides, which follow from the index.
        first = self.offset
        shape = []
        strides = []
        for entry, size, stride in zip(index, self._shape, self.strides):
            if isinstance(entry, (int, np.integer)):
                first += int(entry) * stride
                continue
            start, stop, step = entry.indices(size)
            first += start * stride
            shape.append(len(range(start, stop, step)))
            strides.append(step * stride)
        return first, shape, strides

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        # The stored values of a box. The bytes between its first and its
        # last value are read and viewed with the strides of the box.
        first, shape, strides = self._layout(index)
        dtype = self._dtype
        local = is_local_file(self.source)
        if (
            local
            and self.mmap
            and self.nbytes_stored > 0
            and not _compressed(self.source, self.start)
        ):
            # A view of a map of the whole array, which reads the pages of
            # the box when its values are used.
            low, high = self._whole_span
            buffer = np.memmap(
                self.source,
                dtype=np.uint8,
                mode="r",
                offset=self.start + low,
                shape=(high - low,),
            )
            return np.ndarray.__new__(
                np.memmap,
                shape,
                dtype,
                buffer=buffer,
                offset=first - low,
                strides=strides,
            )
        low, high = _span(shape, strides, dtype.itemsize, first)
        content = self._read_bytes(low, high - low)
        array = np.ndarray(
            shape, dtype, buffer=content, offset=first - low, strides=strides
        )
        return np.array(array)

    def _read_bytes(self, position: int, nbytes: int) -> bytes:
        # `nbytes` bytes of the decoded stream, from `position`.
        if nbytes <= 0:
            return b""
        if isinstance(self.source, Path):
            with self.source.open("rb") as f:
                f.seek(self.start)
                content = _read_at(f, position, nbytes)
        else:
            with preserve_position(self.source):
                self.source.seek(self.start)
                content = _read_at(self.source, position, nbytes)
        if len(content) < nbytes:
            raise ValueError(
                f"{_describe(self.source)} ends before the {nbytes} bytes "
                f"that are read from position {position}."
            )
        return content


class _RestridedProxy(StridedProxy):
    """A strided proxy that a reshape or a transpose of another one gives.

    It reads the bytes of `base`, the proxy that the reader built, with
    other strides, and its `root` is the root of `base`. A reshape or a
    transpose of it holds the same `base`, so these proxies do not
    nest.
    """

    base: tx.Any = None
    """The stored proxy that a reader built, whose bytes it reads."""

    @property
    def root(self) -> StoredProxy:
        """The root of `base`."""
        return self.base.root


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _contiguous_strides(
    shape: tx.Sequence[int], itemsize: int, order: str
) -> tx.List[int]:
    """Return the strides in bytes of a contiguous array in C or F order."""
    strides = [0] * len(shape)
    step = itemsize
    axes = range(len(shape) - 1, -1, -1) if order == "C" else range(len(shape))
    for axis in axes:
        strides[axis] = step
        step *= shape[axis]
    return strides


def _restride(
    shape: tx.Sequence[int],
    strides: tx.Sequence[int],
    new: tx.Sequence[int],
) -> tx.Optional[tx.List[int]]:
    """Return the strides of a reshape in C order that copies nothing.

    The strides are those that NumPy gives to the view that it returns
    when it reshapes an array of shape `shape` and strides `strides`
    without copying it. A reshape groups neighbouring axes that hold as
    many values on both sides, and a view exists when, within each group,
    the old axes are laid out one after the other, each stride being the
    next one times the next length.

    Returns
    -------
    list of int or None
        The new strides, or `None` when NumPy would have to copy the
        values. The arrays hold at least one value.
    """
    # Axes of length one have no layout, so they are left out.
    kept = [(n, s) for n, s in zip(shape, strides) if n != 1]
    old = [n for n, _ in kept]
    old_strides = [s for _, s in kept]
    result = [0] * len(new)
    i = j = 0
    while i < len(new) and j < len(old):
        a, b = i, j
        size_new, size_old = new[i], old[j]
        i += 1
        j += 1
        while size_new != size_old:
            if size_new < size_old:
                size_new *= new[i]
                i += 1
            else:
                size_old *= old[j]
                j += 1
        for k in range(b, j - 1):
            if old_strides[k] != old[k + 1] * old_strides[k + 1]:
                return None
        result[i - 1] = old_strides[j - 1]
        for k in range(i - 1, a, -1):
            result[k - 1] = result[k] * new[k]
    # The axes of length one that end the new shape take any stride.
    for k in range(i, len(new)):
        result[k] = result[i - 1] if i else 0
    return result


def _span(
    shape: tx.Sequence[int],
    strides: tx.Sequence[int],
    itemsize: int,
    first: int,
) -> tx.Tuple[int, int]:
    """
    Return the range of bytes that holds a box of values.

    The box has the shape `shape` and the strides `strides`, and its
    first value starts at the byte `first`. The range starts at the first
    byte of the box and stops past its last byte.
    """
    low = high = first
    for n, stride in zip(shape, strides):
        reach = (n - 1) * stride
        low += min(0, reach)
        high += max(0, reach)
    return low, high + itemsize


def _chunk_bytes() -> int:
    """Return the number of bytes of a block that Dask is configured for."""
    try:
        import dask
        from dask.utils import parse_bytes
    except ImportError:
        return _CHUNK_BYTES
    size = dask.config.get("array.chunk-size", _CHUNK_BYTES)
    return int(parse_bytes(size)) if isinstance(size, str) else int(size)


def _compressed(local: Path, start: int = 0) -> bool:
    # Whether the content of a local file from `start` is compressed,
    # judged by its magic bytes.
    with local.open("rb") as f:
        f.seek(start)
        stream = open_compressed(f)
        return stream is not f


def _read_at(file: tx.BinaryIO, offset: int, nbytes: int) -> bytes:
    # Bytes at an offset of a stream, after decompression. A stream that is
    # not compressed is sought, and a decompressed one is sought as far as
    # it can be, which decompresses up to the offset without keeping it.
    stream = open_compressed(file)
    stream.seek(stream.tell() + offset)
    return stream.read(nbytes)


def _describe(source: tx.Any) -> str:
    if isinstance(source, Path):
        return repr(str(source))
    return repr(source)
