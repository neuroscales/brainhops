"""Lazy arrays that read their values only when they are converted.

A proxy stands for an array whose values lie elsewhere, usually in a
file, and reads them only when NumPy converts it with `np.asarray`.
Indexing a proxy or transposing it returns another proxy, which reads
only the values that it selects. The design is described in the memo
`spec/design/array-proxies.md`.

[`ArrayProxyProtocol`][] states what a consumer of a proxy may rely on,
and [`ArrayProxy`][] is the base class of every proxy of brainhops. The
proxies fall in three groups:

- a [`StoredProxy`][] presents an array as its storage holds it, and
  reads it one box at a time. The proxy of each file format is one, and
  so is a [`PermutedProxy`][], which presents the axes of another
  stored proxy in another order;
- an [`IndexedProxy`][] presents a part of a stored proxy, which an
  index selects. Indexing a stored proxy returns one;
- a [`ScaledProxy`][] presents the values of another proxy multiplied
  by a factor and shifted by an offset, as many formats store
  integers that stand for real values.

Indexing and transposing keep these layers in a fixed order. A scaled
proxy holds an indexed proxy, or a stored one, and an indexed proxy
holds a stored proxy, so a chain of indexes never builds a chain of
proxies. [`stored_proxy`][] returns the stored proxy at the bottom.
"""

__all__ = [
    "ArrayProxyProtocol",
    "ArrayProxy",
    "StoredProxy",
    "PermutedProxy",
    "IndexedProxy",
    "NewAxis",
    "ScaledProxy",
    "stored_proxy",
    "normalize_index",
]

# stdlib
import abc
import math
import operator

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import HideIfDefault, Magic

# internals
from brainhops import backends
from brainhops._core.linalg import ArrayProtocol

_Box: tx.TypeAlias = tx.Tuple[tx.Union[int, slice], ...]
"""The box that [`StoredProxy._read`][brainhops._core.proxy.StoredProxy._read]
receives."""

_Item: tx.TypeAlias = tx.Union[None, int, slice, np.ndarray]
"""An item of an index that [`normalize_index`][] returns."""


@tx.runtime_checkable
class ArrayProxyProtocol(tx.Protocol):
    """What a consumer may rely on when it receives an array or a proxy.

    A NumPy array, a Dask array, a nibabel `ArrayProxy` and an
    [`ArrayProxy`][] all satisfy the protocol. Indexing a proxy returns
    another proxy without reading anything, whereas indexing a NumPy
    array or a nibabel proxy returns a NumPy array. A consumer that needs
    the values of an index must therefore call `np.asarray` on the
    result.
    """

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the array."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """The data type of the values."""
        ...

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        ...

    def __getitem__(self, index: tx.Any) -> "ArrayProxyProtocol": ...

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray: ...


# ----------------------------------------------------------------------
#   COMMON BASE
# ----------------------------------------------------------------------


class ArrayProxy(Magic, eq=False, convert=True, repr=HideIfDefault()):
    """A lazy array, which reads its values when it is converted.

    A proxy is the object that a reader keeps instead of the voxels of a
    file, so that the voxels are read only when they are needed, and
    only the part of them that is needed. Indexing the proxy, as in
    `proxy[index]`, and transposing it return another proxy without
    reading anything. The values are read when NumPy converts the proxy
    with `np.asarray`, or when Dask computes the array that
    [`to_dask`][brainhops._core.proxy.ArrayProxy.to_dask] returns.

    The class defines what every proxy has: its shape, the type of its
    values, the blocks in which it prefers to be read, and the
    conversions to NumPy, CuPy and Dask. A subclass defines `shape`,
    `dtype`, `__getitem__`, `transpose` and `__array__`. A file format
    derives its proxy from [`StoredProxy`][], which implements the last
    three from a method that reads a box.

    The class is a [`Magic`][bagof.magic.Magic] class without fields, so
    that a subclass declares its state as fields and has its constructor
    built from them. The fields of a subclass convert their arguments to
    their types, and its representation shows the fields that do not
    hold their defaults. The methods that a subclass defines are
    abstract, since the metaclass of `Magic` is that of an abstract base
    class, so a subclass that forgets one cannot be built. A proxy is
    compared by identity.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the array that the proxy presents."""

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """The data type of the values."""

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape that the proxy prefers, or `None`.

        The blocks are the units that the storage reads at once, such as
        the chunks of a Zarr array, the tiles of a slide or the slices
        of a stack. `None` means that the storage has no preference.
        """
        return None

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of values."""
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """The number of bytes that the values take once they are read."""
        return self.size * np.dtype(self.dtype).itemsize

    def __len__(self) -> int:
        shape = self.shape
        if not shape:
            raise TypeError("len() of a 0-dimensional array")
        return shape[0]

    @abc.abstractmethod
    def __getitem__(self, index: tx.Any) -> "ArrayProxy":
        """Return a proxy that selects a part of the values.

        A proxy is indexed lazily with integers, slices of any step, the
        ellipsis, `None`, and one array of integers along one axis, and
        the result is what NumPy gives for the same index. Indexing with
        several arrays, or with a boolean mask, would select points
        rather than a box, so it raises an `IndexError` that asks for
        `np.asarray(proxy)[index]` instead.

        Raises
        ------
        IndexError
            If an index is out of bounds, if there are more indices than
            dimensions, or if the index selects points rather than a box.
        TypeError
            If an element of the index has an unsupported type.
        """

    @abc.abstractmethod
    def transpose(self, *axes: tx.Any) -> "ArrayProxy":
        """Return a proxy with its axes in another order, as NumPy does.

        Without an argument, the order of the axes is reversed.

        Parameters
        ----------
        *axes : int or sequence of int
            The permutation, given as integers or as one sequence. Axis
            `i` of the result is axis `axes[i]` of the proxy.

        Returns
        -------
        ArrayProxy
            The transposed proxy.

        Raises
        ------
        ValueError
            If the axes are not a permutation of the axes of the proxy.
        """

    @property
    def T(self) -> "ArrayProxy":
        """The proxy with its axes in reverse order."""
        return self.transpose()

    @abc.abstractmethod
    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        """Read the values.

        Raises
        ------
        ValueError
            If `copy` is `False`, because reading values cannot be done
            without making them.
        """

    def to_numpy(self) -> np.ndarray:
        """Read the values of the proxy into a NumPy array.

        Returns
        -------
        np.ndarray
            The values.
        """
        return np.asarray(self)

    def to_cupy(self, device: backends.DeviceLike = None) -> ArrayProtocol:
        """Read the values of the proxy into a CuPy array on the GPU.

        Returns
        -------
        cupy.ndarray
            The values.

        Raises
        ------
        ImportError
            If CuPy is not installed.
        """
        return backends.to_cupy(self, device=device)

    def to_backend(
        self, backend: tx.Optional[backends.BackendLike] = None
    ) -> ArrayProtocol:
        """Convert the proxy as a backend setting says.

        The settings are described in [`brainhops.backends`][].
        Under `"native"`, and under `"auto"`, a proxy of at most the
        threshold of `"auto"` is read with NumPy and a larger one becomes
        a Dask array.
        A Dask array has the blocks that
        [`to_dask`][brainhops._core.proxy.ArrayProxy.to_dask] gives, and
        reading it reads only the blocks that it needs.

        Parameters
        ----------
        backend : str or module, optional
            The setting. By default, the current setting is used.

        Returns
        -------
        array
            A NumPy, CuPy or Dask array.

        Raises
        ------
        ImportError
            If a package that the setting needs is not installed.
        """
        return backends.to_backend(self, backend)

    def to_dask(
        self, *, device: backends.DeviceLike = "cpu", **kwargs
    ) -> ArrayProtocol:
        """Return a Dask array that reads the proxy block by block.

        The blocks are the `chunks` of the proxy when the storage
        prefers some, and the default blocks of Dask otherwise. Dask
        receives the type of the values through its `meta` argument, so
        it does not read the proxy to learn it. With a CUDA device, each
        block is moved to the GPU once it is read, which gives the
        setting `"dask+cupy"`.

        Parameters
        ----------
        device : str or int, default="cpu"
            The device for the chunks.
        **kwargs : dict
            Options passed to `dask.array.from_array`, which take
            precedence over the ones that this method sets.

        Returns
        -------
        dask.array.Array
            The lazy Dask array.

        Raises
        ------
        ImportError
            If Dask is not installed, or CuPy when a CUDA device is requested.
        """
        da = backends.da
        if da is None:
            raise ImportError("Dask is needed to read a proxy lazily.")
        options = {
            "chunks": self.chunks or "auto",
            "asarray": True,
            "fancy": False,
            "meta": np.empty((0,) * self.ndim, self.dtype),
        }
        options.update(kwargs)
        lazy = da.from_array(self, **options)
        return backends.to_dask(lazy, device=device)


# ----------------------------------------------------------------------
#   STORED PROXIES
# ----------------------------------------------------------------------


class StoredProxy(ArrayProxy):
    """A lazy array that presents an array as its storage holds it.

    The proxy of a file format derives from this class. It describes the
    stored array with two properties, `shape` and `dtype`, and reads a
    box of it with one method, `_read(box)`. It may also define
    `chunks`, the block shape that the storage prefers. The base class
    provides the rest.

    Indexing a stored proxy returns an [`IndexedProxy`][], which holds
    the stored proxy and the index. A read of the indexed proxy calls
    `_read` with the smallest box with a positive step that holds the
    selected values, and then applies in memory the negative steps, the
    arrays of positions and the new axes, so that `_read` never sees
    them. An index that selects every value in order returns the stored
    proxy itself.

    Transposing a stored proxy returns a [`PermutedProxy`][], which is
    itself a stored proxy. A subclass that can present its axes in
    another order without moving any value, as
    [`StridedProxy`][brainhops._core.strided.StridedProxy] does by
    permuting its strides, overrides `transpose` to do so.

    A stored proxy that a reader built is the one that every indexed
    proxy reads through, so a handle that it opens on its first read,
    such as an open store, is opened once and shared by all of them.
    """

    @abc.abstractmethod
    def _read(self, box: _Box) -> np.ndarray:
        """Read a box of the stored array.

        The box has one entry for each axis of the stored array. An
        entry is either an integer, which is a valid position and drops
        the axis, or a slice whose start, stop and step are integers,
        with `0 <= start < stop <= length` and `step >= 1`. The box is
        never empty. Only [`IndexedProxy`][] calls this method.

        Parameters
        ----------
        box : tuple of int or slice
            The box to read.

        Returns
        -------
        np.ndarray
            The values of the box, with the axes that slices keep, in the
            order of the stored array.
        """

    def __getitem__(self, index: tx.Any) -> ArrayProxy:
        """Return a proxy that selects a part of the values.

        The result is an [`IndexedProxy`][] of this proxy, or this proxy
        itself when the index selects every value in order.

        Raises
        ------
        IndexError
            If an index is out of bounds, if there are more indices than
            dimensions, or if the index selects points rather than a box.
        TypeError
            If an element of the index has an unsupported type.
        """
        return IndexedProxy(self, _whole(self.shape))[index]

    def transpose(self, *axes: tx.Any) -> "StoredProxy":
        """Return the proxy with its axes in another order.

        The result is a [`PermutedProxy`][] of this proxy, or this proxy
        itself when the permutation changes nothing.

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
        perm = _permutation(axes, self.ndim)
        if perm == tuple(range(self.ndim)):
            return self
        return PermutedProxy(self, perm)

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        """Read all the values.

        Raises
        ------
        ValueError
            If `copy` is `False`, because reading values cannot be done
            without making them.
        """
        whole = IndexedProxy(self, _whole(self.shape))
        return whole.__array__(dtype=dtype, copy=copy)


class PermutedProxy(StoredProxy):
    """A stored proxy that presents the axes of another one in another order.

    Transposing a [`StoredProxy`][] returns a permuted proxy, so that an
    [`IndexedProxy`][] never has to reorder axes itself: transposing an
    indexed proxy transposes the stored proxy that it holds. Axis `i` of
    the permuted proxy is axis `perm[i]` of `inner`.

    Transposing a permuted proxy composes the two permutations and
    transposes `inner` once, so permuted proxies never nest. A read
    reorders the box for `inner`, reads it, and transposes the values.
    """

    inner: tx.Annotated[
        tx.Any, tx.Doc("The stored proxy whose axes are permuted.")
    ]
    perm: tx.Annotated[
        tx.Tuple[int, ...],
        tx.Doc("The axis of `inner` that each axis of the proxy is."),
    ]

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of `inner`, in the order of the permutation."""
        return tuple(self.inner.shape[p] for p in self.perm)

    @property
    def dtype(self) -> np.dtype:
        """The data type of the values of `inner`."""
        return self.inner.dtype

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape of `inner`, in the order of the permutation."""
        chunks = self.inner.chunks
        if chunks is None:
            return None
        return tuple(chunks[p] for p in self.perm)

    def transpose(self, *axes: tx.Any) -> StoredProxy:
        """Return `inner` transposed by the two permutations, composed.

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
        perm = _permutation(axes, self.ndim)
        return self.inner.transpose([self.perm[p] for p in perm])

    def _read(self, box: _Box) -> np.ndarray:
        inner_box = [None] * len(box)
        for axis, p in enumerate(self.perm):
            inner_box[p] = box[axis]
        values = np.asarray(self.inner[tuple(inner_box)])
        # The values hold the axes that slices keep, in the order of
        # `inner`, and they are put in the order of the permutation.
        kept = [
            p for p, item in enumerate(inner_box) if isinstance(item, slice)
        ]
        order = [kept.index(p) for p in self.perm if p in kept]
        return values.transpose(order)


def _whole(shape: tx.Sequence[int]) -> tx.Tuple[slice, ...]:
    """Return the selection of every value of a stored array, in order."""
    return tuple(slice(0, int(n), 1) for n in shape)


# ----------------------------------------------------------------------
#   INDEXED PROXIES
# ----------------------------------------------------------------------


class NewAxis(tx.NamedTuple):
    """An axis that an [`IndexedProxy`][] presents but that is not stored.

    `None` in an index adds an axis of length one, which holds a single
    value. A later index may keep that value, remove it, or repeat it,
    so the entry keeps the positions of the value that are presented:
    a slice of `range(1)` keeps the value or removes it, and an array
    of zeros repeats it.
    """

    kept: tx.Annotated[
        tx.Union[slice, np.ndarray],
        tx.Doc(
            "The positions of the single value that the axis presents: a "
            "slice of `range(1)` or an array of zeros."
        ),
    ] = slice(0, 1, 1)


_Entry: tx.TypeAlias = tx.Union[int, slice, np.ndarray, NewAxis]
"""An entry of the selection of an [`IndexedProxy`][]."""


class IndexedProxy(ArrayProxy):
    """A lazy array that presents a part of a stored proxy.

    Indexing a [`StoredProxy`][] returns an indexed proxy, which holds
    the stored proxy as `base` and the index as `selection`. The
    selection is an index in the form that NumPy accepts, normalized so
    that it has one entry for each axis of `base`, in order:

    - an integer is a position, and the axis is dropped;
    - a slice has an integer start and step, and its stop is an integer
      or `None`. It keeps the positions that NumPy keeps for it;
    - an array of integers keeps those positions, in that order.

    A [`NewAxis`][] entry is inserted wherever the proxy presents an
    axis that `base` does not store, as `None` does in an index.

    Indexing an indexed proxy composes the index with the selection, and
    transposing it transposes `base` and moves the entries of the
    selection with their axes. The result holds the same stored proxy,
    or a [`PermutedProxy`][] of it, so a chain of indexes never builds a
    chain of proxies, and every read goes through the stored proxy that
    a reader built. An index that selects every value of `base` in order
    returns `base` itself.

    A read calls the `_read` method of `base` with the smallest box with
    a positive step that holds the selected values. The negative steps,
    the arrays of positions and the new axes are then applied in memory.
    """

    base: tx.Annotated[
        tx.Any, tx.Doc("The stored proxy whose values are selected.")
    ]
    selection: tx.Annotated[
        tx.Tuple[tx.Any, ...],
        tx.Doc(
            "One entry for each axis of `base`, in order, with a `NewAxis` "
            "entry for each axis that is not stored."
        ),
    ]

    def _entries(self) -> tx.Iterator[tx.Tuple[tx.Any, int, bool]]:
        """Yield each entry, the length that it indexes, and whether it is new.

        The entry of a new axis is replaced by the positions that it
        keeps, which index an axis of length one.
        """
        stored = iter(self.base.shape)
        for entry in self.selection:
            if isinstance(entry, NewAxis):
                yield entry.kept, 1, True
            else:
                yield entry, int(next(stored)), False

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the selected values."""
        return tuple(
            _length(entry, n)
            for entry, n, _ in self._entries()
            if not isinstance(entry, int)
        )

    @property
    def dtype(self) -> np.dtype:
        """The data type of the values of `base`."""
        return self.base.dtype

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape that the proxy prefers, or `None`.

        The blocks of `base` are carried through the selection. Along an
        axis that a step subsamples, a block holds as many values as the
        stored block covers, and a new axis has blocks of one value.
        """
        base = self.base.chunks
        if base is None:
            return None
        stored = iter(base)
        chunks = []
        for entry, n, new in self._entries():
            if new:
                chunks.append(1)
                continue
            chunk = int(next(stored))
            if isinstance(entry, int):
                continue
            if isinstance(entry, slice):
                chunk = max(1, chunk // abs(entry.step))
            chunks.append(max(1, min(chunk, _length(entry, n))))
        return tuple(chunks)

    def __getitem__(self, index: tx.Any) -> ArrayProxy:
        """Return a proxy that selects a part of the values.

        Raises
        ------
        IndexError
            If an index is out of bounds, if there are more indices than
            dimensions, or if the index selects points rather than a box.
        TypeError
            If an element of the index has an unsupported type.
        """
        items, front = normalize_index(index, self.shape)
        remaining = iter(items)
        selection = []
        for entry, n, new in self._entries():
            if isinstance(entry, int):
                selection.append(entry)
                continue
            item = next(remaining)
            while item is None:
                selection.append(NewAxis())
                item = next(remaining)
            composed = _compose(entry, n, item)
            if not isinstance(composed, int):
                selection.append(NewAxis(composed) if new else composed)
            elif not new:
                selection.append(composed)
        # The items that remain are the new axes after the last one.
        selection.extend(NewAxis() for _ in remaining)
        result = _indexed(self.base, tuple(selection))
        if front is not None:
            # NumPy moves the axis of the array to the front when the
            # array and the integers of the index are not next to each
            # other, and the proxy does the same.
            order = [front] + [i for i in range(result.ndim) if i != front]
            result = result.transpose(order)
        return result

    def transpose(self, *axes: tx.Any) -> ArrayProxy:
        """Return a proxy with its axes in another order, as NumPy does.

        The entries of the selection move with their axes, and the
        stored axes of `base` are transposed to follow them, so that the
        selection stays in the order of the axes of the stored proxy.

        Parameters
        ----------
        *axes : int or sequence of int
            The permutation, given as integers or as one sequence. Axis
            `i` of the result is axis `axes[i]` of the proxy.

        Returns
        -------
        ArrayProxy
            The transposed proxy.

        Raises
        ------
        ValueError
            If the axes are not a permutation of the axes of the proxy.
        """
        perm = _permutation(axes, self.ndim)
        # Each entry with the stored axis that it indexes, or None for a
        # new axis.
        tagged = []
        axis = 0
        for entry in self.selection:
            if isinstance(entry, NewAxis):
                tagged.append((entry, None))
            else:
                tagged.append((entry, axis))
                axis += 1
        # The presented entries move between the places of the presented
        # entries, and the entries that drop an axis keep their places.
        places = [
            i
            for i, entry in enumerate(self.selection)
            if not isinstance(entry, int)
        ]
        moved = list(tagged)
        for k, place in enumerate(places):
            moved[place] = tagged[places[perm[k]]]
        stored = [axis for _, axis in moved if axis is not None]
        base = self.base.transpose(stored)
        return _indexed(base, tuple(entry for entry, _ in moved))

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        """Read the selected values.

        Raises
        ------
        ValueError
            If `copy` is `False`, because reading values cannot be done
            without making them, or if the `_read` method of `base`
            returns an array of the wrong shape.
        """
        _refuse_no_copy(copy)
        return _convert(self._values(), dtype, copy)

    def _values(self) -> np.ndarray:
        """Read the selected values, in the order of the proxy."""
        shape = self.shape
        if 0 in shape:
            return np.empty(shape, self.dtype)
        box = []
        after = []
        for entry, n, new in self._entries():
            if new:
                continue
            if isinstance(entry, int):
                box.append(entry)
            elif isinstance(entry, slice):
                kept = range(*entry.indices(n))
                box.append(slice(min(kept), max(kept) + 1, abs(kept.step)))
                after.append(slice(None, None, -1 if kept.step < 0 else 1))
            else:
                first = int(entry.min())
                box.append(slice(first, int(entry.max()) + 1, 1))
                after.append(entry - first)
        values = np.asanyarray(self.base._read(tuple(box)))
        expected = tuple(
            len(range(item.start, item.stop, item.step))
            for item in box
            if isinstance(item, slice)
        )
        if values.shape != expected:
            raise ValueError(
                f"{type(self.base).__name__}._read returned an array of "
                f"shape {values.shape} for a box of shape {expected}."
            )
        # The negative steps and the arrays are applied one axis at a
        # time, as NumPy applies successive indexes.
        for axis, item in enumerate(after):
            if isinstance(item, slice) and item.step == 1:
                continue
            values = values[(slice(None),) * axis + (item,)]
        axis = 0
        for entry in self.selection:
            if isinstance(entry, int):
                continue
            if isinstance(entry, NewAxis):
                values = np.expand_dims(values, axis)
                if isinstance(entry.kept, np.ndarray):
                    values = values[(slice(None),) * axis + (entry.kept,)]
            axis += 1
        return values


def _length(entry: tx.Union[slice, np.ndarray], n: int) -> int:
    """Return the number of positions that an entry keeps of an axis."""
    if isinstance(entry, slice):
        return len(range(*entry.indices(n)))
    return len(entry)


def _compose(
    entry: tx.Union[slice, np.ndarray], n: int, item: _Item
) -> tx.Union[int, slice, np.ndarray]:
    """Return the entry of an axis of length `n` after an item indexes it."""
    if isinstance(entry, slice):
        entry = range(*entry.indices(n))
    if isinstance(item, np.ndarray):
        if isinstance(entry, range):
            entry = np.arange(entry.start, entry.stop, entry.step)
        return entry[item]
    if isinstance(item, slice):
        if isinstance(entry, range):
            return _as_slice(entry[item])
        return entry[item]
    return int(entry[item])


def _as_slice(kept: range) -> slice:
    """Return the slice that keeps the positions of a range.

    A range that runs backwards to the position 0 stops at -1, which a
    slice reads as the last position, so the stop of the slice is then
    `None`.
    """
    if not kept:
        return slice(0, 0, 1)
    stop = kept.stop if kept.stop >= 0 else None
    return slice(kept.start, stop, kept.step)


def _indexed(
    base: StoredProxy, selection: tx.Tuple[_Entry, ...]
) -> ArrayProxy:
    """Return the proxy of a selection, which is `base` if it is whole."""
    whole = len(selection) == len(base.shape) and all(
        isinstance(entry, slice)
        and range(*entry.indices(int(n))) == range(int(n))
        for entry, n in zip(selection, base.shape)
    )
    if whole:
        return base
    return IndexedProxy(base, selection)


# ----------------------------------------------------------------------
#   SCALED PROXIES
# ----------------------------------------------------------------------


class ScaledProxy(ArrayProxy):
    """A lazy array of the values of another proxy, scaled.

    Many formats store integers that stand for real values, together
    with a factor and an offset: the value is `slope * stored + inter`.
    The factor and the offset may be single numbers, or arrays that
    broadcast against the stored array, such as one factor for each
    volume. A scaled proxy applies them to the values of `inner`, which
    reads the stored integers.

    Indexing or transposing the proxy does the same to `inner` and to
    the factors, so that only the selected values are read and scaled.
    An axis of length one of a factor stays of length one, so the
    factors never grow to the shape of the values. The values are
    floating-point numbers of at least single precision.
    """

    inner: tx.Annotated[
        tx.Any, tx.Doc("The proxy of the values before scaling.")
    ]
    slope: tx.Annotated[
        tx.Optional[np.ndarray],
        tx.Doc(
            "The factor, a number or an array that broadcasts against the "
            "shape of `inner`, or `None`."
        ),
    ] = None
    inter: tx.Annotated[
        tx.Optional[np.ndarray],
        tx.Doc(
            "The offset, a number or an array that broadcasts against the "
            "shape of `inner`, or `None`."
        ),
    ] = None

    def __post_init__(self) -> None:
        # A factor is given one axis for each axis of `inner`, so that an
        # index applies to it axis by axis.
        ndim = len(self.inner.shape)
        for name in ("slope", "inter"):
            factor = getattr(self, name)
            if factor is None:
                continue
            if factor.ndim > ndim:
                raise ValueError(
                    f"The {name} has {factor.ndim} dimensions, more than "
                    f"the {ndim} of the values."
                )
            setattr(
                self,
                name,
                factor.reshape((1,) * (ndim - factor.ndim) + factor.shape),
            )

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of `inner`."""
        return tuple(self.inner.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the scaled values.

        An integer becomes a float32, and a float32 stays one, whatever
        the type of the factors.
        """
        return np.result_type(self.inner.dtype, np.float32)

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape of `inner`."""
        return self.inner.chunks

    def __getitem__(self, index: tx.Any) -> "ScaledProxy":
        """Return the scaled proxy of a part of the values.

        Raises
        ------
        IndexError
            If an index is out of bounds, if there are more indices than
            dimensions, or if the index selects points rather than a box.
        TypeError
            If an element of the index has an unsupported type.
        """
        items, front = normalize_index(index, self.shape)
        return ScaledProxy(
            self.inner[index],
            _index_factor(self.slope, items, self.shape, front),
            _index_factor(self.inter, items, self.shape, front),
        )

    def transpose(self, *axes: tx.Any) -> "ScaledProxy":
        """Return the scaled proxy of the transposed values.

        Parameters
        ----------
        *axes : int or sequence of int
            The permutation, given as integers or as one sequence. Axis
            `i` of the result is axis `axes[i]` of the proxy.

        Returns
        -------
        ScaledProxy
            The transposed proxy.

        Raises
        ------
        ValueError
            If the axes are not a permutation of the axes of the proxy.
        """
        perm = _permutation(axes, self.ndim)
        return ScaledProxy(
            self.inner.transpose(perm),
            None if self.slope is None else self.slope.transpose(perm),
            None if self.inter is None else self.inter.transpose(perm),
        )

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        """Read the values of `inner` and scale them.

        Raises
        ------
        ValueError
            If `copy` is `False`, because reading values cannot be done
            without making them.
        """
        _refuse_no_copy(copy)
        values = np.asarray(self.inner, dtype=self.dtype)
        if self.slope is not None:
            values = values * self.slope
        if self.inter is not None:
            values = values + self.inter
        # The product of arrays of no dimension is a NumPy scalar.
        values = np.asarray(values).astype(self.dtype, copy=False)
        return _convert(values, dtype, copy)


def _index_factor(
    factor: tx.Optional[np.ndarray],
    items: tx.Sequence[_Item],
    shape: tx.Sequence[int],
    front: tx.Optional[int],
) -> tx.Optional[np.ndarray]:
    """Index a factor as [`normalize_index`][] says the values are indexed.

    An axis of length one of the factor stays of length one: an integer
    selects its single value, a slice keeps it unless the slice selects
    nothing, and an array repeats it as many times as the array has
    positions. The array is applied after the other items, so that
    NumPy moves no axis, and the axis that the index moves to the front
    is then moved in the same way.
    """
    if factor is None:
        return None
    index = []
    array = None
    axis = 0
    for item in items:
        if item is None:
            index.append(None)
            continue
        broadcast = factor.shape[axis] == 1
        if isinstance(item, int):
            index.append(0 if broadcast else item)
        elif isinstance(item, np.ndarray):
            place = sum(1 for x in index if not isinstance(x, int))
            index.append(slice(None))
            array = (
                place,
                np.zeros(len(item), np.intp) if broadcast else item,
            )
        elif broadcast:
            kept = _length(item, int(shape[axis]))
            index.append(slice(0, 1 if kept else 0))
        else:
            index.append(item)
        axis += 1
    factor = factor[tuple(index)]
    if array is not None:
        place, positions = array
        factor = factor[(slice(None),) * place + (positions,)]
    if front is not None:
        factor = np.moveaxis(factor, front, 0)
    return factor


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def stored_proxy(proxy: tx.Any) -> tx.Optional[StoredProxy]:
    """Return the stored proxy that a proxy reads, or `None`.

    A writer uses this function to tell whether an array reads a file
    that it is about to write, whatever index, permutation or scaling
    the array applies. The function looks through a [`ScaledProxy`][], an
    [`IndexedProxy`][] and a [`PermutedProxy`][], and returns the stored
    proxy that a reader built.

    Parameters
    ----------
    proxy : array
        A proxy, or any other array.

    Returns
    -------
    StoredProxy or None
        The stored proxy at the bottom of `proxy`, or `None` if `proxy`
        is not a proxy of brainhops.
    """
    while True:
        if isinstance(proxy, ScaledProxy):
            proxy = proxy.inner
        elif isinstance(proxy, IndexedProxy):
            proxy = proxy.base
        elif isinstance(proxy, PermutedProxy):
            proxy = proxy.inner
        elif isinstance(proxy, StoredProxy):
            return proxy
        else:
            return None


def normalize_index(
    index: tx.Any, shape: tx.Sequence[int]
) -> tx.Tuple[tx.List[_Item], tx.Optional[int]]:
    """Return an index with one item for each axis, and the axis it moves.

    The index is one that a proxy accepts: integers, slices, the
    ellipsis, `None`, and at most one array of integers. Each item of
    the result is `None` for a new axis, or an integer, a slice or a
    one-dimensional array of integers for an axis of `shape`. The
    ellipsis is replaced by full slices, missing trailing items are
    appended, and the integers are checked against the shape.

    NumPy moves the axis of an array to the front of the result when the
    array and the integers of the index are not next to each other, as
    in `array[0, :, [1, 2]]`. NumPy decides this on the index as it is
    written, so an ellipsis separates them even when it stands for no
    axis. The second value returned is the axis of the result that is
    moved to the front, or `None`.

    Parameters
    ----------
    index : Any
        The index.
    shape : sequence of int
        The shape of the indexed array.

    Returns
    -------
    items : list of None, int, slice or np.ndarray
        One item for each axis of `shape`, and `None` for each new axis.
    front : int or None
        The axis of the result that NumPy moves to the front, or `None`.

    Raises
    ------
    IndexError
        If an index is out of bounds, if there are more indices than
        dimensions, more than one ellipsis, or an index that selects
        points rather than a box.
    TypeError
        If an element of the index has an unsupported type.
    """
    ndim = len(shape)
    if not isinstance(index, tuple):
        index = (index,)
    written = []
    ellipsis = None
    arrays = 0
    for item in index:
        if item is Ellipsis:
            if ellipsis is not None:
                raise IndexError("An index can only have one ellipsis.")
            ellipsis = len(written)
            written.append(item)
            continue
        if item is None or isinstance(item, slice):
            written.append(item)
            continue
        if isinstance(item, (bool, np.bool_)):
            raise IndexError(_POINTS)
        if isinstance(item, (int, np.integer)):
            written.append(operator.index(item))
            continue
        if isinstance(item, (list, tuple, np.ndarray)) or hasattr(
            item, "__array__"
        ):
            array = np.asarray(item)
            if array.ndim == 0 and array.dtype.kind in "iu":
                written.append(operator.index(array.item()))
                continue
            if array.dtype.kind == "b" or array.ndim != 1:
                raise IndexError(_POINTS)
            if array.size == 0:
                array = array.astype(np.intp)
            if array.dtype.kind not in "iu":
                raise TypeError(
                    f"An array index must hold integers, not {array.dtype}."
                )
            arrays += 1
            written.append(array.astype(np.intp, copy=False))
            continue
        raise TypeError(
            "A proxy is indexed with integers, slices, the ellipsis, None "
            f"and arrays of integers, not {type(item).__name__}."
        )
    if arrays > 1:
        raise IndexError(_POINTS)
    used = sum(
        1 for item in written if item is not None and item is not Ellipsis
    )
    if used > ndim:
        raise IndexError(
            f"Too many indices: the array has {ndim} dimensions, but "
            f"{used} were indexed."
        )
    fill = [slice(None)] * (ndim - used)
    if ellipsis is None:
        items = written + fill
    else:
        items = written[:ellipsis] + fill + written[ellipsis + 1 :]
    axis = 0
    for item in items:
        if item is None:
            continue
        n = int(shape[axis])
        axis += 1
        if isinstance(item, int) and not -n <= item < n:
            raise IndexError(
                f"Index {item} is out of bounds for an axis of length {n}."
            )
        if (
            isinstance(item, np.ndarray)
            and item.size
            and not (-n <= int(item.min()) and int(item.max()) < n)
        ):
            raise IndexError(
                f"An index of {item.tolist()} is out of bounds for an axis "
                f"of length {n}."
            )
    return items, _front(written, items)


def _front(
    written: tx.Sequence[tx.Any], items: tx.Sequence[_Item]
) -> tx.Optional[int]:
    """Return the axis of the result that NumPy moves to the front, or None.

    The array and the integers of an index are next to each other when
    nothing else stands between them in the index as written.
    """
    advanced = [
        i
        for i, item in enumerate(written)
        if isinstance(item, (int, np.ndarray))
    ]
    if not any(isinstance(written[i], np.ndarray) for i in advanced):
        return None
    if advanced[-1] - advanced[0] + 1 == len(advanced):
        return None
    (array,) = [
        i for i, item in enumerate(items) if isinstance(item, np.ndarray)
    ]
    return sum(1 for item in items[:array] if not isinstance(item, int))


def _permutation(axes: tx.Tuple[tx.Any, ...], ndim: int) -> tx.Tuple[int, ...]:
    """Return the permutation that the arguments of `transpose` give.

    Raises
    ------
    ValueError
        If the axes are not a permutation of `ndim` axes.
    """
    if len(axes) == 1 and not isinstance(axes[0], (int, np.integer)):
        axes = tuple(axes[0])
    if not axes:
        axes = tuple(reversed(range(ndim)))
    perm = []
    for axis in axes:
        axis = operator.index(axis)
        if not -ndim <= axis < ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for {ndim} dimensions."
            )
        perm.append(axis % ndim)
    if sorted(perm) != list(range(ndim)):
        raise ValueError(f"The axes {axes} are not a permutation.")
    return tuple(perm)


def _refuse_no_copy(copy: tx.Optional[bool]) -> None:
    """Refuse `copy=False`, since reading values makes them.

    Raises
    ------
    ValueError
        If `copy` is `False`.
    """
    if copy is False:
        raise ValueError(
            "A proxy cannot be converted to an array without reading "
            "its values, which copy=False forbids."
        )


def _convert(
    values: np.ndarray, dtype: tx.Any, copy: tx.Optional[bool]
) -> np.ndarray:
    """Convert values that were read to the type and copy that are asked."""
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    if copy:
        values = np.array(values, copy=True)
    return values


_POINTS = (
    "A proxy is indexed lazily with integers, slices, the ellipsis, None "
    "and one array of integers. Several arrays and boolean masks select "
    "points rather than a box: use np.asarray(proxy)[index] instead."
)
