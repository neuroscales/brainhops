"""Lazy arrays that read their values only when they are converted.

A proxy stands for an array whose values lie elsewhere, usually in a
file, and reads them only when NumPy converts it with `np.asarray`.
Indexing a proxy or transposing it returns another proxy, which reads
only the values that it selects. The design is described in the memo
`spec/design/array-proxies.md`.

[`ArrayProxyProtocol`][] states what a consumer of a proxy may rely on,
and [`ArrayProxy`][] is the base class that implements a proxy from a
single method that reads a box of the stored array.
"""

__all__ = ["ArrayProxyProtocol", "ArrayProxy"]

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

_Selection: tx.TypeAlias = tx.Union[int, range, np.ndarray]
"""What a view selects along one axis of the stored array: a position,
which drops the axis, or the ordered positions that the axis keeps."""

_ReadIndex: tx.TypeAlias = tx.Tuple[tx.Union[int, slice], ...]
"""The index that [`ArrayProxy._read`][] receives."""


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


class ArrayProxy(Magic, eq=False, convert=True, repr=HideIfDefault()):
    """A lazy array, which reads its values when it is converted.

    A proxy is the object that a reader keeps instead of the voxels of a
    file, so that the voxels are read only when they are needed, and
    only the part of them that is needed. Indexing the proxy, as in
    `proxy[index]`, and transposing it return another proxy, called a
    view, which shares the state of its parent and selects a part of its
    values. Nothing is read until NumPy converts the view with
    `np.asarray`, or until Dask computes the array that
    [`to_dask`][brainhops._core.proxy.ArrayProxy.to_dask]
    returns.

    A subclass describes the stored array and knows how to read a box of
    it. It defines three members: `base_shape`, the shape of the stored
    array, `dtype`, the type of its values, and `_read(index)`, which
    reads the values of a box. A subclass may also define `base_chunks`,
    the shape of the blocks in which the storage prefers to be read.

    A view keeps, for each axis of the stored array, either the position
    that the view selects, which drops the axis, or the ordered positions
    that the view keeps. It also keeps the order of its axes, which may
    include new axes of length one. To read a view, the base class reads
    the smallest box with a positive step that holds every selected
    position, and then applies in memory the negative steps, the arrays
    of positions, the order of the axes and the new axes. A subclass
    therefore never sees them.

    A proxy indexes lazily with integers, slices of any step, the
    ellipsis, `None`, and one array of integers along one axis. Indexing
    with several arrays, or with a boolean mask, would select points
    rather than a box, so it raises an `IndexError` that asks for
    `np.asarray(proxy)[index]` instead. Indexing a view again with an
    array along another axis is allowed, and it selects the positions of
    each axis independently, as NumPy does for successive indexes.

    The class is a [`Magic`][bagof.magic.Magic] class without fields, so
    that a subclass may declare its state as fields and have its
    constructor built from them, as
    [`StridedProxy`][brainhops._core.strided.StridedProxy] does. A
    subclass may also write its own `__init__`. The members that a
    subclass defines are abstract, since the metaclass of `Magic` is that
    of an abstract base class, so a subclass that forgets one cannot be
    built. A proxy is compared by identity. The fields of a subclass
    convert their arguments to their types, and its representation shows
    the fields that do not hold their defaults.
    """

    # ------------------------------------------------------------------
    #   WHAT A SUBCLASS DEFINES
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the stored array, before any index."""

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """The data type of the values."""

    @abc.abstractmethod
    def _read(self, index: _ReadIndex) -> np.ndarray:
        """Read a box of the stored array.

        The index has one entry for each axis of the stored array. An
        entry is either an integer, which is a valid position and drops
        the axis, or a slice whose start, stop and step are integers,
        with `0 <= start < stop <= length` and `step >= 1`. The box is
        never empty.

        Parameters
        ----------
        index : tuple of int or slice
            The box to read.

        Returns
        -------
        np.ndarray
            The values of the box, with the axes that slices keep, in the
            order of the stored array.
        """

    @property
    def base_chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape that the stored array prefers, or `None`.

        The blocks are the chunks of a Zarr array, the tiles of a slide
        or the slices of a stack, which are the units that the storage
        reads at once. `None` means that the storage has no preference.
        """
        return None

    # ------------------------------------------------------------------
    #   STATE OF A VIEW
    # ------------------------------------------------------------------

    # A proxy that was never indexed selects everything, so these class
    # attributes stand in for the state that `__getitem__` and
    # `transpose` give a view. A subclass need not set them, and they are
    # class variables so that they are not fields of the constructor.
    _root: tx.ClassVar[tx.Optional["ArrayProxy"]] = None
    _selection: tx.ClassVar[tx.Optional[tx.Tuple[_Selection, ...]]] = None
    _axes: tx.ClassVar[
        tx.Optional[tx.Tuple[tx.Union[int, _Selection], ...]]
    ] = None

    def _view_state(
        self,
    ) -> tx.Tuple[
        tx.Tuple[_Selection, ...], tx.Tuple[tx.Union[int, _Selection], ...]
    ]:
        """Return the selection of each stored axis and the axes of the view.

        An entry of the axes is either the number of a stored axis, or
        the positions that a new axis keeps of its single value: a range
        of length zero or one, or an array of zeros.
        """
        selection = self._selection
        if selection is None:
            selection = tuple(range(int(n)) for n in self.base_shape)
        axes = self._axes
        if axes is None:
            axes = tuple(range(len(selection)))
        return selection, axes

    def _new_view(
        self,
        selection: tx.Tuple[_Selection, ...],
        axes: tx.Tuple[tx.Union[int, _Selection], ...],
    ) -> tx.Self:
        """Return a view that reads through the proxy that a reader built.

        The view copies the attributes of this proxy, so that it answers
        the same questions, but it reads its values through the original
        proxy, which it keeps as `_root`. A handle that the original
        proxy opens, such as an open store, is therefore opened once and
        shared by every view. The view is built without `copy.copy`,
        which would call the `__getstate__` of a subclass.
        """
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        view._root = self._root if self._root is not None else self
        view._selection = selection
        view._axes = axes
        return view

    @property
    def is_view(self) -> bool:
        """Whether the proxy presents something else than the stored array.

        A proxy that a reader built presents the stored array, and so
        does a view whose index and axis order change nothing. Any other
        view selects a part of the stored array, or presents it in
        another order or with new axes. A writer that copies the stored
        bytes of a proxy as they are must check that the proxy is not a
        view, because the bytes would hold the whole stored array.
        """
        if self._selection is None and self._axes is None:
            return False
        selection, axes = self._view_state()
        if axes != tuple(range(len(selection))):
            return True
        return any(
            not isinstance(kept, range) or kept != range(int(n))
            for kept, n in zip(selection, self.base_shape)
        )

    # ------------------------------------------------------------------
    #   ARRAY ATTRIBUTES
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the array that the proxy presents."""
        selection, axes = self._view_state()
        return tuple(
            len(selection[axis]) if isinstance(axis, int) else len(axis)
            for axis in axes
        )

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

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape that the proxy prefers, or `None`.

        The blocks of the stored array, given by `base_chunks`, are
        carried through the index of the view. Along an axis that a step
        subsamples, a block holds as many values as the stored block
        covers, and a new axis has blocks of one value.
        """
        base = self.base_chunks
        if base is None:
            return None
        selection, axes = self._view_state()
        chunks = []
        for axis in axes:
            if not isinstance(axis, int):
                chunks.append(1)
                continue
            kept = selection[axis]
            chunk = int(base[axis])
            if isinstance(kept, range):
                chunk = max(1, chunk // abs(kept.step))
            chunks.append(max(1, min(chunk, len(kept))))
        return tuple(chunks)

    # ------------------------------------------------------------------
    #   LAZY INDEXING
    # ------------------------------------------------------------------

    def __getitem__(self, index: tx.Any) -> tx.Self:
        """Return a view that selects a part of the values.

        Raises
        ------
        IndexError
            If an index is out of bounds, if there are more indices than
            dimensions, or if the index selects points rather than a box
            (several arrays, a boolean mask, or an array of more than one
            dimension).
        TypeError
            If an element of the index has an unsupported type.
        """
        selection, axes = self._view_state()
        items = _expand(index, len(axes))
        selection = list(selection)
        new_axes = []
        position = 0
        for item in items:
            if item is None:
                new_axes.append(range(1))
                continue
            axis = axes[position]
            position += 1
            if not isinstance(axis, int):
                # A new axis repeats its single value, so it keeps the
                # positions of that value that the index selects.
                if isinstance(item, np.ndarray):
                    new_axes.append(np.asarray(axis, dtype=np.intp)[item])
                    continue
                kept = axis[item]
                if not isinstance(kept, (int, np.integer)):
                    new_axes.append(kept)
                continue
            kept = selection[axis]
            if isinstance(item, np.ndarray):
                if isinstance(kept, range):
                    kept = np.arange(kept.start, kept.stop, kept.step)
                selection[axis] = kept[item]
                new_axes.append(axis)
            elif isinstance(item, slice):
                selection[axis] = kept[item]
                new_axes.append(axis)
            else:
                selection[axis] = int(kept[item])
        return self._new_view(tuple(selection), tuple(new_axes))

    def transpose(self, *axes: tx.Any) -> tx.Self:
        """Return a view with its axes in another order, as NumPy does.

        Without an argument, the order of the axes is reversed.

        Parameters
        ----------
        *axes : int or sequence of int
            The permutation, given as integers or as one sequence. Axis
            `i` of the result is axis `axes[i]` of the proxy.

        Returns
        -------
        ArrayProxy
            The transposed view.

        Raises
        ------
        ValueError
            If the axes are not a permutation of the axes of the proxy.
        """
        selection, current = self._view_state()
        ndim = len(current)
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
        return self._new_view(selection, tuple(current[p] for p in perm))

    @property
    def T(self) -> tx.Self:
        """The view with its axes in reverse order."""
        return self.transpose()

    # ------------------------------------------------------------------
    #   READING
    # ------------------------------------------------------------------

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        """Read the values of the view.

        Raises
        ------
        ValueError
            If `copy` is `False`, because reading values cannot be done
            without making them.
        """
        if copy is False:
            raise ValueError(
                "A proxy cannot be converted to an array without reading "
                "its values, which copy=False forbids."
            )
        values = self._load()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        if copy:
            values = np.array(values, copy=True)
        return values

    def _load(self) -> np.ndarray:
        """Read the values of the view.

        Raises
        ------
        ValueError
            If `_read` returns an array of the wrong shape.
        """
        shape = self.shape
        if 0 in shape:
            return np.empty(shape, self.dtype)
        selection, axes = self._view_state()
        index = []
        after = []
        for kept in selection:
            if isinstance(kept, int):
                index.append(kept)
                continue
            if isinstance(kept, range):
                step = abs(kept.step)
                first, last = min(kept), max(kept)
                after.append(slice(None, None, -1 if kept.step < 0 else 1))
            else:
                step = 1
                first, last = int(kept.min()), int(kept.max())
                after.append(kept - first)
            index.append(slice(first, last + 1, step))
        reader = self._root if self._root is not None else self
        values = np.asanyarray(reader._read(tuple(index)))
        expected = tuple(
            len(range(item.start, item.stop, item.step))
            for item in index
            if isinstance(item, slice)
        )
        if values.shape != expected:
            raise ValueError(
                f"{type(self).__name__}._read returned an array of shape "
                f"{values.shape} for a box of shape {expected}."
            )
        for i, item in enumerate(after):
            if isinstance(item, slice) and item.step == 1:
                continue
            values = values[(slice(None),) * i + (item,)]
        # The read axes are the stored axes that the view keeps, in the
        # order of the stored array.
        kept_axes = [
            axis
            for axis, kept in enumerate(selection)
            if not isinstance(kept, int)
        ]
        order = [
            kept_axes.index(axis) for axis in axes if isinstance(axis, int)
        ]
        values = values.transpose(order)
        new = tuple(
            i for i, axis in enumerate(axes) if not isinstance(axis, int)
        )
        if new:
            values = np.expand_dims(values, new)
        for i in new:
            if isinstance(axes[i], np.ndarray):
                values = values[(slice(None),) * i + (axes[i],)]
        return values

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


def _expand(index: tx.Any, ndim: int) -> tx.List[tx.Any]:
    """Return an index with one element for each axis, and the new axes.

    Each element is `None` for a new axis, or an integer, a slice or a
    one-dimensional array of integers for an existing axis. The ellipsis
    is replaced by full slices, and missing trailing indices are
    appended.

    Raises
    ------
    IndexError
        If there are more indices than dimensions, more than one ellipsis,
        or an index that selects points rather than a box.
    TypeError
        If an element of the index has an unsupported type.
    """
    if not isinstance(index, tuple):
        index = (index,)
    items = []
    ellipsis = None
    arrays = 0
    for item in index:
        if item is Ellipsis:
            if ellipsis is not None:
                raise IndexError("An index can only have one ellipsis.")
            ellipsis = len(items)
            items.append(item)
            continue
        if item is None or isinstance(item, slice):
            items.append(item)
            continue
        if isinstance(item, (bool, np.bool_)):
            raise IndexError(_POINTS)
        if isinstance(item, (int, np.integer)):
            items.append(operator.index(item))
            continue
        if isinstance(item, (list, tuple, np.ndarray)) or hasattr(
            item, "__array__"
        ):
            array = np.asarray(item)
            if array.ndim == 0 and array.dtype.kind in "iu":
                items.append(operator.index(array.item()))
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
            items.append(array.astype(np.intp, copy=False))
            continue
        raise TypeError(
            "A proxy is indexed with integers, slices, the ellipsis, None "
            f"and arrays of integers, not {type(item).__name__}."
        )
    if arrays > 1:
        raise IndexError(_POINTS)
    used = sum(
        1 for item in items if item is not None and item is not Ellipsis
    )
    if used > ndim:
        raise IndexError(
            f"Too many indices: the array has {ndim} dimensions, but "
            f"{used} were indexed."
        )
    fill = [slice(None)] * (ndim - used)
    if ellipsis is None:
        return items + fill
    return items[:ellipsis] + fill + items[ellipsis + 1 :]


_POINTS = (
    "A proxy is indexed lazily with integers, slices, the ellipsis, None "
    "and one array of integers. Several arrays and boolean masks select "
    "points rather than a box: use np.asarray(proxy)[index] instead."
)
