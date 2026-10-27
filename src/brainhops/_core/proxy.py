"""Lazy arrays that read their values only when they are needed.

A proxy stands for an array whose values lie elsewhere, usually in a
file, and reads them only when NumPy converts it with `np.asarray`, or
when a computation, such as `proxy + 1` or `np.mean(proxy)`, needs
them. Indexing a proxy, transposing it or reshaping it returns another
proxy, which reads only the values that it needs. The design is
described in the memo `spec/design/array-proxies.md`.

[`ArrayProxyProtocol`][] states what a consumer of a proxy may rely on,
and [`ArrayProxy`][] is the base class of every proxy of brainhops. The
proxies fall in three groups:

- a [`StoredProxy`][] presents an array as its storage holds it, and
  reads it one box at a time. The proxy of each file format is one, and
  so is a [`PermutedProxy`][], which presents the axes of another
  stored proxy in another order, and a [`ReshapedProxy`][], which
  presents the values of another proxy in another shape;
- an [`IndexedProxy`][] presents a part of a stored proxy, which an
  index selects. Indexing a stored proxy returns one;
- a [`ScaledProxy`][] presents the values of another proxy multiplied
  by a factor and shifted by an offset, as many formats store
  integers that stand for real values.

Indexing and transposing keep these layers in a fixed order. A scaled
proxy holds an indexed proxy, or a stored one, and an indexed proxy
holds a stored proxy, so a chain of indexes never builds a chain of
proxies. A reshape that cannot be done otherwise starts the order
again: a reshaped proxy is a stored proxy that holds any other proxy,
and reshaping it again reshapes that proxy. Each layer keeps the proxy
below it as `base`, and the `root` of every proxy is the stored proxy
that a reader built, at the bottom.
"""

__all__ = [
    "ArrayProxyProtocol",
    "ArrayProxy",
    "StoredProxy",
    "PermutedProxy",
    "IndexedProxy",
    "NewAxis",
    "ScaledProxy",
    "ReshapedProxy",
]

# stdlib
import abc
import math
import operator

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import HideIfDefault, Magic
from numpy.lib.mixins import NDArrayOperatorsMixin

# core
from . import backends
from .linalg import ArrayProtocol
from .properties import alias

_Box: tx.TypeAlias = tx.Tuple[tx.Union[int, slice], ...]
"""The box that [`StoredProxy._read`][brainhops._core.proxy.StoredProxy._read]
receives."""

_Item: tx.TypeAlias = tx.Union[None, int, slice, np.ndarray]
"""An item of an index that [`_normalize_index`][] returns."""


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


class ArrayProxy(
    NDArrayOperatorsMixin, Magic, eq=False, convert=True, repr=HideIfDefault()
):
    """A lazy array, which reads its values when they are needed.

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
    class, so a subclass that forgets one cannot be built.

    A proxy can also be used where an array is expected, and it reads
    its values only when a computation needs them. The operators of
    Python, such as `proxy + 1` or `proxy == 0`, the functions of NumPy,
    such as `np.mean(proxy)`, and the methods of a NumPy array, such as
    `proxy.sum()`, all compute on the values. Asking for the shape, the
    type of the values or a transposition reads nothing, through the
    proxy or through NumPy, as in `np.shape(proxy)`, and so does a
    reshape, which returns another proxy. A computation runs in the
    backend that `promote_backends` of `brainhops._core.backends`
    chooses for its operands: next to a CuPy array, the proxy is read
    onto the GPU, next to a Dask array, it becomes a Dask array that
    reads nothing yet, and alone, it becomes what
    [`to_backend`][brainhops._core.proxy.ArrayProxy.to_backend] makes of
    it. The values that a computation reads are not kept, so a second
    computation reads them again. A proxy is read-only: assigning values
    into it, or using it as the output of a computation, raises a
    `TypeError`. Like a NumPy array, a proxy cannot be hashed.
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
    @abc.abstractmethod
    def root(self) -> "StoredProxy":
        """The stored proxy that a reader built, which the values come from.

        An index, a permutation or a scaling of a proxy keeps the proxy
        below it as `base`, and its root is the root of `base`. A writer
        compares the file of the root with the file that it is about to
        write, so that it does not empty a file that the array still
        reads.
        """

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of values."""
        return math.prod(self.shape)

    @property
    def itemsize(self) -> int:
        """The number of bytes that one value takes once it is read."""
        return np.dtype(self.dtype).itemsize

    @property
    def nbytes(self) -> int:
        """The number of bytes that the values take once they are read."""
        return self.size * self.itemsize

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

        Without an argument, or with `None`, the order of the axes is
        reversed.

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

    def reshape(
        self,
        *shape: tx.Any,
        order: str = "C",
        copy: tx.Optional[bool] = None,
    ) -> "ArrayProxy":
        """Return a proxy of the values in another shape, as NumPy does.

        Nothing is read. A reshape that only adds or removes axes of
        length one is an index of the proxy. A strided proxy whose
        values NumPy could reshape without copying them gives another
        strided proxy (see
        [`StridedProxy`][brainhops._core.strided.StridedProxy]). Any
        other reshape gives a [`ReshapedProxy`][], which reads the
        values that a selection needs, and at most a few extra rows.

        Parameters
        ----------
        *shape : int or sequence of int
            The new shape, given as integers or as one sequence. One
            length may be `-1`, and it is then inferred from the others.
        order : {"C", "F"}, default="C"
            The order in which the values are read and placed: the last
            axis varies fastest in C order, and the first one in Fortran
            order.
        copy : bool, optional
            Accepted for compatibility with `np.reshape`. A proxy never
            copies values, since it reads them when they are needed.

        Returns
        -------
        ArrayProxy
            The reshaped proxy.

        Raises
        ------
        ValueError
            If the new shape does not hold as many values as the proxy,
            or `order` is neither `"C"` nor `"F"`.
        """
        new = _new_shape(shape, self.shape)
        if order == "F":
            return self.T.reshape(new[::-1]).T
        if order != "C":
            raise ValueError(f"order must be 'C' or 'F', not {order!r}.")
        if new == tuple(self.shape):
            return self
        index = _singleton_index(tuple(self.shape), new)
        if index is not None:
            return self[index]
        return self._reshaped(new)

    def _reshaped(self, shape: tx.Tuple[int, ...]) -> "ArrayProxy":
        """Return the proxy in another shape, in C order.

        The shape holds as many values as the proxy, and the reshape
        does more than add or remove axes of length one. By default, the
        result is a [`ReshapedProxy`][] of the proxy, and a subclass
        that can do better overrides this method.
        """
        return ReshapedProxy(self, shape)

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
        setting `"dask+cupy"`. The functions of Dask, such as
        `dask.array.transpose`, call this method under the name
        `to_dask_array` when they receive a proxy.

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
        lazy = da.from_array(_DaskSource(self), **options)
        return backends.to_dask(lazy, device=device)

    # Dask looks for a method of this name when one of its functions,
    # such as `dask.array.transpose`, receives a proxy, and calls it
    # instead of wrapping the proxy itself.
    to_dask_array = to_dask

    # --- computations -------------------------------------------------

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: tx.Any, **kwargs: tx.Any
    ) -> tx.Any:
        """Apply a universal function of NumPy, such as `np.add`.

        The operators of Python, such as `proxy + 1`, call this method
        too. The proxies among the operands are
        converted to the backend of the computation, and so are the
        arrays of other backends. When the computation writes into the
        arrays given as `out`, it runs in their backend. Otherwise, the
        backend is the one that `promote_backends` chooses for the
        operands.

        Parameters
        ----------
        ufunc : np.ufunc
            The universal function.
        method : str
            The method of the function, such as `"__call__"` or
            `"reduce"`.
        *inputs : Any
            The operands.
        **kwargs : Any
            The options of the function.

        Returns
        -------
        Any
            The result, or `NotImplemented` when an operand is of a type
            that the proxy does not know.

        Raises
        ------
        TypeError
            If a proxy is given as `out`.
        """
        out = kwargs.get("out", ())
        if any(isinstance(x, ArrayProxy) for x in out):
            raise TypeError(_READ_ONLY)
        if not all(_known(type(x)) for x in inputs + out):
            return NotImplemented
        setting = backends.promote_backends(*_arrays(out or inputs))
        inputs = _to_backend(inputs, setting)
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __array_function__(
        self,
        func: tx.Callable,
        types: tx.Collection[type],
        args: tx.Tuple[tx.Any, ...],
        kwargs: tx.Dict[str, tx.Any],
    ) -> tx.Any:
        """Apply a function of NumPy, such as `np.concatenate`.

        When the computation runs in NumPy, the implementation of NumPy
        receives the proxies themselves. It asks them for what it needs,
        so that `np.shape` reads nothing, `np.transpose` and
        `np.reshape` return proxies, and `np.mean` calls the method of
        the proxy, which reads the values. In any other backend, the
        proxies and the arrays of other backends are converted to the
        backend that `promote_backends` chooses for them, and the
        function of that backend runs.

        Parameters
        ----------
        func : callable
            The function of NumPy.
        types : collection of type
            The types of the operands that define `__array_function__`.
        args : tuple
            The positional arguments of the call.
        kwargs : dict
            The keyword arguments of the call.

        Returns
        -------
        Any
            The result, or `NotImplemented` when an operand is of a type
            that the proxy does not know.
        """
        if not all(_known(t) for t in types):
            return NotImplemented
        arrays = list(_arrays((args, kwargs)))
        setting = backends.promote_backends(*arrays)
        implementation = getattr(func, "_implementation", None)
        in_numpy = all(
            backends.resolve_backend(x, setting) == "numpy" for x in arrays
        )
        if in_numpy and implementation is not None:
            args, kwargs = _to_backend((args, kwargs), "numpy", proxies=False)
            return implementation(*args, **kwargs)
        args, kwargs = _to_backend((args, kwargs), setting)
        return func(*args, **kwargs)

    def __getattr__(self, name: str) -> tx.Any:
        # Python calls this method only for a name that the proxy does
        # not have. The other public attributes of NumPy arrays are
        # taken from the array that `to_backend` makes. A method reads
        # the values when it is called, so that asking whether a proxy
        # has it reads nothing, and another attribute, such as `real`,
        # reads them at once. A name that the class defines comes here
        # only when its property raised an `AttributeError`, which is
        # raised again rather than hidden.
        if name not in _FORWARDED or hasattr(type(self), name):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        if not callable(getattr(np.ndarray, name)):
            return getattr(self.to_backend(), name)

        def method(*args: tx.Any, **kwargs: tx.Any) -> tx.Any:
            return getattr(self.to_backend(), name)(*args, **kwargs)

        method.__name__ = name
        method.__doc__ = getattr(np.ndarray, name).__doc__
        return method

    def __bool__(self) -> bool:
        return bool(self.to_numpy())

    def __int__(self) -> int:
        return int(self.to_numpy())

    def __float__(self) -> float:
        return float(self.to_numpy())

    def __complex__(self) -> complex:
        return complex(self.to_numpy())

    def __index__(self) -> int:
        return operator.index(self.to_numpy())


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

    @property
    def root(self) -> "StoredProxy":
        """The proxy itself, since it reads its values from its storage."""
        return self

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

    def _reshaped_box(
        self, box: _Box, shape: tx.Tuple[int, ...]
    ) -> tx.Optional[ArrayProxy]:
        """Return a box of the proxy in another shape, or `None`.

        An [`IndexedProxy`][] whose selection is a box asks its stored
        proxy for this, so that a stored proxy that can present a box in
        another shape without reading it, as a strided proxy can, does
        so. By default, the stored proxy cannot, and the indexed proxy
        is wrapped in a [`ReshapedProxy`][].

        Parameters
        ----------
        box : tuple of int or slice
            The selection, with one entry for each axis of the proxy.
        shape : tuple of int
            The new shape, which holds as many values as the box.

        Returns
        -------
        ArrayProxy or None
            The reshaped box, or `None`.
        """
        return None

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
    the permuted proxy is axis `perm[i]` of `base`.

    Transposing a permuted proxy composes the two permutations and
    transposes `base` once, so permuted proxies never nest. A read
    reorders the box for `base`, reads it, and transposes the values.
    """

    base: tx.Any
    """The stored proxy whose axes are permuted."""

    perm: tx.Tuple[int, ...]
    """The axis of `base` that each axis of the proxy is."""

    @property
    def root(self) -> StoredProxy:
        """The stored proxy that a reader built, below `base`."""
        return self.base.root

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of `base`, in the order of the permutation."""
        return tuple(self.base.shape[p] for p in self.perm)

    @property
    def dtype(self) -> np.dtype:
        """The data type of the values of `base`."""
        return self.base.dtype

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape of `base`, in the order of the permutation."""
        chunks = self.base.chunks
        if chunks is None:
            return None
        return tuple(chunks[p] for p in self.perm)

    def transpose(self, *axes: tx.Any) -> StoredProxy:
        """Return `base` transposed by the two permutations, composed.

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
        return self.base.transpose([self.perm[p] for p in perm])

    def _read(self, box: _Box) -> np.ndarray:
        base_box = [None] * len(box)
        for axis, p in enumerate(self.perm):
            base_box[p] = box[axis]
        values = np.asarray(self.base[tuple(base_box)])
        # The values hold the axes that slices keep, in the order of
        # `base`, and they are put in the order of the permutation.
        kept = [
            p for p, item in enumerate(base_box) if isinstance(item, slice)
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

    kept: tx.Union[slice, np.ndarray] = slice(0, 1, 1)
    """The positions of the single value that the axis presents: a slice
    of `range(1)` or an array of zeros."""


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

    base: tx.Any
    """The stored proxy whose values are selected."""

    selection: tx.Tuple[tx.Any, ...]
    """One entry for each axis of `base`, in order, with a `NewAxis`
    entry for each axis that is not stored."""

    @property
    def root(self) -> StoredProxy:
        """The stored proxy that a reader built, below `base`."""
        return self.base.root

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
        items, front = _normalize_index(index, self.shape)
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

    def _reshaped(self, shape: tx.Tuple[int, ...]) -> ArrayProxy:
        """Return the selected values in another shape.

        When the selection is a box, made of integers and slices, the
        stored proxy is asked to present the box in the new shape, which
        a strided proxy does by changing its strides. Otherwise, the
        result is a [`ReshapedProxy`][] of this proxy.
        """
        if all(
            isinstance(e, (int, np.integer, slice)) for e in self.selection
        ):
            reshaped = self.base._reshaped_box(self.selection, shape)
            if reshaped is not None:
                return reshaped
        return ReshapedProxy(self, shape)

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
    volume. A scaled proxy applies them to the values of `base`, which
    reads the stored integers.

    Indexing or transposing the proxy does the same to `base` and to
    the factors, so that only the selected values are read and scaled.
    An axis of length one of a factor stays of length one, so the
    factors never grow to the shape of the values.

    The type of the scaled values depends on the stored type only: it is
    the floating-point type that NumPy gives for the stored type and a
    float32, so that integers of 8 or 16 bits and float32 values give
    float32, and other integers and float64 values give float64. The
    factors are cast to that type before they are applied, so a factor
    of a wider type neither widens the result nor makes a wider copy of
    the values.
    """

    base: tx.Any
    """The proxy of the values before scaling."""

    slope: tx.Optional[np.ndarray] = None
    """The factor, a number or an array that broadcasts against the shape
    of `base`, or `None`."""

    inter: tx.Optional[np.ndarray] = None
    """The offset, a number or an array that broadcasts against the shape
    of `base`, or `None`."""

    @property
    def root(self) -> StoredProxy:
        """The stored proxy that a reader built, below `base`."""
        return self.base.root

    def __post_init__(self) -> None:
        # A factor is given one axis for each axis of `base`, so that an
        # index applies to it axis by axis.
        ndim = len(self.base.shape)
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
        """The shape of `base`."""
        return tuple(self.base.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the scaled values, which the stored type decides.

        It is the floating-point type that NumPy gives for the stored
        type and a float32, whatever the type of the factors.
        """
        return np.result_type(self.base.dtype, np.float32)

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape of `base`."""
        return self.base.chunks

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
        items, front = _normalize_index(index, self.shape)
        return ScaledProxy(
            self.base[index],
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
            self.base.transpose(perm),
            None if self.slope is None else self.slope.transpose(perm),
            None if self.inter is None else self.inter.transpose(perm),
        )

    def _reshaped(self, shape: tx.Tuple[int, ...]) -> ArrayProxy:
        """Return the scaled proxy of the values in another shape.

        `base` and the factors are reshaped when each factor still
        broadcasts against the new shape: for each group of axes that
        the reshape splits or merges, a factor has either length one on
        every axis of the group or the full length on all of them. The
        factor of each AFNI sub-brick, which runs along the last axis
        only, does not when that axis is merged with others. The scaled
        proxy is then wrapped in a [`ReshapedProxy`][].
        """
        factors = []
        for factor in (self.slope, self.inter):
            reshaped = _reshape_factor(factor, self.shape, shape)
            if factor is not None and reshaped is None:
                return ReshapedProxy(self, shape)
            factors.append(reshaped)
        return ScaledProxy(self.base.reshape(shape), *factors)

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        """Read the values of `base` and scale them.

        Raises
        ------
        ValueError
            If `copy` is `False`, because reading values cannot be done
            without making them.
        """
        _refuse_no_copy(copy)
        stored = np.asarray(self.base)
        # Values in another type are converted into a new array, which is
        # then scaled in place. Values that are already in the type may be
        # a read-only memory map, or a view of an array that `base` keeps,
        # so the first step writes them into a new array, and the next
        # step is done in place in that array.
        owned = stored.dtype != self.dtype
        values = stored.astype(self.dtype) if owned else stored
        # The factors are cast to the type of the values first, so that a
        # float64 factor does not make a float64 copy of float32 values.
        for factor, apply in ((self.slope, np.multiply), (self.inter, np.add)):
            if factor is None:
                continue
            factor = factor.astype(self.dtype, copy=False)
            if owned:
                apply(values, factor, out=values)
            else:
                # The product of arrays of no dimension is a NumPy scalar,
                # which a later step could not write into.
                values = np.asarray(apply(values, factor))
                owned = True
        return _convert(values, dtype, copy)


# ----------------------------------------------------------------------
#   RESHAPED PROXIES
# ----------------------------------------------------------------------


class ReshapedProxy(StoredProxy):
    """A lazy array of the values of another proxy, in another shape.

    The values are those of `base`, placed in the new shape in C order,
    as `np.reshape` places them. A reshaped proxy is indexed and
    transposed as the proxy of a file is, and its `root` is the root of
    `base`. Reshaping it again reshapes `base`.

    A read gets the rows of `base` that hold the selected values: along
    each group of axes that the reshape splits or merges, at most one
    extra row on each side.

    [`ArrayProxy.reshape`][brainhops._core.proxy.ArrayProxy.reshape]
    returns a reshaped proxy when the reshape does more than add or
    remove axes of length one, unless `base` is a strided proxy whose
    values NumPy could reshape without copying them.
    """

    base: tx.Any
    """The proxy whose values are presented in another shape."""

    # A field cannot implement the abstract property `shape`, so the
    # shape is kept in a private field, given as `shape=`.
    _shape: tx.Tuple[int, ...]
    """The new shape, given as `shape=`."""

    shape = alias("_shape", readonly=True)
    """The new shape."""

    @property
    def dtype(self) -> np.dtype:
        """The type of the values of `base`."""
        return self.base.dtype

    @property
    def root(self) -> StoredProxy:
        """The stored proxy that a reader built, below `base`."""
        return self.base.root

    def __post_init__(self) -> None:
        if math.prod(self._shape) != math.prod(self.base.shape):
            raise ValueError(
                f"A proxy of shape {tuple(self.base.shape)} cannot be "
                f"reshaped to {self._shape}."
            )

    def _reshaped(self, shape: tx.Tuple[int, ...]) -> ArrayProxy:
        """Return `base` in another shape, so that reshapes do not nest."""
        return self.base.reshape(shape)

    def _read(self, box: _Box) -> np.ndarray:
        """Read a box of the reshaped values.

        Parameters
        ----------
        box : tuple of int or slice
            The box to read, as
            [`StoredProxy._read`][brainhops._core.proxy.StoredProxy._read]
            receives it.

        Returns
        -------
        np.ndarray
            The values of the box.
        """
        old_shape = tuple(self.base.shape)
        old_box = []
        picks = []
        sizes = []
        shape = []
        for new_axes, old_axes in _groups(old_shape, self._shape):
            # The positions that the box selects along the group, each
            # given as its position among the values of the group in C
            # order. Positive steps make the first one the smallest and
            # the last one the largest.
            ranges = []
            for axis in new_axes:
                entry = box[axis]
                if isinstance(entry, (int, np.integer)):
                    ranges.append(range(int(entry), int(entry) + 1))
                else:
                    ranges.append(range(entry.start, entry.stop, entry.step))
                    shape.append(len(ranges[-1]))
            group = [self._shape[axis] for axis in new_axes]
            flat = _flat_positions(ranges, group)
            # The box of `base` that holds the positions: the leading
            # axes where the first and the last position agree, then
            # the range between them on the next axis, and the whole of
            # the remaining axes.
            stored = [old_shape[axis] for axis in old_axes]
            first = _unravel(int(flat[0]), stored)
            last = _unravel(int(flat[-1]), stored)
            starts = []
            stops = []
            apart = False
            for low, high, n in zip(first, last, stored):
                if apart:
                    low, high = 0, n - 1
                starts.append(low)
                stops.append(high + 1)
                apart = apart or low != high
            old_box.extend(slice(a, b, 1) for a, b in zip(starts, stops))
            # The box holds the consecutive positions of the group from
            # its first value on.
            picks.append(flat - _ravel(starts, stored))
            sizes.append(math.prod(b - a for a, b in zip(starts, stops)))
        values = np.asarray(self.base[tuple(old_box)])
        values = values.reshape(sizes)[np.ix_(*picks)]
        return values.reshape(shape)


def _new_shape(
    shape: tx.Tuple[tx.Any, ...], old: tx.Sequence[int]
) -> tx.Tuple[int, ...]:
    """Return the shape that the arguments of `reshape` ask for.

    Raises
    ------
    ValueError
        If the shape has more than one `-1`, or does not hold as many
        values as `old`.
    """
    if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):
        shape = tuple(shape[0])
    shape = [operator.index(n) for n in shape]
    size = math.prod(old)
    unknown = [i for i, n in enumerate(shape) if n == -1]
    if len(unknown) > 1:
        raise ValueError("Only one length of a new shape can be -1.")
    if unknown:
        known = math.prod(n for n in shape if n != -1)
        if known == 0 or size % known:
            raise ValueError(
                f"A proxy of shape {tuple(old)} cannot be reshaped to "
                f"{tuple(shape)}."
            )
        shape[unknown[0]] = size // known
    if any(n < 0 for n in shape) or math.prod(shape) != size:
        raise ValueError(
            f"A proxy of shape {tuple(old)} cannot be reshaped to "
            f"{tuple(shape)}."
        )
    return tuple(shape)


def _singleton_index(
    old: tx.Tuple[int, ...], new: tx.Tuple[int, ...]
) -> tx.Optional[tx.Tuple[tx.Any, ...]]:
    """Return the index that reshapes `old` into `new`, or `None`.

    The index exists when the reshape only adds or removes axes of length
    one: it holds `None` for each added axis, `0` for each removed one,
    and a whole slice for each kept one.
    """
    if [n for n in old if n != 1] != [n for n in new if n != 1]:
        return None
    index = []
    i = j = 0
    while i < len(old) or j < len(new):
        if i < len(old) and j < len(new) and old[i] == new[j]:
            index.append(slice(None))
            i += 1
            j += 1
        elif i < len(old) and old[i] == 1:
            index.append(0)
            i += 1
        else:
            index.append(None)
            j += 1
    return tuple(index)


def _groups(
    old: tx.Sequence[int], new: tx.Sequence[int]
) -> tx.List[tx.Tuple[range, range]]:
    """Return the groups of axes that a reshape splits or merges.

    Each group is a range of axes of `new` and a range of axes of `old`
    that hold as many values, and the groups are the smallest such
    ranges. A range may be empty on one side when the other side holds
    axes of length one only. The arrays hold at least one value.
    """
    groups = []
    i = j = 0
    while i < len(new) or j < len(old):
        a, b = i, j
        size_new = size_old = 1
        if i < len(new):
            size_new *= new[i]
            i += 1
        if j < len(old):
            size_old *= old[j]
            j += 1
        while size_new != size_old:
            if size_new < size_old:
                size_new *= new[i]
                i += 1
            else:
                size_old *= old[j]
                j += 1
        groups.append((range(a, i), range(b, j)))
    return groups


def _flat_positions(
    ranges: tx.Sequence[range], shape: tx.Sequence[int]
) -> np.ndarray:
    """Return the positions in C order of the values that ranges select.

    The positions are those of the values of an array of shape `shape`
    whose index along each axis lies in the range of that axis, in the
    order of these values.
    """
    flat = np.zeros((1,) * len(shape), dtype=np.intp)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        positions = np.asarray(ranges[axis], dtype=np.intp) * stride
        flat = flat + positions.reshape(
            (1,) * axis + (-1,) + (1,) * (len(shape) - axis - 1)
        )
        stride *= shape[axis]
    return flat.reshape(-1)


def _ravel(index: tx.Sequence[int], shape: tx.Sequence[int]) -> int:
    """Return the position in C order of an index in an array of a shape."""
    position = 0
    for i, n in zip(index, shape):
        position = position * n + i
    return position


def _unravel(position: int, shape: tx.Sequence[int]) -> tx.List[int]:
    """Return the index of a position in C order in an array of a shape."""
    index = []
    for n in reversed(shape):
        position, i = divmod(position, n)
        index.append(i)
    return index[::-1]


def _reshape_factor(
    factor: tx.Optional[np.ndarray],
    old: tx.Tuple[int, ...],
    new: tx.Tuple[int, ...],
) -> tx.Optional[np.ndarray]:
    """Return a factor of a scaled proxy for the values in another shape.

    The factor has one axis for each axis of the values. For each group
    of axes that the reshape splits or merges, it must have length one
    on every axis of the group, or the full length on all of them.

    Returns
    -------
    np.ndarray or None
        The reshaped factor, or `None` when the factor is `None` or does
        not broadcast against the new shape.
    """
    if factor is None or not math.prod(old):
        return None
    lengths = []
    for new_axes, old_axes in _groups(old, new):
        stored = [factor.shape[axis] for axis in old_axes]
        if all(n == 1 for n in stored):
            lengths.extend(1 for _ in new_axes)
        elif stored == [old[axis] for axis in old_axes]:
            lengths.extend(new[axis] for axis in new_axes)
        else:
            return None
    return factor.reshape(lengths)


def _index_factor(
    factor: tx.Optional[np.ndarray],
    items: tx.Sequence[_Item],
    shape: tx.Sequence[int],
    front: tx.Optional[int],
) -> tx.Optional[np.ndarray]:
    """Index a factor as [`_normalize_index`][] says the values are indexed.

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


def _normalize_index(
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
    if axes == (None,):
        # `np.transpose(proxy)` calls `proxy.transpose(None)`.
        axes = ()
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


# ----------------------------------------------------------------------
#   COMPUTATIONS
# ----------------------------------------------------------------------

_IN_PLACE = frozenset(
    ("fill", "itemset", "partition", "put", "resize", "setfield", "sort")
)
"""The methods of NumPy arrays that change the array in place.

A proxy does not take them from the array that it reads, because the
change would be made to values that are then thrown away.
"""

_STORAGE = frozenset(
    ("base", "ctypes", "data", "device", "flags", "setflags", "strides")
)
"""The attributes of NumPy arrays that describe their memory.

A proxy does not take them from the array that it reads, because they
would describe a temporary array rather than the proxy.
"""

_FORWARDED = (
    frozenset(name for name in dir(np.ndarray) if not name.startswith("_"))
    - _IN_PLACE
    - _STORAGE
)
"""The attributes that a proxy takes from the array that it reads."""

_READ_ONLY = (
    "A proxy is read-only. Read its values with np.asarray(proxy), or "
    "convert the object that holds it with as_numpy(), before changing "
    "them."
)


class _DaskSource:
    """The view of a proxy that Dask reads block by block.

    `dask.array.from_array` copies an array-like argument that has a
    `copy` method, and a proxy has one, which reads all its values.
    Dask also keeps a block as it is when the block behaves as an array,
    as a proxy does, so the blocks would reach the computation unread.
    Dask therefore receives this view, which only has what Dask reads:
    the shape, the type of the values, the preferred blocks, and the
    indexing, which reads a block into a NumPy array.
    """

    def __init__(self, proxy: ArrayProxy) -> None:
        self.proxy = proxy

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        return self.proxy.shape

    @property
    def dtype(self) -> np.dtype:
        return self.proxy.dtype

    @property
    def ndim(self) -> int:
        return self.proxy.ndim

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        return self.proxy.chunks

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        return np.asarray(self.proxy[index])


def _known(cls: type) -> bool:
    """Tell whether a proxy can compute with operands of a type.

    The type is a proxy, a NumPy, CuPy or Dask array, or a type that
    does not take part in the dispatch of NumPy, such as a number.
    """
    if not hasattr(cls, "__array_ufunc__"):
        return True
    if issubclass(cls, (ArrayProxy, np.ndarray)):
        return True
    if backends.cp and issubclass(cls, backends.cp.ndarray):
        return True
    return bool(backends.da) and issubclass(cls, backends.da.Array)


def _is_array(x: tx.Any) -> bool:
    """Tell whether an operand is a proxy or a NumPy, CuPy or Dask array."""
    return isinstance(x, ArrayProxy) or backends._is_loaded(x)


def _arrays(values: tx.Any) -> tx.Iterator[tx.Any]:
    """Yield the proxies and the arrays among the arguments of a call.

    The arguments may be nested in lists, tuples and dictionaries, as
    the arrays that `np.concatenate` receives.
    """
    if _is_array(values):
        yield values
    elif isinstance(values, (list, tuple)):
        for value in values:
            yield from _arrays(value)
    elif isinstance(values, dict):
        for value in values.values():
            yield from _arrays(value)


def _to_backend(
    values: tx.Any, setting: tx.Optional[str], *, proxies: bool = True
) -> tx.Any:
    """Convert the proxies and the arrays among the arguments of a call.

    The lists, tuples and dictionaries that hold them are rebuilt around
    the converted arrays, and the other arguments are kept as they are.

    Parameters
    ----------
    values : Any
        The arguments.
    setting : str or None
        The backend setting of the conversion, or `None` for the
        current setting.
    proxies : bool, default=True
        Whether the proxies are converted. Otherwise they are kept, and
        only the arrays are converted.

    Returns
    -------
    Any
        The converted arguments.
    """
    if isinstance(values, ArrayProxy):
        return backends.to_backend(values, setting) if proxies else values
    if backends._is_loaded(values):
        return backends.to_backend(values, setting)
    if isinstance(values, (list, tuple)):
        converted = [_to_backend(x, setting, proxies=proxies) for x in values]
        if isinstance(values, tuple) and hasattr(values, "_fields"):
            return type(values)(*converted)
        return type(values)(converted)
    if isinstance(values, dict):
        return {
            key: _to_backend(x, setting, proxies=proxies)
            for key, x in values.items()
        }
    return values


_POINTS = (
    "A proxy is indexed lazily with integers, slices, the ellipsis, None "
    "and one array of integers. Several arrays and boolean masks select "
    "points rather than a box: use np.asarray(proxy)[index] instead."
)
