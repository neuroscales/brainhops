import itertools
import math
import operator

import numpy as np
import typing_extensions as _tx

try:
    import torch
except ImportError:
    torch = None


class oob_slice:
    """Out-of-bound slice, which selects no element.

    When `newaxis` is true, the slice stands in for a new axis and does not
    consume an input dimension.
    """

    newaxis: bool = False

    def __init__(self, newaxis: bool = False) -> None:
        self.newaxis = newaxis

    def __repr__(self) -> str:
        if self.newaxis:
            return "oob_slice(newaxis=True)"
        else:
            return "oob_slice()"

    __str__ = __repr__


IndexLike = _tx.Union[int, slice, oob_slice, type(None), type(Ellipsis)]
NDIndexLike = _tx.Union[IndexLike, _tx.Tuple[IndexLike, ...]]
INDEX_LIKE = (int, slice, oob_slice, type(None), type(Ellipsis))


def is_newaxis(index: IndexLike) -> bool:
    """Return whether an index inserts a new axis."""
    return index is None or (isinstance(index, oob_slice) and index.newaxis)


def is_droppedaxis(index: IndexLike) -> bool:
    """Return whether an index drops an axis, which is the case for an int."""
    return isinstance(index, int)


def is_sliceaxis(index: IndexLike) -> bool:
    """Return whether an index is a slice or an out-of-bound slice."""
    return isinstance(index, (slice, oob_slice))


@_tx.overload
def neg2pos(index: int, shape: int) -> int: ...


@_tx.overload
def neg2pos(
    index: _tx.Tuple[IndexLike, ...], shape: _tx.Tuple[int, ...]
) -> _tx.Tuple[IndexLike, ...]: ...


def neg2pos(index, shape):
    """Convert negative indices, which count from the end, to positive ones.

    The start and stop of a slice are converted, and its step is unchanged.
    `None`, `Ellipsis` and [`oob_slice`][] are returned as is. When `index` is
    a sequence, `shape` holds one length per index that is not `None`, and a
    tuple is returned.

    !!! warning
        This function must be applied only once, to user input, because it
        can flip an index that has already been converted:

        ```python
        neg2pos(-5, 3) = -2                              # correct
        neg2pos(neg2pos(-5, 3), 3) = neg2pos(-2, 3) = 1  # wrong
        ```

    Raises
    ------
    ValueError
        If `shape` is negative or inconsistent with `index`.
    TypeError
        If `index` or `shape` has an unsupported type.
    """

    if not isinstance(index, INDEX_LIKE):
        # `None` consumes no dimension.
        shape0 = shape
        shape = []
        for _d, idx in enumerate(index):
            if idx is None:
                shape.append(None)
            else:
                shp, *shape0 = shape0
                shape.append(shp)
        index = list(index)
        if len(shape0) > 0 or len(index) != len(shape):
            raise ValueError("shape and index vectors not consistent.")
        return tuple(neg2pos(idx, shp) for idx, shp in zip(index, shape))

    try:
        shape0 = shape
        shape = int(shape0)
        if shape != shape0:
            raise TypeError("Shape should be an integer")
    except TypeError:
        raise TypeError("Shape should be an integer") from None
    if shape < 0:
        raise ValueError("Shape should be a nonnegative integer")

    if isinstance(index, slice):
        return slice(
            neg2pos(index.start, shape), neg2pos(index.stop, shape), index.step
        )
    elif isinstance(index, int):
        if index is not None and index < 0:
            index = shape + index
        return index
    elif isinstance(index, (oob_slice, type(None), type(Ellipsis))):
        return index
    raise TypeError("Index should be an int, slice, Ellipsis or None")


@_tx.overload
def is_fullslice(
    index: IndexLike, shape: int, do_neg2pos: bool = True
) -> bool: ...


@_tx.overload
def is_fullslice(
    index: _tx.Tuple[IndexLike, ...],
    shape: _tx.Tuple[int, ...],
    do_neg2pos: bool = True,
) -> _tx.Tuple[bool, ...]: ...


def is_fullslice(index, shape, do_neg2pos=True):
    """Return whether an index covers a whole dimension.

    An index is a full slice if it is a new axis, a slice equivalent to `:`
    or `::-1`, or `0` on a singleton dimension. A sequence index is expanded
    with [`expand_index`][] and yields one result per dimension.
    """
    if index is None:
        return True
    elif isinstance(index, slice):
        index = simplify_slice(index, shape, do_neg2pos=do_neg2pos)
        return (
            index.start is None
            and index.stop is None
            and index.step in (None, 1, -1)
        )
    elif isinstance(index, int):
        if do_neg2pos:
            index = neg2pos(index, shape)
        return index == 0 and shape == 1
    elif isinstance(index, oob_slice):
        return oob_slice.newaxis
    else:
        index = expand_index(index, shape)
        # `None` consumes no dimension.
        shape0 = shape
        shape = []
        for _d, idx in enumerate(index):
            if idx is None:
                shape.append(None)
            else:
                shp, *shape0 = shape0
                shape.append(shp)
        return tuple(is_fullslice(idx, shp) for idx, shp in zip(index, shape))


def slice_length(index: slice, shape: int, do_neg2pos: bool = True) -> int:
    """Return the number of elements that a slice selects.

    Unless `do_neg2pos` is false, the slice must not have been converted
    with [`neg2pos`][] beforehand. The same holds for the other slice helpers.
    """

    def sign(x: float) -> int:
        return 1 if x > 0 else -1 if x < 0 else 0

    if do_neg2pos:
        index = neg2pos(index, shape)
    start = index.start
    stop = index.stop

    step = index.step
    step = 1 if step is None else step

    if step < 0:
        if stop is None or stop < 0:
            stop = -1
        if start is None or start >= shape:
            start = shape - 1
    else:
        if stop is None or stop > shape:
            stop = shape
        if start is None or start < 0:
            start = 0
    return max(1 + (stop - start - sign(step)) // step, 0)


def simplify_slice(index: slice, shape: int, do_neg2pos: bool = True) -> slice:
    """Replace the redundant parts of a slice with `None`.

    A slice that selects nothing is replaced with an [`oob_slice`][].
    """

    length = slice_length(index, shape)
    if do_neg2pos:
        index = neg2pos(index, shape)
    start = index.start

    step = index.step or 1
    if step < 0:
        if start is None or start >= shape - 1:
            start = shape - 1
        stop = start + length * step
        if stop >= start:
            return oob_slice()
        if stop < 0:
            stop = None
        if start >= shape - 1:
            start = None
    else:
        if start is None or start <= 0:
            start = 0
        stop = start + length * step
        if stop <= start:
            return oob_slice()
        if stop >= shape:
            stop = None
        if start <= 0:
            start = None
        if step == 1:
            step = None

    return slice(start, stop, step)


def invert_slice(index: slice, shape: int, do_neg2pos: bool = True) -> slice:
    """Return the slice that selects the same elements in reverse order."""

    def sign(x: float) -> int:
        return 1 if x > 0 else -1 if x < 0 else 0

    start, step, length = slice_navigator(index, shape, do_neg2pos)

    start = start + (length - 1) * step
    step = -step
    stop = start + (length - 1) * step + sign(step)
    return simplify_slice(slice(start, stop, step), shape, do_neg2pos=False)


def slice_navigator(
    index: slice, shape: int, do_neg2pos: bool = True
) -> _tx.Tuple[int, int, int]:
    """Return the explicit `(start, step, length)` of a slice."""
    length = slice_length(index, shape, do_neg2pos)
    index = simplify_slice(index, shape, do_neg2pos)
    start = index.start
    step = index.step or 1

    if step < 0:
        if start is None:
            start = shape - 1
    else:
        if start is None:
            start = 0

    return start, step, length


def is_slice_equivalent(
    index1: slice,
    index2: slice,
    shape: int,
    same_sign: bool = True,
    do_neg2pos: bool = True,
) -> bool:
    """Return whether two slices select the same data.

    If `same_sign` is false, slices with a negative step are inverted before
    the comparison, so that the order of the elements does not matter.
    """
    if do_neg2pos:
        index1 = neg2pos(index1, shape)
        index2 = neg2pos(index2, shape)
    if not same_sign:
        if index1.step is not None and index1.step < 0:
            index1 = invert_slice(index1, shape, False)
        if index2.step is not None and index2.step < 0:
            index2 = invert_slice(index2, shape, False)
    start1, step1, length1 = slice_navigator(index1, shape, False)
    start2, step2, length2 = slice_navigator(index2, shape, False)
    return (start1, step1, length1) == (start2, step2, length2)


def guess_shape(
    index: _tx.Sequence[IndexLike], shape: _tx.Sequence[int]
) -> _tx.Tuple[int, ...]:
    """Return the shape of the result of indexing an array.

    New axes have size 1, out-of-bound slices have size 0 and integer
    indices drop their axis. A `ValueError` is raised if `shape` has more
    dimensions than `index` consumes.
    """
    index = expand_index(index, shape)

    output_shape = []
    while len(index) > 0:
        idx, *index = index
        if idx is None:
            # A new axis has size 1.
            output_shape.append(1)
            continue
        if isinstance(idx, oob_slice):
            # Size 0; consumes an input dimension unless it is a new axis.
            output_shape.append(0)
            if not idx.newaxis:
                _, *shape = shape
            continue
        sz, *shape = shape
        if isinstance(idx, int):
            # An integer drops the axis.
            continue
        if isinstance(idx, slice):
            output_shape.append(slice_length(idx, sz))
            continue

    if len(shape) > 0:
        raise ValueError("Shape to long for this index vector")

    return tuple(output_shape)


def expand_index(
    index: NDIndexLike, shape: _tx.Tuple[int, ...]
) -> _tx.Tuple[IndexLike, ...]:
    """Convert an index into its canonical form.

    The ellipsis is replaced with full slices, slices are simplified,
    implicit trailing slices are appended and negative indices are made
    positive, so that the result contains only `None`, integers, slices and
    [`oob_slice`][] objects. Unlike `nibabel.fileslice.canonical_slicers`,
    the function rejects floating-point indices and keeps negative steps.
    Zero-dimensional integer arrays and tensors are accepted as integers.

    Raises
    ------
    TypeError
        If an element of `index` is not a supported index.
    ValueError
        If `index` contains more than one ellipsis or a non-scalar array.
    IndexError
        If an integer index is out of bounds.
    """
    index = list(index)
    shape = list(shape)
    nb_dim = len(shape)

    def is_int(elem: _tx.Any) -> bool:
        if torch.is_tensor(elem):
            return elem.dtype in (torch.int32, torch.int64) and not elem.shape
        elif np and isinstance(elem, np.ndarray):
            return elem.dtype in (np.int32, np.int64) and not elem.shape
        elif isinstance(elem, int):
            return True
        else:
            return False

    # Number of input and output dimensions of each kind of index:
    #
    #    type        | in             | out         | supported
    #   --------------------------------------------------------
    #    None        | 0              | 1           | yes
    #    slice       | 1              | 1           | yes
    #    int         | 1              | 0           | yes
    #    ellipsis    | (dim - others) | same        | yes
    #    list[int]   | 1              | 1           | no
    #    list[bool]  | 1              | 1           | no
    #    array[int]  | 1              | array.dim() | no
    #    array[bool] | array.dim()    | 1           | no
    #
    # Like nibabel, advanced indexing is not supported: its broadcasting
    # rules are less intuitive than Matlab's.

    nb_dim_in = []
    nb_dim_out = []
    ind_ellipsis = None
    for n_ind, ind in enumerate(index):
        if ind is None:
            nb_dim_in.append(0)
            nb_dim_out.append(1)
        elif isinstance(ind, slice):
            nb_dim_in.append(1)
            nb_dim_out.append(1)
        elif isinstance(ind, oob_slice):
            nb_dim_in.append(0 if ind.newaxis else 1)
            nb_dim_out.append(1)
        elif ind is Ellipsis:
            if ind_ellipsis is not None:
                raise ValueError("Cannot have more than one ellipsis.")
            ind_ellipsis = n_ind
            nb_dim_in.append(-1)
            nb_dim_out.append(-1)
        elif is_int(ind):
            ind = torch.as_tensor(ind, dtype=torch.int64)
            if ind.dim() > 0:
                raise ValueError(
                    "Integer indices should be scalars "
                    f"Got array with shape {ind.shape}."
                )
            nb_dim_in.append(1)
            nb_dim_out.append(ind.dim())
            index[n_ind] = ind.item()
        else:
            raise TypeError(
                "Indices should be integers, slices "
                f"or ellipses. Got {type(ind)}."
            )

    # The ellipsis absorbs the remaining dimensions. If it is absent, it is
    # appended implicitly.
    nb_known_dims = sum(n for n in nb_dim_in if n > 0)
    if ind_ellipsis is not None:
        nb_dim_in[ind_ellipsis] = max(0, nb_dim - nb_known_dims)
        nb_dim_out[ind_ellipsis] = nb_dim_in[ind_ellipsis]
    else:
        index.append(Ellipsis)
        nb_dim_in.append(max(0, nb_dim - nb_known_dims))
        nb_dim_out.append(nb_dim_in[-1])

    nb_ind = 0
    index0 = index
    index = []
    for d, ind in enumerate(index0):
        if ind is None:
            # A new axis consumes no input dimension.
            nb_ind += nb_dim_in[d]
            index.append(None)
        elif isinstance(ind, slice):
            index.append(simplify_slice(ind, shape[nb_ind]))
            nb_ind += nb_dim_in[d]
        elif isinstance(ind, oob_slice):
            index.append(ind)
            nb_ind += nb_dim_in[d]
        elif ind is Ellipsis:
            # The ellipsis becomes one full slice per dimension it covers.
            for _dd in range(nb_ind, nb_ind + nb_dim_in[d]):
                index.append(slice(None))
                nb_ind += 1
        else:
            assert isinstance(ind, int)  # validated in the first loop
            ind = neg2pos(ind, shape[nb_ind])
            if ind < 0 or ind >= shape[nb_ind]:
                raise IndexError(
                    f"Out-of-bound index in dimension {nb_ind} "
                    f"({ind} not in [0, {shape[nb_ind] - 1}])"
                )
            index.append(ind)
            nb_ind += nb_dim_in[d]

    return tuple(index)


def compose_index(
    parent: _tx.Sequence[IndexLike],
    child: _tx.Sequence[IndexLike],
    full_shape: _tx.Sequence[int],
) -> _tx.Tuple[IndexLike]:
    """Combine a parent index and a child index into a single index.

    The child index applies to the result of the parent index, and the
    combined index applies to the original array of shape `full_shape`. An
    `IndexError` is raised if an integer of the child index is out of
    bounds, or if the child index has more indices than there are dimensions.
    """

    def oob(i: IndexLike) -> None:
        raise IndexError(f"Index out-of-bound in parent dimension {i}.")

    parent = expand_index(parent, full_shape)
    sub_shape = guess_shape(parent, full_shape)
    child = list(expand_index(child, sub_shape))

    i_parent = -1
    new_parent = []
    while parent:
        # New axes of the child have no counterpart in the parent.
        while child and child[0] is None:
            new_parent = [*new_parent, None]
            child = child[1:]

        # Probably unreachable.
        if not child:
            new_parent += parent
            break

        p, *parent = parent
        i_parent += 1

        if isinstance(p, int):
            # Already dropped by the parent: only the original dim is consumed.
            sz0, *full_shape = full_shape
            new_parent.append(p)
            continue

        c, *child = child
        sz, *sub_shape = sub_shape

        if p is None:
            # The parent dimension is a new axis, of size 1.
            if isinstance(c, int):
                if c != 0:
                    oob(i_parent)
                continue
            if isinstance(c, slice):
                if slice_length(c, 1) == 0:
                    new_parent.append(oob_slice(newaxis=True))
                else:
                    new_parent.append(None)
                continue
            if isinstance(c, oob_slice):
                new_parent.append(oob_slice(newaxis=True))
            raise AssertionError(f"p is None and c is {c}")

        if isinstance(p, oob_slice):
            if not p.newaxis:
                sz0, *full_shape = full_shape
            # The parent dimension is empty.
            if isinstance(c, int):
                oob(i_parent)
                continue
            if isinstance(c, (slice, oob_slice)):
                new_parent.append(p)
                continue
            raise AssertionError(f"p is oob_slice(newaxis=True) and c is {c}")

        sz0, *full_shape = full_shape

        if isinstance(p, slice):
            if isinstance(c, int):
                # Absolute position, which depends on the sign of the step.
                if c < 0 or c >= sz:
                    oob(i_parent)
                if p.step is not None and p.step < 0:
                    start = sz0 - 1 if p.start is None else p.start
                    new_parent.append(start + c * p.step)
                else:
                    new_parent.append((p.start or 0) + c * (p.step or 1))
                continue
            if isinstance(c, slice):
                # Compose the two slices.
                length = slice_length(c, sz)
                if length == 0:
                    new_parent.append(oob_slice())
                    continue
                if c.step is not None and c.step < 0:
                    start = sz - 1 if c.start is None else c.start
                    step = c.step
                else:
                    start = 0 if c.start is None else c.start
                    step = 1 if c.step is None else c.step
                if p.step is not None and p.step < 0:
                    start0 = sz0 - 1 if p.start is None else p.start
                    step0 = p.step
                else:
                    start0 = 0 if p.start is None else p.start
                    step0 = 1 if p.step is None else p.step
                start = start0 + start * step0
                step = step0 * step
                stop = start + length * step
                if step < 0 and stop < 0:
                    # A negative stop would wrap around in simplify_slice.
                    stop = None
                new_slice = simplify_slice(
                    slice(start, stop, step), sz0, do_neg2pos=False
                )
                new_parent.append(new_slice)
                continue
            if isinstance(c, oob_slice):
                new_parent.append(c)
            raise AssertionError(f"p is slice and c is {c}")

    while child:
        c, *child = child
        if c is not None:
            raise IndexError("More indices than dimensions")
        new_parent.append(c)

    return tuple(new_parent)


def split_operation(
    perm: _tx.Sequence[int],
    slicer: _tx.Sequence[IndexLike],
    direction: _tx.Literal["r", "w"],
) -> tuple:
    """Split a permuted and sliced view into simpler operations.

    A symbolic view `sub` of an array `full` combines a permutation and an
    index, such that `sub.data()` equals:

    ```python
    >>> full_data = full.data()
    >>> sub_data = full_data.permute(sub.permutation)[sub.slicer]
    ```

    Because the index may add new axes or drop axes, the operation is split
    into three steps. When reading (`direction="r"`), `slicer_sub` is the
    unpermuted index without new axes, `perm` is the permutation without the
    dropped axes, and `slicer_add` inserts the new axes:

    ```python
    >>> dat = full[slicer_sub].transpose(perm)[slicer_add]
    ```

    When writing (`direction="w"`), `slicer_drop` removes the new axes and
    `perm` is the inverse of the permutation without the dropped axes:

    ```python
    >>> full[slicer_sub] = dat[slicer_drop].transpose(perm)
    ```

    Returns
    -------
    tuple of tuple
        `(slicer_sub, perm, slicer_add)` when reading, and
        `(slicer_drop, perm, slicer_sub)` when writing.

    Raises
    ------
    ValueError
        If `direction` does not start with `"r"` or `"w"`, in either case.
    """

    def remap(perm: _tx.Sequence[int]) -> list:
        """Renumber the dimensions to 0, ..., n - 1."""
        remaining_dims = sorted(perm)
        dim_map = {}
        for new, old in enumerate(remaining_dims):
            dim_map[old] = new
        remapped_dim = [dim_map[d] for d in perm]
        return remapped_dim

    def select(index: _tx.Sequence[IndexLike], seq: list) -> list:
        return [seq[idx] for idx in list(index)]

    slicer_nonew = list(filter(lambda x: not is_newaxis(x), slicer))
    slicer_nodrop = filter(lambda x: not is_droppedaxis(x), slicer)
    slicer_sub = select(invert_permutation(perm), slicer_nonew)
    perm_nodrop = [
        d for d, idx in zip(perm, slicer_nonew) if not is_droppedaxis(idx)
    ]
    perm_nodrop = remap(perm_nodrop)

    if direction.lower().startswith("r"):
        slicer_add = map(
            lambda x: slice(None) if x is not None else x, slicer_nodrop
        )
        return tuple(slicer_sub), tuple(perm_nodrop), tuple(slicer_add)
    elif direction.lower().startswith("w"):
        slicer_drop = map(
            lambda x: 0 if x is None else slice(None), slicer_nodrop
        )
        inv_perm_nodrop = invert_permutation(perm_nodrop)
        return tuple(slicer_drop), tuple(inv_perm_nodrop), tuple(slicer_sub)
    else:
        raise ValueError(
            f"direction should be in ('read' ,'write') but got {direction}."
        )


def invert_permutation(perm: _tx.Sequence[int]) -> _tx.List[int]:
    """Return the inverse of a permutation of `range(len(perm))`."""
    iperm = [0] * len(perm)
    for i, p in enumerate(perm):
        iperm[p] = i
    return iperm


def slicer_sub2ind(
    slicer: _tx.Sequence[_tx.Union[slice, int]], shape: _tx.Sequence[int]
) -> _tx.Union[slice, int, _tx.List[int]]:
    """Convert a multidimensional index into an index into the flat array.

    Contiguous dimensions are merged, so that the result is a single slice
    or integer whenever possible, and a list of linear indices otherwise. A
    `ValueError` is raised if `slicer` contains a negative step or a new axis.
    """

    slicer = expand_index(slicer, shape)
    shape_out = guess_shape(slicer, shape)
    if any(
        isinstance(idx, slice) and idx.step and idx.step < 0 for idx in slicer
    ):
        raise ValueError("sub2ind does not like negative strides")
    if any(is_newaxis(idx) for idx in slicer):
        raise ValueError("sub2ind does not like new axes")

    _slicer0 = slicer
    shape0 = shape

    # Merge full slices from the last (fastest) dimension onwards.
    slicer = list(reversed(slicer))
    shape = list(reversed(shape))
    new_slicer = slice(None)
    new_shape = 1
    while len(slicer) > 0:
        idx, *slicer = slicer
        shp, *shape = shape

        if isinstance(idx, slice):
            if idx == slice(None):
                new_shape *= shp
                continue
            else:
                if idx.step in (1, None):
                    start = idx.start or 0
                    stop = idx.stop or shp
                    new_slicer = slice(start * new_shape, stop * new_shape)
                    new_shape *= shp
                    new_slicer = simplify_slice(new_slicer, new_shape)
                    new_slicer = [new_slicer] + slicer
                    new_shape = [new_shape] + shape
                else:
                    if new_shape != 1:
                        new_slicer = [new_slicer, idx] + slicer
                        new_shape = [new_shape, shp] + shape
                    else:
                        new_slicer = [idx] + slicer
                        new_shape = [shp] + shape
                break

        elif isinstance(idx, int):
            if shp == 1:
                continue
            else:
                new_slicer = slice(idx * new_shape, (idx + 1) * new_shape)
                new_shape *= shp
                new_slicer = simplify_slice(new_slicer, new_shape)
                if new_shape != 1:
                    new_slicer = [new_slicer] + slicer
                    new_shape = [new_shape] + shape
                else:
                    new_slicer = [idx] + slicer
                    new_shape = [shp] + shape
                break

    new_slicer = list(new_slicer)
    new_shape = list(new_shape)

    assert math.prod(shape0) == math.prod(new_shape), (
        f"Oops: lost something: {math.prod(shape0)} vs {math.prod(new_shape)}"
    )

    # A single remaining index is returned as is.
    if len(new_slicer) == 1:
        return new_slicer[0]

    # Otherwise, list the linear indices explicitly.
    strides = [1] + list(itertools.accumulate(new_shape[1:], operator.mul))
    new_index = []
    for idx, shp, stride in zip(new_slicer, new_shape, strides):
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or shp
            step = idx.step or 1
            idx = list(range(start, stop, step))
        else:
            idx = [idx]
        idx = [i * stride for i in idx]
        if new_index:
            new_index = list(itertools.product(idx, new_index))
            new_index = [sum(idx) for idx in new_index]
        else:
            new_index = idx

    assert len(new_index) == math.prod(shape_out), (
        f"Oops: lost something: {len(new_index)} vs {math.prod(shape_out)}"
    )

    return new_index
