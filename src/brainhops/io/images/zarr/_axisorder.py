"""The axis-order seam between brainhops and OME-Zarr.

All axis-order reconciliation between the brainhops data model and an
OME-Zarr store passes through this module, and through nowhere else. A
brainhops object uses nibabel's F-order: the spatial axes ``x``, ``y``,
``z`` first, then time, then channel, and the vector component axis of a
field last. An OME-Zarr store uses C-order, most often
``(t, c, z, y, x)``. The two are reconciled by a permutation computed over
the full ordered axis list, not the spatial axes alone.

[to_canonical][brainhops.io.images.zarr._axisorder.to_canonical] permutes
a store's axes into the brainhops order, and is used when reading.
[to_storage][brainhops.io.images.zarr._axisorder.to_storage] permutes
brainhops axes into the store order, and is used when writing. Each of the
two orders sorts the axes by the same axis-role ranking, and one order
reverses the other, so a value written and read back is returned unchanged.
"""

# dependencies
import typing_extensions as tx

#: The brainhops canonical axis order, most significant group first. The
#: vector component axis of a field is always placed last.
CANONICAL_ORDER = ("space", "time", "channel", "other", "vector")

#: The OME-Zarr storage axis order, most significant group first.
STORAGE_ORDER = ("time", "channel", "other", "space", "vector")

#: The vector-field component axis types, which are always placed last.
_VECTOR_TYPES = ("displacement", "coordinate")

#: The canonical position of a named spatial axis, so the spatial axes
#: sort into x, y, z.
_CANONICAL_SPACE = {"x": 0, "y": 1, "z": 2}

#: The storage position of a named spatial axis, so the spatial axes sort
#: into z, y, x.
_STORAGE_SPACE = {"z": 0, "y": 1, "x": 2}


def _group(axis: tx.Any) -> str:
    """The role group of an axis, one of the names in the order tuples."""
    type_ = getattr(axis, "type", None)
    if type_ in _VECTOR_TYPES:
        return "vector"
    if type_ in ("space", "time", "channel"):
        return type_
    return "other"


def _space_name_rank(axis: tx.Any, table: tx.Mapping[str, int]) -> int:
    # The within-space position of a spatial axis, read from its name. An
    # unnamed spatial axis, or one whose name is not x, y or z, sorts after
    # the named ones and keeps its original order.
    name = getattr(axis, "name", None)
    if isinstance(name, str) and name in table:
        return table[name]
    return len(table)


def _permutation(
    axes: tx.Sequence[tx.Any],
    order: tx.Sequence[str],
    space: tx.Mapping[str, int],
) -> tx.List[int]:
    # The permutation that sorts `axes` into `order`, breaking ties by the
    # within-space name table and then by original position, so the sort is
    # stable and the two directions invert each other.
    def key(item: tx.Tuple[int, tx.Any]) -> tx.Tuple[int, int, int]:
        position, axis = item
        group = _group(axis)
        return (order.index(group), _space_name_rank(axis, space), position)

    ranked = sorted(enumerate(axes), key=key)
    return [position for position, _ in ranked]


def to_canonical(axes: tx.Sequence[tx.Any]) -> tx.List[int]:
    """Return the permutation that reorders store `axes` into brainhops order.

    The result is a list of indices. Element ``i`` is the position, in the
    stored order, of the axis that becomes axis ``i`` in the brainhops
    order. The same permutation reorders the array, the axis list, and any
    per-axis vector such as a scale or a translation.
    """
    return _permutation(axes, CANONICAL_ORDER, _CANONICAL_SPACE)


def to_storage(axes: tx.Sequence[tx.Any]) -> tx.List[int]:
    """Return the permutation that reorders brainhops `axes` into store order.

    The result is a list of indices. Element ``i`` is the position, in the
    brainhops order, of the axis that becomes axis ``i`` in the stored
    order. This permutation reverses the reordering that
    [to_canonical][brainhops.io.images.zarr._axisorder.to_canonical]
    applies when reading.
    """
    return _permutation(axes, STORAGE_ORDER, _STORAGE_SPACE)


def permute(
    seq: tx.Sequence[tx.Any], perm: tx.Sequence[int]
) -> tx.List[tx.Any]:
    """Return the elements of `seq` in the order given by `perm`."""
    return [seq[p] for p in perm]
