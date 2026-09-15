"""The axis-order seam between brainhops and OME-Zarr.

All axis-order reconciliation between the brainhops data model and an
OME-Zarr store passes through this module, and through nowhere else. A
brainhops object uses nibabel's F-order: the spatial axes first, then
time, then the channel axis. An OME-Zarr store uses C-order, most often
``(t, c, z, y, x)``. The two are reconciled by a permutation computed over
the full ordered axis list, not the spatial axes alone.

The component axis of a field, whose values are the components of a
displacement or a coordinate vector, is not a separate trailing role. It
occupies the same position as the channel axis, in the brainhops order and
in the store. This matches how nibabel and OME-Zarr both store a field:
the displacement components share the dimension a channel would use.

[to_canonical][brainhops.io.images.zarr._axisorder.to_canonical] permutes
a store's axes into the brainhops order, and is used when reading.
[to_storage][brainhops.io.images.zarr._axisorder.to_storage] permutes
brainhops axes into the store order, and is used when writing. Each of the
two orders sorts the axes by the same axis-type ranking, and one order
reverses the other, so a value written and read back is returned unchanged.

The ordering keys on the axis *type*, never on the axis name. OME RFC-3
allows further, arbitrarily named axes, so a name is consulted only to
break ties between spatial axes of the same type.
"""

# dependencies
import typing_extensions as tx

#: The brainhops canonical axis order, most significant group first. The
#: spatial axes lead, then time, then the channel group.
CANONICAL_ORDER = ("space", "time", "channel", "other")

#: The OME-Zarr storage axis order, most significant group first. The
#: non-spatial axes lead and the spatial axes trail, giving ``(t, c, z, y,
#: x)`` for a plain image.
STORAGE_ORDER = ("time", "channel", "other", "space")

#: The axis types whose axis is a vector component axis. A field stores its
#: displacement or coordinate components along such an axis, in the
#: position a channel axis would otherwise occupy.
_VECTOR_TYPES = ("displacement", "coordinate")

#: The canonical position of a named spatial axis, used only to break ties
#: between spatial axes, so they sort into x, y, z.
_CANONICAL_SPACE = {"x": 0, "y": 1, "z": 2}

#: The storage position of a named spatial axis, used only to break ties
#: between spatial axes, so they sort into z, y, x.
_STORAGE_SPACE = {"z": 0, "y": 1, "x": 2}


def _group(axis: tx.Any) -> str:
    """The role group of an axis, one of the names in the order tuples.

    A displacement or coordinate component axis is grouped with the
    channel axis, since a field stores its components where a channel
    would sit.
    """
    type_ = getattr(axis, "type", None)
    if type_ in _VECTOR_TYPES or type_ == "channel":
        return "channel"
    if type_ in ("space", "time"):
        return type_
    return "other"


def is_vector(axis: tx.Any) -> bool:
    """Whether `axis` is a displacement or coordinate component axis."""
    return getattr(axis, "type", None) in _VECTOR_TYPES


def _space_name_rank(axis: tx.Any, table: tx.Mapping[str, int]) -> int:
    # The within-space position of a spatial axis, read from its name. An
    # unnamed spatial axis, or one whose name is not in the table, sorts
    # after the named ones and keeps its original order. The name only ever
    # breaks a tie between spatial axes; it never decides the group.
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


def vector_flip(
    axes: tx.Sequence[tx.Any], perm: tx.Sequence[int]
) -> tx.Optional[tx.Tuple[int, tx.List[int]]]:
    """Return how to reindex the values along a field's component axis.

    When `axes` includes a displacement or coordinate component axis, the
    values along that axis are per-spatial-axis components. Reordering the
    spatial axes by `perm` must reorder those components in lockstep, so a
    component keeps referring to the spatial axis it belongs to.

    The result is ``(position, component_perm)`` when `axes` has one
    component axis. `position` is the component axis's index after `perm`
    is applied. `component_perm` reorders the values along it, so a caller
    applies it with a backend ``take`` along `position`. The result is
    `None` when `axes` has no component axis, and also when the number of
    components does not match the number of spatial axes, in which case the
    correspondence is unknown and the values are left untouched.
    """
    vectors = [i for i, axis in enumerate(axes) if is_vector(axis)]
    if len(vectors) != 1:
        return None
    v = vectors[0]
    position = list(perm).index(v)

    spatial = [i for i, axis in enumerate(axes) if _group(axes[i]) == "space"]
    spatial_out = [p for p in perm if _group(axes[p]) == "space"]
    if len(spatial) != len(spatial_out):
        return None
    component_perm = [spatial.index(p) for p in spatial_out]
    return position, component_perm
