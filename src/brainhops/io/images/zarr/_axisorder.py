"""The axis order of OME-Zarr images, and its conversion to the brainhops
order.

Every reconciliation of axis orders between brainhops and OME-Zarr goes through
this module. brainhops follows the F-ordered convention of nibabel, with the
spatial axes first, then time, channels and any other axes. An OME-Zarr store
is C-ordered, most often `(t, c, z, y, x)`. The two orders are reconciled by a
permutation of the full list of axes, not of the spatial axes alone:
[`to_canonical`][] reorders the stored axes when reading, and [`to_storage`][]
reorders the brainhops axes when writing.

Both orders sort axes by their type, never by their name, since OME RFC-3
allows extra axes with arbitrary names. A name only breaks ties between spatial
axes, which sort as x, y, z in brainhops and as z, y, x in the store. Other
ties keep the original order, so that writing and then reading restores the
brainhops order.

The component axis of a field, which holds the vector components of a
displacement or coordinate field, is not a separate trailing axis. It takes the
position of the channel axis, both in brainhops and in the store, as in nibabel
and OME-Zarr. Permuting the axes does not reorder the component values,
because the components of a field are expressed in its output coordinate
system, which this module leaves untouched.
"""

import typing_extensions as tx

from brainhops.datamodel.axes import Axis

_T = tx.TypeVar("_T")

# The groups of axes in the brainhops order, from the first to the last.
CANONICAL_ORDER = ("space", "time", "channel", "other")

# In the stored order, the non-spatial axes come first and the spatial axes
# last, which gives (t, c, z, y, x) for a plain image.
STORAGE_ORDER = ("time", "channel", "other", "space")

# These axis types hold the vector components of a field, which the field
# stores where a channel axis would sit.
_VECTOR_TYPES = ("displacement", "coordinate")

# These ranks break ties between named spatial axes to sort them as x, y, z.
_CANONICAL_SPACE = {"x": 0, "y": 1, "z": 2}

# These ranks break ties between named spatial axes to sort them as z, y, x.
_STORAGE_SPACE = {"z": 0, "y": 1, "x": 2}


def _group(axis: Axis) -> str:
    """Return the group of an axis, as named in `CANONICAL_ORDER` and
    `STORAGE_ORDER`.

    Component axes of displacement and coordinate fields are grouped with
    channels, since a field stores its components where a channel would sit.
    """
    type_ = getattr(axis, "type", None)
    if type_ in _VECTOR_TYPES or type_ == "channel":
        return "channel"
    if type_ in ("space", "time"):
        return type_
    return "other"


def _space_name_rank(axis: Axis, table: tx.Mapping[str, int]) -> int:
    # The name only breaks ties within a group. Unnamed spatial axes, and those
    # with unknown names, sort after the named ones, in their original order.
    name = getattr(axis, "name", None)
    if isinstance(name, str) and name in table:
        return table[name]
    return len(table)


def _permutation(
    axes: tx.Sequence[Axis],
    order: tx.Sequence[str],
    space: tx.Mapping[str, int],
) -> tx.List[int]:
    # Ties are broken by the name table and then by the original position, so
    # that the sort is stable.
    def key(item: tx.Tuple[int, Axis]) -> tx.Tuple[int, int, int]:
        position, axis = item
        group = _group(axis)
        return (order.index(group), _space_name_rank(axis, space), position)

    ranked = sorted(enumerate(axes), key=key)
    return [position for position, _ in ranked]


def to_canonical(axes: tx.Sequence[Axis]) -> tx.List[int]:
    """Return the permutation that reorders stored axes into the brainhops
    order.

    Element `i` is the position, in the stored order, of the axis that becomes
    axis `i` in the brainhops order. The same permutation reorders the array,
    the list of axes and per-axis vectors such as scales and translations.
    """
    return _permutation(axes, CANONICAL_ORDER, _CANONICAL_SPACE)


def to_storage(axes: tx.Sequence[Axis]) -> tx.List[int]:
    """Return the permutation that reorders brainhops axes into the stored
    order.

    Element `i` is the position, in the brainhops order, of the axis that
    becomes axis `i` in the stored order. Reading the result back with
    [`to_canonical`][] restores the brainhops order.
    """
    return _permutation(axes, STORAGE_ORDER, _STORAGE_SPACE)


def permute(seq: tx.Sequence[_T], perm: tx.Sequence[int]) -> tx.List[_T]:
    """Return the elements of `seq` in the order given by `perm`."""
    return [seq[p] for p in perm]
