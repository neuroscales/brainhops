"""Reading of the arrays and axes of an OME-Zarr field node.

A field is stored in its own node. In OME-Zarr 0.6, that node carries `ome`
metadata that names typed axes, including the axis of vector components. An
older node is a bare array, which can only name its axes through its
`dimension_names`. The functions of this module read a node without deciding
its layout: they return the array as stored, and its axes as
[`Axis`][brainhops.datamodel.axes.Axis] objects, and callers reorder them as
they need.
"""

import typing_extensions as tx

from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.io.transformations.zarr._axes import _to_axis

# Axis types that carry the components of a field's vectors.
_VECTOR_TYPES = ("displacement", "coordinate")


def follow_path(node: tx.Any, path: str) -> tx.Any:
    """Follow a slash-separated path of subgroups from a node.

    The path is, for example, `"coordinateTransformations/dfield"`. Leading and
    trailing slashes are ignored.
    """
    current = node
    for segment in path.strip("/").split("/"):
        current = current[segment]
    return current


def coordinate_systems(field_node: tx.Any) -> tx.List[tx.Any]:
    """Return the coordinate systems declared in a node's OME metadata.

    The systems are those at the top level, followed by those of each
    multiscale. A node without OME metadata declares none.
    """
    try:
        ome = field_node.ome
    except Exception:
        return []
    if ome is None:
        return []
    systems = list(getattr(ome, "coordinateSystems", None) or [])
    for multiscale in getattr(ome, "multiscales", None) or []:
        systems.extend(getattr(multiscale, "coordinateSystems", None) or [])
    return systems


def typed_axes(field_node: tx.Any, ndim: int) -> tx.Optional[tx.List[Axis]]:
    """Return a field node's axes, in the stored order of its array.

    The axes are read from a coordinate system of the typed OME metadata that
    has one axis per dimension of the array. A system that names a displacement
    or coordinate axis is preferred, since that axis locates the vector
    components. `None` is returned when no system matches.
    """
    fallback = None  # type: tx.Optional[tx.List[Axis]]
    for system in coordinate_systems(field_node):
        axes = [_to_axis(axis.to_json()) for axis in system.axes]
        if len(axes) != ndim:
            continue
        if any(axis.type in _VECTOR_TYPES for axis in axes):
            return axes
        if fallback is None:
            fallback = axes
    return fallback


def dimension_names(field_node: tx.Any) -> tx.Optional[tx.List[str]]:
    """Return the dimension names, or `None` if any is missing."""
    metadata = getattr(field_node, "metadata", None)
    names = getattr(metadata, "dimension_names", None)
    if not names or None in names or len(names) != field_node.ndim:
        return None
    return list(names)


def read_array(field_node: tx.Any) -> tx.Any:
    """Read the field's array through the active array backend."""
    return get_array_backend().asarray(field_node[...])
