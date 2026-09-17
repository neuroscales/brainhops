"""Read an OME-Zarr field node into arrays and axes.

An OME-Zarr coordinate or displacement field is stored in its own node. A
0.6 field node is a full OME-Zarr node whose ``ome`` metadata names the
field's typed axes, including the axis that holds the vector components. An
older field node is a bare array that names its axes only with
``dimension_names``.

The functions here read such a node without deciding how the array is laid
out. The array is returned exactly as stored, together with the field's
axes as brainhops [`Axis`][brainhops.datamodel.axes.Axis] objects. Each
caller then orders the array as it needs. The image reader lays the field
out in the brainhops order, and the multiscale field reader keeps the
stored order.
"""

# dependencies
import typing_extensions as tx

# backends
from brainhops.backends import get_array_backend

# internals
from brainhops.datamodel.axes import Axis
from brainhops.io.transformations.zarr._axes import _to_axis

#: The axis types whose axis carries the components of a field's vectors.
_VECTOR_TYPES = ("displacement", "coordinate")


def follow_path(node: tx.Any, path: str) -> tx.Any:
    """Return the node reached by following `path` from `node`.

    The path may descend through subgroups, so
    ``"coordinateTransformations/dfield"`` opens the ``dfield`` node inside
    the ``coordinateTransformations`` subgroup. A leading or trailing
    slash is ignored.
    """
    current = node
    for segment in path.strip("/").split("/"):
        current = current[segment]
    return current


def coordinate_systems(field_node: tx.Any) -> tx.List[tx.Any]:
    """Return the coordinate systems a node declares in its OME metadata.

    Both the top-level systems and those each multiscale carries are
    collected. A node with no OME metadata contributes no systems.
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
    """Return the field node's own axes, read from its typed OME metadata.

    The axes are returned as brainhops [`Axis`][brainhops.datamodel.axes.Axis]
    objects, in the field array's stored dimension order. The coordinate
    system that describes the field array is the one with one axis per array
    dimension. A system that names a displacement or coordinate component
    axis is preferred, since that axis identifies where the vector
    components are stored. The result is `None` when the node declares no
    such system.
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
    """Return a field node's dimension names, or `None`.

    The result is the list of names when the node names every dimension,
    and `None` when the node names no dimensions or leaves one unnamed.
    """
    metadata = getattr(field_node, "metadata", None)
    names = getattr(metadata, "dimension_names", None)
    if not names or None in names or len(names) != field_node.ndim:
        return None
    return list(names)


def read_array(field_node: tx.Any) -> tx.Any:
    """Return the field node's array through the active array backend."""
    return get_array_backend().asarray(field_node[...])
