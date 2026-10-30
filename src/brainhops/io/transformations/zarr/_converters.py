"""
Private converters into the OME-Zarr field format.

`t.to(OmeZarrField)`, `OmeZarrField.from_any(t)` and
`io.save(t, "warp.ome.zarr")` all run this converter. It returns a field
that is the map `t` is, raises a `LossyConversionError` that carries an
approximation of it, or raises a `ConversionError` that says what the
format cannot hold.

| Source                             | Format          |
| ---------------------------------- | --------------- |
| any field, `Sequence`              | `OmeZarrField`  |

OME-Zarr does not name the world of a field, and a field read from a
store has no known endpoints. The converter writes the field in the RAS
world, in millimetres, as NIfTI does: the endpoints of `t` are bridged
exactly to RAS, and an endpoint that is not known is taken to be RAS.
A field of coordinates is stored as coordinates and any other field as
displacements, in millimetres, so that the kind of the field is kept.
The affines around a field of displacements are carried into it when
they do not undo each other, and a velocity is integrated.

The field is written as one level, as OME-NGFF 0.6 lays out a field:
an array whose first axis holds the components of the vectors (an axis
of the type `displacement` or `coordinate`), followed by the spatial
axes `x`, `y` and `z`, which are the axes of the RAS world. The level is
placed by a scale and a translation, the only placement that a dataset
of OME-NGFF 0.6 holds. A grid whose axes run along the world axes in
another order or direction is placed so by reordering and flipping the
axes of the array, which is exact and reads nothing. A field on an
oblique grid is sampled on the nearest grid along the world axes, and a
field interpolated with another degree than 1 is sampled at its nodes.
Both are losses: the converter raises a `LossyConversionError`, and
`t.to(OmeZarrField, on_loss="ignore")` returns the sampled field. The
arrays are built with the operations of their backend, so a Dask array
stays lazy until the field is written.

A field that is already of this format is changed by the rules of its
family, which keep its type.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.backends import array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import converter, smart_replace
from brainhops.datamodel.transformations import Affine, Identity

# io
from brainhops.io.transformations.base._internal_api import (
    FIELD_SOURCES,
    FieldParts,
    cardinal_parts,
    coordinate_parts,
    displacement_parts,
    displacements_of,
    format_options,
    held,
    register_converter,
    split_field_chain,
)

# this format
from ._axes import to_axis
from ._xforms import OmeZarrField, dataset_placement, field_ome

RAS = _systems.RASmm()
"""The world that the converter writes a field in."""

_SPACE = ("x", "y", "z")
"""The names of the spatial axes, which are those of the RAS world."""


@register_converter(FIELD_SOURCES, OmeZarrField)
def _(
    t: _xforms.Transformation, cls: tx.Type[OmeZarrField], **kwargs
) -> OmeZarrField:
    # One level of coordinates or of displacements in RAS millimetres, on
    # a grid along the world axes. What loses is reported.
    cls = OmeZarrField
    format_options(t, cls, kwargs)
    _, field, _ = split_field_chain(t, RAS, RAS, cls)
    if isinstance(field, _xforms.CoordinateField):
        kind = "coordinate"
        parts = coordinate_parts(t, RAS, RAS, cls)
        parts = cardinal_parts(parts, t, RAS, RAS, cls)
        values = parts.field.values
    else:
        kind = "displacement"
        parts = displacement_parts(t, RAS, cls)
        parts = displacements_of(cardinal_parts(parts, t, RAS, RAS, cls))
        values = _world_vectors(parts)
    values, scale, translation = _along_world(values, parts.grid2world)
    return held(_field(values, kind, scale, translation), parts)


@converter
def _(t: OmeZarrField, cls: tx.Type[OmeZarrField], **kwargs) -> OmeZarrField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   BUILDING THE FIELD
# ----------------------------------------------------------------------


def _world_vectors(parts: FieldParts) -> tx.Any:
    """Return the displacements of parts in millimetres of the world.

    The displacements of the parts are in the voxels of their grid, and
    only the linear part of the grid-to-world affine acts on them.
    """
    values = parts.field.values
    backend = array_backend(values)
    ndim = len(parts.grid2world) - 1
    linear = parts.grid2world[:ndim, :ndim].astype(values.dtype)
    return backend.matmul(backend.asarray(linear), values[..., None])[..., 0]


def _along_world(
    values: tx.Any, grid2world: np.ndarray
) -> tx.Tuple[tx.Any, np.ndarray, np.ndarray]:
    """
    Reorder and flip the axes of a field so that they run along the world.

    The grid is cardinal: each of its axes runs along one world axis,
    in either direction. Axis `k` of the result runs along world axis `k`
    in its positive direction, so that the grid is placed by a positive
    scale and a translation. The vectors themselves do not change, since
    they are given in the world.

    Returns
    -------
    values : array, shape `(*shape, D)`
        The field on the reordered grid.
    scale : array, shape `(D,)`
        The voxel size along each world axis.
    translation : array, shape `(D,)`
        The world position of the first node.
    """
    backend = array_backend(values)
    ndim = len(grid2world) - 1
    linear = grid2world[:ndim, :ndim]
    shape = [int(n) for n in values.shape[:-1]]
    order = [int(np.argmax(np.abs(linear[k]))) for k in range(ndim)]
    scale = np.array([linear[k, order[k]] for k in range(ndim)])
    translation = np.array(grid2world[:ndim, ndim], dtype=np.float64)
    flips = []
    for k in range(ndim):
        if scale[k] < 0:
            translation[k] += scale[k] * (shape[order[k]] - 1)
            flips.append(order[k])
    if flips:
        index = tuple(
            slice(None, None, -1) if axis in flips else slice(None)
            for axis in range(ndim + 1)
        )
        values = values[index]
    if order != list(range(ndim)):
        values = backend.transpose(values, (*order, ndim))
    return values, np.abs(scale), translation


def _field(
    values: tx.Any,
    kind: str,
    scale: np.ndarray,
    translation: np.ndarray,
) -> OmeZarrField:
    """
    Build the OME-Zarr field of one level, with its OME metadata.

    The level is stored with the axis of the vectors first, as OME-NGFF
    0.6 lays out a field, and the field is built as
    [`OmeZarrField.from_node`][brainhops.io.transformations.zarr.OmeZarrField.from_node]
    builds it from the metadata.
    """
    backend = array_backend(values)
    stored = backend.moveaxis(values, -1, 0)
    ndim = len(scale)
    axes = [{"name": "c", "type": kind, "discrete": True}] + [
        {"name": name, "type": "space", "unit": "millimeter"}
        for name in _SPACE[:ndim]
    ]
    matrix = np.concatenate([np.diag(scale), translation[:, None]], axis=1)
    ome = field_ome([to_axis(axis) for axis in axes], [Affine(matrix=matrix)])
    dataset = ome.multiscales[0].datasets[0]
    spatial = list(range(1, ndim + 1))
    voxel2world = dataset_placement(dataset, spatial, ndim + 1)
    return OmeZarrField(
        raw_levels=[stored],
        voxel2world=voxel2world,
        level_transforms=[Identity()],
        axes=[to_axis(axis) for axis in axes],
        ome=ome,
        input=RAS,
        output=RAS,
    )
