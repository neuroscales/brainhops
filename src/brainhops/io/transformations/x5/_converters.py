"""
Private converters into the X5 format.

`t.to(X5Transform)`, `X5Transform.from_any(t)` and
`io.save(t, "warp.x5")` all run this converter. It returns a chain of X5
transforms that is the map `t` is, raises a `LossyConversionError` that
carries an approximation of it, or raises a `ConversionError` that says
what X5 cannot hold.

| Source                                     | Format         |
| ------------------------------------------ | -------------- |
| any affine, any field, any `Sequence`      | `X5Transform`  |

Every transform of an X5 file maps the RAS world to the RAS world, so
the endpoints of `t` are bridged exactly to RAS, and an endpoint that
is not known is taken to be RAS. The chain is then cut into pieces that
each map RAS to RAS, as
[`world_pieces`][brainhops.io.transformations.base._conversions.world_pieces]
cuts it, and each piece becomes one transform of the file:

| Piece                                        | X5 transform               |
| -------------------------------------------- | -------------------------- |
| affines                                      | `linear`, an `Affine`      |
| a field of displacements                     | [`X5DisplacementField`][]  |
| a field of displacements of cubic B-splines  | [`X5BSplineField`][]       |
| a field of coordinates                       | [`X5CoordinatesField`][]   |

A field of displacements is stored as its values at the nodes of its
grid, which are read back with linear interpolation, and a velocity is
integrated. A field interpolated with cubic B-splines is stored as its
coefficients, which X5 reads back with cubic B-splines and a zero
boundary. A field interpolated with another degree, or with cubic
B-splines and another boundary, which X5 refits with a zero boundary,
is not the same map near the edges or between the nodes, which is a
loss: the converter raises a `LossyConversionError`, and
`t.to(X5Transform, lossy=True)` returns the approximation.

The conversion takes `metadata=`, and without it the metadata of `t` is
converted into the metadata of X5. An `X5Transform` that is converted
to its own class is changed by the rules of any chain, which keep its
type.
"""

import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    FIELD_SOURCES,
    FieldParts,
    coordinates_parts,
    displacement_parts,
    format_options,
    held,
    register_converter,
    split_field_chain,
    world_pieces,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import RASCoordinatesField

from ._blocks import X5BSplineField, X5CoordinatesField, X5DisplacementField
from ._xform import X5Transform

RAS = _systems.RASmm()
"""The RAS world, in millimetres, that every X5 transform maps."""

VOXEL = _systems.VoxelCoordinateSystem()
"""The voxels of the grid of a field."""

SOURCES = tuple(dict.fromkeys((*AFFINE_SOURCES, *FIELD_SOURCES)))
"""The families that the converter takes: every affine and every field."""


def _(
    t: _xforms.Transformation, cls: tx.Type[X5Transform], **kwargs
) -> X5Transform:
    # One X5 transform per piece of the chain that maps RAS to RAS. What
    # loses is reported once, for the whole chain.
    cls = X5Transform
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    options = format_options(t, cls, kwargs, ("metadata",))
    # The metadata of `t` rides along, converted by the field into the
    # metadata of X5 without the record of another format.
    options.setdefault("metadata", t.metadata)
    elements = []
    losses = []
    lossy = None
    for piece in world_pieces(t, RAS, cls):
        element, parts = _element(piece, cls)
        elements.append(element)
        if parts is not None and parts.loss is not None:
            losses.append(parts.loss)
            lossy = parts
    result = cls(transformations=elements, **options)
    if lossy is None:
        return result
    return held(result, lossy._replace(loss=" ".join(losses)))


register_converter(SOURCES, X5Transform)(_)


def _element(piece: _xforms.Transformation, cls: type) -> tx.Tuple:
    """
    Return the X5 transform of a piece, and the parts it was built from.

    The parts are `None` for an affine, which is held exactly.
    """
    if isinstance(piece, _xforms.Affine):
        affine = _xforms.Affine(matrix=piece.matrix, input=RAS, output=RAS)
        return affine, None
    before, field, _ = split_field_chain(piece, RAS, RAS, cls)
    if isinstance(field, _xforms.CoordinatesField):
        parts = coordinates_parts(piece, RAS, RAS, cls)
        element = X5CoordinatesField(
            (
                RASToVoxel(matrix=parts.world2grid[:-1]),
                RASCoordinatesField(field=parts.field.values),
            )
        )
        return element, parts
    if int(field.degree) == 3 and not field.log and field.data is not None:
        # A cubic B-spline is kept as one, and the writer stores it as
        # coefficients with a zero boundary. A piece of displacements is
        # closed by the inverse of its first affine.
        loss = None
        if field.bound != BoundaryCondition.zeros:
            loss = (
                f"its cubic B-spline has the boundary {field.bound!s}, and "
                f"X5 refits it with a zero boundary, which changes the map "
                f"near the edges of its grid."
            )
        parts = FieldParts(before, field, np.linalg.inv(before), loss)
        element = X5BSplineField(
            transformations=(
                RASToVoxel(matrix=parts.world2grid[:-1]),
                # X5 stores the coefficients of the B-spline, so values are
                # turned into the coefficients that interpolate them.
                replace(field, input=VOXEL, output=VOXEL).to(
                    store=StoreEnum.coefficients
                ),
                VoxelToRAS(matrix=parts.grid2world[:-1]),
            )
        )
        return element, parts
    parts = displacement_parts(piece, RAS, cls)
    element = X5DisplacementField(
        transformations=(
            RASToVoxel(matrix=parts.world2grid[:-1]),
            parts.field,
            VoxelToRAS(matrix=parts.grid2world[:-1]),
        )
    )
    return element, parts
