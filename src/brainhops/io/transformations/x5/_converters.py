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

A field is converted as every field format converts it (see
[`displacement_parts`][brainhops.io.transformations.base._conversions.displacement_parts]):
a field interpolated with cubic B-splines is stored as its
coefficients, which X5 reads back with cubic B-splines, and any other
field as its values at the nodes of its grid, which are read back with
linear interpolation. A velocity is integrated, and the affines around
a field are carried into it. A field interpolated with a degree other
than 1 or 3 is evaluated at its nodes, which is a loss: the converter
raises a `LossyConversionError`, and `t.to(X5Transform, lossy=True)`
returns the approximation. How a field is extended beyond its grid,
which X5 does with a zero boundary, is not counted as a loss.

The conversion takes `metadata=`, and without it the metadata of `t` is
converted into the metadata of X5. An `X5Transform` that is converted
to its own class is changed by the rules of any chain, which keep its
type.
"""

import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.datamodel.enums import StoreEnum
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    FIELD_SOURCES,
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
    if int(field.degree) == 3 and field.data is not None:
        # A cubic B-spline is kept as one, with its coefficients, and a
        # cubic velocity is integrated into one. The affines around the
        # field are carried into its coefficients.
        parts = displacement_parts(piece, RAS, cls, degree=3)
        element = X5BSplineField(
            transformations=(
                RASToVoxel(matrix=parts.world2grid[:-1]),
                parts.field.to(store=StoreEnum.coefficients),
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
