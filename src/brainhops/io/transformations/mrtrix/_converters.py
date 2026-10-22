"""
Private converters into the MRtrix transformation formats.

`t.to(MrtrixCoordinatesField)`, `MrtrixCoordinatesField.from_any(t)` and
`io.save(t, "warp.mif")` all run these converters. Each one returns an
instance of its format that is the map `t` is, raises a
`LossyConversionError` that carries an approximation of it, or raises a
`ConversionError` that says what the format cannot hold.

| Source                             | Format                       |
| ---------------------------------- | ---------------------------- |
| any affine, `Sequence` of affines  | `MrtrixLinearTransform`      |
| any field, `Sequence`              | `MrtrixCoordinatesField`     |
| any field, `Sequence`              | `MrtrixDisplacementField`    |
| `Sequence`                         | `MrtrixWarpFull`, refused    |

In this table, "any affine" means every family that has an affine
form, which is `Identity`, `Translation`, `Scaling`, `Permutation`,
`Linear` and `Rotation`, `Affine`, their tangents, and
`SubspaceTransformation`. "Any field" means every family of fields. Each
family is registered with its own converter, so that its endpoints are
bridged to scanner RAS, which every MRtrix transformation maps from and
to. An endpoint that is not known is taken to be scanner RAS.

A field is stored as its values at the nodes of its grid, which MRtrix
reads back with linear interpolation, and which brainhops extends with
the nearest value outside the grid. A field of coordinates is written
to a displacement file as the displacement of each node, and a field of
displacements to a deformation file as the position of each node, since
the two are the same map inside the grid and only extend differently
beyond it. A velocity is integrated. The
affines on either side of a field of displacements are carried into
its values when they do not undo each other. A field interpolated with
another degree than 1 is sampled at its nodes, which is a loss: the
converter raises a `LossyConversionError`, and
`t.to(Format, lossy=True)` returns the sampled field.

A warpfull file holds four midway-space warps and two linear
transforms, and no chain can be split back into them, so its converter
refuses every chain. The file is built with `MrtrixWarpFull.from_warps`
instead.

A conversion takes no option, since every option would change the map.
An object that is already in its format is changed by the rules of its
family, which keep its type.
"""

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    smart_replace,
)

# io
from brainhops.io.transformations.base._conversions import (
    FIELD_SOURCES,
    affine_between,
    coordinates_parts,
    displacement_parts,
    format_options,
    held,
    register_converter,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import RASCoordinatesField

# this format
from ._linear import MrtrixLinearTransform
from ._warps import (
    MrtrixCoordinatesField,
    MrtrixDisplacementField,
    MrtrixWarpFull,
)

RAS = _systems.RASmm()
"""The scanner RAS world, in millimetres, that MRtrix maps from and to."""


# ----------------------------------------------------------------------
#   LINEAR TRANSFORMS
# ----------------------------------------------------------------------


@converter(_xforms.Identity, MrtrixLinearTransform)
@converter(_xforms.Translation, MrtrixLinearTransform)
@converter(_xforms.Scaling, MrtrixLinearTransform)
@converter(_xforms.Permutation, MrtrixLinearTransform)
@converter(_xforms.Linear, MrtrixLinearTransform)
@converter(_xforms.AffineExponential, MrtrixLinearTransform)
@converter(_xforms.SubspaceTransformation, MrtrixLinearTransform)
@converter(_xforms.Sequence, MrtrixLinearTransform)
@converter
def _(
    t: _xforms.Affine, cls: tx.Type[MrtrixLinearTransform], **kwargs
) -> MrtrixLinearTransform:
    # Exactly, or raise: `affine_between` bridges the endpoints of `t` to
    # scanner RAS, which flips an affine between LPS spaces, and reduces a
    # chain to one affine. An affine whose systems are not known is taken
    # to map scanner RAS to scanner RAS.
    cls = MrtrixLinearTransform
    format_options(t, cls, kwargs)
    matrix = affine_between(t, RAS, RAS, cls)
    return cls(matrix=matrix[:-1])


@converter
def _(
    t: MrtrixLinearTransform,
    cls: tx.Type[MrtrixLinearTransform],
    **kwargs,
) -> MrtrixLinearTransform:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return _convert_withlog(t, cls, "matrix", **kwargs)


# ----------------------------------------------------------------------
#   DEFORMATIONS AND DISPLACEMENTS
# ----------------------------------------------------------------------


def _(
    t: _xforms.Transformation,
    cls: tx.Type[MrtrixCoordinatesField],
    **kwargs,
) -> MrtrixCoordinatesField:
    # An affine from scanner RAS to a grid, then the scanner RAS position
    # of each node of that grid. A field of displacements becomes the
    # position of each node, a velocity is integrated, and a field that is
    # not linear is sampled at its nodes, which loses.
    cls = MrtrixCoordinatesField
    format_options(t, cls, kwargs)
    parts = coordinates_parts(t, RAS, RAS, cls)
    chain = (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        RASCoordinatesField(field=parts.field.values),
    )
    return held(cls(transformations=chain), parts)


register_converter(FIELD_SOURCES, MrtrixCoordinatesField)(_)


def _(
    t: _xforms.Transformation,
    cls: tx.Type[MrtrixDisplacementField],
    **kwargs,
) -> MrtrixDisplacementField:
    # One field of displacements, between a world-to-grid affine and its
    # inverse. A field of coordinates becomes the displacement of each
    # node, a velocity is integrated, the affines around the field are
    # carried into it when they do not undo each other, and a field that
    # is not linear is sampled at its nodes, which loses.
    cls = MrtrixDisplacementField
    format_options(t, cls, kwargs)
    parts = displacement_parts(t, RAS, cls)
    chain = (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        parts.field,
        VoxelToRAS(matrix=parts.grid2world[:-1]),
    )
    return held(cls(transformations=chain), parts)


register_converter(FIELD_SOURCES, MrtrixDisplacementField)(_)


@converter
def _(
    t: MrtrixCoordinatesField,
    cls: tx.Type[MrtrixCoordinatesField],
    **kwargs,
) -> MrtrixCoordinatesField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: MrtrixDisplacementField,
    cls: tx.Type[MrtrixDisplacementField],
    **kwargs,
) -> MrtrixDisplacementField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   WARPFULL
# ----------------------------------------------------------------------


@converter
def _(
    t: _xforms.Sequence, cls: tx.Type[MrtrixWarpFull], **kwargs
) -> MrtrixWarpFull:
    # A warpfull file holds the pieces of a registration, which a chain
    # cannot be split back into, so no chain is held.
    raise unrepresentable(
        t,
        MrtrixWarpFull,
        "a warpfull file holds four midway-space warps and two linear "
        "transforms, and a chain cannot be split back into them. Build the "
        "file with MrtrixWarpFull.from_warps.",
    )


@converter
def _(
    t: MrtrixWarpFull, cls: tx.Type[MrtrixWarpFull], **kwargs
) -> MrtrixWarpFull:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
