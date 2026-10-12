"""
Private converters into the MRtrix transformation formats.

`t.to(MrtrixDeformationField)`, `MrtrixDeformationField.from_any(t)` and
`io.save(t, "warp.mif")` all run these converters. Each one either
returns an instance of its format that is exactly the map `t` is, or
raises a `ConversionError` that says what the format cannot hold.

| Source                             | Format                       |
| ---------------------------------- | ---------------------------- |
| any affine, `Sequence` of affines  | `MrtrixLinearTransform`      |
| any field, `Sequence`              | `MrtrixDeformationField`     |
| any field, `Sequence`              | `MrtrixDisplacementField`    |
| `Sequence`                         | `MrtrixWarpFull`, refused    |

In this table, "any affine" means every family that has an affine
form, which is `Identity`, `Translation`, `Scaling`, `Permutation`,
`Linear` and `Rotation`, `Affine`, their tangents, and
`SubspaceTransformation`. "Any field" means every family of fields. Each
family is registered with its own converter, so that its endpoints are
bridged to scanner RAS, which every MRtrix transformation maps from and
to. An endpoint that is not known is taken to be scanner RAS.

A field is held only as MRtrix stores it, which means that its values
are read back with linear interpolation and with the nearest value
outside the grid. A field of displacements is held only between a
world-to-grid affine and its inverse, and a field of coordinates only
after a world-to-grid affine. A field of displacements is not stored as
one of coordinates, or the reverse, because the two extend differently
outside their grid.

A warpfull file holds four midway-space warps and two linear
transforms, and no chain can be split back into them, so its converter
refuses every chain. The file is built with `MrtrixWarpFull.from_warps`
instead.

Each converter makes the decision "exactly, or raise" in one place,
and nothing is resampled or approximated. A conversion takes no option,
since every option would change the map. An object that is already in
its format is changed by the rules of its family, which keep its type.
"""

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    smart_replace,
)
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.transformations.base._conversions import (
    affine_between,
    apply_affine,
    format_options,
    split_field_chain,
    undoes,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import RASCoordinatesField

# this format
from ._linear import MrtrixLinearTransform
from ._warps import (
    MrtrixDeformationField,
    MrtrixDisplacementField,
    MrtrixWarpFull,
)

VOXEL = _systems.VoxelCoordinateSystem()
"""The voxels of the grid of an MRtrix warp."""

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


@converter(_xforms.StationaryVelocityField, MrtrixDeformationField)
@converter(_xforms.CartesianField, MrtrixDeformationField)
@converter(_xforms.DisplacementField, MrtrixDeformationField)
@converter(_xforms.Sequence, MrtrixDeformationField)
@converter
def _(
    t: _xforms.CoordinatesField,
    cls: tx.Type[MrtrixDeformationField],
    **kwargs,
) -> MrtrixDeformationField:
    # Exactly, or raise: an affine from scanner RAS to a grid, then one
    # field of coordinates on that grid, sampled as MRtrix stores it, and
    # an affine into scanner RAS, which is carried into the coordinates. A
    # field of displacements is refused there, with the reason.
    cls = MrtrixDeformationField
    format_options(t, cls, kwargs)
    ras2voxel, field, voxel2ras = split_field_chain(t, RAS, RAS, cls)
    if not isinstance(field, _xforms.CoordinatesField):
        raise unrepresentable(
            t,
            cls,
            f"it holds a {type(field).__name__}, and the format holds "
            f"coordinates. The two extend differently outside their grid, "
            f"so one is not the other there.",
        )
    _check_sampling(field, t, cls)
    if field.data is None:
        raise unrepresentable(
            t,
            cls,
            "it holds no coordinates, and an MRtrix deformation stores the "
            "coordinates of every voxel of its grid.",
        )
    coordinates = apply_affine(voxel2ras, field.values)
    chain = (
        RASToVoxel(matrix=ras2voxel[:-1]),
        RASCoordinatesField(field=coordinates),
    )
    return cls(transformations=chain)


@converter(_xforms.CoordinatesField, MrtrixDisplacementField)
@converter(_xforms.Sequence, MrtrixDisplacementField)
@converter
def _(
    t: _xforms.DisplacementField,
    cls: tx.Type[MrtrixDisplacementField],
    **kwargs,
) -> MrtrixDisplacementField:
    # Exactly, or raise: one field of displacements, between a world-to-grid
    # affine and its inverse, sampled as MRtrix stores it. A field of
    # coordinates is refused there, with the reason.
    cls = MrtrixDisplacementField
    format_options(t, cls, kwargs)
    ras2voxel, field, voxel2ras = split_field_chain(t, RAS, RAS, cls)
    if not isinstance(field, _xforms.DisplacementField):
        raise unrepresentable(
            t,
            cls,
            f"it holds a {type(field).__name__}, and the format holds "
            f"displacements. The two extend differently outside their "
            f"grid, so one is not the other there.",
        )
    if field.log:
        raise unrepresentable(
            t,
            cls,
            "its field is a stationary velocity, and an MRtrix displacement "
            "stores the displacement itself. Integrate the velocity first.",
        )
    _check_sampling(field, t, cls)
    if not undoes(ras2voxel, voxel2ras):
        raise unrepresentable(
            t,
            cls,
            "the affines on either side of its field do not undo each "
            "other, and an MRtrix displacement adds its vectors to the "
            "point it moves, in scanner RAS.",
        )
    chain = (
        RASToVoxel(matrix=ras2voxel[:-1]),
        replace(field, input=VOXEL, output=VOXEL),
        VoxelToRAS(matrix=voxel2ras[:-1]),
    )
    return cls(transformations=chain)


@converter
def _(
    t: MrtrixDeformationField,
    cls: tx.Type[MrtrixDeformationField],
    **kwargs,
) -> MrtrixDeformationField:
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


# ----------------------------------------------------------------------
#   WHAT AN MRTRIX WARP HOLDS
# ----------------------------------------------------------------------


def _check_sampling(
    field: _xforms.Transformation, t: _xforms.Transformation, cls: type
) -> None:
    """
    Refuse a field that MRtrix does not read back as the same map.

    MRtrix stores the values of a field, which it composes with linear
    interpolation, and which brainhops extends with the nearest value
    outside the grid. A field of values interpolated otherwise is a
    different map between its samples, or outside them.
    """
    degree = int(field.degree)
    if degree != 1 or field.bound != BoundaryCondition.nearest:
        raise unrepresentable(
            t,
            cls,
            f"its field is interpolated with degree {degree} and bound "
            f"{field.bound!s}, and an MRtrix warp stores values that are "
            f"read with degree 1 and bound 'nearest'.",
        )
