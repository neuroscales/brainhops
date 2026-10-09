"""
Private converters into the NIfTI transformation formats.

`t.to(NiftiVoxelToRAS)`, `NiftiVoxelToRAS.from_any(t)` and
`io.save(t, "affine.nii.gz")` all run these converters. Each one either
returns an instance of its format that is exactly the map `t` is, or
raises a `ConversionError` that says what the format cannot hold.

| Source                             | Format                        |
| ---------------------------------- | ----------------------------- |
| any affine, `Sequence` of affines  | `NiftiVoxelToRAS`             |
| any affine, `Sequence` of affines  | `NiftiRASToVoxel`             |
| any field, `Sequence`              | `NiftiRASDisplacementField`   |
| any field, `Sequence`              | `NiftiRASCoordinatesField`    |

"Any affine" is each family that has an affine form (`Identity`,
`Translation`, `Scaling`, `Permutation`, `Linear` and `Rotation`, `Affine`,
their tangents, and `SubspaceTransformation`), and "any field" each
family of fields. Each is named, because the converters of the data model
would otherwise rebuild it as the format, whatever its endpoints.

The endpoints are those of `t`, bridged to the format's (see
[`brainhops.io.transformations.base.conversions`][]): an affine to LPS
is flipped into RAS, and one whose systems are not known is taken to map
the format's. A field is held only as NIfTI stores it -- its values,
read back with linear interpolation and the nearest value outside the
grid -- and a displacement field only between a world-to-grid affine and
its inverse. A field of displacements is not stored as one of
coordinates, or the reverse: the two extend differently outside their
grid, so they are not the same map there.

Each converter makes the decision "exactly, or raise" in one place: the
helper it calls first (`_ras_displacement`, `_ras_coordinates`, or
`affine_between`). Nothing is resampled or approximated. The options a
converter is given are the format's own (`header=`, and `log=` and
`steps=` for a displacement field); one that would change the map, such
as `input=` or `matrix=`, is refused (see
[`format_options`][brainhops.io.transformations.base.conversions.\
format_options]).
"""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    _convert_withsplines,
    smart_replace,
)
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.conversions import (
    affine_between,
    apply_affine,
    format_options,
    split_field_chain,
    undoes,
    unrepresentable,
)

from .affines import NiftiRASToVoxel, NiftiVoxelToRAS
from .fields import NiftiRASCoordinatesField, NiftiRASDisplacementField

VOXEL = _systems.VoxelCoordinateSystem()
"""The voxels of the grid a NIfTI file describes."""

RAS = _systems.RASmm()
"""The RAS world, in millimetres, that a NIfTI file maps its grid to."""

AFFINE_OPTIONS = ("header",)
"""What a conversion to a NIfTI affine may set: the header it writes."""

DISPLACEMENT_OPTIONS = ("header", "log", "steps")
"""What a conversion to a NIfTI displacement field may set: the header,
and whether a velocity is written as one (`log`, `steps`)."""

COORDINATES_OPTIONS = ("header",)
"""What a conversion to a NIfTI field of coordinates may set: the header,
whose affine places the grid without changing the map."""


# ----------------------------------------------------------------------
#   AFFINES
# ----------------------------------------------------------------------


@converter(_xforms.Identity, NiftiVoxelToRAS)
@converter(_xforms.Translation, NiftiVoxelToRAS)
@converter(_xforms.Scaling, NiftiVoxelToRAS)
@converter(_xforms.Permutation, NiftiVoxelToRAS)
@converter(_xforms.Linear, NiftiVoxelToRAS)
@converter(_xforms.AffineExponential, NiftiVoxelToRAS)
@converter(_xforms.SubspaceTransformation, NiftiVoxelToRAS)
@converter(_xforms.Sequence, NiftiVoxelToRAS)
@converter
def _(
    t: _xforms.Affine,
    cls: tx.Type[NiftiVoxelToRAS],
    **kwargs,
) -> NiftiVoxelToRAS:
    # Exactly, or raise: the bridges, and the reduction to one affine.
    cls = NiftiVoxelToRAS
    options = format_options(t, cls, kwargs, AFFINE_OPTIONS)
    matrix = affine_between(t, VOXEL, RAS, cls)
    return cls(matrix=matrix[:-1], **options)


@converter(_xforms.Identity, NiftiRASToVoxel)
@converter(_xforms.Translation, NiftiRASToVoxel)
@converter(_xforms.Scaling, NiftiRASToVoxel)
@converter(_xforms.Permutation, NiftiRASToVoxel)
@converter(_xforms.Linear, NiftiRASToVoxel)
@converter(_xforms.AffineExponential, NiftiRASToVoxel)
@converter(_xforms.SubspaceTransformation, NiftiRASToVoxel)
@converter(_xforms.Sequence, NiftiRASToVoxel)
@converter
def _(
    t: _xforms.Affine,
    cls: tx.Type[NiftiRASToVoxel],
    **kwargs,
) -> NiftiRASToVoxel:
    # Exactly, or raise: the bridges, and the reduction to one affine.
    cls = NiftiRASToVoxel
    options = format_options(t, cls, kwargs, AFFINE_OPTIONS)
    matrix = affine_between(t, RAS, VOXEL, cls)
    return cls(matrix=matrix[:-1], **options)


@converter(NiftiRASToVoxel, NiftiRASToVoxel)
@converter
def _(
    t: NiftiVoxelToRAS,
    cls: tx.Type[NiftiVoxelToRAS],
    **kwargs,
) -> NiftiVoxelToRAS:
    # Within its own format, an affine is changed by the rules of any
    # affine (a new `matrix=`, say), which keep its type.
    return _convert_withlog(t, cls, "matrix", **kwargs)


# ----------------------------------------------------------------------
#   FIELDS
# ----------------------------------------------------------------------


@converter(_xforms.CoordinatesField, NiftiRASDisplacementField)
@converter(_xforms.Sequence, NiftiRASDisplacementField)
@converter
def _(
    t: _xforms.DisplacementField,
    cls: tx.Type[NiftiRASDisplacementField],
    **kwargs,
) -> NiftiRASDisplacementField:
    # Exactly, or raise: one field of displacements, between a world-to-grid
    # affine and its inverse, sampled as NIfTI stores it, and written in the
    # encoding it holds. A field of coordinates is refused there, with the
    # reason.
    options = format_options(
        t, NiftiRASDisplacementField, kwargs, DISPLACEMENT_OPTIONS
    )
    ras2voxel, field, voxel2ras = _ras_displacement(t, options)
    chain = (
        RASToVoxel(matrix=ras2voxel[:-1]),
        replace(field, input=VOXEL, output=VOXEL),
        VoxelToRAS(matrix=voxel2ras[:-1]),
    )
    return NiftiRASDisplacementField(transformations=chain, **options)


@converter(_xforms.StationaryVelocityField, NiftiRASCoordinatesField)
@converter(_xforms.CartesianField, NiftiRASCoordinatesField)
@converter(_xforms.DisplacementField, NiftiRASCoordinatesField)
@converter(_xforms.Sequence, NiftiRASCoordinatesField)
@converter
def _(
    t: _xforms.CoordinatesField,
    cls: tx.Type[NiftiRASCoordinatesField],
    **kwargs,
) -> NiftiRASCoordinatesField:
    # Exactly, or raise: one field of coordinates on the file's voxels,
    # sampled as NIfTI stores it, followed by an affine into RAS. A field
    # of displacements is refused there, with the reason.
    cls = NiftiRASCoordinatesField
    options = format_options(t, cls, kwargs, COORDINATES_OPTIONS)
    field, voxel2ras = _ras_coordinates(t, cls)
    coordinates = ras_coordinate_values(t, field, voxel2ras, cls)
    return cls(field=coordinates, **options)


@converter
def _(
    t: NiftiRASDisplacementField,
    cls: tx.Type[NiftiRASDisplacementField],
    **kwargs,
) -> NiftiRASDisplacementField:
    # Within its own format, a chain is changed by the rules of any chain,
    # which keep its type.
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: NiftiRASCoordinatesField,
    cls: tx.Type[NiftiRASCoordinatesField],
    **kwargs,
) -> NiftiRASCoordinatesField:
    # Within its own format, a field is changed by the rules of any field
    # of coordinates (a new `field=`, say), which keep its type.
    return _convert_withsplines(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   WHAT A NIFTI FIELD HOLDS
# ----------------------------------------------------------------------


def _ras_displacement(t: _xforms.Transformation, options: dict) -> tuple:
    """
    `t` as a NIfTI field of RAS displacements, or the reason it is not.

    Returns `(ras2voxel, field, voxel2ras)`: the homogeneous world-to-grid
    affine, the displacement field in the voxels of its grid, and the
    grid-to-world affine, which undoes the first. `options` are those the
    format is built with: `log=True` writes the velocity of the field,
    which only a velocity has.
    """
    cls = NiftiRASDisplacementField
    ras2voxel, field, voxel2ras = split_field_chain(t, RAS, RAS, cls)
    if not isinstance(field, _xforms.DisplacementField):
        raise unrepresentable(
            t,
            cls,
            f"it holds a {type(field).__name__}, and the format holds "
            f"displacements. The two extend differently outside their "
            f"grid, so one is not the other there.",
        )
    _check_nifti_sampling(field, t, cls)
    if not undoes(ras2voxel, voxel2ras):
        raise unrepresentable(
            t,
            cls,
            "the affines on either side of its field do not undo each "
            "other, and a NIfTI displacement field adds its vectors to the "
            "point it moves, in RAS.",
        )
    if options.get("log") and not field.log:
        raise unrepresentable(
            t,
            cls,
            "log=True writes the velocity of its field, and its field holds "
            "a displacement, whose logarithm brainhops does not compute.",
        )
    if options.get("steps") is not None and not options.get("log"):
        raise unrepresentable(
            t,
            cls,
            "steps= is the number of squaring steps of a velocity, so it "
            "is given with log=True.",
        )
    return ras2voxel, field, voxel2ras


def _ras_coordinates(t: _xforms.Transformation, cls: type) -> tuple:
    """
    `t` as a NIfTI field of RAS coordinates, or the reason it is not.

    Returns `(field, voxel2ras)`: the field of coordinates, on the voxels
    of the file, and the affine its coordinates are carried into RAS by.
    """
    voxel2grid, field, voxel2ras = split_field_chain(t, VOXEL, RAS, cls)
    check_coordinates(field, t, cls)
    # The voxels of the file are the grid of the field: nothing (but
    # rounding) may come between them.
    if not undoes(voxel2grid, np.eye(len(voxel2grid))):
        raise unrepresentable(
            t,
            cls,
            "an affine comes before its field, and the voxels of the file "
            "are those of the grid the field is sampled on.",
        )
    return field, voxel2ras


def ras_coordinate_values(
    t: _xforms.Transformation,
    field: _xforms.CoordinatesField,
    voxel2ras: np.ndarray,
    cls: type,
) -> tx.Any:
    """
    The values of a field of coordinates, carried into RAS by `voxel2ras`.

    A field without data is the identity, as a displacement field without
    data is, and is held without data. The affine after it then cannot be
    carried into its values, so it must be the identity too.
    """
    if field.data is None:
        if not undoes(voxel2ras, np.eye(len(voxel2ras))):
            raise unrepresentable(
                t,
                cls,
                "it holds no coordinates, and an affine after them has no "
                "values to be carried into.",
            )
        return None
    return apply_affine(voxel2ras, field.field)


def check_coordinates(
    field: _xforms.Transformation, t: _xforms.Transformation, cls: type
) -> None:
    """Refuse a field that is not one of coordinates, sampled as NIfTI
    stores it."""
    if not isinstance(field, _xforms.CoordinatesField):
        raise unrepresentable(
            t,
            cls,
            f"it holds a {type(field).__name__}, and the format holds "
            f"coordinates. The two extend differently outside their grid, "
            f"so one is not the other there.",
        )
    _check_nifti_sampling(field, t, cls)


def _check_nifti_sampling(
    field: _xforms.Transformation, t: _xforms.Transformation, cls: type
) -> None:
    """
    Refuse a field that NIfTI does not read back as the same map.

    NIfTI stores the values of a field, which are read back with linear
    interpolation and extended with the nearest value outside the grid.
    A field of values interpolated otherwise is a different map between
    its samples, or outside them.
    """
    degree = int(field.degree)
    if degree != 1 or field.bound != BoundaryCondition.nearest:
        raise unrepresentable(
            t,
            cls,
            f"its field is interpolated with degree {degree} and bound "
            f"{field.bound!s}, and a NIfTI file stores values that are read "
            f"with degree 1 and bound 'nearest'.",
        )
