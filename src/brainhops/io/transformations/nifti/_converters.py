"""
Private converters into the NIfTI transformation formats.

`t.to(NiftiVoxelToRAS)`, `NiftiVoxelToRAS.from_any(t)` and
`io.save(t, "affine.nii.gz")` all run these converters. Each one
returns an instance of its format that is the map `t` is, raises a
`LossyConversionError` that carries an approximation of it, or raises a
`ConversionError` that says what the format cannot hold.

| Source                             | Format                        |
| ---------------------------------- | ----------------------------- |
| any affine, `Sequence` of affines  | `NiftiVoxelToRAS`             |
| any affine, `Sequence` of affines  | `NiftiRASToVoxel`             |
| any field, `Sequence`              | `NiftiRASDisplacementField`   |
| any field, `Sequence`              | `NiftiRASCoordinatesField`    |

In this table, "any affine" means every family that has an affine
form, which is `Identity`, `Translation`, `Scaling`, `Permutation`,
`Linear` and `Rotation`, `Affine`, their tangents, and
`SubspaceTransformation`. "Any field" means every family of fields. Each
family is registered with its own converter so that its endpoints are
bridged to those of the format. Otherwise, the converters of the data
model would rebuild the transformation as the format with its original
endpoints, and the format refuses endpoints that are not compatible
with its own.

The endpoints are those of `t`, bridged exactly to those of the format.
An affine to LPS is flipped into RAS, and an affine whose systems are
not known is taken to map the systems of the format. A field is stored
as its values at the nodes of its grid, which NIfTI reads back with
linear interpolation and with the nearest value outside the grid. A
field of coordinates is stored as the displacement of each node, and a
field of displacements as the position of each node, since the two are
the same map inside the grid and only extend differently beyond it.
The affines on either side of the field are carried into its values
when they do not undo each other. A field interpolated with another
degree than 1 is sampled at its nodes, which is a loss: the converter
raises a `LossyConversionError`, and `t.to(Format, lossy=True)` returns
the sampled field.

A NIfTI-1 header stores its affine in single precision. An affine whose
matrix does not read back as the same numbers in single precision is
therefore written with rounded numbers, which is a loss: the converter
builds the format and raises a `LossyConversionError` that carries it,
unless the header of its metadata is a NIfTI-2 header, which stores the
affine in double precision.

The options a converter is given are the format's own (`metadata=`,
and `log=` and `steps=` for a displacement field); one that would
change the map, such as `input=` or `matrix=`, is refused by
`format_options`. Without `metadata=`, the format receives the metadata
of `t`, which its field converts into `NiftiMetadata` without a record,
so that the vocabulary of an X5 transform, say, is carried into the
NIfTI file.
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    _convert_withsplines,
    smart_replace,
)
from brainhops.errors import ConversionError, LossyConversionError
from brainhops.io.common.nifti._header import _header

# io
from brainhops.io.transformations.base._conversions import (
    FieldParts,
    affine_between,
    coordinates_parts,
    displacement_parts,
    format_options,
    held,
    split_field_chain,
    undoes,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

from ._affines import NiftiRASToVoxel, NiftiVoxelToRAS
from ._fields import NiftiRASCoordinatesField, NiftiRASDisplacementField

VOXEL = _systems.VoxelCoordinateSystem()
"""The voxels of the grid a NIfTI file describes."""

RAS = _systems.RASmm()
"""The RAS world, in millimetres, that a NIfTI file maps its grid to."""

AFFINE_OPTIONS = ("metadata",)
"""What a conversion to a NIfTI affine may set: the metadata, whose
header is the base of the header it writes."""

DISPLACEMENT_OPTIONS = ("metadata", "log", "steps")
"""What a conversion to a NIfTI displacement field may set: the metadata,
and whether a velocity is written as one (`log`, `steps`)."""

COORDINATES_OPTIONS = ("metadata",)
"""What a conversion to a NIfTI field of coordinates may set: the metadata,
whose header places the grid without changing the map."""


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
    # The metadata of `t` rides along, converted by the field.
    options.setdefault("metadata", t.metadata)
    matrix = affine_between(t, VOXEL, RAS, cls)
    return _single(cls(matrix=matrix[:-1], **options), matrix)


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
    # The metadata of `t` rides along, converted by the field.
    options.setdefault("metadata", t.metadata)
    matrix = affine_between(t, RAS, VOXEL, cls)
    return _single(cls(matrix=matrix[:-1], **options), np.linalg.inv(matrix))


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
@converter(_xforms.CartesianField, NiftiRASDisplacementField)
@converter(_xforms.Sequence, NiftiRASDisplacementField)
@converter
def _(
    t: _xforms.DisplacementField,
    cls: tx.Type[NiftiRASDisplacementField],
    **kwargs,
) -> NiftiRASDisplacementField:
    # One field between a world-to-grid affine and its inverse, sampled as
    # NIfTI stores it, and written in the encoding it holds. A field of
    # coordinates becomes the displacement of each node, and a field that
    # is not linear is sampled at its nodes, which loses.
    cls = NiftiRASDisplacementField
    options = format_options(t, cls, kwargs, DISPLACEMENT_OPTIONS)
    # The metadata of `t` rides along, converted by the field.
    options.setdefault("metadata", t.metadata)
    parts = _ras_displacement(t, options)
    chain = (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        parts.field,
        VoxelToRAS(matrix=parts.grid2world[:-1]),
    )
    return held(cls(transformations=chain, **options), parts)


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
    # One field of coordinates on the file's voxels, sampled as NIfTI
    # stores it, carried into RAS. A field of displacements becomes the
    # position of each node, and a field that is not linear is sampled at
    # its nodes, which loses.
    cls = NiftiRASCoordinatesField
    options = format_options(t, cls, kwargs, COORDINATES_OPTIONS)
    # The metadata of `t` rides along, converted by the field.
    options.setdefault("metadata", t.metadata)
    if _empty_coordinates(t, cls) is not None:
        return cls(field=None, **options)
    parts = coordinates_parts(t, VOXEL, RAS, cls)
    # The voxels of the file are the grid of the field: nothing (but
    # rounding) may come between them.
    if not undoes(parts.world2grid, np.eye(len(parts.world2grid))):
        raise unrepresentable(
            t,
            cls,
            "an affine comes before its field, and the voxels of the file "
            "are those of the grid the field is sampled on. A map from RAS "
            "is written as an SPM deformation, whose file name starts with "
            "'y_'.",
        )
    return held(cls(field=parts.field.values, **options), parts)


@converter
def _(
    t: NiftiRASDisplacementField,
    cls: tx.Type[NiftiRASDisplacementField],
    **kwargs,
) -> NiftiRASDisplacementField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
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


def _single(result: tx.Any, voxel2ras: np.ndarray) -> tx.Any:
    """
    Return a NIfTI affine, or report that its header rounds it.

    A NIfTI-1 header stores the voxel-to-RAS affine `voxel2ras` in single
    precision, and a NIfTI-2 header in double precision.

    Raises
    ------
    LossyConversionError
        If the header stores single precision and the affine does not
        read back as the same numbers in it.
    """
    if isinstance(_header(result), nb.Nifti2Header):
        return result
    rounded = voxel2ras.astype(np.float32).astype(np.float64)
    if np.array_equal(rounded, voxel2ras):
        return result
    raise LossyConversionError(
        f"The {type(result).__name__} that this transformation converts "
        f"to is not the same map: a NIfTI-1 header stores its affine in "
        f"single precision, which rounds this one by up to "
        f"{np.abs(rounded - voxel2ras).max():.3g}. Pass lossy=True to "
        f"accept it, or give the format the metadata of a NIfTI-2 header.",
        result=result,
    )


# ----------------------------------------------------------------------
#   WHAT A NIFTI FIELD HOLDS
# ----------------------------------------------------------------------


def _ras_displacement(t: _xforms.Transformation, options: dict) -> FieldParts:
    """
    `t` as a NIfTI field of RAS displacements, or the reason it is not.

    Returns the parts that
    [`displacement_parts`][brainhops.io.transformations.base._conversions.displacement_parts]
    gives: the world-to-grid affine, the displacement field in the voxels
    of its grid, and the inverse affine. `options` are those the format is
    built with: `log=True` writes the velocity of the field, which only a
    velocity has.
    """
    cls = NiftiRASDisplacementField
    log = bool(options.get("log"))
    if options.get("steps") is not None and not log:
        raise unrepresentable(
            t,
            cls,
            "steps= is the number of squaring steps of a velocity, so it "
            "is given with log=True.",
        )
    # A velocity is kept, and is written as its displacement unless the
    # format is given log=True.
    parts = displacement_parts(t, RAS, cls, keep_log=True)
    if log and not parts.field.log:
        raise unrepresentable(
            t,
            cls,
            "log=True writes the velocity of its field, and its field holds "
            "a displacement, whose logarithm brainhops does not compute.",
        )
    return parts


def _empty_coordinates(
    t: _xforms.Transformation, cls: type
) -> tx.Optional[_xforms.CoordinatesField]:
    """
    The field of `t` when it holds no coordinates, or `None`.

    A field without data is the identity, as a displacement field without
    data is, and is held without data. Nothing may then come before or
    after it, since an affine has no values to be carried into.
    """
    voxel2grid, field, voxel2ras = _voxel_chain(t, cls)
    if field.data is not None:
        return None
    identity = np.eye(len(voxel2ras))
    if not undoes(voxel2grid, identity) or not undoes(voxel2ras, identity):
        raise unrepresentable(
            t,
            cls,
            "it holds no coordinates, and an affine before or after them "
            "has no values to be carried into.",
        )
    return field


def _voxel_chain(t: _xforms.Transformation, cls: type) -> tuple:
    """
    `t` taken apart around its field, read from the voxels of the file.

    A NIfTI field of coordinates maps the voxels of its file, so a
    transformation that maps from a world cannot be read from them. The
    error then says that such a map is written as an SPM deformation.
    """
    try:
        return split_field_chain(t, VOXEL, RAS, cls)
    except ConversionError as error:
        raise unrepresentable(
            t,
            cls,
            f"a NIfTI field of coordinates maps the voxels of the file, and "
            f"this transformation does not. A map from RAS is written as an "
            f"SPM deformation, whose file name starts with 'y_'. {error}",
        ) from error
