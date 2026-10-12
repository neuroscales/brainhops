"""
Private converters into the AFNI transformation formats.

`t.to(AfniAffine)`, `AfniAffine.from_any(t)` and
`io.save(t, "x.aff12.1D")` all run these converters. Each one either
returns an instance of its format that is exactly the map `t` is, or
raises a `ConversionError` that says what the format cannot hold.

| Source                             | Format                       |
| ---------------------------------- | ---------------------------- |
| any affine, `Sequence` of affines  | `AfniAffine`                 |
| any field, `Sequence`              | `AfniBrikWarp`               |
| any field, `Sequence`              | `AfniNiftiWarp`              |

In this table, "any affine" means every family that has an affine
form, which is `Identity`, `Translation`, `Scaling`, `Permutation`,
`Linear` and `Rotation`, `Affine`, their tangents, and
`SubspaceTransformation`. "Any field" means every family of fields. Each
family is registered with its own converter, so that its endpoints are
bridged to those of the format. An affine is bridged to the DICOM (LPS)
world of AFNI, and a warp to RAS, in which it is read. An endpoint that
is not known is taken to be the system of the format.

A warp is held only as AFNI stores it: a field of displacements between
a world-to-grid affine and its inverse, read back with linear
interpolation and with the nearest value outside the grid, on a grid
whose axes run along the DICOM axes. A field of coordinates is refused,
because the two extend differently outside their grid.

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
from brainhops.io.common.afni._constants import DICOM_TO_RAS
from brainhops.io.transformations.base._conversions import (
    affine_between,
    format_options,
    split_field_chain,
    undoes,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

# this format
from ._affine import AfniAffine
from ._warp import AfniBrikWarp, _cardinal

VOXEL = _systems.VoxelCoordinateSystem()
"""The voxels of the grid of an AFNI warp."""

RAS = _systems.RASmm()
"""The RAS world, in millimetres, in which an AFNI warp is read."""

LPS = _systems.LPSmm()
"""The DICOM world of AFNI, which is LPS in millimetres."""


# ----------------------------------------------------------------------
#   AFFINES
# ----------------------------------------------------------------------


@converter(_xforms.Identity, AfniAffine)
@converter(_xforms.Translation, AfniAffine)
@converter(_xforms.Scaling, AfniAffine)
@converter(_xforms.Permutation, AfniAffine)
@converter(_xforms.Linear, AfniAffine)
@converter(_xforms.AffineExponential, AfniAffine)
@converter(_xforms.SubspaceTransformation, AfniAffine)
@converter(_xforms.Sequence, AfniAffine)
@converter
def _(t: _xforms.Affine, cls: tx.Type[AfniAffine], **kwargs) -> AfniAffine:
    # Exactly, or raise: `affine_between` bridges the endpoints of `t` to
    # DICOM, which flips an affine between RAS spaces, and reduces a chain
    # to one affine. An affine whose systems are not known is taken to map
    # DICOM to DICOM.
    cls = AfniAffine
    format_options(t, cls, kwargs)
    matrix = affine_between(t, LPS, LPS, cls)
    return cls(matrix=matrix[:-1])


@converter
def _(t: AfniAffine, cls: tx.Type[AfniAffine], **kwargs) -> AfniAffine:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return _convert_withlog(t, cls, "matrix", **kwargs)


# ----------------------------------------------------------------------
#   WARPS
# ----------------------------------------------------------------------


@converter(_xforms.CoordinatesField, AfniBrikWarp)
@converter(_xforms.Sequence, AfniBrikWarp)
@converter
def _(
    t: _xforms.DisplacementField, cls: tx.Type[AfniBrikWarp], **kwargs
) -> AfniBrikWarp:
    # Exactly, or raise: one field of displacements, between a world-to-grid
    # affine and its inverse, on a cardinal grid, sampled as AFNI stores
    # it.
    cls = AfniBrikWarp
    format_options(t, cls, kwargs)
    return cls(transformations=_afni_displacement(t, cls))


@converter
def _(t: AfniBrikWarp, cls: tx.Type[AfniBrikWarp], **kwargs) -> AfniBrikWarp:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   WHAT AN AFNI WARP HOLDS
# ----------------------------------------------------------------------


def _afni_displacement(t: _xforms.Transformation, cls: type) -> tuple:
    """
    Return `t` as the chain of an AFNI warp, or the reason it is not one.

    The chain is `(ras2voxel, displacement, voxel2ras)`: the world-to-grid
    affine, the displacement field in the voxels of its grid, and the
    grid-to-world affine, which undoes the first and runs along the DICOM
    axes.
    """
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
            "its field is a stationary velocity, and an AFNI warp stores "
            "the displacement itself. Integrate the velocity first.",
        )
    degree = int(field.degree)
    if degree != 1 or field.bound != BoundaryCondition.nearest:
        raise unrepresentable(
            t,
            cls,
            f"its field is interpolated with degree {degree} and bound "
            f"{field.bound!s}, and an AFNI warp stores values that are read "
            f"with degree 1 and bound 'nearest'.",
        )
    if not undoes(ras2voxel, voxel2ras):
        raise unrepresentable(
            t,
            cls,
            "the affines on either side of its field do not undo each "
            "other, and an AFNI warp adds its vectors to the point it "
            "moves.",
        )
    if not _cardinal(DICOM_TO_RAS @ voxel2ras):
        raise unrepresentable(
            t,
            cls,
            "its grid is oblique, and AFNI places a warp on a grid whose "
            "axes run along the DICOM axes.",
        )
    return (
        RASToVoxel(matrix=ras2voxel[:-1]),
        replace(field, input=VOXEL, output=VOXEL),
        VoxelToRAS(matrix=voxel2ras[:-1]),
    )
