"""
Private converters into the AFNI transformation formats.

`t.to(AfniAffine)`, `AfniAffine.from_any(t)` and
`io.save(t, "x.aff12.1D")` all run these converters. Each one returns
an instance of its format that is the map `t` is, raises a
`LossyConversionError` that carries an approximation of it, or raises a
`ConversionError` that says what the format cannot hold.

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

A warp is stored as AFNI stores it: a field of displacements between a
world-to-grid affine and its inverse, read back with linear
interpolation and with the nearest value outside the grid, on a grid
whose axes run along the DICOM axes. A field of coordinates becomes the
displacement of each node, since the two are the same map inside the
grid and only extend differently beyond it. A velocity is integrated,
and the affines on either side of a field are carried into its values
when they do not undo each other. A field interpolated with another
degree than 1 is sampled at its nodes, and a field on an oblique grid
is sampled on the nearest cardinal grid. Both are losses: the converter
raises a `LossyConversionError`, and `t.to(Format, on_loss="ignore")`
returns the sampled field.

A conversion to an `AfniBrikWarp` takes no option, and a conversion to
an `AfniNiftiWarp` takes `metadata=`, since every other option would
change the map. An object that is already in its format is changed by
the rules of its family, which keep its type.
"""

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import (
    convert_withlog,
    converter,
    smart_replace,
)

# io
from brainhops.io.transformations.base._internal_api import (
    FIELD_SOURCES,
    FieldParts,
    affine_between,
    cardinal_parts,
    displacement_parts,
    displacements_of,
    format_options,
    held,
    register_converter,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

# this format
from ._affine import AfniAffine
from ._warp import AfniBrikWarp

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
    return convert_withlog(t, cls, "matrix", **kwargs)


# ----------------------------------------------------------------------
#   WARPS
# ----------------------------------------------------------------------


@register_converter(FIELD_SOURCES, AfniBrikWarp)
def _(
    t: _xforms.Transformation, cls: tx.Type[AfniBrikWarp], **kwargs
) -> AfniBrikWarp:
    # One field of displacements, between a world-to-grid affine and its
    # inverse, on a cardinal grid, sampled as AFNI stores it. Anything
    # else is turned into one, and what loses is reported.
    cls = AfniBrikWarp
    format_options(t, cls, kwargs)
    parts = afni_displacement(t, cls)
    return held(cls(transformations=afni_chain(parts)), parts)


@converter
def _(t: AfniBrikWarp, cls: tx.Type[AfniBrikWarp], **kwargs) -> AfniBrikWarp:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   WHAT AN AFNI WARP HOLDS
# ----------------------------------------------------------------------


def afni_displacement(t: _xforms.Transformation, cls: type) -> FieldParts:
    """
    Return `t` as the parts of an AFNI warp, or the reason it is not one.

    The parts are those of
    [`displacement_parts`][brainhops.io.transformations.base._conversions.displacement_parts],
    in RAS: the world-to-grid affine, the displacement field in the
    voxels of its grid, and the grid-to-world affine, which undoes the
    first. A field of coordinates becomes the displacement of each node,
    a velocity is integrated, and the affines around the field are
    carried into it when they do not undo each other. A field that is
    not linear is sampled at its nodes, and a field on an oblique grid
    is sampled on the nearest grid whose axes run along the DICOM axes,
    since AFNI places a warp on such a grid. Both are losses, which the
    parts report.
    """
    parts = displacement_parts(t, RAS, cls)
    return displacements_of(cardinal_parts(parts, t, RAS, RAS, cls))


def afni_chain(parts: FieldParts) -> tuple:
    """Return the chain `(ras2voxel, displacement, voxel2ras)` of parts."""
    return (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        parts.field,
        VoxelToRAS(matrix=parts.grid2world[:-1]),
    )
