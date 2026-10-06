"""
Readers and writers for NiftyReg transformations.

| Written by           | `intent_p1`         | Class                          |
| -------------------- | ------------------- | ------------------------------ |
| `reg_aladin -aff`    | --                  | [`NiftyRegAffine`][]           |
| `reg_f3d -cpp`       | 2 `CUB_SPLINE_GRID` | [`NiftyRegControlPointGrid`][] |
| (linear grid)        | 6 `LIN_SPLINE_GRID` | [`NiftyRegControlPointGrid`][] |
| `reg_transform -def` | 0 `DEF_FIELD`       | [`NiftyRegDeformationField`][] |
| `reg_transform -disp`| 1 `DISP_FIELD`      | [`NiftyRegDisplacementField`][]|
| `reg_f3d -vel -cpp`  | 5 `SPLINE_VEL_GRID` | [`NiftyRegVelocityGrid`][]     |
| (dense velocity)     | 3 `DEF_VEL_FIELD`   | [`NiftyRegVelocityField`][]    |
| (dense velocity)     | 4 `DISP_VEL_FIELD`  | [`NiftyRegVelocityField`][]    |

Hints: `niftyreg.aladin`, `niftyreg.cpp` (or `niftyreg.f3d`),
`niftyreg.deformation` (`niftyreg.def`), `niftyreg.displacement`
(`niftyreg.disp`) and `niftyreg.velocity` (`niftyreg.vel`); `niftyreg`
alone selects any of them.

The NIfTI files are claimed from their header alone; the affine, a bare
matrix, only with a hint (`hint="niftyreg"`).

Every reader writes back what it reads.

## Conventions

Checked against the NiftyReg sources (KCL-BMEIS/niftyreg, commit
`79a1762`, September 2026):

- **Direction.** Every NiftyReg transformation maps the *reference*
  image's world to the *floating* image's world -- `reg_aladin` states
  it as `Affine * Reference = Floating` -- which is the direction the
  data model uses to resample a moving (floating) image onto a
  reference. Nothing is inverted.
- **World.** World coordinates are the NIfTI ones, RAS millimetres, with
  no sign flip: NiftyReg takes an image's sform when `sform_code > 0`,
  and its qform otherwise (`sto_xyz` / `qto_xyz` in `reg-lib`). With
  both codes zero, `nifti1_io` makes the qform a diagonal of pixel sizes
  with no offset, which is what is used here (not `nibabel`'s centred
  fallback).
- **The NIfTI files** are `VECTOR` (1007) images named `"NREG_TRANS"`,
  of shape `(X, Y, Z, 1, 3)`, whose `intent_p1` holds the
  `NREG_TRANS_TYPE` (`reg-lib/cpu/Maths.hpp`). The generic NIfTI and ITK
  field readers decline a file with that name.
- **Deformation vs displacement.** A deformation field holds positions,
  a displacement field holds `position - voxel world position`
  (`reg_getDisplacementFromDeformation`): the sign is +1.
- **Control-point grids hold positions**, not displacements: the
  floating world position of each control point
  (`reg_createControlPointGrid` initialises them to the identity). The
  grid's header places it: the reference orientation, scaled to the
  control-point spacing, with its origin one control point before the
  reference origin. The spline is the centred cubic B-spline in the
  grid's voxel units (`reg_cubic_spline_getDeformationField3D`).
- **Beyond the grid**, NiftyReg slides the displacement of the nearest
  grid point (`get_SlidedValues`): every field is read as displacements
  with the `nearest` boundary condition, which reproduces that.

- **Velocities** (`reg_f3d -vel`) are read as the same chain, whose
  field is a `StationaryVelocityField` (`log=True`): the velocity, in
  voxel units -- the grid's cubic coefficients for a velocity grid --
  integrated by scaling and squaring with `|intent_p2|` steps (the
  default rule when it is zero). A negative `intent_p2` marks a backward
  field, whose velocity is negated
  (`reg_defField_getDeformationFieldFromFlowField`). A velocity read
  from a file is written back as it was read; one built from a chain is
  written as positions, with its steps in `intent_p2`.

## Not supported

- **Velocities with an affine in their extensions** (a symmetric
  registration): NiftyReg removes that affine before integrating the
  velocity and composes it back after, which is not decoded. They are
  read and written back, but using one raises `NotImplementedError`.
  Integrate them with `reg_transform -def` and read the deformation
  field instead.
- **2-D fields and grids** (two components) are not decoded.
- The non-composition shortcut NiftyReg uses on the reference grid
  (`reg_cubic_spline_getDeformationField3D` with `composition=false`)
  places the grid by the ratio of pixel sizes rather than by the
  header; the two agree for a grid made on that reference, which is
  the case NiftyReg supports.
"""

__all__ = ["NiftyRegAffine"]

# internals
from brainhops._core.dependencies import HAS_NIBABEL
from brainhops.io.base._dispatch import register_missing_format

from ._affine import NiftyRegAffine

# The affine is plain text; the fields and grids are NIfTI files, read
# with nibabel, which is optional.
if HAS_NIBABEL:
    from ._fields import (
        NiftyRegControlPointGrid,
        NiftyRegDeformationField,
        NiftyRegDisplacementField,
        NiftyRegField,
        NiftyRegSequence,
        NiftyRegVelocity,
        NiftyRegVelocityField,
        NiftyRegVelocityGrid,
    )

    __all__ += [
        "NiftyRegControlPointGrid",
        "NiftyRegDeformationField",
        "NiftyRegDisplacementField",
        "NiftyRegField",
        "NiftyRegSequence",
        "NiftyRegVelocity",
        "NiftyRegVelocityField",
        "NiftyRegVelocityGrid",
    ]
else:
    register_missing_format(
        [
            "niftyreg.cpp",
            "niftyreg.f3d",
            "niftyreg.deformation",
            "niftyreg.def",
            "niftyreg.displacement",
            "niftyreg.disp",
            "niftyreg.velocity",
            "niftyreg.vel",
        ],
        "nibabel",
        "nibabel",
    )
