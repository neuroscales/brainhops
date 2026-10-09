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

The hints are `niftyreg.aladin`, `niftyreg.cpp` (`niftyreg.f3d`),
`niftyreg.deformation` (`niftyreg.def`), `niftyreg.displacement`
(`niftyreg.disp`) and `niftyreg.velocity` (`niftyreg.vel`), and `niftyreg`
alone selects any of them. NIfTI files are claimed from their header alone,
while the affine, a bare matrix, is only read with `hint="niftyreg"`. Every
reader writes back what it reads.

## Conventions

These conventions were checked against KCL-BMEIS/niftyreg at commit 79a1762
(September 2026).

- Direction: every NiftyReg transformation maps the reference world to the
  floating world (for `reg_aladin`, `Affine * Reference = Floating`). This is
  the resampling direction of the data model, so nothing is inverted.
- World: NIfTI RAS millimetres, with no sign flip. The sform is used when
  `sform_code > 0`, and the qform otherwise. When both codes are zero, the
  diagonal of pixel sizes that `nifti1_io` builds is used, not the centred
  fallback of `nibabel`.
- NIfTI files: `VECTOR` images (1007) named `"NREG_TRANS"`, with shape (X, Y,
  Z, 1, 3) and the transformation type (`NREG_TRANS_TYPE`) in `intent_p1`.
- Deformations and displacements: a deformation holds positions, and a
  displacement holds the position minus the world position of its voxel.
- Control-point grids hold the floating world position of each control point.
  The grid header copies the orientation of the reference, scaled to the
  control-point spacing, with its origin one control point before the reference
  origin. The spline is a centred cubic B-spline in grid voxel units.
- Beyond the grid, NiftyReg slides the displacement of the nearest grid point,
  so every field is read as displacements with the `nearest` boundary.
- Velocities (`reg_f3d -vel`) give the same chain with a stationary velocity
  field (`log=True`) in voxel units, integrated by scaling and squaring with
  `|intent_p2|` steps, or the default rule when it is zero. A negative
  `intent_p2` marks a backward field, whose velocity is negated. A velocity
  read from a file is written back as read, and one built from a chain is
  written as positions with its steps in `intent_p2`.

## Not supported

- A velocity with an affine in its header extensions, as written by a symmetric
  registration. NiftyReg removes the affine before integration and composes it
  back after, which is not decoded. Such a file is read and written back, but
  using it raises `NotImplementedError`; integrating it with `reg_transform
  -def` gives a deformation field that can be read.
- 2-D fields and grids, which have two components.
- The non-composition shortcut of `reg_cubic_spline_getDeformationField3D`,
  which places the grid from the ratio of pixel sizes rather than from the
  header. Both agree for a grid made on that reference.
"""

__all__ = ["NiftyRegAffine"]

from brainhops._core.dependencies import HAS_NIBABEL
from brainhops.io.base._dispatch import register_missing_format

from ._affine import NiftyRegAffine

# The affine is plain text, but the fields and grids are NIfTI files,
# which need the optional nibabel.
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
