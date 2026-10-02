"""
Readers and writers for MRtrix transformations: warps stored in MRtrix
images, and linear transforms stored as text.

- [`MrtrixDeformationField`][]: a 4-D warp of absolute positions;
  hints `"mrtrix.deformation"`, `"coordinates"`.
- [`MrtrixDisplacementField`][]: a 4-D warp of relative
  displacements; hints `"mrtrix.displacement"`, `"displacements"`.
- [`MrtrixWarpFull`][]: the 5-D file of `mrregister -nl_warp_full`;
  hint `"mrtrix.warpfull"`.
- [`MrtrixLinearTransform`][]: a 3x4 / 4x4 text matrix (`.txt`); hint
  `"mrtrix.linear"`.

The three warps share the abstract [`MrtrixWarp`][] (hint
`"mrtrix.warp"`), which reads the `.mif`, `.mif.gz` and `.mih` variants
through the same header parser and codec as the MRtrix image reader
([`brainhops.io.base.mrtrix`][]). They are read and written with no
dependency beyond numpy.

```python
import brainhops.io as io

warp = io.load("warp.mif")                    # MrtrixDeformationField
disp = io.load("disp.mif", hint="mrtrix.displacement")
full = io.load("warps.mif")                   # MrtrixWarpFull (5-D)
lin = io.load("affine.txt", hint="mrtrix.linear")
io.save(warp, "warp.mih")
```

The conventions below were checked against the MRtrix3 sources
(`cmd/mrtransform.cpp`, `cmd/warpconvert.cpp`, `cmd/warpinit.cpp`,
`cmd/warpinvert.cpp`, `cmd/mrregister.cpp`,
`src/registration/warp/{convert,compose,helpers}.h`,
`src/registration/nonlinear.{h,cpp}`, `core/adapter/{warp,reslice}.h`,
`core/math/math.h`, `core/file/key_value.cpp`).

## Direction

Every MRtrix transformation uses the "reverse" (pull-back) convention:
it maps points of the *template* (the grid of the output) to points of
the *moving* image (the one that is sampled). That is also the
direction of a brainhops transformation, so nothing is inverted: each
one is read as a map from scanner RAS to scanner RAS millimetres
(`RASmm` to `RASmm`).

## Deformations and displacements

A 4-D image with three volumes (`x, y, z`) is a warp
(`check_warp`, `helpers.h`). Its grid is the template's, and its
voxel-to-scanner affine is the image's (`transform @ diag(vox)`, or
MRtrix's centred default when the header has none).

- A **deformation** holds, at each voxel, the scanner RAS position its
  centre maps to (`mrtransform -warp`, `warpinit`). It is read as a
  field of RAS coordinates, as an SPM `y_` file or an X5
  `deformations` field: the chain `ras2voxel`, `coordinates`.
- A **displacement** holds the offset from the voxel's own scanner
  position: `deformation = voxel2scanner(v) + displacement`
  (`displacement2deformation`, `convert.h`). It is read as a field of RAS
  displacements, as a NIfTI `DISPVECT` file: the chain `ras2voxel`,
  `displacement` (in voxel units), `voxel2ras`, built and split by the
  shared helpers of
  [`brainhops.io.transformations.base.fields`][].

**Which is which.** The header does not say. Every MRtrix command
takes deformations (`mrtransform -warp`, `warpinvert` by default,
`mrregister -nl_init` via the 5-D format), so a warp is read as a
**deformation** unless the last line of its `command_history` (which
MRtrix appends to every image it writes) names a command that writes
displacements: `warpconvert ... deformation2displacement` /
`warpfull2displacement`, or `warpinvert -displacement`. The values are
never looked at. To read a file the other way, ask for it:
`hint="mrtrix.displacement"` (or `"mrtrix.deformation"`), or call
`MrtrixDisplacementField.from_file(path)`.

A 4-D image with three volumes (or a 5-D one of shape `(X, Y, Z, 3,
4)`) is also a valid MRtrix image; the image reader scores it `WEAK`, so
`io.load` returns a warp, and `io.images.load` an image.

Values of `nan` in a deformation mark points outside the field of view
of the composition that made it (`ComposeHalfwayKernel`); they map to
`nan`. Fields are interpolated linearly, as MRtrix composes them.
Outside the grid, MRtrix gives up (a composed point becomes `nan`, a
resampled voxel the out-of-bounds value); here, as for the other field
readers, the field is extended with its nearest value.

## The 5-D *warpfull* format

`mrregister -nl_warp_full` writes one 5-D image of shape
`(X, Y, Z, 3, 4)`, sampled on the *midway* grid, which holds four
deformations and, in the header keys `linear1` and `linear2`, the two
halves of the linear registration ([`MrtrixWarpFull`][]). Each one is
named after the image it moves, in the reverse convention, so as point
maps they read:

| Index | Name         | Maps                                     |
| ----- | ------------ | ---------------------------------------- |
| 0     | `im1_to_mid` | midway -> image 1, before `linear1`       |
| 1     | `mid_to_im1` | image 1, after `linear1⁻¹` -> midway      |
| 2     | `im2_to_mid` | midway -> image 2, before `linear2`       |
| 3     | `mid_to_im2` | image 2, after `linear2⁻¹` -> midway      |

`linear1` (`linear2`) maps midway points to image-1 (image-2) points:
it is the half of `mrregister`'s affine (which maps image-2 points to
image-1 points) that `nonlinear.h` stores as `im1_to_mid_linear`
(`im2_to_mid_linear`).

The object is the map `mrtransform -warp_full` applies
(`compute_full_deformation`, `compose.h`): by default the map from
image 2 to image 1, which warps image 1 onto image 2 (`-from 1`):
`linear2⁻¹`, `mid_to_im2`, `im1_to_mid`, `linear1`. The reading options
`from_image=2` and `midway=True` select the other maps (`-from 2`,
`-midway_space`); the warps and linear transforms are also available
one by one.

A warpfull file keeps its linear transforms in its header, so only a
`.mif` / `.mih` file holds them all (MRtrix warns that a NIfTI file
needs its JSON sidecar, which is not read here).

## Linear transforms

`mrtransform -linear` and `mrregister -rigid` / `-affine` read and
write a text matrix (`load_transform` / `save_transform`,
`core/math/math.h`): 3 or 4 rows of 4 values, separated by spaces,
commas, semicolons or tabs, after `# key: value` comments
(`command_history`, `centre`). Only the first three rows are read. It
maps template scanner RAS points to moving scanner RAS points, and is
read as such, unchanged ([`MrtrixLinearTransform`][]). It is written
back with its comments and a `0 0 0 1` row.

A text matrix says nothing about where it comes from, and plain
matrices, FLIRT `.mat` files and MRtrix transforms look alike. A file
is therefore read as an MRtrix transform only when its comments carry
MRtrix's signature (a `command_history` entry), or when asked for with
`hint="mrtrix.linear"`; otherwise a `.txt` matrix stays a
[`TxtMatrixAffine`][brainhops.io.transformations.matrix.TxtMatrixAffine].

## NIfTI warps

MRtrix also writes its 4-D warps as NIfTI, with no intent code. Such a
file is read by the NIfTI readers, as a field of RAS coordinates
([`NiftiRASCoordinatesField`][brainhops.io.transformations.nifti.NiftiRASCoordinatesField])
-- the deformation reading.

## Header without a transform

MRtrix's default transform is not the identity: it centres the field of
view on the origin (`-0.5 * (size - 1) * vox`, `core/transform.h`). It
is applied by the shared header parser.
"""

__all__ = [
    "MrtrixWarp",
    "MrtrixDeformationField",
    "MrtrixDisplacementField",
    "MrtrixWarpFull",
    "MrtrixLinearTransform",
]

from ._linear import MrtrixLinearTransform
from ._warps import (
    MrtrixDeformationField,
    MrtrixDisplacementField,
    MrtrixWarp,
    MrtrixWarpFull,
)
