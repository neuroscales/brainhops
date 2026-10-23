"""
Readers and writers for MRtrix transformations: warps stored in MRtrix
images, and linear transforms stored as text.

MRtrix3 writes its nonlinear registrations as images and its linear ones
as small text files. Each kind of file has a format class here.

| Class                         | File                         |
| ----------------------------- | ---------------------------- |
| [`MrtrixCoordinateField`][]  | 4-D warp of positions        |
| [`MrtrixDisplacementField`][] | 4-D warp of displacements    |
| [`MrtrixWarpFull`][]          | 5-D `-nl_warp_full` file     |
| [`MrtrixLinearTransform`][]   | 3x4 or 4x4 matrix (`.txt`)   |

Each class answers to the hint `"mrtrix"` followed by one of its own
names. Long and short names are both accepted, as for the NiftyReg
fields.

| Class                         | Hints                                  |
| ----------------------------- | -------------------------------------- |
| [`MrtrixCoordinateField`][]  | `mrtrix.coordinates`, `mrtrix.coord`   |
| [`MrtrixDisplacementField`][] | `mrtrix.displacement`, `mrtrix.disp`   |
| [`MrtrixWarpFull`][]          | `mrtrix.warpfull`, `mrtrix.warp_full`  |
| [`MrtrixLinearTransform`][]   | `mrtrix.linear`, `mrtrix.affine`       |

The three warps share the abstract [`MrtrixWarp`][], which answers to
the hint `"mrtrix.field"`, so that `"mrtrix.field.coord"` and
`"mrtrix.field.disp"` are accepted as well. They read the `.mif`,
`.mif.gz` and `.mih` variants with the record and the lazy array of the
MRtrix image, which are described in [`brainhops.io.common.mrtrix`][],
and they need no dependency beyond NumPy.

Each warp holds the header of its file as [`MrtrixMetadata`][], whose
record is an [`MrtrixRaw`][], and the values of the file in `rawdata`,
which is read lazily. A linear transform is short and is read in one
pass, so its whole file, matrix included, is held as an
[`MrtrixLinearRaw`][] record by its [`MrtrixLinearMetadata`][]. The
record and `rawdata` describe the file as it was read. Values, chains
and matrices that are assigned are pending until `consolidate()` encodes
them into a copy, or until the object is written, which writes such a
copy and leaves the object as it was. An object that is read and written
again without changes keeps the bytes of its file, and the free-form
keys of its header, such as `command_history`, are kept when it is
changed.

```python
import brainhops.io as io

warp = io.load("warp.mif")                    # MrtrixCoordinateField
disp = io.load("disp.mif", hint="mrtrix.disp")
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

Every MRtrix transformation follows the "reverse", or pull-back,
convention. It maps points of the *template*, which is the grid of the
output, to points of the *moving* image, which is the image that is
sampled. That is also the direction of a brainhops transformation, so
nothing is inverted. Each transformation is read as a map from scanner
RAS millimetres to scanner RAS millimetres (`RASmm` to `RASmm`).

## Deformations and displacements

A 4-D image with three volumes (`x`, `y` and `z`) is a warp
(`check_warp`, in `helpers.h`). Its grid is the grid of the template,
and its voxel-to-scanner affine is the affine of the image, which is
`transform @ diag(vox)`, or the centred default of MRtrix when the
header has no transform.

- A deformation holds, at each voxel, the scanner RAS position that the
  centre of the voxel maps to (`mrtransform -warp`, `warpinit`). It is
  read as a field of RAS coordinates, as an SPM `y_` file or an X5
  `deformations` field is, which is the chain `ras2voxel`,
  `coordinates`.
- A displacement holds the offset from the scanner position of the
  voxel itself, so that the deformation is
  `voxel2scanner(v) + displacement` (`displacement2deformation`, in
  `convert.h`). It is read as a field of RAS displacements, as a NIfTI
  `DISPVECT` file is, which is the chain `ras2voxel`, `displacement` in
  voxel units, and `voxel2ras`. The chain is built and split by the
  shared helpers of [`brainhops.io.transformations.base.fields`][].

The header does not say which of the two a file holds, and MRtrix
users do not say it either when they apply a warp: every MRtrix command
takes a deformation (`mrtransform -warp`, `transformcompose`,
`warpcorrect`, and `warpinvert` by default), and a displacement must
first be converted with `warpconvert displacement2deformation`. Only
`warpconvert`, by its conversion type, and `warpinvert`, by its
`-displacement` option, are told which kind they read. A warp is
therefore read from these clues, in this order.

1. A field that brainhops wrote records its kind in the header key
   `brainhops_field_type` (`coordinates` or `displacement`), and appends
   `brainhops  (version=...)` to `command_history`, as MRtrix appends
   each command that writes a file. MRtrix keeps the key without reading
   it. While brainhops wrote the file last, the key decides, with
   `Confidence.CERTAIN`.
2. The last line of `command_history` decides, with
   `Confidence.LIKELY`, when it names a command that writes one kind of
   warp: `warpconvert` with `*2displacement` or `*2deformation`,
   `warpinvert` with or without `-displacement`, `warpinit`,
   `warpcorrect` and `mrregister`. MRtrix copies the key of brainhops
   into the files that its commands derive, so such a command overrides
   a key that it made stale.
3. The key decides, with `Confidence.LIKELY`, under a later command that
   does not change the kind of warp, such as `mrconvert`.
4. The values decide, with `Confidence.MAYBE`. Coordinates follow the
   scanner position of their voxel and spread across the field of view,
   while displacements stay close to zero, so the warp is read as a
   field of coordinates when its values spread more than their
   difference from the scanner positions of the voxels. This is the
   test that the FNIRT reader applies to tell absolute fields from
   relative ones. A large warp is subsampled, and the values are not
   read when the data file of a `.mih` header cannot be found.
5. Otherwise, the warp is read as a field of coordinates, which is what
   MRtrix takes.

The values cannot decide for every field, for example a field of a
single voxel or a displacement as large as the field of view, so a file
may still need a hint: `hint="mrtrix.disp"` or `hint="mrtrix.coord"`,
or `MrtrixDisplacementField.from_file(path)`. A hint always overrides
the clues. A field may be written without the key, with
`keyval={"brainhops_field_type": None}`.

A 4-D image with three volumes, or a 5-D image of shape
`(X, Y, Z, 3, 4)`, is also a valid MRtrix image. The image reader scores
it `Confidence.WEAK`, so `io.load` returns a warp, and `io.images.load`
returns an image.

Values of `nan` in a deformation mark the points that lie outside the
field of view of the composition that made it (`ComposeHalfwayKernel`),
and they map to `nan`. Fields are interpolated linearly, as MRtrix
composes them. Outside the grid, MRtrix gives up: a composed point
becomes `nan`, and a resampled voxel takes the out-of-bounds value.
Here, as for the other field readers, the field is extended with its
nearest value.

A warp is written with the options `layout`, `datatype`, `scaling` and
`keyval` of the MRtrix image writer. A warp whose chain was assigned is
written from the chain, whose grid replaces the geometry of the record.

## The 5-D warpfull format

`mrregister -nl_warp_full` writes one 5-D image of shape
`(X, Y, Z, 3, 4)`, sampled on the *midway* grid, which holds four
deformations, and the two halves of the linear registration in the
header keys `linear1` and `linear2` ([`MrtrixWarpFull`][]). Each
deformation is named after the image that it moves, in the reverse
convention, so that as maps of points they read as follows.

| Index | Name         | Maps                                     |
| ----- | ------------ | ---------------------------------------- |
| 0     | `im1_to_mid` | midway -> image 1, before `linear1`       |
| 1     | `mid_to_im1` | image 1, after `linear1⁻¹` -> midway      |
| 2     | `im2_to_mid` | midway -> image 2, before `linear2`       |
| 3     | `mid_to_im2` | image 2, after `linear2⁻¹` -> midway      |

`linear1` maps midway points to the points of image 1, and `linear2` to
the points of image 2. Each is the half of the affine of `mrregister`,
which maps the points of image 2 to those of image 1, that
`nonlinear.h` stores as `im1_to_mid_linear` or `im2_to_mid_linear`.

The object is the map that `mrtransform -warp_full` applies
(`compute_full_deformation`, in `compose.h`). By default, it is the map
from image 2 to image 1, which warps image 1 onto image 2 (`-from 1`):
`linear2⁻¹`, `mid_to_im2`, `im1_to_mid` and `linear1`. The reading
options `from_image=2` and `midway=True` select the other maps (`-from
2` and `-midway_space`), and the warps and the linear transforms are
also available one by one. A chain cannot be split back into these
pieces, so a warpfull file is written from its warps and its record, and
a new one is built with `MrtrixWarpFull.from_warps`.

A warpfull file keeps its linear transforms in its header, so only a
`.mif` or `.mih` file holds all of it. MRtrix warns that a NIfTI file
needs its JSON sidecar, which is not read here.

## Linear transforms

`mrtransform -linear` and `mrregister -rigid` or `-affine` read and
write a text matrix (`load_transform` and `save_transform`, in
`core/math/math.h`). The matrix has 3 or 4 rows of 4 values, separated
by spaces, commas, semicolons or tabs, after `# key: value` comments
such as `command_history` and `centre`. Only the first three rows are
read. The matrix maps the scanner RAS points of the template to those of
the moving image, and is read as it is ([`MrtrixLinearTransform`][]). A
changed matrix is written with the comments of its file and a `0 0 0 1`
row.

A text matrix says nothing about where it comes from, and plain
matrices, FLIRT `.mat` files and MRtrix transforms look alike. A file is
therefore read as an MRtrix transform only when its comments carry the
signature of MRtrix, which is a `command_history` or a `centre` entry,
or when it is asked for with `hint="mrtrix.linear"`. Otherwise a `.txt`
matrix stays a
[`TxtMatrixAffine`][brainhops.io.transformations.matrix.TxtMatrixAffine].
A transform that brainhops writes from an affine without a file names
brainhops in its `command_history`, so that it is read back as an MRtrix
transform. For the same reason, `io.save` writes an affine between RAS
spaces to a `.txt` name as a NiftyReg affine, and an MRtrix transform is
written from `t.to(MrtrixLinearTransform)`.

## NIfTI warps

MRtrix also writes its 4-D warps as NIfTI files, without an intent
code. Such a file is read by the NIfTI readers as a field of RAS
coordinates
([`NiftiRASCoordinateField`][brainhops.io.transformations.nifti.NiftiRASCoordinateField]),
which is the reading of a deformation.

## Header without a transform

The default transform of MRtrix is not the identity: it centres the
field of view on the origin (`-0.5 * (size - 1) * vox`, in
`core/transform.h`). The record of the header applies it.
"""

__all__ = [
    "MrtrixCoordinateField",
    "MrtrixDisplacementField",
    "MrtrixLinearMetadata",
    "MrtrixLinearRaw",
    "MrtrixLinearTransform",
    "MrtrixMetadata",
    "MrtrixRaw",
    "MrtrixWarp",
    "MrtrixWarpFull",
]

from brainhops.io.common.mrtrix import MrtrixMetadata, MrtrixRaw

from ._linear import MrtrixLinearTransform
from ._metadata import MrtrixLinearMetadata
from ._raw import MrtrixLinearRaw
from ._warps import (
    MrtrixCoordinateField,
    MrtrixDisplacementField,
    MrtrixWarp,
    MrtrixWarpFull,
)

# Importing the private converters registers them with the data model, so
# that a conversion into one of these formats, such as `t.to(Format)`,
# holds the very map or fails with a reason.
from . import _converters  # noqa: E402, F401  isort: skip


# -- deprecated names ----------------------------------------------------
# The old spellings of the coordinate fields (#547) still import, with a
# DeprecationWarning, for one release.
from brainhops._core.deprecation import (  # noqa: E402
    renamed_attributes as _renamed_attributes,
)

__getattr__ = _renamed_attributes(
    __name__,
    {
        "MrtrixCoordinatesField": "MrtrixCoordinateField",
    },
)
