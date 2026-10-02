"""
ITK nonlinear warps stored as NIfTI vector images, in LPS space.

ITK (and therefore ANTs, which writes its warps through ITK) stores a
dense displacement field as a NIfTI *vector image*. Three facts define
the encoding, and each one differs from the RAS NIfTI fields of
[`brainhops.io.transformations.nifti`][]:

1. **Layout.** The array is five-dimensional, `(X, Y, Z, 1, 3)`: three
   spatial axes, a singleton time axis, and the vector components in the
   fifth axis. ITK's `NiftiImageIO` always writes `dim[0] = 5` for a
   vector image, sets `dim[4] = 1` when the image has fewer than four
   dimensions, and stores the components in `dim[5]`.

2. **Intent code.** ITK writes `NIFTI_INTENT_VECTOR` (1007) for a vector
   image. Since ITK 5.4 it writes `NIFTI_INTENT_DISPVECT` (1006) instead
   only when the image's metadata dictionary explicitly asks for it.

3. **Frame of the vectors.** The voxel-to-world affine in the header is
   the usual NIfTI RAS sform/qform: ITK negates the first two rows of its
   LPS direction and origin when it writes them. The *vector values* are
   not converted for a `VECTOR` (1007) file, so they stay in ITK's LPS
   physical space. A `DISPVECT` (1006) file is the exception: since ITK
   5.4, ITK flips the first two components of a three-component
   `DISPVECT` image between RAS (in the file) and LPS (in memory), both on
   reading and on writing.

So, for a point `x` of the field's grid, an ITK displacement field maps
`x_lps -> x_lps + u_lps(x)`. In RAS world, the same displacement is
`(-u_x, -u_y, u_z)`.

Sources
-------
- ITK, `Modules/IO/NIFTI/src/itkNiftiImageIO.cxx`
  (`WriteImageInformation`, `SetNIfTIOrientationFromImageIO`,
  `ReadImageInformation`, and `ConvertRASToFromLPS_*`), and
  `Modules/IO/NIFTI/include/itkNiftiImageIO.h`
  (`ConvertRASVectors`, off by default, and
  `ConvertRASDisplacementVectors`, on by default, both new in v5.4.0).
  <https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/IO/NIFTI/src/itkNiftiImageIO.cxx>
- NiTransforms, `nitransforms/io/itk.py`, `ITKDisplacementsField`: it
  requires the `(X, Y, Z, 1, 2|3)` shape and the `vector` intent, and it
  negates components 0 and 1 to go between ITK (LPS) and RAS.
  <https://github.com/nipy/nitransforms/blob/master/nitransforms/io/itk.py>
- ANTs writes the warps of `antsRegistration`
  (`<prefix><n>Warp.nii.gz`, `<prefix><n>InverseWarp.nii.gz`) through
  ITK's image writers, so they follow the encoding above.

Telling an ITK field from a RAS NIfTI field
-------------------------------------------
NIfTI has no field that says which frame the vector values live in, so
the header alone cannot always tell. The readers here are honest about
that:

| Header                         | Read as, without a hint                  |
| ------------------------------ | ---------------------------------------- |
| `VECTOR` (1007), `(X,Y,Z,1,3)` | ambiguous: `AmbiguousFormatError`        |
| `DISPVECT` (1006)              | a RAS NIfTI field                        |
| no intent (0)                  | a RAS NIfTI field (SPM's convention)     |

- `VECTOR` is exactly what ITK writes, but it is a generic code that any
  software may use, and ITK itself does not treat its vectors as
  spatial by default. So the ITK displacement reader and the RAS reader
  claim it with equal confidence, and loading such a file without more
  evidence raises `AmbiguousFormatError` rather than guessing a frame.
- An explicit hint decides: `load(path, hint="itk")` reads the file as
  an ITK displacement field, whatever its intent code, and
  `hint="itk.coordinates"` as an ITK coordinates field. Calling the class
  directly, `ITKNiftiDisplacementField.from_file(path)`, does the same.
- `DISPVECT` and no intent stay with the RAS readers, which is also what
  ITK 5.4 and later assumes of a `DISPVECT` file. Read with a hint, a
  three-component `DISPVECT` file has its RAS vectors converted to LPS,
  as ITK does.
- ITK and ANTs only ever write *displacements*. A field of absolute LPS
  coordinates is never claimed from the file's content, because nothing
  in a NIfTI header separates it from a displacement field.
- ANTs' file names (`*Warp.nii.gz`) are not used as evidence: they are
  a convention of one program, not of the format.

Only three-dimensional fields are supported. A two-dimensional ITK field
(`(X, Y, 1, 1, 2)`) is not claimed.
"""

__all__ = [
    "ITKNiftiCoordinatesField",
    "ITKNiftiDisplacementField",
    "ITKNiftiField",
]

from ._fields import (
    ITKNiftiCoordinatesField,
    ITKNiftiDisplacementField,
    ITKNiftiField,
)
