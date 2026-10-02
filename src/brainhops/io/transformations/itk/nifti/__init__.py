"""
ITK nonlinear warps stored as NIfTI vector images, in LPS space.

ITK (and therefore ANTs, which writes its warps through ITK) stores a
dense displacement field as a NIfTI *vector image*. Three facts define
the encoding, and each one differs from the RAS NIfTI fields of
[`brainhops.io.transformations.nifti`][]:

1. **Layout.** The array is five-dimensional: `(X, Y, Z, 1, 3)` for a
   3-D field, `(X, Y, 1, 1, 2)` for a 2-D one -- the spatial axes,
   singletons in place of the dimensions the image does not have, and
   the vector components in the fifth axis. ITK's `NiftiImageIO` always
   writes `dim[0] = 5` for a vector image, sets `dim[4] = 1` when the
   image has fewer than four dimensions, and stores the components in
   `dim[5]`.

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
   reading and on writing. It never converts a two-component one.

So, for a point `x` of the field's grid, an ITK displacement field maps
`x_lps -> x_lps + u_lps(x)`. In RAS world, the same displacement is
`(-u_x, -u_y, u_z)`.

The same holds in 2-D. ITK negates the x and y rows of the geometry of an
image of any dimension when it writes or reads a NIfTI, so a 2-D ITK
image lives in (L, P), and a 2-D displacement `(u_x, u_y)` is
`(-u_x, -u_y)` in (R, A). Both dimensions are read by the same classes;
the endpoints are the ITK spaces of the field's dimension, `LPSmm` in
3-D and (L, P) in millimetres in 2-D -- the spaces the `.tfm` and `.h5`
readers use, so a warp and an affine from ITK name the same space.

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
  (`<prefix><n>Warp.nii.gz`, `<prefix><n>InverseWarp.nii.gz`) with an
  `itk::ImageFileWriter` of the displacement field image
  (`itk::ants::WriteTransform`, `Utilities/itkantsReadWriteTransform.h`;
  file names from `RegTypeToFileName`,
  `Examples/antsRegistrationTemplateHeader.cxx`), so they follow the
  encoding above. These readers therefore also answer to
  `hint="ants"`, exactly as they do to `hint="itk"`.

Telling an ITK field from a RAS NIfTI field
-------------------------------------------
NIfTI has no field that says which frame the vector values live in, so
the header alone cannot always tell. The readers here are honest about
that:

| Header                             | Read as, without a hint            |
| ---------------------------------- | ---------------------------------- |
| `VECTOR` (1007), ITK's layout      | ambiguous: `AmbiguousFormatError`  |
| `VECTOR` (1007), named `"Mapping"` | RAS coordinates (SPM12, brainhops) |
| `DISPVECT` (1006)                  | RAS displacements                  |
| no intent (0)                      | RAS coordinates                    |

The RAS readers are `NiftiRASCoordinatesField` and
`NiftiRASDisplacementField`, in [`brainhops.io.transformations.nifti`][].

- `VECTOR` is exactly what ITK writes, but it is a generic code that any
  software may use, and ITK itself does not treat its vectors as
  spatial by default. So the ITK displacement reader and the RAS reader
  claim it with equal confidence, and loading such a file without more
  evidence raises `AmbiguousFormatError` rather than guessing a frame.
- The intent name is that evidence when it is `"Mapping"`: SPM12 names
  its `y_` deformations so, and so does brainhops' RAS coordinates
  writer, while ITK's `NiftiImageIO` never writes an intent name (and
  ANTs writes through it). The ITK reader does not claim such a file.
- An explicit hint decides: `load(path, hint="itk")` (or `"ants"`) reads
  the file as an ITK displacement field, whatever its intent code, and
  `hint="itk.coordinates"` as an ITK coordinates field. Calling the class
  directly, `ItkNiftiDisplacementField.from_file(path)`, does the same.
- `DISPVECT` and no intent stay with the RAS readers. A `DISPVECT` file
  holds RAS displacements, which is also what ITK 5.4 and later assumes
  of one. Read with a hint, a three-component `DISPVECT` file has its
  RAS vectors converted to LPS, as ITK does.
- ITK and ANTs only ever write *displacements*. A field of absolute LPS
  coordinates is never claimed from the file's content, because nothing
  in a NIfTI header separates it from a displacement field.
- ANTs' file names (`*Warp.nii.gz`) are not used as evidence: they are
  a convention of one program, not of the format.

Two- and three-dimensional fields are supported. ITK can also write a
four-dimensional vector image, with its fourth dimension in the NIfTI
time axis, but NIfTI has no geometry for that axis, and it is not read.
"""

__all__ = [
    "ItkNiftiCoordinatesField",
    "ItkNiftiDisplacementField",
    "ItkNiftiField",
]

from ._fields import (
    ItkNiftiCoordinatesField,
    ItkNiftiDisplacementField,
    ItkNiftiField,
)
