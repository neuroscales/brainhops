"""ITK non-linear warps stored as NIfTI vector images, in LPS.

ITK and ANTs store a dense displacement field as a NIfTI vector image. The
encoding differs from the RAS fields of
[`brainhops.io.transformations.nifti`][] in three ways.

1. Layout. The image is 5-D: `(X, Y, Z, 1, 3)` in 3-D and
   `(X, Y, 1, 1, 2)` in 2-D, with singletons for the missing dimensions
   and the components on the fifth axis.
2. Intent. ITK writes `NIFTI_INTENT_VECTOR` (1007). Since ITK 5.4, it
   writes `NIFTI_INTENT_DISPVECT` (1006) only when the metadata of the
   image explicitly asks for it.
3. Frame. The header affine is the usual RAS sform or qform, but the
   vector values of a `VECTOR` file are not converted and stay in LPS.
   Since ITK 5.4, the first two components of a three-component
   `DISPVECT` file are flipped between RAS in the file and LPS in memory;
   two-component files are never converted.

An ITK field therefore maps `x_lps` to `x_lps + u_lps(x)`, and the same
displacement in RAS is `(-u_x, -u_y, u_z)`. In 2-D, ITK negates the x and
y rows of the geometry as well, so a 2-D image lives in (L, P) and the
displacement `(u_x, u_y)` is `(-u_x, -u_y)` in (R, A). The same classes
read both dimensionalities, and their endpoints are the ITK spaces used by
the `.tfm` and `.h5` readers, so that a warp and an affine from ITK name
the same space.

## Telling an ITK field from a RAS NIfTI field

The header alone cannot always tell the frame:

| Header                             | Read as, without a hint            |
| ---------------------------------- | ---------------------------------- |
| `VECTOR` (1007), ITK's layout      | ambiguous: `AmbiguousFormatError`  |
| `VECTOR` (1007), named `"Mapping"` | RAS coordinates (SPM12, brainhops) |
| `DISPVECT` (1006)                  | RAS displacements                  |
| no intent (0)                      | RAS coordinates                    |

- `VECTOR` is what ITK writes, but it is a generic code. The ITK reader
  and the RAS reader claim it equally, so loading such a file without
  further evidence raises `AmbiguousFormatError`.
- The intent name `"Mapping"`, used by SPM12 `y_` deformations and by the
  brainhops RAS writer, is such evidence, since ITK never writes an
  intent name.
- A hint decides: `load(path, hint="itk")` (or `"ants"`) reads the file as
  an ITK displacement field whatever its intent, and
  `hint="itk.coordinates"` as an ITK coordinates field. Calling
  `ItkNiftiDisplacementField.from_file(path)` does the same. A
  three-component `DISPVECT` file read this way has its RAS vectors
  converted to LPS, as ITK does.
- ITK and ANTs only write displacements, so an absolute LPS coordinate
  field is never claimed from the content alone. ANTs file names such as
  `*Warp.nii.gz` are not used as evidence either.

Fields in 2-D and 3-D are supported. ITK can also write 4-D vector images,
but these are not read, because NIfTI has no geometry for the fourth axis.

Sources
-------
- ITK `Modules/IO/NIFTI/src/itkNiftiImageIO.cxx` (`WriteImageInformation`,
  `SetNIfTIOrientationFromImageIO`, `ReadImageInformation`,
  `ConvertRASToFromLPS_*`) and `Modules/IO/NIFTI/include/itkNiftiImageIO.h`
  (`ConvertRASVectors`, off by default, and `ConvertRASDisplacementVectors`,
  on by default, both new in v5.4.0).
  <https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/IO/NIFTI/src/itkNiftiImageIO.cxx>
- NiTransforms `nitransforms/io/itk.py`, `ITKDisplacementsField`, which
  requires the `(X, Y, Z, 1, 2|3)` shape and the vector intent, and negates
  components 0 and 1 between ITK (LPS) and RAS.
  <https://github.com/nipy/nitransforms/blob/master/nitransforms/io/itk.py>
- ANTs writes the `antsRegistration` warps (`<prefix><n>Warp.nii.gz`,
  `<prefix><n>InverseWarp.nii.gz`) with `itk::ImageFileWriter`
  (`itk::ants::WriteTransform` in `Utilities/itkantsReadWriteTransform.h`),
  so they follow this encoding, and the readers answer `hint="ants"` as
  they answer `hint="itk"`.
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
