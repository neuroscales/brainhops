"""
Readers and writers for ITK transformation formats.

| Format                      | Module    | Reader                      |
| --------------------------- | --------- | --------------------------- |
| text (`.tfm`, `.txt`)       | `tfm`     | `TFMTransform`              |
| HDF5 (`.h5`)                | `h5`      | `H5Transform`               |
| binary MATLAB (`.mat`)      | `mat`     | `MATTransform`              |
| NIfTI warp (`.nii[.gz]`)    | `nifti`   | `ITKNiftiDisplacementField` |

## ANTs

ANTs writes its transformations through ITK's own writers
(`itk::ants::WriteTransform` in `Utilities/itkantsReadWriteTransform.h`),
so they are ITK files, and every reader here also answers to
`hint="ants"`, exactly as it does to `hint="itk"`:

- warps, `<prefix><n>Warp.nii.gz` and `<prefix><n>InverseWarp.nii.gz`,
  are ITK NIfTI displacement fields, in 2-D or 3-D;
- composite transforms, `<prefix>Composite.h5` and
  `<prefix>InverseComposite.h5`, are ITK HDF5 files;
- B-spline (`BSpline.txt`) transforms, and the output of
  `ConvertTransformFile` in text mode, are ITK text files;
- linear transforms, `<prefix><n>GenericAffine.mat` (and `Rigid.mat`,
  `Affine.mat`, `Similarity.mat`, `Translation.mat` and
  `DerivedInitialMovingTranslation.mat`), are ITK binary MATLAB files.

Not every ANTs output can be read yet:

- time-varying velocity fields, `<prefix><n>VelocityField.nii.gz`, are
  `(D + 1)`-dimensional vector images that must be integrated, not
  displacement fields.
- with `--minc`, ANTs writes MINC `.xfm`/`.mnc` files instead.
"""

__all__ = [
    "ITKAffineBase",
    "ITKBlockBase",
    "ITKDisplacementBase",
    "ITKPrecision",
    "ITKStruct",
    "ITKTransform",
    "ITKTransformClass",
    "mat",
    "tfm",
]

from . import mat, tfm
from ._common import (
    ITKAffineBase,
    ITKBlockBase,
    ITKDisplacementBase,
    ITKPrecision,
    ITKStruct,
    ITKTransformClass,
)
from ._xform import ITKTransform

# The h5 reader needs h5py, which is optional. It is imported only when
# h5py is available, mirroring how the transformations package imports
# its own optional-dependency submodules.
try:
    from . import h5

    __all__ += ["h5"]
except ImportError:  # h5py is optional
    pass

# The NIfTI field readers need nibabel, which is optional too.
try:
    from . import nifti

    __all__ += ["nifti"]
except ImportError:  # nibabel is optional
    pass
