"""Readers and writers for ITK transform files.

| Format                   | Module  | Class                       | Writes |
| ------------------------ | ------- | --------------------------- | ------ |
| text (`.tfm`, `.txt`)    | `tfm`   | `TfmTransform`              | no     |
| HDF5 (`.h5`)             | `h5`    | `H5Transform`               | no     |
| binary MATLAB (`.mat`)   | `mat`   | `MatTransform`              | yes    |
| NIfTI warp (`.nii[.gz]`) | `nifti` | `ItkNiftiDisplacementField` | yes    |

The blocks of a `CompositeTransform` are read in application order, which is
the reverse of the file order.

## ANTs

ANTs writes its transforms with the ITK writers, so every reader accepts
`hint="ants"` as well as `hint="itk"`:

- warps (`<prefix><n>Warp.nii.gz`, `<prefix><n>InverseWarp.nii.gz`) are NIfTI
  displacement fields, in 2-D or 3-D;
- composites (`<prefix>Composite.h5`, `<prefix>InverseComposite.h5`) are HDF5
  files;
- B-splines (`BSpline.txt`) and `ConvertTransformFile` output are text files;
- linear transforms (`<prefix><n>GenericAffine.mat` and the other `.mat`
  outputs) are binary MATLAB files.

[`brainhops.io.transformations.itk.mat`][] explains how an ANTs transform list
maps to a sequence. Velocity fields (`VelocityField.nii.gz`), which need
integration, and MINC output (`--minc`) are not supported.
"""

__all__ = [
    "ItkAffineBase",
    "ItkBlockBase",
    "ItkDisplacementBase",
    "ItkPrecision",
    "ItkStruct",
    "ItkTransform",
    "ItkTransformClass",
    "mat",
    "tfm",
]

from . import mat, tfm
from ._common import (
    ItkAffineBase,
    ItkBlockBase,
    ItkDisplacementBase,
    ItkPrecision,
    ItkStruct,
    ItkTransformClass,
)
from ._xform import ItkTransform

# The HDF5 reader needs the optional h5py.
try:
    from . import h5

    __all__ += ["h5"]
except ImportError:
    pass

# The NIfTI readers need the optional nibabel.
try:
    from . import nifti

    __all__ += ["nifti"]
except ImportError:
    pass
