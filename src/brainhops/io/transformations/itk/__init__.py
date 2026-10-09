"""Readers and writers for ITK transform files.

| Format                   | Module  | Class                       | Writes |
| ------------------------ | ------- | --------------------------- | ------ |
| text (`.tfm`, `.txt`)    | `tfm`   | `TfmTransform`              | no     |
| HDF5 (`.h5`)             | `h5`    | `H5Transform`               | no     |
| binary MATLAB (`.mat`)   | `mat`   | `MatTransform`              | yes    |
| NIfTI warp (`.nii[.gz]`) | `nifti` | `ItkNiftiDisplacementField` | yes    |

The blocks of a `CompositeTransform` are read in the order in which they apply
to points. This order is the reverse of their order in the file, because ITK
applies the last block of a composite first.

## ANTs

ANTs writes its transforms with the ITK writers, so every reader accepts
`hint="ants"` as well as `hint="itk"`:

- warps (`<prefix><n>Warp.nii.gz`, `<prefix><n>InverseWarp.nii.gz`) are NIfTI
  displacement fields, in 2-D or 3-D;
- composites (`<prefix>Composite.h5`, `<prefix>InverseComposite.h5`) are HDF5
  files;
- B-spline transforms (`BSpline.txt`) and the text output of
  `ConvertTransformFile` are text files;
- linear transforms (`<prefix><n>GenericAffine.mat` and the other `.mat`
  outputs) are binary MATLAB files.

[`brainhops.io.transformations.itk.mat`][] explains how an ANTs transform list
maps to a sequence. Time-varying velocity fields (`VelocityField.nii.gz`) are
not supported, because they must be integrated rather than read as
displacement fields. The MINC files that ANTs writes with `--minc` are not
supported either.
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

# The converters into these formats register themselves when this module
# is imported, here, so that `t.to(Format)` refuses rather than relabels.
from . import _converters  # noqa: E402, F401  isort: skip
