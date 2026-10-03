"""
Readers and writers for ITK transformation formats.

| Format                   | Module  | Class                       | Writes |
| ------------------------ | ------- | --------------------------- | ------ |
| text (`.tfm`, `.txt`)    | `tfm`   | `TfmTransform`              | no     |
| HDF5 (`.h5`)             | `h5`    | `H5Transform`               | no     |
| binary MATLAB (`.mat`)   | `mat`   | `MatTransform`              | yes    |
| NIfTI warp (`.nii[.gz]`) | `nifti` | `ItkNiftiDisplacementField` | yes    |

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

How an ANTs transform list (`-t A -t B`, `[file.mat,1]`) maps onto a
brainhops `Sequence` is described in
[`brainhops.io.transformations.itk.mat`][].

The blocks of a `CompositeTransform` (in any ITK file) are read in the
order they apply to points, which is the reverse of their order in the
file: ITK applies the last block of a composite first.

## Metadata

`.tfm` and `.mat` files store no metadata: their `metadata` is an
[`ItkMetadata`][], for which every field is unsupported, and a `.mat`
write reports any field that was set as lost. An `.h5` file records the
version of ITK that wrote it, read as `generated_by`
([`ItkH5Metadata`][]). The blocks of a chain have no metadata of their
own: composition does not merge. NIfTI warps carry the metadata of their
NIfTI header.

Not every ANTs output can be read yet:

- time-varying velocity fields, `<prefix><n>VelocityField.nii.gz`, are
  `(D + 1)`-dimensional vector images that must be integrated, not
  displacement fields.
- with `--minc`, ANTs writes MINC `.xfm`/`.mnc` files instead.
"""

__all__ = [
    "ItkAffineBase",
    "ItkBlockBase",
    "ItkDisplacementBase",
    "ItkH5Metadata",
    "ItkMetadata",
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
from ._metadata import ItkH5Metadata, ItkMetadata
from ._xform import ItkTransform

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
