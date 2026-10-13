"""Readers and writers of the affine matrices of Greedy and Convert3D.

Greedy, `c3d_affine_tool` (Convert3D) and ITK-SNAP save an affine as a
text file of sixteen numbers, the rows of a homogeneous (4, 4) matrix that
maps the RAS world coordinates of the fixed image to those of the moving
image. [`C3dAffine`][] reads and writes these files. The content of such a
file holds no marker, so the format is selected with `hint="c3d"`,
`hint="greedy"` or `hint="itksnap"`.

The file is short and is read in one pass. Its whole content is held as a
[`C3dAffineRaw`][] record by [`C3dAffineMetadata`][], the metadata of the
file, and a [`C3dAffine`][] holds that metadata. A transformation that is
read and written again without changes is written as its file stored it.

!!! example "Reading the affine of a Greedy registration"
    ```python
    from brainhops.io.transformations import load

    xform = load("affine.mat", hint="greedy")
    ```

Greedy writes its deformable warps (the options `-o` and `-oinv` of a
deformable registration) with ITK, as NIfTI vector images that hold the
displacement from each point of the fixed image to the moving image, in
LPS millimetres, under the `VECTOR` intent (`vimg_write` in
`src/lddmm_data.cxx` and `VoxelToPhysicalWarpFunctor` in
`src/WarpFunctors.h` of <https://github.com/pyushkevich/greedy>, commit
`f5212c5`). These are the fields that
[`ItkNiftiDisplacementField`][brainhops.io.transformations.itk.nifti.ItkNiftiDisplacementField]
reads with `hint="itk"`, so this package does not read them.
"""

__all__ = [
    "C3dAffine",
    "C3dAffineMetadata",
    "C3dAffineRaw",
]

from ._affine import C3dAffine
from ._metadata import C3dAffineMetadata
from ._raw import C3dAffineRaw

# Importing the private converters registers them with the data model,
# so that a conversion into this format, such as `t.to(C3dAffine)`, is
# exact.
from . import _converters  # noqa: E402, F401  isort: skip
