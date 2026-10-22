"""FSL FLIRT linear transforms.

A FLIRT `.mat` file holds a (4, 4) affine that maps the scaled millimetre
coordinates of the moving image to the scaled millimetre coordinates of the
reference image.

A FLIRT file is short and is read in one pass. Its matrix is held as a
[`FlirtRaw`][] record by [`FlirtMetadata`][], the metadata of the file, and
a [`FlirtTransform`][] holds that metadata. The file holds no image
geometry, so the moving and reference images that place the matrix in
world coordinates are options of the transformation, `moving=` and
`reference=`, which are never part of the record. A FLIRT transform is
written as FLIRT writes its matrices, and another affine is converted to
one when both images are given, as in
`t.to(FlirtTransform, reference=ref, moving=mov)`.
"""

__all__ = ["FlirtMetadata", "FlirtRaw", "FlirtTransform"]

from ._metadata import FlirtMetadata
from ._raw import FlirtRaw
from ._xform import FlirtTransform

# Importing the private converters registers them with the data model,
# so that a conversion into this format, `t.to(FlirtTransform)`,
# computes the FLIRT matrix from the images that it is given.
from . import _converters  # noqa: E402, F401  isort: skip
