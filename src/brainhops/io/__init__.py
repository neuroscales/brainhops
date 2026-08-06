"""
This module should implement generic APIs for interacting with
transformations, images, meshes, or streamlines that are stored in
some format on disk or on the clkoud.

Each format should implement the more general
transformation/images/meshes/streamlines API that does not assume that
the data is stored at a specific path.

E.g.
```python
# -------------------
# in brainhops.images
# -------------------
class Image(Struct): ...
    data: ArrayLike
    coordinateSystem: CoordinateSystem | None = None
    coordinateTransformations: List[CoordinateTransformation] | None = None


class MultiscaleImage(Struct):
    datasets: List[Image]
    coordinateSystem: CoordinateSystem | None = None
    coordinateTransformations: List[CoordinateTransformation] | None = None

class 2DImage(Image): ...
class 3DImage(Image): ...
class ScalarField(Image): ...
class VectorField(Image): ...
class MatrixField(Image): ...

# ---------------------------
# in brainhops.io.images.base
# ---------------------------
from brainhops.images import Image

class ImageFile(Image): ...
    path: str | None = None
    fileobj: FileLike | None = None

# ----------------------------
# in brainhops.io.images.nifti
# ----------------------------
from brainhops.io.images import ImageFile, ScalarField

class NiftiImageFile(ImageFile): ...
class NiftiScalarField(NiftiImageFile, ScalarField): ...

```
"""

from . import transformations
from . import images
from . import vectors
