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

__all__ = ["images", "transformations", "vectors"]


import os

import typing_extensions as _tx

from brainhops.datamodel.base import DataModelBase

from . import images, transformations, vectors

entries = images.image_entries + transformations.transformation_entries


def load(file_name: str, hint: _tx.Optional[str] = None) -> DataModelBase:
    base = os.path.basename(file_name)
    if hint is not None:
        hint = hint.lower().replace(" ", "_").removeprefix(".")
        for i in entries:
            if hint in i.hints:
                return i.class_value.from_file(file_name)
    for i in entries:
        if base.endswith(i.extension) and (
            i.prefix is None or base.startswith(i.prefix)
        ):
            return i.class_value.from_file(file_name)
    for i in entries:
        if i.class_value.sniff_file(file_name):
            return i.class_value.from_file(file_name)
    raise NotImplementedError(f"can't parse the file: {file_name}")
