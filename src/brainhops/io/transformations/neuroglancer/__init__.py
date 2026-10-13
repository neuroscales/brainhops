"""Readers and writers of Neuroglancer coordinate transforms.

Neuroglancer, a viewer of large volumes in the browser, places each data
source of a layer with an affine coordinate transform, a JSON object
written in the state of the viewer as the `transform` of the layer
`source`. This package reads and writes that object on its own, so that
an affine computed by brainhops can be pasted into the state of a
viewer, or read back from it (#227).

The JSON is short and is read in one pass. Its whole content is held as
a [`NeuroglancerTransformRaw`][] record by
[`NeuroglancerTransformMetadata`][], and a [`NeuroglancerTransform`][]
holds that metadata. The matrix of the affine lives in the record, so
the transformation decodes its matrix and its coordinate systems from
the record, and a transformation that is read and written again without
changes is written as its file stored it.

!!! example "Exporting an affine to Neuroglancer"
    ```python
    from brainhops.datamodel.systems import RASmm
    from brainhops.datamodel.transformations import Affine
    from brainhops.io.transformations.neuroglancer import (
        NeuroglancerTransform,
    )

    affine = Affine([[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30]],
                    input=RASmm(), output=RASmm())
    text = NeuroglancerTransform.from_any(affine).to_text()
    ```
"""

__all__ = [
    "NeuroglancerTransform",
    "NeuroglancerTransformMetadata",
    "NeuroglancerTransformRaw",
]

from ._metadata import NeuroglancerTransformMetadata
from ._raw import NeuroglancerTransformRaw
from ._xform import NeuroglancerTransform

# Importing the private converters registers them with the data model,
# so that a conversion into this format, such as `t.to(Format)`, is exact
# or fails with a reason.
from . import _converters  # noqa: E402, F401  isort: skip
