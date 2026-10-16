"""The registrations of BigDataViewer and BigStitcher datasets.

BigDataViewer and BigStitcher keep, in the XML file of a dataset, a list
of affine transforms for each view, and BigStitcher adds a transform to
the list each time it registers the views. A view is one setup, such as
a channel, an angle or a tile, at one time point. The layout of the file
and its conventions are described in [`brainhops.io.common.bdv`][].

```python
from brainhops import io
from brainhops.datamodel.transformations import Affine

xform = io.transformations.load("dataset.xml", setup=1, timepoint=0)
xform.transformations          # the affines, in the order applied
xform.to(Affine).matrix        # voxels of the view -> global space
xform.save("copy.xml")         # the same file, byte for byte
```

[`BdvViewRegistration`][] reads the transforms of one view as a chain of
affines. The last transform of the file is applied first, so the chain
lists them in the reverse order of the file. Its input is the voxel
system of the view, with axes x, y and z and integers at the centers of
the voxels, and its output is the global space, which has no unit,
because BigStitcher measures it in units of the smallest voxel size of
the dataset. A dataset stored in HDF5 is also an image, and
[`brainhops.io.load`][brainhops.io.load] reads it as one, with the
registration as its transformation.

A registration is written back into the same XML file: only the element
of its view changes when its chain is assigned, and the rest of the file,
with its images and its other views, is kept byte for byte.
"""

__all__ = [
    "BdvMetadata",
    "BdvRaw",
    "BdvViewRegistration",
]

from brainhops.io.common.bdv import BdvMetadata, BdvRaw

from ._xform import BdvViewRegistration

from . import _converters  # noqa: E402, F401  isort: skip
