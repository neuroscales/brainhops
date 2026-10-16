"""How the data of a JPEG 2000 image is decoded from the samples that
Pillow decodes, and encoded back.

Pillow decodes a level as `(rows, columns)`, or `(rows, columns,
samples)` for several samples per pixel. The data of an image is indexed
`(x, y)` or `(x, y, c)`, as for every raster image, so the two functions
below swap the first two axes. The proxy that decodes the samples is
read with the array backend, which decodes it with NumPy and keeps it
lazy with Dask. The two functions are pure and invert each other, so
that `_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.base._utils_proxies import load_proxy

# this format
from ._data import _Jpeg2000Proxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the samples that Pillow decodes.

    A proxy is read with the array backend. The data is a view of the
    samples with the columns first, and a NumPy view is made read-only,
    because the writer writes the file that `rawdata` reads and not the data,
    so an edit made in place would be lost. A Dask array stays lazy.
    """
    if isinstance(raw, _Jpeg2000Proxy):
        raw = load_proxy(raw)
    perm = [1, 0] + list(range(2, len(raw.shape)))
    data = get_array_backend(raw).transpose(raw, perm)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the samples, as Pillow decodes them, from the data.

    The rows come first, then the columns, then the samples.
    """
    perm = [1, 0] + list(range(2, len(data.shape)))
    return get_array_backend(data).transpose(data, perm)
