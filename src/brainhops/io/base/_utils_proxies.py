"""How the format views turn a lazy array of a file into data."""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.proxy import ArrayProxy
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import resolve_backend, to_backend


def load_proxy(raw: ArrayProxy, **kwargs: tx.Any) -> ArrayProtocol:
    """Return the data of a lazy array with the current backend setting.

    The format views call this function when the `rawdata` of an object
    is a lazy array of its file. The setting decides the backend (see
    [`resolve_backend`][brainhops.backends.resolve_backend]). Under the
    default setting `None`, a small array is read with NumPy and a large
    one becomes a Dask array. A Dask array has the blocks that the
    storage prefers (see
    [`ArrayProxy.to_dask`][brainhops._core.proxy.ArrayProxy.to_dask]),
    and nothing is read. A NumPy array that the read gives is made
    read-only, because the writer writes `rawdata` and not the data, so
    an edit in place would be lost.

    Parameters
    ----------
    raw : ArrayProxy
        The lazy array.
    **kwargs : dict
        Options of `to_dask`, used only when the data is a Dask array.

    Returns
    -------
    ArrayProtocol
        The data, as an array of the current backend.
    """
    target = resolve_backend(raw)
    if target in ("dask", "dask+cupy"):
        return to_backend(raw.to_dask(**kwargs), target)
    data = to_backend(raw, target)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data
