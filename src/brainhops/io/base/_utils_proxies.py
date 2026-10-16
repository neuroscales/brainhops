"""How the format views turn a lazy array of a file into data."""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.dependencies import da
from brainhops._core.typing import ArrayProtocol
from brainhops._ext.npfileobj.proxy import ArrayProxy
from brainhops.backends import get_array_backend


def load_proxy(raw: ArrayProxy, **kwargs: tx.Any) -> ArrayProtocol:
    """Return the data of a lazy array with the current array backend.

    The format views call this function when the `rawdata` of an object
    is a lazy array of its file. With the Dask backend, the data is a Dask
    array whose blocks are the blocks that the storage prefers (see
    [`ArrayProxy.to_dask`][brainhops._ext.npfileobj.proxy.ArrayProxy.to_dask]),
    and nothing is read. With any other backend, the values are read. A
    NumPy array that the read gives is made read-only, because the writer
    writes `rawdata` and not the data, so an edit in place would be lost.

    Parameters
    ----------
    raw : ArrayProxy
        The lazy array.
    **kwargs : dict
        Options of `to_dask`, used only with the Dask backend.

    Returns
    -------
    ArrayProtocol
        The data, as an array of the current backend.
    """
    backend = get_array_backend()
    if da is not None and backend is da:
        return raw.to_dask(**kwargs)
    data = backend.asarray(raw)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data
