"""Computational backends.

`brainhops` supports multiple computational backends, which allow users
to make optimal user of their computational infrastructure (threads,
processes, GPUs, computing clusters, etc.).

Backends can be set globally with [`set_backend`][] or locally with the
[`backend`][] context manager.  When a backend is selected, accessing
data from any `brainhops` object triggers an implicit conversion to the
selected backend. That is, if the backend is set to `"cupy"`, accessing
an image's data will return a CuPY array, even if the image stores it as
a NumPy array internally. When the backend is set to `"native"`, the data is
returned as it is stored, without any conversion. In this case, if an
operation involves multiple arrays that live on different backends, they
are "promoted" to the same backend, according to the following rules:

- If one of the arrays is a Dask array, all arrays are promoted to Dask.
- If one of the arrays is a CuPy array, all arrays are promoted to CuPy.
- Otherwise, all arrays are promoted to NumPy.

The following backend settings are supported:

| Setting       | On CPU or GPU memory   | Eagerly or lazily        |
|---------------|------------------------|--------------------------|
| `"native"`    | as the array is        | as the array is          |
| `"auto"`      | CPU memory             | depends on array size    |
| `"auto+cupy"` | GPU memory             | depends on array size    |
| `"numpy"`     | CPU memory             | eagerly                  |
| `"cupy"`      | GPU memory             | eagerly                  |
| `"dask"`      | CPU memory             | lazily, in NumPy chunks  |
| `"dask+cupy"` | GPU memory             | lazily, in CuPy chunks   |

When dask is installed and the `"auto"` setting is selected, arrays that
are larger than [`get_auto_threshold`][] bytes are promoted to Dask arrays,
while smaller arrays are kept as NumPy arrays.

The `"auto+cupy"` setting behaves similarly to `"auto"`, but returns
either a CuPy array, or a Dask array with CuPy chunks.

A setting can be set by a module (`numpy`, `cupy` or `dask.array`)
instead of a name.
"""

__all__ = [
    "backends",
    "get_backend",
    "set_backend",
    "get_auto_threshold",
    "set_auto_threshold",
    "backend",
]

from brainhops._core.backends import (
    backend,
    backends,
    get_auto_threshold,
    get_backend,
    set_auto_threshold,
    set_backend,
)
