"""Computational backends.

`brainhops` supports multiple computational backends, which allow users
to make optimal user of their computational infrastructure (threads,
processes, GPUs, computing clusters, etc.).

Backends can be set globally with [`set_backend`][] or locally with the
[`backend`][] context manager. The setting decides in which backend a
computation runs and where its result lives, for example when an image
is resliced or when the lazy array of a file is summed. Reading the
`data` of an object does not convert it: `data` returns the array that
the object stores, and the `to_*` and `as_*` methods convert it on
request. Under the default setting, `"native"`, a computation runs where
its operands live, and operands that live in different backends are
promoted to one backend (see `promote_backends` in
`brainhops._core.backends`):

- if one of the operands is a Dask array, the computation is lazy;
- if one of the operands is on a GPU, the computation runs on the GPU;
- otherwise, the computation runs with NumPy.

The lazy array of a file counts as the array that `"auto"` would make
of it.

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
