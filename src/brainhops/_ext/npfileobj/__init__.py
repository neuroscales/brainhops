"""
Implements an array-like API for files stored contiguously behind
a file-like object, such as a memory-mapped file or a file on disk.

The idea is to implement "virtual memory mapping" for files that are
not necessarily local. E.g., any file that can be opened by fsspec.

We only need to implement "non-smart" indexing (slices, integers,
and tuples of those). Data is only loaded when required by a call to
`compute()` (like in Dask). Otherwise, slicing operates lazily.

On `compute()`, we need to find an efficient "chunking" strategy, which
balances the number of reads with the amount of data read.

nibabel's ArrayProxy is a good example of this, but it only works for
local files. We need to implement something similar, but that can work
with any file-like object. Its chunking strategy also needs to be
adapted to different fsspec backends.
"""
# TODO: almost everything!
# (I had bits of it in nitorch, but it called some nibabel functions
# under the hood, which we should avoid here).
# This is not high-priority. We can simply wrap nibabel/zarr/etc for now.
