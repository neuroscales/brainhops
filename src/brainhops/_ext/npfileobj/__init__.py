"""Array-like access to data stored contiguously in a file-like object.

The goal is a form of virtual memory mapping for files that are not
necessarily local, such as any file that `fsspec` can open. Only basic
indexing (integers, slices and tuples of them) is needed. Indexing is lazy,
and data is read only by `compute()`, which must choose a chunking strategy
that balances the number of reads against the volume read. Nibabel's
`ArrayProxy` follows a similar model, but only for local files.
"""
# TODO: This module is mostly unimplemented. Earlier versions in nitorch
# relied on nibabel functions, which should be avoided here.
