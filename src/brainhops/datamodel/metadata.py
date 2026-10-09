"""Descriptive information about an image or a transformation."""

__all__ = ["Metadata"]

from .base import DataModelBase


class Metadata(DataModelBase, kw_only=True):
    """Information that describes an image or a transformation.

    Metadata is what a file records beyond the data and its geometry, such
    as a repetition time, a description or a gradient table. The data model
    represents it in a vocabulary shared by all formats, so that it can be
    carried from one format to another.

    The class has no fields yet. The shared vocabulary is designed in
    [#233](https://github.com/neuroscales/brainhops/issues/233) and arrives
    with that work. Until then, a file format keeps what its header records
    in its own subclass of this class, which a reader of that format
    returns and which the writer of the same format reads back.
    """
