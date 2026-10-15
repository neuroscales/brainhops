"""Descriptive information about an image or a transformation."""

__all__ = ["Metadata"]

from .base import DataModelBase


class Metadata(DataModelBase, kw_only=True):
    """Information that describes an image or a transformation.

    Metadata is attached to objects that hold data, such as [`Image`][]
    and [`Transformation`][] and contains information that is not
    necessarily required for these objects to function, but that is
    useful to keep track of, and that one may want to write to disk
    along with the data itself.

    Most file formats to encode some form of metadata, and some of this
    metadata have sibling representations in other formats.
    """
