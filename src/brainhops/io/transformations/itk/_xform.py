from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base import FileBasedTransformation


class ItkTransform(_xforms.MutableSequence, FileBasedTransformation):
    """Base class for transformations stored in ITK files.

    An ITK file holds a chain of blocks, each of which is parsed directly into
    a transformation. A file may hold any number of blocks in any order, so the
    chain is a
    [`MutableSequence`][brainhops.datamodel.transformations.MutableSequence].
    The class is not registered as a format; its concrete subclasses are.
    """

    HINTS = ("itk", "ants")
