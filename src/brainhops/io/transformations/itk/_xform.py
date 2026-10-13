from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base import TransformationFormat


class ItkTransform(_xforms.MutableSequence, TransformationFormat):
    """Base class for transformations stored in ITK files.

    An ITK file holds a chain of blocks, and the record of each format
    holds every block of its file, in the order of the file. Each block is
    read directly as a transformation, such as an
    [`ItkAffineBase`][brainhops.io.transformations.itk.ItkAffineBase] or an
    [`ItkDisplacementBase`][brainhops.io.transformations.itk.ItkDisplacementBase],
    and the chain of a format holds the blocks of one top-level transform
    of its record. A file may hold any number of blocks in any order, so
    the chain can be edited and the class is a
    [`MutableSequence`][brainhops.datamodel.transformations.MutableSequence].
    The class itself is not registered as a format, but its concrete
    subclasses are.
    """

    HINTS = ("itk", "ants")
