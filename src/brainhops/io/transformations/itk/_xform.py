from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base import TransformationFormat


class ItkTransform(_xforms.MutableSequence, TransformationFormat):
    """Base class for transformations stored in ITK files.

    An ITK file holds a chain of blocks, and the parser turns each block
    directly into a transformation, such as an
    [`ItkAffineBase`][brainhops.io.transformations.itk.ItkAffineBase] or an
    [`ItkDisplacementBase`][brainhops.io.transformations.itk.ItkDisplacementBase].
    A file may hold any number of blocks in any order, so the chain can be
    edited and the class is a
    [`MutableSequence`][brainhops.datamodel.transformations.MutableSequence].
    The class itself is not registered as a format, but its concrete
    subclasses are.
    """

    HINTS = ("itk", "ants")
