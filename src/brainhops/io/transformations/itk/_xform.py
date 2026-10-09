# io
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base import FileBasedTransformation


class ItkTransform(_xforms.MutableSequence, FileBasedTransformation):
    """
    A transformation that is stored in an ITK file.

    ITK transforms are chains of transform blocks, so this class is a
    `Sequence`. Each block that a parser reads is itself a
    `Transformation` -- an [`ItkAffineBase`][] or an
    [`ItkDisplacementBase`][] -- so the parser stores the blocks
    directly in `transformations`, and nothing has to be converted
    afterwards.

    An ITK file holds any number of blocks, in any order, so the chain is
    editable: this is a
    [`MutableSequence`][brainhops.datamodel.transformations.MutableSequence],
    unlike the formats whose structure the file fixes, which are an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence].

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete ITK transformations inherit from it
    and register themselves.
    """

    HINTS = ("itk", "ants")
