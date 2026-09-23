# io
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base import FileBasedTransformation


class ITKTransform(_xforms.Sequence, FileBasedTransformation):
    """
    A transformation that is stored in an ITK file.

    ITK transforms are chains of transform blocks, so this class is a
    `Sequence`. Each block that a parser reads is itself a
    `Transformation` -- an [`ITKAffineBase`][] or an
    [`ITKDisplacementBase`][] -- so the parser stores the blocks
    directly in `transformations`, and nothing has to be converted
    afterwards.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete ITK transformations inherit from it
    and register themselves.
    """

    HINTS = ("itk",)
