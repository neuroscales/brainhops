# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.transformations.base import FileBasedTransformation


class ITKTransform(_xforms.Sequence, FileBasedTransformation):
    """
    A transformation that is stored in an ITK file.

    ITK transforms are chains of transform blocks, so this class is a
    `Sequence`. Concrete subclasses provide the `transform_group`
    attribute, and each block turns into a `Transformation` through its
    `to_transform` method.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete ITK transformations inherit from it
    and register themselves.
    """

    HINTS = ("itk",)

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations, decoded from `transform_group`."""
        return [t.to_transform() for t in self.transform_group]

    @transformations.setter
    def transformations(
        self, value: tx.Tuple[_xforms.Transformation, ...]
    ) -> None:
        self._transformations = value
