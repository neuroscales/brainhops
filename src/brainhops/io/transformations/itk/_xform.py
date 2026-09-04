# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import transformations as _xforms


class ITKTransform(_xforms.Sequence):
    """
    Base class for ITK transformations.

    Concrete classes implement the `transform_group` attribute.
    """

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        return [t.to_transform() for t in self.transform_group]

    @transformations.setter
    def transformations(
        self, value: tx.Tuple[_xforms.Transformation, ...]
    ) -> None:
        self._transformations = value
