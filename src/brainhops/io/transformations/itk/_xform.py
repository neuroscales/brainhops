# dependencies
import typing_extensions as _tx

# datamodel
from brainhops.datamodel import transformations as _xforms


class ITKTransform(_xforms.Sequence):
    """
    Base class for ITK transformations.

    Concrete classes implement the `transform_group` attribute.
    """

    @property
    def transformations(self) -> _tx.Tuple[_xforms.Transformation, ...]:
        return [t.to_transform() for t in self.transform_group]

    @transformations.setter
    def transformations(self, value: _tx.Tuple[_xforms.Transformation, ...]) -> None:
        self._transformations = value
