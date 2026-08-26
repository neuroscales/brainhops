import typing_extensions as _tx

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.tirl._parser import TIRLParser


class TIRLTransform(_xforms.Sequence, TIRLParser):
    """
    Base class for TIRL transformations.

    Concrete classes implement the `transform_group` attribute.
    """

    @property
    def transformations(self) -> _tx.Tuple[_xforms.Transformation, ...]:
        if self._transformations is not None:
            return self._transformations
        self._transformations = \
            self.loaded_object.to_transform().transformations
        return self._transformations

    @transformations.setter
    def transformations(self,
                        value: _tx.Tuple[_xforms.Transformation, ...]) -> None:
        self._transformations = value
