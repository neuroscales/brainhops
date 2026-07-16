# dependencies
import typing_extensions as _tx

# datamodel
from brainhops.datamodel import transformations as _xforms

# locals
from ._parser import TFMTransformParser
from .._xform import ITKTransform


class TFMTransform(TFMTransformParser, ITKTransform):

    @property
    def transformations(self) -> _tx.Tuple[_xforms.Transformation, ...]:
        if self._transformations is None:
            self._transformations = [t.to_transform()
                                     for t in self.transform_group]
        return self._transformations

    @transformations.setter
    def transformations(self, value: _tx.Tuple[_xforms.Transformation, ...]) -> None:
        self._transformations = value
