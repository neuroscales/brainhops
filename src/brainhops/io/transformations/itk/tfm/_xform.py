# dependencies
import typing_extensions as _tx

# datamodel
from brainhops.datamodel import transformations as _xforms

from .._xform import ITKTransform

# locals
from ._parser import TFMTransformParser


class TFMTransform(TFMTransformParser, ITKTransform):

    @property
    def transformations(self) -> _tx.Tuple[_xforms.Transformation, ...]:
        if self._transformations is None:
            self._transformations = [t.to_transform()
                                     for t in self.transform_group]
        return self._transformations

    @transformations.setter
    def transformations(self,
                        value: _tx.Tuple[_xforms.Transformation, ...]) -> None:
        self._transformations = value
