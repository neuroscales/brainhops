# dependencies
import typing_extensions as _tx

# datamodel
from brainhops.datamodel import transformations as _xforms

# locals
from ._parser import H5TransformParser
from .._xform import ITKTransform


class H5Transform(H5TransformParser, ITKTransform):

    @property
    def transformations(self) -> _tx.Tuple[_xforms.Transformation, ...]:
        return [t.to_transform() for t in self.transform_group]

    @transformations.setter
    def transformations(self, value: _tx.Tuple[_xforms.Transformation, ...]) -> None:
        self._transformations = value
