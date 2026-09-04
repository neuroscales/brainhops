# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import transformations as _xforms

# locals
from .._xform import ITKTransform
from ._parser import H5TransformParser


class H5Transform(H5TransformParser, ITKTransform):
    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        if self._transformations is None:
            self._transformations = [
                t.to_transform() for t in self.transform_group
            ]
        return self._transformations

    @transformations.setter
    def transformations(
        self, value: tx.Tuple[_xforms.Transformation, ...]
    ) -> None:
        self._transformations = value
