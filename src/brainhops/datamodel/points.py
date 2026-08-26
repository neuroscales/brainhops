# externals
import typing_extensions as _tx

from brainhops._core.typing import ArrayProtocol

# internals
from .base import DataModelBase
from .systems import CoordinateSystem
from .transformations import CoordinatesField, Sequence, Transformation


class Points(DataModelBase):
    data: ArrayProtocol
    coordinateSystem: CoordinateSystem
    coordinateTransforms: _tx.List[Transformation] = []

    def compute(self) -> ArrayProtocol:
        steps = list(self.coordinateTransforms)

        transformation = Sequence(
            transformations=[CoordinatesField(
                field=self.data, output=self.coordinateSystem)] + steps,
            output=steps[-1].output,
        ).compute()
        return transformation.to(CoordinatesField).field
