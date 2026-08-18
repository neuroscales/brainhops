# externals
import typing_extensions as _tx

from brainhops._core.bsplines import (
    pull,
    pull_affine,
)
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.hierarchy import AffineTransformation

# internals
from .base import DataModelBase
from .systems import CoordinateSystem
from .transformations import (
    Affine,
    CoordinatesField,
    Sequence,
    Transformation,
    _adapt,
    is_identity,
)


class Image(DataModelBase):
    parameter_names: _tx.ClassVar[str] = "coordinateTranforms"

    data: _tx.Optional[ArrayProtocol] = None
    coordinateSystem: _tx.Optional[CoordinateSystem] = None
    coordinateTransforms: _tx.Optional[_tx.List[Transformation]] = None

    def compute(self) -> ArrayProtocol:
        steps = []  # list(self.coordinateTransforms)

        transformation = Sequence(
            transformations=steps,
            input=steps[0].input,
            output=steps[-1].output,
        ).compute()
        return self._compute_helper(self.data, transformation)

    def _compute_helper(self,
                        data: ArrayProtocol,
                        transformation: Transformation,
                        output_space: _tx.Optional[CoordinateSystem] = None
                        ) -> ArrayProtocol:
        if output_space is None:
            output_space = self.coordinateSystem
        adapter_transformation = _adapt(transformation, CoordinatesField(
            field=data, output=output_space))
        if not is_identity(adapter_transformation):
            transformation = Sequence(transformations=[
                                      adapter_transformation, transformation],
                                      input=adapter_transformation.input,
                                      output=transformation.output).compute()
        if isinstance(transformation, Sequence):
            for i in transformation.transformations:
                data = self._compute_helper(
                    data, i, output_space=output_space)
                output_space = i.output

            return data
        if isinstance(transformation, AffineTransformation):
            affine_transformation = transformation.to(Affine)
            return pull_affine(data, affine_transformation.matrix, order=3, bound=0.0)
        coord_transform = transformation.to(CoordinatesField)
        return pull(data, coord_transform.field, 3, 0.0)


class MultiImage(Image):
    images: _tx.Optional[_tx.List[Image]] = None

    @property
    def data(self) -> ArrayProtocol:
        return self.images[0].data

    @data.setter
    def data(self, value: ArrayProtocol) -> None:
        if self.images is not None and len(self.images) > 0:
            self.images[0].data = value

    @property
    def coordinateSystem(self) -> CoordinateSystem:
        return self.images[0].coordinateSystem

    @coordinateSystem.setter
    def coordinateSystem(self, value: CoordinateSystem) -> None:
        if self.images is not None and len(self.images) > 0:
            self.images[0].coordinateSystem = value

    @property
    def coordinateTransforms(self) -> _tx.List[Transformation]:
        return self.images[0].coordinateTransforms

    @coordinateTransforms.setter
    def coordinateTransforms(self, value: _tx.List[Transformation]) -> None:
        if self.images is not None and len(self.images) > 0:
            self.images[0].coordinateTransforms = value
