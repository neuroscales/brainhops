import typing_extensions as _tx

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.tirl._parser import TIRLParser


class TIRLTransform(TIRLParser, _xforms.Sequence):
    """
    Base class for TIRL transformations.

    Concrete classes implement the `transform_group` attribute.
    """

    @property
    def transformations(self) -> _tx.Tuple[_xforms.Transformation, ...]:
        return self.loaded_object.to_transform().transformations

    @transformations.setter
    def transformations(
        self, value: _tx.Tuple[_xforms.Transformation, ...]
    ) -> None:
        NotImplementedError("cannot set transformations from loaded file")
