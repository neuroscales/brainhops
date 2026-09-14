import typing_extensions as tx

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.tirl._parser import TIRLParser


class TIRLTransform(TIRLParser, _xforms.Sequence):
    """
    Base class for TIRL transformations.

    Concrete classes implement the `transform_group` attribute.
    """

    @property
    def transformations(self) -> tx.List[_xforms.Transformation]:
        if getattr(self, "_transformation", None) is None:
            self._tranformation = (
                self.loaded_object.to_transform().transformations
            )
        return self._tranformation

    @transformations.setter
    def transformations(self, value: tx.List[_xforms.Transformation]) -> None:
        self._tranformation = value
