# dependencies
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base._base import register_format

# locals
from .._xform import ITKTransform
from ._parser import H5TransformParser


@register_format
class H5Transform(
    H5TransformParser,
    ITKTransform,
    mapping=HIDE_IF_NONE,
):
    """A transformation stored in an ITK binary (`.h5`) file."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5", ".hdf5")

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations, decoded from `transform_group`
        and cached after the first access."""
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
