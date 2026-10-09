__all__ = ["FileBasedTransformation"]

from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import (
    FileBasedObject,
    _FileBasedModelMixin,
    format_registry,
)

from ._formats import TransformationFormat


@format_registry
class FileBasedTransformation(
    _FileBasedModelMixin, TransformationFormat, Transformation, FileBasedObject
):
    """Base class of transformations that are stored in files.

    Loading a file through this class selects the concrete format that
    matches the file and delegates the reading to it. A reader of a concrete
    format derives from the class and opts in with the `@register_format`
    decorator, after which [`load`][brainhops.io.transformations.load] can
    select that reader.
    """
