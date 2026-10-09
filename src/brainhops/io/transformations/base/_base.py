__all__ = ["FileBasedTransformation", "WritableFileBasedTransformation"]

from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import (
    FileBasedObject,
    WritableFileBasedObject,
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


@format_registry
class WritableFileBasedTransformation(
    FileBasedTransformation, WritableFileBasedObject
):
    """File-based transformation that can also be written."""
