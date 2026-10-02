__all__ = ["FileBasedTransformation", "WritableFileBasedTransformation"]

# internals
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import (
    FileBasedObject,
    WritableFileBasedObject,
    _FileBasedModel,
    format_registry,
)

from ._formats import TransformationFormat


@format_registry
class FileBasedTransformation(
    _FileBasedModel, TransformationFormat, Transformation, FileBasedObject
):
    """
    A transformation that is stored in a file.

    This is the dispatcher for transformation formats: concrete readers
    inherit from it and opt in with `@register_format`, which is what
    makes `brainhops.io.transformations.load` work.
    """


@format_registry
class WritableFileBasedTransformation(
    FileBasedTransformation, WritableFileBasedObject
):
    """A transformation that is stored in a file and can be written."""
