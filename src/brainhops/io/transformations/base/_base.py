__all__ = ["TransformationFormat"]

from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._base import (
    FileBasedObject,
    _FileBasedModelMixin,
    format_registry,
)


@format_registry
class TransformationFormat(
    _FileBasedModelMixin,
    DataModelBase,
    FileBasedObject,
    reverse=True,
    eq=False,
    kw_only=True,
):
    """Format dispatcher and common base for stored transformations.

    Loading a file through this class selects the concrete format that
    matches the file and delegates the reading to it. A reader of a concrete
    format derives from the class and opts in with the `@register_format`
    decorator, after which [`load`][brainhops.io.transformations.load] can
    select that reader.
    """

    HINTS = ("xform",)
