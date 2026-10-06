__all__ = [
    "ConvertedFormat",
    "FileBasedTransformation",
    "WritableFileBasedTransformation",
]

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel._transformations.convert import convert
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


class ConvertedFormat:
    """
    A transformation format that other transformations are converted to.

    The converters registered into a format (`t.to(Format)`) say when a
    transformation can be held by it exactly, and refuse it otherwise.
    A format that has them builds itself from another transformation
    through them, so that `Format.from_other(t)`,
    `Format.from_instance(t)` and `t.to(Format)` are one conversion, with
    the same result and the same errors. An instance of the format
    itself is copied, as by any data model.

    It comes first in the bases of the format, ahead of the data model
    and of the file machinery that it defers to for anything else.
    """

    @classmethod
    def from_other(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create an instance from a file, a mapping, or another object.

        Another transformation is converted, as with `t.to(cls)`.
        Anything else is read as the format's other bases read it.
        """
        if _is_foreign_transformation(other, cls):
            return cls.from_instance(other, *args, **kwargs)
        return super().from_other(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create an instance from another transformation.

        Another transformation is converted, as with `t.to(cls)`: it is
        held exactly, or a `ConversionError` says why it cannot be. An
        instance of this format is copied.
        """
        if not _is_foreign_transformation(other, cls):
            return super().from_instance(other, *args, **kwargs)
        if args:
            raise TypeError(
                f"{cls.__name__}.from_instance() converts a transformation "
                f"with keyword options only, but was given {len(args)} "
                f"positional argument(s)."
            )
        return convert(other, cls, **kwargs)


def _is_foreign_transformation(other: tx.Any, cls: type) -> bool:
    """Whether `other` is a transformation that is not of format `cls`."""
    return isinstance(other, Transformation) and not isinstance(other, cls)
