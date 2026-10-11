__all__ = ["TransformationFormat"]

import typing_extensions as tx

from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._base import (
    Format,
    _FileBasedModelMixin,
    format_registry,
)
from brainhops.io.transformations.base._conversions import (
    convert_instance,
    converts_to,
)


@format_registry
class TransformationFormat(
    _FileBasedModelMixin,
    DataModelBase,
    Format,
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

    A format builds itself from another transformation by converting it,
    so that `Format.from_any(t)`, `Format.from_instance(t)` and
    `t.to(Format)` are one conversion. The conversion runs the exact
    converter of the format when it has one, and the converter that
    relabels a transformation as the format otherwise (see
    [`relabel`][brainhops.io.transformations.base._converters.relabel]).
    A file, a mapping and an instance of the format itself are read or
    copied as the bases of this class read or copy them.
    """

    HINTS = ("xform",)

    # --- conversions --------------------------------------------------
    # A class whose data model comes before this class in its method
    # resolution order, such as the NIfTI affines, reaches the `from_instance`
    # of its data model first, so it converts in an override of its own.

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build an object from a file, a transformation or any other value.

        A transformation that is not of this format is converted exactly as
        `other.to(cls)` converts it. Anything else is read or built as
        [`Format.from_any`][brainhops.io.base.Format.from_any] does.

        Parameters
        ----------
        other : object
            A file, its content, a transformation, a mapping, or an
            instance of a similar class.
        *args : Any
            Constructor arguments. A file and a transformation that is
            converted take keyword options only.
        **kwargs : Any
            Options of the reader or of the conversion, and field values
            otherwise.

        Returns
        -------
        Self
            The new object.

        Raises
        ------
        TypeError
            If positional arguments accompany a file or a transformation
            that is converted.
        ConversionError
            If the format cannot hold the transformation.
        """
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy an instance, or convert a transformation to this format.

        A transformation that is not of this format is converted exactly as
        `other.to(cls)` converts it. An instance of this format is copied
        as [`Format.from_instance`][brainhops.io.base.Format.from_instance]
        copies it.

        Parameters
        ----------
        other : object
            The instance to copy or convert.
        *args : Any
            Constructor arguments. A transformation that is converted
            takes keyword options only.
        **kwargs : Any
            Options of the conversion, or field values of the copy.

        Returns
        -------
        Self
            The new object.

        Raises
        ------
        TypeError
            If positional arguments accompany a transformation that is
            converted.
        ConversionError
            If the format cannot hold the transformation.
        """
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_instance(other, *args, **kwargs)
