__all__ = ["StrEnum", "IntEnum", "Enum", "EnumConverter"]

from enum import Enum, IntEnum

import typing_extensions as tx

from .compat import StrEnum


class EnumConverter:
    """
    The converter of a free-text value that has a list of known terms.

    The known terms are the members of an enum. A string that is the
    value of a member becomes that member, any other string stays a
    string, and `None` stays `None`. A field annotated with
    `ConvertTo(EnumConverter(SpaceEnum))` therefore holds a `SpaceEnum`
    member for a known space and the plain string for any other space.

    `bagof` does not convert a `Union[Enum, str]` this way by itself. Its
    union converter returns a value that already fits one branch of the
    union as it is, so a string stays a string, and the enum branch is
    never tried. Registering a converter for the union does not help
    either, because `bagof` matches a parametrised union by equality:
    each enum, and each wrapping of its union (`Optional[...]`, or the
    metadata `Maybe[...]`, which flatten into other unions), would need a
    registration of its own, and a shape that was left out would silently
    keep the string. An explicit converter on the field cannot be missed.

    The converter is a class rather than a closure so that it has a
    readable `repr` and can be pickled, since it is stored in the
    metadata of a field.

    Examples
    --------
    ```pycon
    >>> from brainhops.datamodel.enums import SpaceEnum
    >>> convert = EnumConverter(SpaceEnum)
    >>> convert("scanner")
    <SpaceEnum.scanner: 'scanner'>
    >>> convert("my-template")
    'my-template'
    >>> convert
    EnumConverter(SpaceEnum)
    ```
    """

    __slots__ = ("enum",)

    def __init__(self, enum: tx.Type[Enum]) -> None:
        """
        Parameters
        ----------
        enum : type
            The enum whose members are the known terms.
        """
        self.enum = enum

    def __call__(self, value: tx.Any) -> tx.Any:
        """
        Convert a value.

        Parameters
        ----------
        value : Enum, str or None
            The value to convert.

        Returns
        -------
        Enum, str or None
            The member whose value is `value`, or `value` itself when it
            is `None`, a member, or a string that names no member.

        Raises
        ------
        TypeError
            If `value` is neither a member, a string nor `None`.
        """
        enum = self.enum
        if value is None or isinstance(value, enum):
            return value
        if isinstance(value, str):
            try:
                return enum(value)
            except ValueError:
                return str(value)
        raise TypeError(
            f"Expected a {enum.__name__} or a str, not {type(value).__name__}."
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.enum is other.enum

    def __hash__(self) -> int:
        return hash((type(self), self.enum))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.enum.__name__})"

    def __reduce__(self) -> tx.Tuple[tx.Any, ...]:
        return (type(self), (self.enum,))
