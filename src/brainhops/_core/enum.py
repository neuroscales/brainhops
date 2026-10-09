__all__ = ["StrEnum", "IntEnum", "Enum", "EnumConverter", "enum_name"]

from enum import Enum, IntEnum

import typing_extensions as tx

from .compat import StrEnum


class EnumConverter:
    """Converter from free text to the known terms of an enum.

    A string that equals the value of a member of the enum is converted to
    that member. Any other string is kept as a string, and `None` is kept as
    `None`. A field annotated with `ConvertTo(EnumConverter(SpaceEnum))`
    therefore holds a `SpaceEnum` member when the text names a known space,
    and a plain string otherwise.

    The converter is declared on each field for two reasons. The union
    converter of bagof keeps a string as it is, without trying the enum
    branch of the union. A converter registered for the union type instead
    would need one registration for each enum and for each wrapper around
    it, such as `Optional[...]`. The converter is a class rather than a
    closure so that it has a readable representation and can be pickled
    with the field metadata.

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
        """Create a converter for an enum.

        Parameters
        ----------
        enum : type of Enum
            Enum whose members are the known terms.
        """
        self.enum = enum

    def __call__(self, value: tx.Any) -> tx.Any:
        """Convert a value to a member of the enum when possible.

        Parameters
        ----------
        value : Any
            A member of the enum, a string, or `None`.

        Returns
        -------
        Enum or str or None
            The member whose value equals `value`. A member, `None`, and a
            string that names no member are returned unchanged.

        Raises
        ------
        TypeError
            If `value` is not a member of the enum, a string, or `None`.
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


def enum_name(value: tx.Union[Enum, int, str]) -> str:
    """Return the name of an enum member, or the text of a plain value.

    This function is meant for error messages. Formatting an enum that
    derives from `int` or `str` in an f-string gives its value on older
    versions of Python and its qualified name on newer ones, so reading
    the name explicitly keeps a message identical on every version.

    Parameters
    ----------
    value : Enum or int or str
        An enum member, or a plain value that was not converted to one.

    Returns
    -------
    name : str
        The name of the member, or the value converted to text.

    Examples
    --------
    ```pycon
    >>> from enum import IntEnum
    >>> class Color(IntEnum):
    ...     RED = 1
    >>> enum_name(Color.RED)
    'RED'
    >>> enum_name(7)
    '7'
    ```
    """
    return value.name if isinstance(value, Enum) else str(value)
