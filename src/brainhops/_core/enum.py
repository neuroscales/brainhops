__all__ = ["StrEnum", "IntEnum", "Enum", "EnumConverter"]

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

    The converter is declared on each field because the union converter of
    bagof keeps a string as it is, without trying the enum branch, and a
    converter registered for the union would need one registration per
    enum and per wrapping, such as `Optional[...]`. It is a class rather
    than a closure so that it has a readable representation and can be
    pickled with the field metadata.

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
        enum
            Enum whose members are the known terms.
        """
        self.enum = enum

    def __call__(self, value: tx.Any) -> tx.Any:
        """Convert a value to a member of the enum when possible.

        Parameters
        ----------
        value
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
