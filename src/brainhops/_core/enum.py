__all__ = ["StrEnum", "IntEnum", "Enum", "enum_name"]

from enum import Enum, IntEnum

import typing_extensions as tx

from .compat import StrEnum


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
