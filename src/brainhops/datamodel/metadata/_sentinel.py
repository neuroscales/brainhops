"""The `UNSUPPORTED` sentinel and the `Maybe` value type."""

__all__ = ["ALL", "UNSUPPORTED", "Maybe", "Unsupported"]

# externals
import typing_extensions as tx


class Unsupported:
    """
    The type of [`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED].

    A field of a format that cannot store it holds `UNSUPPORTED`, which
    is distinct from `None` ("nobody set it"). There is one instance:
    it is falsy, never equal to anything but itself, and survives
    copying and pickling as the same object.
    """

    __slots__ = ()
    _instance: tx.ClassVar[tx.Optional["Unsupported"]] = None

    def __new__(cls) -> "Unsupported":
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNSUPPORTED"

    def __reduce__(self) -> str:
        # A string makes pickle (and `copy`) look the singleton up by name.
        return "UNSUPPORTED"


UNSUPPORTED = Unsupported()
"""The format cannot store this field. Singleton, falsy, not `None`."""

_T = tx.TypeVar("_T")

Maybe = tx.Union[_T, None, Unsupported]
"""
A vocabulary value: a value, `None` (unknown) or `UNSUPPORTED`.

`Maybe[float]` is `Union[float, None, Unsupported]`.
"""

ALL = "all"
"""`supports=ALL` declares that a format can store every field."""
