import typing_extensions as tx

_T = tx.TypeVar("_T", bound=type)


def singleton(cls: _T) -> _T:
    """Make a class a singleton, so that only one instance of it can exist.

    Calling the class again returns the existing instance. Subclassing a
    singleton raises a TypeError, because a subclass would introduce a second
    variant of an object that is meant to be unique. The orientation classes
    in this module are also frozen: every axis with a given orientation holds
    the same instance, so mutating that instance would change all of them.
    """

    registry = {}

    cls.__orig_new__ = __orig__new__ = cls.__new__

    def __new__(cls: tx.Type[tx.Self], *args, **kwargs) -> tx.Self:
        obj = registry.get(cls)
        if obj is None:
            registry[cls] = obj = __orig__new__(cls, *args, **kwargs)
        return obj

    def __init_subclass__(sub: type, **kwargs) -> None:
        raise TypeError(
            f"{cls.__name__} is a singleton: it cannot be subclassed."
        )

    cls.__new__ = __new__
    cls.__init_subclass__ = classmethod(__init_subclass__)

    return cls
