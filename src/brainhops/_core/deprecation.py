"""Old public names that still import, with a warning, after a rename."""

__all__ = ["renamed_attributes"]

import sys
import warnings

import typing_extensions as tx


def renamed_attributes(
    module: str, renamed: tx.Mapping[str, str]
) -> tx.Callable[[str], tx.Any]:
    """Return a module `__getattr__` that serves the old names of objects.

    When a public object is renamed, its old name stays importable for
    one release so that code written against the old name keeps working.
    A module assigns the returned function to its own `__getattr__`.
    Python calls that function only for a name that the module does not
    define, and the function then emits a `DeprecationWarning` and
    returns the object under its new name.

    Parameters
    ----------
    module : str
        The name of the module, usually `__name__`.
    renamed : Mapping[str, str]
        The new name of each object, keyed by its old name.

    Returns
    -------
    getattr : Callable[[str], Any]
        The function to assign to the `__getattr__` of the module.
    """

    def __getattr__(name: str) -> tx.Any:
        if name not in renamed:
            raise AttributeError(
                f"module {module!r} has no attribute {name!r}"
            )
        new = renamed[name]
        warnings.warn(
            f"{module}.{name} is deprecated and will be removed in a "
            f"future release; use {module}.{new} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(sys.modules[module], new)

    return __getattr__
