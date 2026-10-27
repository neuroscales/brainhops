"""Old public names and keywords that still work, with a warning."""

__all__ = ["pop_renamed_keyword", "renamed_attributes"]

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


def pop_renamed_keyword(
    kwargs: tx.Dict[str, tx.Any],
    old: str,
    new: str,
    owner: str,
    default: tx.Any = None,
    *,
    stacklevel: int = 3,
) -> tx.Any:
    """Remove a deprecated keyword argument and return its value.

    When a keyword argument is renamed, its old name stays accepted for
    one release so that calls written against it keep working. A
    function that takes `**kwargs` passes them to this function, which
    removes the old keyword, emits a `DeprecationWarning` that names the
    new one, and returns the value that was given. The caller then
    reads the value as it reads the new keyword, possibly after a
    conversion when the two keywords do not take the same values.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments of the call, changed in place.
    old : str
        The deprecated name of the keyword.
    new : str
        The name that replaces it.
    owner : str
        The name of the function, as the warning shows it, such as
        `"Transformation.to"`.
    default : Any, optional
        The value returned when `old` was not given.
    stacklevel : int, default=3
        The stack level of the warning, counted from this function, so
        that the default points at the caller of the function that
        calls it.

    Returns
    -------
    value : Any
        The value given as `old`, or `default`.

    Raises
    ------
    TypeError
        If both `old` and `new` were given.
    """
    if old not in kwargs:
        return default
    if new in kwargs:
        raise TypeError(
            f"{owner}() got both {old}= and {new}=; pass {new}= only."
        )
    warnings.warn(
        f"The {old}= keyword of {owner}() is deprecated and will be "
        f"removed in a future release; use {new}= instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return kwargs.pop(old)
