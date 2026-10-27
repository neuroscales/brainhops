"""Old public names and keywords that still work, with a warning."""

__all__ = ["external_stacklevel", "pop_renamed_keyword", "renamed_attributes"]

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
        The lowest stack level of the warning, counted from this
        function. The warning points at the first frame at or above it
        that is outside brainhops (see [`external_stacklevel`][]).

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
        stacklevel=external_stacklevel(stacklevel),
    )
    return kwargs.pop(old)


# The packages whose frames a warning skips, so that it points at the code
# that called them. bagof calls brainhops back from its field converters.
_INTERNAL_PACKAGES = ("brainhops", "bagof")


def external_stacklevel(stacklevel: int = 1) -> int:
    """Return the stack level of a warning about the caller's own code.

    A warning issued deep inside brainhops, such as the warning of a loss
    of metadata or of a deprecated keyword, should point at the line of
    the user's code that started the call, and the number of frames in
    between depends on the path that the call took through decorators
    and helpers. The function that issues the warning passes the result
    of this function to `warnings.warn` as its `stacklevel`. The frames
    are counted from that function, which is level 1, and the result is
    the first level, at or above `stacklevel`, whose frame belongs to
    neither brainhops nor bagof.

    Parameters
    ----------
    stacklevel : int, default=1
        The lowest level to return, counted as `warnings.warn` counts it
        from the function that calls this one.

    Returns
    -------
    int
        The stack level, or the level of the outermost frame when every
        frame belongs to brainhops or bagof.
    """
    frame = sys._getframe(1)
    level = 1
    while level < stacklevel or _is_internal(frame):
        if frame.f_back is None:
            return level
        frame = frame.f_back
        level += 1
    return level


def _is_internal(frame: tx.Any) -> bool:
    name = frame.f_globals.get("__name__") or ""
    return name.split(".", 1)[0] in _INTERNAL_PACKAGES
