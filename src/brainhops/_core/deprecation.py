"""Old public names that still import, and the stack level of warnings."""

__all__ = ["external_stacklevel", "renamed_attributes"]

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
